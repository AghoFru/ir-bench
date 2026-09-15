"""Run explicit dataset and engine matrices and retain every failure."""

import json
import re
import statistics
from pathlib import Path

from .catalog import prepare
from .dataset import dataset_paths
from .run import evaluate_file, load_engine, run, write_report


def run_suite(config, work, output):
    allowed = {
        "datasets",
        "engines",
        "depth",
        "measures",
        "provider",
        "repeats",
        "warmup",
        "seed",
        "exclude_self_matches",
    }
    if set(config) - allowed:
        raise ValueError("Unknown suite configuration field.")
    datasets, engines = config["datasets"], config["engines"]
    if not datasets or not engines or len(datasets) * len(engines) > 10000:
        raise ValueError("A suite must contain from 1 through 10000 engine and dataset pairs.")
    for group in (datasets, engines):
        names = [entry["name"] for entry in group]
        if len(names) != len(set(names)) or any(
            not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", n) for n in names
        ):
            raise ValueError("Use unique names with letters, numbers, underscores, or hyphens.")
    for entry in datasets:
        if set(entry) - {"name", "source", "split", "doc_fields", "query_fields"}:
            raise ValueError("Unknown dataset configuration field.")
    for entry in engines:
        if set(entry) - {"name", "adapter", "config", "runs"}:
            raise ValueError("Unknown engine configuration field.")
        if ("adapter" in entry) == ("runs" in entry):
            raise ValueError("Select exactly one of adapter or runs for each engine.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty suite output directory to prevent stale reports.")
    summary = {"schema_version": 2, "complete": False, "configuration": config, "results": []}
    write_report(output / "suite.json", summary)
    for dataset_config in datasets:
        try:
            dataset = prepare(
                dataset_config["source"],
                work,
                dataset_config.get("doc_fields"),
                dataset_config.get("query_fields"),
            )
            dataset_error = None
        except Exception as error:
            dataset_error = str(error)
        for engine_config in engines:
            row = {"dataset": dataset_config["name"], "engine": engine_config["name"]}
            try:
                if dataset_error is not None:
                    raise ValueError(dataset_error)
                split = dataset_config.get("split", "test")
                if "runs" in engine_config:
                    run_path = Path(engine_config["runs"][dataset_config["name"]])
                    report = evaluate_file(
                        run_path,
                        dataset_paths(dataset, split)["qrels"],
                        config.get("depth", 100),
                        config.get("measures"),
                        config.get("provider"),
                        exclude_self_matches=config.get("exclude_self_matches", False),
                    )
                else:
                    engine = load_engine(engine_config["adapter"], engine_config.get("config", {}))
                    options = {
                        key: config[key]
                        for key in (
                            "depth",
                            "measures",
                            "provider",
                            "repeats",
                            "warmup",
                            "seed",
                            "exclude_self_matches",
                        )
                        if key in config
                    }
                    report = run(engine, dataset, work / "cache", split=split, **options)
                filename = f"{row['dataset']}/{row['engine']}.json"
                report["configuration"] = {"dataset": dataset_config, "engine": engine_config}
                write_report(output / filename, report)
                row.update(status="success", report=filename, metrics=report["metrics"])
            except Exception as error:
                row.update(status="failed", error=str(error))
            summary["results"].append(row)
            print(f"{row['dataset']} / {row['engine']}: {row['status']}", flush=True)
            write_report(output / "suite.json", summary)
    summary["complete"] = all(row["status"] == "success" for row in summary["results"])
    summary["dataset_medians"] = dataset_medians(summary, output)
    write_report(output / "suite.json", summary)
    return 0 if summary["complete"] else 1


def dataset_medians(summary, output):
    if not summary["complete"]:
        return None
    datasets = [entry["name"] for entry in summary["configuration"]["datasets"]]
    engines = [entry["name"] for entry in summary["configuration"]["engines"]]
    expected = {(dataset, engine) for dataset in datasets for engine in engines}
    found = {(row["dataset"], row["engine"]) for row in summary["results"]}
    if found != expected or len(summary["results"]) != len(expected):
        raise ValueError("Dataset medians require one result for every dataset and system.")
    medians = {}
    for engine in engines:
        reports = []
        for row in summary["results"]:
            if row["engine"] == engine:
                if row["status"] != "success":
                    raise ValueError("Dataset medians require every comparison to succeed.")
                reports.append(json.loads((Path(output) / row["report"]).read_text()))
        measures = set(reports[0]["metrics"])
        if any(set(report["metrics"]) != measures for report in reports):
            raise ValueError("Dataset medians require the same measures for every dataset.")
        medians[engine] = {
            "metrics": {
                measure: statistics.median(report["metrics"][measure] for report in reports)
                for measure in sorted(measures)
            },
            "p50_us": statistics.median(report["latency"]["p50_us"] for report in reports)
            if all(report["latency"] is not None for report in reports)
            else None,
            "ingestion_seconds": statistics.median(
                report["build"]["build_seconds"] for report in reports
            )
            if all(report["build"] is not None for report in reports)
            else None,
        }
    return {"datasets": datasets, "weighting": "equal per dataset", "engines": medians}
