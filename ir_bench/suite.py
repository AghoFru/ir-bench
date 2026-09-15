"""Run explicit dataset and engine matrices and retain every failure."""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .cache import digest
from .catalog import prepare
from .dataset import dataset_paths
from .run import evaluate_file, load_engine, run, write_report


def run_suite(config, work, output, *, resume=False):
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
        if set(entry) - {
            "name",
            "source",
            "split",
            "doc_fields",
            "query_fields",
            "expected_missing_qrel_docs",
            "group",
            "expected_documents",
        }:
            raise ValueError("Unknown dataset configuration field.")
        if "group" in entry and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", entry["group"]):
            raise ValueError("Use a dataset group with letters, numbers, underscores, or hyphens.")
    for entry in engines:
        if set(entry) - {"name", "adapter", "config", "runs"}:
            raise ValueError("Unknown engine configuration field.")
        if ("adapter" in entry) == ("runs" in entry):
            raise ValueError("Select exactly one of adapter or runs for each engine.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise ValueError("Use an empty suite output directory to prevent stale reports.")
    summary = {"schema_version": 2, "complete": False, "configuration": config, "results": []}
    previous = {}
    if resume and (output / "suite.json").exists():
        saved = json.loads((output / "suite.json").read_text())
        if saved["configuration"] != config:
            raise ValueError("Resume requires the original suite configuration.")
        previous = {(row["dataset"], row["engine"]): row for row in saved["results"]}
        summary["results"] = list(previous.values())
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
                saved_row = previous.get((row["dataset"], row["engine"]))
                if saved_row and saved_row["status"] == "success":
                    verify_resume(saved_row, output, dataset, split, engine_config)
                    print(
                        f"{row['dataset']} / {row['engine']}: verified existing report", flush=True
                    )
                    continue
                print(f"{row['dataset']} / {row['engine']}: running", flush=True)
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
                    report = run(
                        engine,
                        dataset,
                        work / "cache",
                        split=split,
                        expected_missing_qrel_docs=dataset_config.get(
                            "expected_missing_qrel_docs", ()
                        ),
                        expected_documents=dataset_config.get("expected_documents"),
                        **options,
                    )
                filename = f"{row['dataset']}/{row['engine']}.json"
                report["configuration"] = {"dataset": dataset_config, "engine": engine_config}
                write_report(output / filename, report)
                row.update(
                    status="success",
                    report=filename,
                    metrics=report["metrics"],
                    report_sha256=digest(output / filename),
                )
            except Exception as error:
                row.update(status="failed", error=str(error))
            summary["results"] = [
                existing
                for existing in summary["results"]
                if (existing["dataset"], existing["engine"]) != (row["dataset"], row["engine"])
            ]
            summary["results"].append(row)
            print(f"{row['dataset']} / {row['engine']}: {row['status']}", flush=True)
            write_report(output / "suite.json", summary)
    summary["complete"] = all(row["status"] == "success" for row in summary["results"])
    summary["dataset_medians"] = dataset_medians(summary, output)
    summary["dataset_means"] = dataset_means(summary, output)
    write_report(output / "suite.json", summary)
    return 0 if summary["complete"] else 1


def verify_resume(row, output, dataset, split, engine_config):
    path = output / row["report"]
    if digest(path) != row.get("report_sha256"):
        raise ValueError("The saved report changed or has no checksum. Use a new output directory.")
    report = json.loads(path.read_text())
    if "runs" in engine_config:
        expected = {
            "run": digest(Path(engine_config["runs"][row["dataset"]])),
            "qrels": digest(dataset_paths(dataset, split)["qrels"]),
        }
        valid = report["input_files"] == expected
    else:
        expected = {name: digest(path) for name, path in dataset_paths(dataset, split).items()}
        engine = load_engine(engine_config["adapter"], engine_config.get("config", {}))
        valid = report["dataset"]["files"] == expected and report["engine"] == engine.identity()
        harness = {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        valid = valid and report["harness"] == harness
    if not valid:
        raise ValueError("Benchmark inputs changed. Use a new output directory.")


def dataset_medians(summary, output):
    return dataset_aggregate(summary, output, statistics.median)


def dataset_means(summary, output):
    return dataset_aggregate(summary, output, statistics.mean)


def dataset_aggregate(summary, output, aggregate):
    if not summary["complete"]:
        return None
    datasets = [entry["name"] for entry in summary["configuration"]["datasets"]]
    engines = [entry["name"] for entry in summary["configuration"]["engines"]]
    expected = {(dataset, engine) for dataset in datasets for engine in engines}
    found = {(row["dataset"], row["engine"]) for row in summary["results"]}
    if found != expected or len(summary["results"]) != len(expected):
        raise ValueError("Dataset summaries require one result for every dataset and system.")
    groups = defaultdict(list)
    for entry in summary["configuration"]["datasets"]:
        groups[entry.get("group", entry["name"])].append(entry["name"])
    aggregates = {}
    for engine in engines:
        by_dataset = {}
        for row in summary["results"]:
            if row["engine"] == engine:
                if row["status"] != "success":
                    raise ValueError("Dataset summaries require every comparison to succeed.")
                by_dataset[row["dataset"]] = json.loads((Path(output) / row["report"]).read_text())
        reports = [group_report([by_dataset[name] for name in names]) for names in groups.values()]
        measures = set(reports[0]["metrics"])
        if any(set(report["metrics"]) != measures for report in reports):
            raise ValueError("Dataset summaries require the same measures for every dataset.")
        aggregates[engine] = {
            "metrics": {
                measure: aggregate(report["metrics"][measure] for report in reports)
                for measure in sorted(measures)
            },
            "p50_us": aggregate(report["latency"]["p50_us"] for report in reports)
            if all(report["latency"] is not None for report in reports)
            else None,
            "ingestion_seconds": aggregate(report["build"]["build_seconds"] for report in reports)
            if all(report["build"] is not None for report in reports)
            else None,
        }
        if aggregate is statistics.mean:
            aggregates[engine]["mean_us"] = (
                statistics.mean(report["latency"]["mean_us"] for report in reports)
                if all(report["latency"] and "mean_us" in report["latency"] for report in reports)
                else None
            )
    result = {"datasets": list(groups), "weighting": "equal per dataset", "engines": aggregates}
    if any(len(names) > 1 for names in groups.values()):
        result["groups"] = dict(groups)
    return result


def group_report(reports):
    if len(reports) == 1:
        return reports[0]
    measures = set(reports[0]["metrics"])
    if any(set(report["metrics"]) != measures for report in reports):
        raise ValueError("Dataset groups require the same measures for every subset.")
    latency = None
    if all(report["latency"] is not None for report in reports):
        observations = [
            sample["latency_us"]
            for report in reports
            for sample in report["latency"]["observations"]
        ]
        latency = {
            "p50_us": statistics.median(observations),
            "mean_us": statistics.mean(observations),
        }
    return {
        "metrics": {
            measure: statistics.mean(report["metrics"][measure] for report in reports)
            for measure in measures
        },
        "latency": latency,
        "build": {"build_seconds": sum(report["build"]["build_seconds"] for report in reports)}
        if all(report["build"] is not None for report in reports)
        else None,
    }
