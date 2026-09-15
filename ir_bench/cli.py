"""Evaluate runs or execute reproducible engine and dataset comparisons."""

import argparse
import json
import os
import sys
from pathlib import Path

from .catalog import catalog, prepare
from .run import ADAPTERS, evaluate_file, load_engine, run, write_report
from .trec import write_run


def evaluation_options(parser):
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--measure", action="append", dest="measures")
    parser.add_argument(
        "--exclude-self-matches",
        action="store_true",
        help="Exclude results whose document ID equals the query ID, as in BEIR.",
    )
    parser.add_argument(
        "--provider", help="Explicit ir_measures provider, or its default provider chain."
    )


def dataset_options(parser):
    parser.add_argument("--dataset", required=True, help="Local directory or irds:catalog/split.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--doc-fields", nargs="+")
    parser.add_argument("--query-fields", nargs="+")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path.cwd() / "work")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "datasets", help="List catalog entries and their available components."
    )
    listing.add_argument("--match", default="")
    commands.add_parser("engines", help="List built-in engine integration paths.")
    importing = commands.add_parser(
        "prepare", help="Materialize a catalog split as portable files."
    )
    dataset_options(importing)
    live = commands.add_parser("run", help="Build or reuse an index, retrieve, and evaluate.")
    dataset_options(live)
    evaluation_options(live)
    live.add_argument("--adapter", required=True, help="Built-in alias or Python module:factory.")
    live.add_argument("--config", type=Path)
    live.add_argument("--repeats", type=int, default=1)
    live.add_argument("--warmup", type=int, default=0)
    live.add_argument("--seed", type=int, default=0)
    live.add_argument(
        "--missing-qrel-doc",
        action="append",
        default=[],
        dest="expected_missing_qrel_docs",
        help="Declare an expected absent judged document. Retain its judgments when scoring.",
    )
    live.add_argument("--output", required=True, type=Path)
    live.add_argument("--run-output", type=Path, help="Export the rankings in TREC format.")
    offline = commands.add_parser(
        "evaluate", help="Evaluate TREC or BEIR JSON results from any engine."
    )
    offline.add_argument("--run", required=True, type=Path)
    offline.add_argument("--qrels", required=True, type=Path)
    offline.add_argument("--output", required=True, type=Path)
    evaluation_options(offline)
    matrix = commands.add_parser("suite", help="Run a dataset by engine matrix.")
    matrix.add_argument("config", type=Path)
    matrix.add_argument("--output", required=True, type=Path)
    matrix.add_argument("--resume", action="store_true", help="Verify and reuse completed reports.")
    arguments = parser.parse_args(argv)
    work = arguments.work.resolve()
    os.environ.setdefault("HF_HOME", str(work / "huggingface"))
    os.environ.setdefault("PYTERRIER_HOME", str(work / "pyterrier"))
    os.environ.setdefault("IR_DATASETS_HOME", str(work / "ir-datasets"))
    try:
        if arguments.command == "datasets":
            print(json.dumps(list(catalog(work, arguments.match)), indent=2))
        elif arguments.command == "engines":
            print(json.dumps(ADAPTERS, indent=2))
        elif arguments.command == "prepare":
            print(prepare(arguments.dataset, work, arguments.doc_fields, arguments.query_fields))
        elif arguments.command == "run":
            config = json.loads(arguments.config.read_text()) if arguments.config else {}
            dataset = prepare(arguments.dataset, work, arguments.doc_fields, arguments.query_fields)
            report = run(
                load_engine(arguments.adapter, config),
                dataset,
                work / "cache",
                arguments.depth,
                split=arguments.split,
                measures=arguments.measures,
                provider=arguments.provider,
                repeats=arguments.repeats,
                warmup=arguments.warmup,
                seed=arguments.seed,
                exclude_self_matches=arguments.exclude_self_matches,
                expected_missing_qrel_docs=arguments.expected_missing_qrel_docs,
            )
            report["configuration"] = config
            report["dataset_source"] = arguments.dataset
            if arguments.run_output:
                arguments.run_output.parent.mkdir(parents=True, exist_ok=True)
                write_run(arguments.run_output, report["rankings"])
            write_report(arguments.output, report)
            print(json.dumps(report["metrics"], indent=2))
        elif arguments.command == "evaluate":
            report = evaluate_file(
                arguments.run,
                arguments.qrels,
                arguments.depth,
                arguments.measures,
                arguments.provider,
                exclude_self_matches=arguments.exclude_self_matches,
            )
            write_report(arguments.output, report)
            print(json.dumps(report["metrics"], indent=2))
        else:
            from .suite import run_suite

            return run_suite(
                json.loads(arguments.config.read_text()),
                work,
                arguments.output,
                resume=arguments.resume,
            )
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0
