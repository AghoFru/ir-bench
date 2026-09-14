"""Run one engine and dataset through the shared evaluator."""

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from .cache import digest, ensure_artifact
from .dataset import load_qrels, load_queries
from .metrics import evaluate


def load_engine(adapter, config):
    module, name = adapter.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    return factory(config)


def run(engine, dataset, cache, depth=100):
    queries = load_queries(dataset / "queries.jsonl")
    qrels = load_qrels(dataset / "qrels" / "test.tsv")
    missing = qrels.keys() - queries.keys()
    if missing:
        raise ValueError(f"Relevance judgments reference missing queries: {sorted(missing)[:5]}")
    artifact, build = ensure_artifact(engine, dataset / "corpus.jsonl", cache)
    scores, latencies, rankings = [], [], {}
    with engine.open(artifact) as search:
        for query_id, judgments in qrels.items():
            started = time.perf_counter_ns()
            try:
                hits = search(queries[query_id], depth)
                elapsed = (time.perf_counter_ns() - started) / 1000
                scores.append(evaluate(hits, judgments, depth))
            except Exception as error:
                raise RuntimeError(
                    f"Query {query_id} failed. No aggregate was produced."
                ) from error
            latencies.append(elapsed)
            rankings[query_id] = hits
    return {
        "schema_version": 1,
        "metrics_version": "qrels-ideal-dcg-v1",
        "engine": engine.identity(),
        "dataset": {
            name: digest(dataset / name)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
        },
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "build": build,
        "artifact_bytes": sum(
            path.stat().st_size for path in artifact.rglob("*") if path.is_file()
        ),
        "query_count": len(scores),
        "metrics": {name: statistics.mean(score[name] for score in scores) for name in scores[0]},
        "latency": {
            "boundary": "adapter call, including transport when present, without result cache",
            "samples": len(latencies),
            "mean_us": statistics.mean(latencies),
            "p50_us": statistics.median(latencies),
            "p95_us": sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        },
        "rankings": rankings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Python module:factory for the engine.")
    parser.add_argument("--config", type=Path, help="Engine configuration JSON file.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=Path(__file__).resolve().parents[1] / "work")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 10 <= args.depth <= 10000:
        parser.error("depth must be from 10 through 10000")
    try:
        config = json.loads(args.config.read_text()) if args.config else {}
        engine = load_engine(args.adapter, config)
        report = run(engine, args.dataset.resolve(), args.work.resolve() / "cache", args.depth)
        report["configuration"] = config
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["metrics"], indent=2))
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
