"""Run retrieval through one validated evaluation and measurement path."""

import importlib
import json
import math
import os
import platform
import random
import statistics
import tempfile
import time
from pathlib import Path

from .cache import digest, ensure_artifact
from .dataset import dataset_paths, load_qrels, load_queries, validate_corpus
from .metrics import evaluate, validate_ranking
from .trec import read_run

ADAPTERS = {
    "dense": "ir_bench.neural:Dense",
    "splade": "ir_bench.neural:Splade",
    "weaviate": "ir_bench.weaviate:Weaviate",
    "sift": "ir_bench.adapters:Sift",
    "sqlite": "ir_bench.adapters:SQLiteFTS5",
    "command": "ir_bench.bridges:Command",
    "http": "ir_bench.bridges:HTTP",
    "terrier": "ir_bench.terrier:Terrier",
    "pyterrier": "ir_bench.terrier:Pipeline",
}


def load_engine(adapter, config):
    module, name = ADAPTERS.get(adapter, adapter).split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    return factory(config)


def machine():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }


def latency_report(samples, repeats, warmup, seed, boundary):
    ordered = sorted(sample["latency_us"] for sample in samples)
    return {
        "boundary": boundary,
        "concurrency": 1,
        "warmup_passes": warmup,
        "measured_passes": repeats,
        "query_order_seed": seed,
        "samples": len(samples),
        "mean_us": statistics.mean(ordered),
        "p50_us": ordered[math.ceil(len(ordered) * 0.5) - 1],
        "p95_us": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "p99_us": ordered[math.ceil(len(ordered) * 0.99) - 1],
        "stdev_us": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "observations": samples,
    }


def run(
    engine,
    dataset,
    cache,
    depth=100,
    *,
    split="test",
    measures=None,
    provider=None,
    repeats=1,
    warmup=0,
    seed=0,
):
    if not 1 <= repeats <= 100 or not 0 <= warmup <= 10:
        raise ValueError("Use 1 through 100 measured passes and 0 through 10 warmup passes.")
    paths = dataset_paths(dataset, split)
    initial = {name: digest(path) for name, path in paths.items()}
    queries, qrels = load_queries(paths["queries"]), load_qrels(paths["qrels"])
    query_ids = sorted({row.query_id for row in qrels})
    missing = set(query_ids) - queries.keys()
    if missing:
        raise ValueError(f"Relevance judgments reference missing queries: {sorted(missing)[:5]}")
    # Reject unsupported metrics before building an engine or making service calls.
    evaluate({}, qrels, depth, measures, provider)
    initial_engine = engine.identity()
    rankings, samples, repeat_evaluations = {}, [], []
    randomizer = random.Random(seed)
    with validate_corpus(paths["corpus"], qrels, Path(cache).parent) as (check_ids, count):
        artifact, build = ensure_artifact(engine, paths["corpus"], cache)
        artifacts_before = dict(build["files"])
        with engine.open(artifact) as search:
            for pass_number in range(-warmup, repeats):
                current_rankings = {}
                order = query_ids.copy()
                randomizer.shuffle(order)
                for query_id in order:
                    started = time.perf_counter_ns()
                    try:
                        hits = search(queries[query_id], depth)
                        elapsed = (time.perf_counter_ns() - started) / 1000
                        validate_ranking(hits, depth)
                        check_ids(hits)
                        if pass_number >= 0:
                            current_rankings[query_id] = hits
                    except Exception as error:
                        raise RuntimeError(
                            f"Query {query_id} failed: {error}. No aggregate was produced."
                        ) from error
                    if pass_number >= 0:
                        samples.append(
                            {
                                "query_id": query_id,
                                "pass": pass_number,
                                "latency_us": elapsed,
                                "hits": len(hits),
                            }
                        )
                if pass_number == 0:
                    rankings = current_rankings
                elif pass_number > 0:
                    result = evaluate(current_rankings, qrels, depth, measures, provider)
                    repeat_evaluations.append(
                        {
                            "pass": pass_number,
                            "metrics": result["metrics"],
                            "ranking_changes": {
                                query_id: hits
                                for query_id, hits in current_rankings.items()
                                if hits != rankings[query_id]
                            },
                        }
                    )
        from .cache import inventory

        if inventory(artifact) != artifacts_before:
            raise ValueError("The engine changed its cached artifact during retrieval.")
    if {name: digest(path) for name, path in paths.items()} != initial:
        raise ValueError("Dataset inputs changed during the benchmark.")
    if engine.identity() != initial_engine:
        raise ValueError("Engine inputs changed during the benchmark.")
    provenance = Path(dataset).parent / "provenance.json"
    return {
        "schema_version": 2,
        "engine": engine.identity(),
        "dataset": {
            "files": initial,
            "split": split,
            "documents": count,
            "provenance": json.loads(provenance.read_text()) if provenance.is_file() else None,
        },
        "machine": machine(),
        "build": None if getattr(engine, "external_index", False) else build,
        "build_identity": build["identity"],
        "retrieval_depth": depth,
        "artifact_bytes": None
        if getattr(engine, "external_index", False) or getattr(engine, "remote_index", False)
        else sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file()),
        **evaluate(rankings, qrels, depth, measures, provider),
        "latency": latency_report(
            samples,
            repeats,
            warmup,
            seed,
            getattr(
                engine,
                "boundary",
                "adapter call, including transport if present, excluding validation and evaluation",
            ),
        ),
        "rankings": rankings,
        "ranking_pass": 0,
        "repeat_evaluations": repeat_evaluations,
        "harness": {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    }


def evaluate_file(run_file, qrels_file, depth=100, measures=None, provider=None):
    initial = {"run": digest(run_file), "qrels": digest(qrels_file)}
    qrels, rankings = load_qrels(qrels_file), read_run(run_file, depth)
    result = evaluate(rankings, qrels, depth, measures, provider)
    if {"run": digest(run_file), "qrels": digest(qrels_file)} != initial:
        raise ValueError("Evaluation inputs changed while they were read.")
    return {
        "schema_version": 2,
        "input_files": initial,
        "retrieval_depth": depth,
        "ordering": "score descending, then document ID descending, rank column ignored",
        "build": None,
        "latency": None,
        "artifact_bytes": None,
        "rankings": rankings,
        **result,
    }


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="report-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    # Preserve the old module entry point while the CLI adds distinct operations.
    import sys

    from .cli import main as entry

    return entry(["run", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
