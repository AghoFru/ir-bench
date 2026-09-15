"""Check standard metric agreement, common formats, and real integration boundaries."""

import gzip
import json
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ir_measures
import pytrec_eval

from ir_bench.adapters import SQLiteFTS5
from ir_bench.bridges import HTTP, Command
from ir_bench.cache import digest
from ir_bench.catalog import catalog, prepare, text_fields
from ir_bench.dataset import dataset_paths, load_qrels
from ir_bench.metrics import evaluate
from ir_bench.run import evaluate_file, run, write_report
from ir_bench.suite import dataset_medians, run_suite
from ir_bench.trec import read_run, write_run

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("IR_DATASETS_HOME", str(ROOT / "work/ir-datasets"))
os.environ.setdefault("PYTERRIER_HOME", str(ROOT / "work/pyterrier"))


class InterchangeTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.dataset = self.work / "dataset"
        shutil.copytree(ROOT / "examples/tiny", self.dataset)

    def test_matches_direct_trec_eval_for_graded_negative_tied_and_missing_results(self):
        generator = random.Random(7)
        qrels = {f"q{q}": {f"d{d}": generator.randint(-1, 4) for d in range(18)} for q in range(6)}
        qrels["zero"] = {"none": 0}
        scores = {
            f"q{q}": {f"d{d}": float(generator.randrange(5)) for d in range(14)} for q in range(5)
        }
        qrel_path = self.work / "qrels.txt"
        qrel_path.write_text(
            "".join(f"{q} 0 {d} {v}\n" for q, docs in qrels.items() for d, v in docs.items())
        )
        run_path = self.work / "run.json"
        run_path.write_text(json.dumps(scores))
        report = evaluate_file(run_path, qrel_path, 100)
        direct = pytrec_eval.RelevanceEvaluator(
            qrels, {"ndcg_cut_10", "map_cut_100", "recall_100", "P_10"}
        ).evaluate(scores)
        for query_id, values in report["per_query"].items():
            expected = direct.get(query_id, {})
            for actual, reference in [
                ("nDCG@10", "ndcg_cut_10"),
                ("AP@100", "map_cut_100"),
                ("R@100", "recall_100"),
                ("P@10", "P_10"),
            ]:
                self.assertAlmostEqual(values[actual], expected.get(reference, 0.0))
        self.assertEqual(report["query_count"], 7)
        self.assertEqual(report["evaluation"]["missing_result_queries"], ["q5", "zero"])
        self.assertIsNone(report["latency"])
        self.assertIsNone(report["build"])

    def test_hand_calculated_gain_and_relevance_threshold(self):
        qrels = [ir_measures.Qrel("q", "high", 2), ir_measures.Qrel("q", "low", 1)]
        result = evaluate(
            {"q": ["low", "high"]}, qrels, measures=["nDCG@10", "RR(rel=2)@10", "R(rel=2)@10"]
        )
        self.assertAlmostEqual(
            result["metrics"]["nDCG@10"], (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
        )
        self.assertEqual(result["metrics"]["RR(rel=2)@10"], 0.5)
        self.assertEqual(result["metrics"]["R(rel=2)@10"], 1.0)

    def test_trec_scores_override_rank_column_and_invalid_runs_fail(self):
        path = self.work / "run.trec"
        path.write_text("q Q0 a 1 2 system\nq Q0 b 99 2 system\nq Q0 c 3 3 system\n")
        self.assertEqual(read_run(path), {"q": ["c", "b", "a"]})
        output = self.work / "roundtrip.trec"
        write_run(output, read_run(path))
        self.assertEqual(read_run(output), read_run(path))
        for invalid in [
            "q Q0 a 1 NaN x\n",
            "q Q0 a 1 1 x\nq Q0 a 2 2 x\n",
            "q Q0 a 1 1 x\nq Q0 b 2 2 y\n",
        ]:
            path.write_text(invalid)
            with self.assertRaises(ValueError):
                read_run(path)
        path = self.work / "run.json"
        path.write_text('{"q":{"a":1,"a":2}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_run(path)

    def test_tsv_compressed_inputs_and_subtopic_judgments(self):
        (self.dataset / "corpus.jsonl").unlink()
        (self.dataset / "queries.jsonl").unlink()
        (self.dataset / "qrels/test.tsv").unlink()
        with gzip.open(self.dataset / "collection.tsv.gz", "wt") as stream:
            stream.write("cat\tA cat sleeps.\tCat\ndog\tA dog plays.\tDog\n")
        (self.dataset / "queries.tsv").write_text("q\tcat\n")
        (self.dataset / "qrels.txt").write_text("q 0 cat 2\nq 0 dog -1\n")
        report = run(SQLiteFTS5({}), self.dataset, self.work / "cache")
        self.assertEqual(report["rankings"], {"q": ["cat"]})
        self.assertEqual(report["metrics"]["nDCG@10"], 1.0)
        (self.dataset / "qrels.txt").write_text("q intent-1 cat 1\nq intent-2 cat 2\n")
        rows = load_qrels(dataset_paths(self.dataset)["qrels"])
        self.assertEqual([row.iteration for row in rows], ["intent-1", "intent-2"])
        with self.assertRaisesRegex(ValueError, "diversity"):
            evaluate({"q": ["cat"]}, rows)

    def test_command_bridge_uses_a_real_external_process_and_reuses_its_index(self):
        script = ROOT / "examples/sqlite_command.py"
        engine = Command(
            {
                "identity": {"engine": "sqlite-fts5-example", "version": 1},
                "build": [sys.executable, str(script), "build", "{corpus}", "{artifact}"],
                "search": [sys.executable, str(script), "search", "{artifact}"],
                "files": [str(script)],
            }
        )
        report = run(engine, self.dataset, self.work / "cache", repeats=2, warmup=1)
        self.assertEqual(report["rankings"]["q1"], ["cat"])
        self.assertEqual(report["latency"]["samples"], 6)
        self.assertEqual(report["latency"]["warmup_passes"], 1)
        self.assertTrue(run(engine, self.dataset, self.work / "cache")["build"]["cache_hit"])

    def test_http_bridge_uses_real_search_and_does_not_claim_remote_build_or_storage(self):
        artifact = self.work / "http-index"
        artifact.mkdir()
        sqlite = SQLiteFTS5({})
        sqlite.build(self.dataset / "corpus.jsonl", artifact)

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                with sqlite.open(artifact) as search:
                    hits = search(body["query"], body["size"])
                encoded = json.dumps(
                    {"hits": {"hits": [{"_id": value} for value in hits]}}
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            engine = HTTP(
                {
                    "url": f"http://127.0.0.1:{server.server_port}/search",
                    "identity": {
                        "engine": "sqlite-fts5",
                        "revision": "test-corpus-v1",
                        "corpus_sha256": digest(self.dataset / "corpus.jsonl"),
                    },
                    "request": {"query": "{query}", "size": "{depth}"},
                    "hits_path": "hits.hits",
                    "id_field": "_id",
                }
            )
            report = run(engine, self.dataset, self.work / "cache")
            self.assertEqual(report["rankings"]["q1"], ["cat"])
            self.assertIsNone(report["build"])
            self.assertIsNone(report["artifact_bytes"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_catalog_and_explicit_field_mapping(self):
        entries = list(catalog(self.work))
        self.assertGreater(len(entries), 500)
        self.assertTrue(any(row["id"] == "cranfield" and row["qrels"] for row in entries))
        from collections import namedtuple

        record = namedtuple("Record", "title body")("Title", "Body")
        self.assertEqual(text_fields(record, ["title", "body"]), "Title\nBody")
        with self.assertRaises(ValueError):
            text_fields(record, None)

    def test_suite_retains_failures_and_never_averages_them_away(self):
        output = self.work / "suite"
        config = {
            "datasets": [{"name": "tiny", "source": str(self.dataset)}],
            "engines": [
                {"name": "sqlite", "adapter": "sqlite"},
                {"name": "broken", "adapter": "missing_module:Factory"},
            ],
        }
        self.assertEqual(run_suite(config, self.work, output), 1)
        report = json.loads((output / "suite.json").read_text())
        self.assertFalse(report["complete"])
        self.assertIsNone(report["dataset_medians"])
        self.assertEqual([row["status"] for row in report["results"]], ["success", "failed"])
        with self.assertRaisesRegex(ValueError, "empty"):
            run_suite(config, self.work, output)

    def test_dataset_medians_give_each_dataset_equal_weight(self):
        summary = {
            "complete": True,
            "configuration": {
                "datasets": [{"name": name} for name in ("a", "b", "c")],
                "engines": [{"name": "system"}],
            },
            "results": [],
        }
        for name, score, latency, count in [
            ("a", 0.1, 1, 1000),
            ("b", 0.5, 10, 1),
            ("c", 0.9, 100, 1),
        ]:
            filename = name + ".json"
            write_report(
                self.work / filename,
                {
                    "metrics": {"nDCG@10": score},
                    "query_count": count,
                    "latency": {"p50_us": latency},
                    "build": {"build_seconds": latency * 2},
                },
            )
            summary["results"].append(
                {
                    "dataset": name,
                    "engine": "system",
                    "status": "success",
                    "report": filename,
                }
            )
        result = dataset_medians(summary, self.work)
        self.assertEqual(result["datasets"], ["a", "b", "c"])
        self.assertEqual(
            result["engines"]["system"],
            {
                "metrics": {"nDCG@10": 0.5},
                "p50_us": 10,
                "ingestion_seconds": 20,
            },
        )
        summary["configuration"]["datasets"].pop()
        summary["results"].pop()
        self.assertEqual(dataset_medians(summary, self.work)["engines"]["system"]["p50_us"], 5.5)
        summary["results"].pop()
        with self.assertRaisesRegex(ValueError, "every dataset"):
            dataset_medians(summary, self.work)

    def test_atomic_report_does_not_replace_a_valid_report_with_invalid_values(self):
        path = self.work / "report.json"
        write_report(path, {"value": 1})
        with self.assertRaises(ValueError):
            write_report(path, {"value": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"value": 1})

    @unittest.skipUnless(
        os.environ.get("IR_BENCH_LIVE"), "Set IR_BENCH_LIVE=1 for catalog and Java checks."
    )
    def test_real_catalog_and_terrier_models(self):
        from ir_bench.terrier import Terrier

        dataset = prepare("irds:cranfield", ROOT / "work")
        for number, model in enumerate(["BM25", "DPH", "PL2"]):
            report = run(Terrier({"wmodel": model}), dataset, self.work / "cache")
            self.assertEqual(report["build"]["cache_hit"], number > 0)
            self.assertEqual(report["query_count"], 225)
            self.assertEqual(report["dataset"]["documents"], 1400)
            self.assertGreater(report["metrics"]["nDCG@10"], 0.1)
            self.assertTrue(
                run(Terrier({"wmodel": model}), dataset, self.work / "cache")["build"]["cache_hit"]
            )

    @unittest.skipUnless(os.environ.get("IR_BENCH_LIVE"), "Set IR_BENCH_LIVE=1 for PyTerrier.")
    def test_real_pyterrier_pipeline(self):
        from ir_bench.terrier import Pipeline

        config = json.loads((ROOT / "examples/pyterrier-pipeline.json").read_text())
        engine = Pipeline(config)
        report = run(engine, self.dataset, self.work / "cache", repeats=2)
        reference = run(SQLiteFTS5({}), self.dataset, self.work / "cache")
        self.assertEqual(report["rankings"], reference["rankings"])
        self.assertTrue(run(engine, self.dataset, self.work / "cache")["build"]["cache_hit"])

    def test_count_metrics_use_provider_aggregation(self):
        qrels = [
            ir_measures.Qrel("q1", "a", 1),
            ir_measures.Qrel("q2", "b", 1),
            ir_measures.Qrel("q2", "c", 1),
        ]
        report = evaluate(
            {"q1": ["a"], "q2": ["b", "c"]},
            qrels,
            measures=["NumQ", "NumRel", "NumRet"],
        )
        self.assertEqual(report["metrics"], {"NumQ": 2, "NumRel": 3, "NumRet": 3})
        with self.assertRaisesRegex(ValueError, "Select"):
            evaluate({}, qrels, measures=[])
