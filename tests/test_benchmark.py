"""Check metric failures, cache invalidation, and real engine integrations."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from ir_measures import Qrel

from ir_bench.adapters import Sift, SQLiteFTS5
from ir_bench.cache import cache_key, ensure_artifact
from ir_bench.dataset import load_qrels, validate_corpus
from ir_bench.metrics import evaluate as evaluate_run
from ir_bench.run import evaluate_file, latency_report, run
from ir_bench.trec import write_run

ROOT = Path(__file__).resolve().parents[1]


def evaluate(hits, judgments, depth=100):
    return evaluate_run(
        {"q": hits}, [Qrel("q", doc, grade) for doc, grade in judgments.items()], depth
    )["metrics"]


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.dataset = self.work / "dataset"
        shutil.copytree(ROOT / "examples" / "tiny", self.dataset)

    def test_latency_median_averages_the_two_middle_samples(self):
        samples = [{"latency_us": value} for value in (9, 1)]
        self.assertEqual(latency_report(samples, 1, 0, 0, "test")["p50_us"], 5)

    def test_metrics_include_unretrieved_judgments(self):
        scores = evaluate(["a"], {"a": 1, "b": 1})
        self.assertAlmostEqual(scores["nDCG@10"], 1 / (1 + 1 / math.log2(3)))
        self.assertEqual(scores["RR@10"], 1)
        self.assertEqual(scores["R@100"], 0.5)
        self.assertEqual(evaluate([], {"a": 1})["nDCG@10"], 0)
        self.assertEqual(evaluate(["a"], {"a": 0})["nDCG@10"], 0)

    def test_metrics_grades_cutoffs_and_invalid_results(self):
        scores = evaluate(["b", "a"], {"a": 2, "b": 1})
        self.assertAlmostEqual(scores["nDCG@10"], (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))
        hits = [str(i) for i in range(11)]
        self.assertEqual(evaluate(hits, {"10": 1})["RR@10"], 0)
        for hits, judgments in [(["a", "a"], {"a": 1}), ([None], {"a": 1}), ([], {"a": 1.5})]:
            with self.assertRaises(ValueError):
                evaluate(hits, judgments)
        with self.assertRaises(ValueError):
            evaluate([str(i) for i in range(101)], {"a": 1})

    def test_sqlite_real_search_and_cache(self):
        engine = SQLiteFTS5({})
        report = run(engine, self.dataset, self.work / "cache")
        self.assertFalse(report["build"]["cache_hit"])
        self.assertEqual(report["query_count"], 3)
        self.assertEqual(report["rankings"], {"q1": ["cat"], "q2": ["dog"], "q3": []})
        self.assertAlmostEqual(report["metrics"]["R@100"], 0.5)
        second = run(engine, self.dataset, self.work / "cache")
        self.assertTrue(second["build"]["cache_hit"])
        self.assertEqual(report["metrics"], second["metrics"])
        corpus = self.dataset / "corpus.jsonl"
        corpus.write_text(corpus.read_text().replace("kitten chases", "cat chases"))
        changed = run(engine, self.dataset, self.work / "cache")
        self.assertFalse(changed["build"]["cache_hit"])
        self.assertNotEqual(report["build"]["key"], changed["build"]["key"])

    def test_query_changes_reuse_index_but_change_report_identity(self):
        engine = SQLiteFTS5({})
        first = run(engine, self.dataset, self.work / "cache")
        queries = self.dataset / "queries.jsonl"
        queries.write_text(queries.read_text().replace('"cat"', '"kitten"'))
        second = run(engine, self.dataset, self.work / "cache")
        self.assertTrue(second["build"]["cache_hit"])
        self.assertNotEqual(first["dataset"], second["dataset"])
        self.assertEqual(second["rankings"]["q1"], ["kitten"])

    def test_self_match_exclusion_preserves_cache_and_saved_run_metrics(self):
        corpus = [{"_id": doc, "text": "cat"} for doc in ("cat", "kitten")]
        (self.dataset / "corpus.jsonl").write_text("\n".join(map(json.dumps, corpus)))
        (self.dataset / "queries.jsonl").write_text(json.dumps({"_id": "cat", "text": "cat"}))
        qrels = self.dataset / "qrels/test.tsv"
        qrels.write_text("cat 0 kitten 1\n")
        cache = self.work / "cache"
        original = run(SQLiteFTS5({}), self.dataset, cache)
        self.assertEqual(original["rankings"], {"cat": ["cat", "kitten"]})
        filtered = run(SQLiteFTS5({}), self.dataset, cache, exclude_self_matches=True, repeats=2)
        self.assertEqual(filtered["rankings"], {"cat": ["kitten"]})
        self.assertEqual(filtered["metrics"]["nDCG@10"], 1)
        self.assertTrue(filtered["build"]["cache_hit"])
        self.assertEqual(filtered["repeat_evaluations"][0]["metrics"], filtered["metrics"])
        saved = self.work / "results.trec"
        write_run(saved, original["rankings"])
        self.assertEqual(
            evaluate_file(saved, qrels, exclude_self_matches=True)["metrics"], filtered["metrics"]
        )
        empty = run(
            SQLiteFTS5({}),
            self.dataset,
            cache,
            depth=1,
            measures=["nDCG@1"],
            exclude_self_matches=True,
        )
        self.assertEqual(empty["rankings"], {"cat": []})
        self.assertEqual(empty["metrics"]["nDCG@1"], 0)
        with self.assertRaisesRegex(ValueError, "true or false"):
            run(SQLiteFTS5({}), self.dataset, cache, exclude_self_matches="false")

    def test_declared_missing_judgments_still_contribute_to_scores(self):
        qrels = self.dataset / "qrels/test.tsv"
        qrels.write_text("q1 0 cat 1\nq1 0 missing 1\nq2 0 missing 1\n")
        with self.assertRaisesRegex(ValueError, "Missing judged documents"):
            run(SQLiteFTS5({}), self.dataset, self.work / "cache")
        report = run(
            SQLiteFTS5({}),
            self.dataset,
            self.work / "cache",
            expected_missing_qrel_docs=["missing"],
        )
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["metrics"]["R@100"], 0.25)
        self.assertEqual(report["dataset"]["missing_judged_documents"], ["missing"])
        self.assertEqual(report["per_query"]["q2"]["nDCG@10"], 0)
        with validate_corpus(
            self.dataset / "corpus.jsonl", load_qrels(qrels), self.work, ["missing"]
        ) as (check, _):
            with self.assertRaisesRegex(ValueError, "Unknown corpus"):
                check(["missing"])
        with self.assertRaisesRegex(ValueError, "Missing judged documents"):
            run(
                SQLiteFTS5({}),
                self.dataset,
                self.work / "cache",
                expected_missing_qrel_docs=["missing", "unexpected"],
            )

    def test_corrupt_artifact_is_rejected(self):
        engine = SQLiteFTS5({})
        artifact, _ = ensure_artifact(engine, self.dataset / "corpus.jsonl", self.work / "cache")
        (artifact / "index.sqlite").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "damaged"):
            ensure_artifact(engine, self.dataset / "corpus.jsonl", self.work / "cache")

    def test_failed_build_cleans_stage_and_lock(self):
        corpus = self.dataset / "corpus.jsonl"
        corpus.write_text(corpus.read_text() * 2)
        with self.assertRaises(Exception):
            ensure_artifact(SQLiteFTS5({}), corpus, self.work / "cache")
        self.assertEqual(list((self.work / "cache").iterdir()), [])

    def test_failed_query_cannot_disappear_from_average(self):
        engine = SQLiteFTS5({})
        original = engine.open

        from contextlib import contextmanager

        @contextmanager
        def failing_open(artifact):
            with original(artifact) as search:

                def fail(query, depth):
                    if query == "dog":
                        raise RuntimeError("Injected failure.")
                    return search(query, depth)

                yield fail

        engine.open = failing_open
        with self.assertRaisesRegex(RuntimeError, "Query q2 failed"):
            run(engine, self.dataset, self.work / "cache")

    def test_cache_key_preserves_argument_boundaries(self):
        self.assertNotEqual(cache_key(["a b", "c"]), cache_key(["a", "b c"]))

    def test_changed_inputs_cannot_publish_a_cache_entry(self):
        engine = SQLiteFTS5({})
        original = engine.build

        def changing_build(corpus, artifact):
            original(corpus, artifact)
            corpus.write_text(corpus.read_text().replace("domestic", "sleepy"))

        engine.build = changing_build
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            ensure_artifact(engine, self.dataset / "corpus.jsonl", self.work / "cache")
        self.assertEqual(list((self.work / "cache").iterdir()), [])

    def test_cache_lock_rejects_a_concurrent_writer(self):
        engine = SQLiteFTS5({})
        corpus, cache = self.dataset / "corpus.jsonl", self.work / "cache"
        _, build = ensure_artifact(engine, corpus, cache)
        lock = cache / (build["key"] + ".lock")
        lock.mkdir()
        with self.assertRaisesRegex(RuntimeError, "locked"):
            ensure_artifact(engine, corpus, cache)
        self.assertTrue(lock.is_dir())

    def test_sift_identity_tracks_binary_model_and_build_options(self):
        binary = self.work / "sift"
        binary.write_bytes(b"engine-v1")
        model = self.work / "model"
        model.mkdir()
        for name in ("tokenizer.json", "model.safetensors"):
            (model / name).write_bytes(b"model-v1")
        config = {"binary": str(binary), "model": str(model)}
        first = Sift(config).identity()
        binary.write_bytes(b"engine-v2")
        second = Sift(config).identity()
        self.assertNotEqual(first, second)
        (model / "model.safetensors").write_bytes(b"model-v2")
        third = Sift(config).identity()
        self.assertNotEqual(second, third)
        fourth = Sift({**config, "build_args": ["--k-expand", "0"]}).identity()
        self.assertNotEqual(third, fourth)
        changed = Sift({**config, "query_params": {"blend_alpha": 0.2}})
        self.assertNotEqual(third, changed.identity())
        self.assertEqual(Sift(config).build_identity(), changed.build_identity())

    def test_missing_query_is_rejected(self):
        (self.dataset / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "cat"}))
        with self.assertRaisesRegex(ValueError, "missing queries"):
            run(SQLiteFTS5({}), self.dataset, self.work / "cache")

    @unittest.skipUnless(
        os.environ.get("SIFT_BIN") and os.environ.get("SIFT_MODEL"),
        "Set SIFT_BIN and SIFT_MODEL to run the native Sift integration.",
    )
    def test_sift_real_search_and_reuse(self):
        engine = Sift(
            {
                "binary": os.environ["SIFT_BIN"],
                "model": os.environ["SIFT_MODEL"],
                "build_args": ["--stop-df", "1.0"],
            }
        )
        report = run(engine, self.dataset, self.work / "cache")
        self.assertEqual(report["query_count"], 3)
        self.assertIn("cat", report["rankings"]["q1"])
        self.assertEqual(report["rankings"]["q2"][0], "dog")
        second = run(engine, self.dataset, self.work / "cache")
        self.assertTrue(second["build"]["cache_hit"])
        self.assertEqual(report["rankings"], second["rankings"])
        engine.query_params = {"blend_alpha": 0.2}
        changed = run(engine, self.dataset, self.work / "cache")
        self.assertTrue(changed["build"]["cache_hit"])
        self.assertNotEqual(report["engine"], changed["engine"])

        (self.dataset / "corpus.jsonl").write_text(
            "".join(
                json.dumps({"_id": f"d{number:03}", "text": "cat " * (number + 1)}) + "\n"
                for number in range(300)
            )
        )
        (self.dataset / "queries.jsonl").write_text(json.dumps({"_id": "q", "text": "cat"}) + "\n")
        (self.dataset / "qrels/test.tsv").write_text("q 0 d299 1\n")
        paged = run(engine, self.dataset, self.work / "cache", depth=250, repeats=2)
        self.assertEqual(len(paged["rankings"]["q"]), 250)

    def test_repeated_rankings_are_scored_without_using_warmup_results(self):
        class Varying(SQLiteFTS5):
            @contextmanager
            def open(self, artifact):
                calls = 0
                with super().open(artifact) as actual_search:

                    def search(query, depth):
                        nonlocal calls
                        calls += 1
                        hits = actual_search(query, depth)
                        return hits if calls % 2 else []

                    yield search

        (self.dataset / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "cat"}))
        (self.dataset / "qrels/test.tsv").write_text("q1 0 cat 1\n")
        report = run(Varying({}), self.dataset, self.work / "cache", repeats=2, warmup=1)
        self.assertEqual(report["rankings"], {"q1": []})
        self.assertEqual(report["ranking_pass"], 0)
        self.assertEqual(report["metrics"]["R@100"], 0)
        repeated = report["repeat_evaluations"][0]
        self.assertEqual(repeated["metrics"]["R@100"], 1)
        self.assertEqual(repeated["ranking_changes"], {"q1": ["cat"]})
        self.assertEqual(report["latency"]["samples"], 2)

    def test_evaluator_changes_reuse_indexes_but_reader_changes_rebuild(self):
        isolated = self.work / "isolated"
        package = isolated / "ir_bench"
        shutil.copytree(ROOT / "ir_bench", package, ignore=shutil.ignore_patterns("__pycache__"))
        program = (
            "import json,sys; from pathlib import Path; "
            "from ir_bench.adapters import SQLiteFTS5; from ir_bench.run import run; "
            "print(json.dumps(run(SQLiteFTS5({}),Path(sys.argv[1]),Path(sys.argv[2]))['build']['cache_hit']))"
        )

        def cached():
            result = subprocess.run(
                [sys.executable, "-c", program, str(self.dataset), str(self.work / "cache")],
                cwd=isolated,
                check=True,
                text=True,
                capture_output=True,
                timeout=60,
            )
            return json.loads(result.stdout)

        self.assertFalse(cached())
        with (package / "metrics.py").open("a") as stream:
            stream.write("\n# Evaluation-only change.\n")
        self.assertTrue(cached())
        with (package / "dataset.py").open("a") as stream:
            stream.write("\n# Input-reader change.\n")
        self.assertFalse(cached())


if __name__ == "__main__":
    unittest.main()
