"""Check real neural retrievers and a live Weaviate server."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ir_bench.cache import ensure_artifact
from ir_bench.dataset import documents, load_queries
from ir_bench.neural import Dense, Splade
from ir_bench.run import run
from ir_bench.weaviate import Weaviate

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "work/huggingface"))


@unittest.skipUnless(os.environ.get("IR_BENCH_NEURAL"), "Set IR_BENCH_NEURAL=1 for model checks.")
class RetrieverTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.dataset = self.work / "dataset"
        shutil.copytree(ROOT / "examples/tiny", self.dataset)
        self.corpus = self.dataset / "corpus.jsonl"
        self.cache = self.work / "cache"

    def test_dense_models_match_exhaustive_cosine_and_reuse_query_changes(self):
        import numpy as np

        for filename in ("dense-e5.json", "dense-bge.json"):
            config = json.loads((ROOT / "examples" / filename).read_text())
            config["batch_size"] = 2
            engine = Dense(config)
            report = run(engine, self.dataset, self.cache, repeats=2, warmup=1)
            rows = list(documents(self.corpus))
            model = engine.encoder.load()
            vectors = model.encode(
                [config["document_prompt"] + title + "\n" + text for _, title, text in rows],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for query_id, text in load_queries(self.dataset / "queries.jsonl").items():
                query = model.encode(
                    config["query_prompt"] + text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                scores = np.dot(vectors, query)
                expected = [
                    doc for _, doc in sorted(zip(scores, [row[0] for row in rows]), reverse=True)
                ]
                self.assertEqual(report["rankings"][query_id], expected)
            self.assertGreater(report["artifact_bytes"], 0)
            self.assertIsNotNone(report["engine"]["encoder"]["model"]["revision"])
            changed = Dense({**config, "query_prompt": "different query: "})
            self.assertEqual(engine.build_identity(), changed.build_identity())
            self.assertNotEqual(engine.identity(), changed.identity())
            self.assertTrue(run(changed, self.dataset, self.cache)["build"]["cache_hit"])

    def test_splade_matches_mlm_pooling_and_merges_sparse_shards(self):
        import numpy as np
        import torch
        from scipy.sparse import load_npz, save_npz

        config = json.loads((ROOT / "examples/splade.json").read_text())
        config["batch_size"] = 2
        engine = Splade(config)
        report = run(engine, self.dataset, self.cache, repeats=2)
        artifact, reused = ensure_artifact(engine, self.corpus, self.cache)
        self.assertTrue(reused["cache_hit"])
        rows = list(documents(self.corpus))
        model = engine.encoder.load()
        tokens = model.tokenize(["cat"])
        with torch.no_grad():
            logits = model[0].auto_model(**tokens).logits
            expected = (
                torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1)
            ).amax(dim=1)
        actual = engine.encoder.encode(["cat"], "query").toarray()
        np.testing.assert_allclose(actual, expected.cpu().numpy(), atol=1e-5)

        vectors = engine.encoder.encode(
            [title + "\n" + text for _, title, text in rows], "document"
        )
        for query_id, text in load_queries(self.dataset / "queries.jsonl").items():
            scores = (vectors @ engine.encoder.encode([text], "query").T).toarray().ravel()
            expected = [
                doc
                for score, doc in sorted(zip(scores, [row[0] for row in rows]), reverse=True)
                if score > 0
            ]
            self.assertEqual(report["rankings"][query_id], expected)

        # Split one real index to exercise the global top-k merge across persisted shards.
        matrix = load_npz(artifact / "00000000.npz")
        ids = json.loads((artifact / "00000000.json").read_text())
        for number, bounds in enumerate((slice(0, 1), slice(1, None))):
            save_npz(artifact / f"{number:08}.npz", matrix[bounds])
            (artifact / f"{number:08}.json").write_text(json.dumps(ids[bounds]))
        with engine.open(artifact) as search:
            for query_id, text in load_queries(self.dataset / "queries.jsonl").items():
                self.assertEqual(search(text, 2), report["rankings"][query_id][:2])

    @unittest.skipUnless(
        os.environ.get("IR_BENCH_WEAVIATE"), "Set IR_BENCH_WEAVIATE to a server URL."
    )
    def test_weaviate_vector_hybrid_bm25_reuse_and_remote_loss(self):
        config = json.loads((ROOT / "examples/weaviate-vector.json").read_text())
        config["url"] = os.environ["IR_BENCH_WEAVIATE"]
        engine = Weaviate(config)
        reference = run(Dense(config["encoder"]), self.dataset, self.cache)
        collection = None
        try:
            report = run(engine, self.dataset, self.cache, repeats=2)
            artifact, build = ensure_artifact(engine, self.corpus, self.cache)
            collection = json.loads((artifact / "weaviate.json").read_text())["collection"]
            self.assertEqual(report["rankings"], reference["rankings"])
            self.assertIsNone(report["artifact_bytes"])
            self.assertFalse(report["build"]["cache_hit"])
            for mode in ("bm25", "hybrid"):
                result = run(Weaviate({**config, "mode": mode}), self.dataset, self.cache)
                self.assertTrue(result["build"]["cache_hit"])
                self.assertEqual(result["build"]["key"], build["key"])
                self.assertEqual(result["rankings"]["q2"][0], "dog")
            with engine.open(artifact) as search:
                self.assertEqual(len(search('cat "} } query {', 2)), 2)
            engine.request("/v1/schema/" + collection, method="DELETE")
            collection = None
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                run(engine, self.dataset, self.cache)
        finally:
            if collection is not None:
                engine.request("/v1/schema/" + collection, method="DELETE")

    def test_invalid_retriever_settings_fail_before_download_or_indexing(self):
        for options in ({"batch_size": 0}, {"normalize": "yes"}, {"max_length": 0}):
            with self.assertRaises(ValueError):
                Dense({"model": "unused", **options})
        with self.assertRaises(ValueError):
            Splade({"model": "unused", "normalize": True})
        with self.assertRaises(ValueError):
            Weaviate({"url": "http://user:secret@localhost", "mode": "bm25"})
