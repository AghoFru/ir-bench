"""Run dense encoders with FAISS and SPLADE with an exact sparse index."""

import heapq
import importlib.metadata
import itertools
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .cache import inventory
from .dataset import documents, identifier


def batches(rows, size):
    iterator = iter(rows)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def versions(*packages):
    return {package: importlib.metadata.version(package) for package in packages}


class Encoder:
    def __init__(self, config, kind="dense"):
        allowed = {
            "model",
            "revision",
            "device",
            "batch_size",
            "max_length",
            "query_prompt",
            "document_prompt",
            "normalize",
        }
        if set(config) - allowed or not isinstance(config.get("model"), str):
            raise ValueError(
                "Provide a model identifier or directory and supported encoder settings."
            )
        self.config = dict(config)
        self.kind = kind
        self.batch_size = config.get("batch_size", 32)
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 1024:
            raise ValueError("Encoder batch_size must be from 1 through 1024.")
        if "max_length" in config and (
            type(config["max_length"]) is not int or not 1 <= config["max_length"] <= 16384
        ):
            raise ValueError("Encoder max_length must be from 1 through 16384 tokens.")
        for name in ("query_prompt", "document_prompt", "device", "revision"):
            if name in config and not isinstance(config[name], str):
                raise ValueError(f"{name} must be a string.")
        if type(config.get("normalize", True)) is not bool:
            raise ValueError("normalize must be true or false.")
        if kind == "splade" and "normalize" in config:
            raise ValueError("SPLADE uses unnormalized dot products. Remove normalize.")
        self.local = Path(config["model"]).is_dir()
        self.source = str(Path(config["model"]).resolve()) if self.local else config["model"]
        self.revision = None
        if not self.local:
            from huggingface_hub import HfApi

            self.revision = (
                HfApi().model_info(self.source, revision=config.get("revision"), timeout=60).sha
            )
        self.model = None

    def identity(self):
        import torch

        return {
            "kind": self.kind,
            "configuration": self.config,
            "model": {"path": self.source, "files": inventory(Path(self.source))}
            if self.local
            else {"repository": self.source, "revision": self.revision},
            "versions": versions("sentence-transformers", "transformers", "torch"),
            "device": self.config.get("device", "cpu"),
            "dtype": "float32",
            "threads": torch.get_num_threads(),
        }

    def build_identity(self):
        identity = self.identity()
        identity["configuration"] = {
            key: value for key, value in self.config.items() if key != "query_prompt"
        }
        return identity

    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer, SparseEncoder

            model_type = SparseEncoder if self.kind == "splade" else SentenceTransformer
            cache = Path(os.environ.get("HF_HOME", Path.cwd() / "work/huggingface"))
            cache.mkdir(parents=True, exist_ok=True)
            self.model = model_type(
                self.source,
                revision=self.revision,
                device=self.config.get("device", "cpu"),
                cache_folder=str(cache / "hub"),
                trust_remote_code=False,
                model_kwargs={"dtype": "float32"},
            )
            if "max_length" in self.config:
                self.model.max_seq_length = self.config["max_length"]
        return self.model

    def encode(self, texts, role):
        import numpy as np

        model = self.load()
        options = {"batch_size": self.batch_size, "show_progress_bar": False}
        prompt = self.config.get(f"{role}_prompt")
        if prompt is not None:
            options["prompt"] = prompt
        encode = model.encode_query if role == "query" else model.encode_document
        if self.kind == "splade":
            from scipy.sparse import coo_matrix

            tensor = encode(texts, convert_to_sparse_tensor=True, **options).cpu().coalesce()
            coordinates, values = tensor.indices().numpy(), tensor.values().detach().numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("SPLADE returned invalid token weights.")
            return coo_matrix((values, coordinates), shape=tuple(tensor.shape)).tocsr()
        vectors = np.asarray(
            encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.config.get("normalize", True),
                **options,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("The dense encoder returned invalid vectors.")
        return np.ascontiguousarray(vectors)


class Dense:
    boundary = "query encoding and exact FAISS inner-product retrieval on CPU"

    def __init__(self, config):
        self.encoder = Encoder(config)

    def identity(self):
        return {
            "engine": "dense-faiss",
            "encoder": self.encoder.identity(),
            "index": "IndexFlatIP",
            **versions("faiss-cpu"),
        }

    def build_identity(self):
        return {**self.identity(), "encoder": self.encoder.build_identity()}

    def build(self, corpus, artifact):
        import faiss

        index = None
        with (artifact / "ids.jsonl").open("w", encoding="utf-8") as ids:
            for rows in batches(documents(corpus), self.encoder.batch_size):
                vectors = self.encoder.encode(
                    [title + "\n" + text for _, title, text in rows], "document"
                )
                if index is None:
                    index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                for doc_id, _, _ in rows:
                    ids.write(json.dumps(doc_id) + "\n")
        if index is None:
            raise ValueError("The corpus is empty.")
        faiss.write_index(index, str(artifact / "index.faiss"))

    @contextmanager
    def open(self, artifact):
        import faiss
        import numpy as np

        self.encoder.load()
        ids = [
            identifier(json.loads(line))
            for line in (artifact / "ids.jsonl").read_text().splitlines()
        ]
        index = faiss.read_index(str(artifact / "index.faiss"))
        if index.ntotal != len(ids):
            raise ValueError("FAISS vector and document counts do not match.")

        def search(query, depth):
            vector = self.encoder.encode([query], "query")
            scores, positions = index.search(vector, min(depth, len(ids)))
            if not np.isfinite(scores).all() or (positions < 0).any():
                raise ValueError("FAISS returned invalid scores or positions.")
            return [
                doc
                for _, doc in sorted(
                    (
                        (float(score), ids[int(position)])
                        for score, position in zip(scores[0], positions[0])
                    ),
                    reverse=True,
                )
            ]

        yield search


class Splade:
    boundary = "query encoding and exact sparse dot-product retrieval on CPU"

    def __init__(self, config):
        self.encoder = Encoder(config, "splade")

    def identity(self):
        return {
            "engine": "splade",
            "encoder": self.encoder.identity(),
            "index": "sharded CSC dot product",
            **versions("scipy"),
        }

    def build_identity(self):
        return {**self.identity(), "encoder": self.encoder.build_identity()}

    def build(self, corpus, artifact):
        from scipy.sparse import save_npz, vstack

        for number, rows in enumerate(batches(documents(corpus), 8192)):
            encoded = [
                self.encoder.encode([title + "\n" + text for _, title, text in batch], "document")
                for batch in batches(rows, self.encoder.batch_size)
            ]
            save_npz(artifact / f"{number:08}.npz", vstack(encoded).tocsc())
            (artifact / f"{number:08}.json").write_text(json.dumps([row[0] for row in rows]))
        if not any(artifact.glob("*.npz")):
            raise ValueError("The corpus is empty.")

    @contextmanager
    def open(self, artifact):
        import numpy as np
        from scipy.sparse import load_npz

        self.encoder.load()
        shards = []
        for path in sorted(artifact.glob("*.npz")):
            ids = [identifier(value) for value in json.loads(path.with_suffix(".json").read_text())]
            matrix = load_npz(path)
            if matrix.shape[0] != len(ids):
                raise ValueError("SPLADE vector and document counts do not match.")
            shards.append((ids, matrix))

        def search(query, depth):
            vector = self.encoder.encode([query], "query")
            found = []
            for ids, matrix in shards:
                scores = (matrix @ vector.T).tocoo()
                if not np.isfinite(scores.data).all():
                    raise ValueError("SPLADE returned nonfinite scores.")
                for row, score in zip(scores.row, scores.data):
                    if score > 0:
                        entry = (float(score), ids[int(row)])
                        if len(found) < depth:
                            heapq.heappush(found, entry)
                        elif entry > found[0]:
                            heapq.heapreplace(found, entry)
            return [doc_id for _, doc_id in sorted(found, reverse=True)]

        yield search
