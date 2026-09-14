"""Benchmark Weaviate BM25, vector, and hybrid search in isolated collections."""

import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from urllib import error, parse, request

from .dataset import documents, identifier
from .neural import Encoder, batches


class Weaviate:
    boundary = "query encoding when enabled, HTTP transport, and Weaviate retrieval"
    remote_index = True

    def __init__(self, config):
        allowed = {"url", "mode", "encoder", "alpha", "index", "api_key_env", "timeout"}
        if set(config) - allowed:
            raise ValueError("Unknown Weaviate configuration field.")
        self.config = dict(config)
        self.url = config.get("url", "http://127.0.0.1:8080").rstrip("/")
        address = parse.urlsplit(self.url)
        if address.scheme not in {"http", "https"} or not address.netloc:
            raise ValueError("Provide a Weaviate HTTP or HTTPS URL.")
        if address.username or address.password or address.query or address.fragment:
            raise ValueError("Use api_key_env for authentication. Remove credentials from the URL.")
        self.mode = config.get("mode", "vector")
        if self.mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("Weaviate mode must be bm25, vector, or hybrid.")
        self.alpha = config.get("alpha", 0.5)
        if type(self.alpha) not in (int, float) or not 0 <= self.alpha <= 1:
            raise ValueError("Hybrid alpha must be from 0 through 1.")
        self.timeout = float(config.get("timeout", 60))
        if not 0 < self.timeout <= 3600:
            raise ValueError("Weaviate timeout must be positive and at most 3600 seconds.")
        self.encoder = Encoder(config["encoder"]) if "encoder" in config else None
        if self.mode != "bm25" and self.encoder is None:
            raise ValueError("Vector and hybrid search require an encoder configuration.")
        self.index = {
            "distance": "cosine",
            "ef": 128,
            "efConstruction": 128,
            "maxConnections": 16,
            "flatSearchCutoff": 0,
            **config.get("index", {}),
        }
        if set(self.index) - {
            "distance",
            "ef",
            "efConstruction",
            "maxConnections",
            "flatSearchCutoff",
        }:
            raise ValueError("Unknown Weaviate HNSW setting.")
        if self.index["distance"] not in {"cosine", "dot", "l2-squared"}:
            raise ValueError("Weaviate distance must be cosine, dot, or l2-squared.")
        for key in ("ef", "efConstruction", "maxConnections", "flatSearchCutoff"):
            if type(self.index[key]) is not int or not 0 <= self.index[key] <= 100000:
                raise ValueError(f"Invalid HNSW setting: {key}.")

    def request(self, path, body=None, method=None):
        headers = {"content-type": "application/json"}
        if "api_key_env" in self.config:
            headers["authorization"] = "Bearer " + os.environ[self.config["api_key_env"]]
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        req = request.Request(self.url + path, data=encoded, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
        except error.HTTPError as failure:
            detail = failure.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Weaviate {path} returned HTTP {failure.code}: {detail}"
            ) from failure
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Weaviate response exceeds 16 MiB.")
        return json.loads(raw) if raw else None

    def graphql(self, query):
        response = self.request("/v1/graphql", {"query": query})
        if response.get("errors"):
            raise ValueError(f"Weaviate query failed: {response['errors']}")
        return response["data"]

    def identity(self):
        return {
            "engine": "weaviate",
            "version": self.request("/v1/meta")["version"],
            "configuration": self.config,
            "hnsw": self.index,
            "encoder": self.encoder.identity() if self.encoder else None,
            "hybrid_fusion": "relativeScoreFusion",
            "bm25_properties": ["title^2", "text"],
        }

    def build_identity(self):
        identity = self.identity()
        identity["configuration"] = {
            key: value
            for key, value in self.config.items()
            if key not in {"mode", "alpha", "encoder"}
        }
        identity["encoder"] = self.encoder.build_identity() if self.encoder else None
        return identity

    def build(self, corpus, artifact):
        collection = "IrBench" + uuid.uuid4().hex
        schema = {
            "class": collection,
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": self.index,
            "properties": [
                {
                    "name": "doc_id",
                    "dataType": ["text"],
                    "tokenization": "field",
                    "indexSearchable": False,
                },
                {"name": "title", "dataType": ["text"]},
                {"name": "text", "dataType": ["text"]},
            ],
        }
        self.request("/v1/schema", schema)
        try:
            count = 0
            size = self.encoder.batch_size if self.encoder else 32
            for rows in batches(documents(corpus), size):
                vectors = (
                    self.encoder.encode(
                        [title + "\n" + text for _, title, text in rows], "document"
                    )
                    if self.encoder
                    else None
                )
                objects = []
                for number, (doc_id, title, text) in enumerate(rows):
                    obj = {
                        "class": collection,
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, collection + "/" + doc_id)),
                        "properties": {"doc_id": doc_id, "title": title, "text": text},
                    }
                    if vectors is not None:
                        obj["vector"] = vectors[number].tolist()
                    objects.append(obj)
                response = self.request("/v1/batch/objects", {"objects": objects})
                if len(response) != len(objects) or any(
                    entry.get("result", {}).get("status") != "SUCCESS" for entry in response
                ):
                    raise ValueError(f"Weaviate did not index every document: {response}")
                count += len(objects)
            if not count:
                raise ValueError("The corpus is empty.")
            record = {
                "collection": collection,
                "documents": count,
                "schema": self.request("/v1/schema/" + collection),
            }
            self.verify(record)
            (artifact / "weaviate.json").write_text(json.dumps(record, indent=2) + "\n")
        except BaseException:
            self.request("/v1/schema/" + collection, method="DELETE")
            raise

    def verify(self, record):
        collection = record["collection"]
        if not re.fullmatch(r"IrBench[0-9a-f]{32}", collection):
            raise ValueError("Invalid benchmark collection identity.")
        schema = self.request("/v1/schema/" + collection)
        if schema != record["schema"]:
            raise ValueError("Weaviate collection settings changed. Rebuild the benchmark index.")
        result = self.graphql("{Aggregate{" + collection + "{meta{count}}}}")
        count = result["Aggregate"][collection][0]["meta"]["count"]
        if count != record["documents"]:
            raise ValueError("The Weaviate document count changed. Rebuild the benchmark index.")

    @contextmanager
    def open(self, artifact):
        record = json.loads((artifact / "weaviate.json").read_text())
        self.verify(record)
        collection = record["collection"]
        if self.encoder:
            self.encoder.load()

        def search(query, depth):
            quoted = json.dumps(query)
            if self.mode == "bm25":
                operator = f'bm25:{{query:{quoted},properties:["title^2","text"]}}'
            else:
                vector = json.dumps(self.encoder.encode([query], "query")[0].tolist())
                operator = f"nearVector:{{vector:{vector}}}"
                if self.mode == "hybrid":
                    operator = (
                        f"hybrid:{{query:{quoted},vector:{vector},alpha:{self.alpha},"
                        'fusionType:relativeScoreFusion,properties:["title^2","text"]}'
                    )
            score_field = "distance" if self.mode == "vector" else "score"
            rows = self.graphql(
                f"{{Get{{{collection}(limit:{depth},{operator})"
                f"{{doc_id _additional{{{score_field}}}}}}}}}"
            )["Get"][collection]
            found = []
            for row in rows:
                score = float(row["_additional"][score_field])
                if not math.isfinite(score):
                    raise ValueError("Weaviate returned a nonfinite score.")
                found.append(
                    (-score if self.mode == "vector" else score, identifier(row["doc_id"]))
                )
            return [doc_id for _, doc_id in sorted(found, reverse=True)]

        yield search
        self.verify(record)
