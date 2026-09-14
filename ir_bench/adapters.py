"""Engine adapters use public interfaces and return ranked document identifiers."""

import json
import re
import socket
import sqlite3
import subprocess
import time
from contextlib import closing, contextmanager
from pathlib import Path
from urllib import error, request

from .cache import digest
from .dataset import documents


class SQLiteFTS5:
    def __init__(self, config):
        if config:
            raise ValueError("The SQLite FTS5 adapter does not accept configuration fields.")

    def identity(self):
        return {"engine": "sqlite-fts5", "version": sqlite3.sqlite_version, "adapter_version": 1}

    def build(self, corpus, artifact):
        with closing(sqlite3.connect(artifact / "index.sqlite")) as connection:
            with connection:
                connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
                connection.execute(
                    "CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, title, text)"
                )
                count = 0
                for doc_id, title, text in documents(corpus):
                    connection.execute("INSERT INTO ids VALUES (?)", (doc_id,))
                    connection.execute("INSERT INTO docs VALUES (?, ?, ?)", (doc_id, title, text))
                    count += 1
                if not count:
                    raise ValueError("The corpus is empty.")
                connection.execute("INSERT INTO docs(docs) VALUES ('optimize')")

    @contextmanager
    def open(self, artifact):
        uri = (artifact / "index.sqlite").resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:

            def search(query, depth):
                tokens = re.findall(r"\w+", query, flags=re.UNICODE)
                if not tokens:
                    return []
                expression = " OR ".join('"' + token + '"' for token in tokens)
                rows = connection.execute(
                    "SELECT id FROM docs WHERE docs MATCH ? "
                    "ORDER BY bm25(docs, 0, 2, 1), id LIMIT ?",
                    (expression, depth),
                )
                return [row[0] for row in rows]

            yield search


class Sift:
    def __init__(self, config):
        unknown = set(config) - {"binary", "model", "build_args", "query_params"}
        if unknown or not {"binary", "model"} <= config.keys():
            raise ValueError("Sift requires binary and model paths. Unknown configuration fields.")
        self.binary = Path(config["binary"]).resolve(strict=True)
        self.model = Path(config["model"]).resolve(strict=True)
        self.build_args = config.get("build_args", [])
        self.query_params = config.get("query_params", {})
        if not isinstance(self.build_args, list) or not all(
            isinstance(value, str) for value in self.build_args
        ):
            raise ValueError("build_args must be a list of strings.")
        reserved = {"--input", "--out", "--model", "--format", "--spell-dictionary"}
        if any(value.split("=", 1)[0] in reserved for value in self.build_args):
            raise ValueError(
                "Build arguments cannot override input, output, model, or external files."
            )
        if (
            not isinstance(self.query_params, dict)
            or {"index", "q", "k", "cache", "with_payload"} & self.query_params.keys()
        ):
            raise ValueError("Query parameters cannot override benchmark request fields.")

    def identity(self):
        return {
            "engine": "sift",
            "binary_sha256": digest(self.binary),
            "model": {
                name: digest(self.model / name) for name in ("tokenizer.json", "model.safetensors")
            },
            "build_args": self.build_args,
            "adapter_version": 1,
        }

    def build(self, corpus, artifact):
        normalized = artifact / "corpus.jsonl"
        # A unique identifier is required for a meaningful comparison across engines.
        with closing(sqlite3.connect(artifact / "validation.sqlite")) as validation:
            validation.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            with normalized.open("w", encoding="utf-8") as stream:
                count = 0
                for doc_id, title, text in documents(corpus):
                    validation.execute("INSERT INTO ids VALUES (?)", (doc_id,))
                    stream.write(json.dumps({"_id": doc_id, "title": title, "text": text}) + "\n")
                    count += 1
                if not count:
                    raise ValueError("The corpus is empty.")
        with (artifact / "build.log").open("wb") as log:
            subprocess.run(
                [
                    str(self.binary),
                    "build",
                    "--input",
                    str(normalized),
                    "--out",
                    str(artifact / "docs.sift"),
                    "--format",
                    "beir",
                    "--model",
                    str(self.model),
                    *self.build_args,
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        normalized.unlink()
        (artifact / "validation.sqlite").unlink()
        (artifact / "build.log").unlink()

    @contextmanager
    def open(self, artifact):
        # The server owns the port after startup. A bind race fails without a retry.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen(
            [
                str(self.binary),
                "serve",
                "--artifacts",
                str(artifact),
                "--bind",
                f"127.0.0.1:{port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError(f"Sift exited during startup with status {proc.returncode}.")
                try:
                    with request.urlopen(base + "/healthz", timeout=0.2):
                        break
                except (error.URLError, TimeoutError):
                    time.sleep(0.05)
            else:
                raise RuntimeError("Sift did not become ready within the startup budget.")

            def search(query, depth):
                if depth > 200:
                    raise ValueError("The Sift HTTP API supports at most 200 hits per request.")
                body = {
                    **self.query_params,
                    "index": "docs",
                    "q": query,
                    "k": depth,
                    "cache": False,
                    "with_payload": False,
                }
                req = request.Request(
                    base + "/search",
                    data=json.dumps(body).encode(),
                    headers={"content-type": "application/json"},
                )
                with request.urlopen(req, timeout=60) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("Sift response exceeds 16 MiB.")
                return [hit["doc_id"] for hit in json.loads(raw)["hits"]]

            yield search
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
