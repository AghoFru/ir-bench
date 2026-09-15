"""Connect Terrier weighting models and existing PyTerrier pipelines."""

import importlib
import importlib.metadata
import inspect
import math
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .cache import digest, inventory
from .dataset import documents, identifier


class Terrier:
    boundary = "PyTerrier query transform in the current process, including the Java boundary"

    def __init__(self, config):
        if set(config) - {"wmodel", "controls", "tokeniser", "stemmer", "stopwords"}:
            raise ValueError("Unknown Terrier configuration field.")
        self.config = config

    def identity(self):
        import pyterrier as pt

        return {
            "engine": "terrier",
            "pyterrier": importlib.metadata.version("pyterrier"),
            "terrier": pt.terrier.version(),
            "configuration": self.config,
            "query_input": "plain text",
        }

    def build_identity(self):
        identity = self.identity()
        identity.pop("query_input")
        identity["configuration"] = {
            key: value for key, value in self.config.items() if key not in {"wmodel", "controls"}
        }
        return identity

    def build(self, corpus, artifact):
        import pyterrier as pt

        def rows():
            for doc_id, title, text in documents(corpus):
                if len(doc_id) > 4096:
                    raise ValueError("Terrier document identifiers cannot exceed 4096 characters.")
                yield {"docno": doc_id, "text": title + "\n" + text}

        indexer = pt.terrier.IterDictIndexer(
            str(artifact / "index"),
            meta={"docno": 4096},
            stemmer=self.config.get("stemmer", "porter"),
            stopwords=self.config.get("stopwords", "terrier"),
            tokeniser=self.config.get("tokeniser", "english"),
        )
        indexer.index(rows())

    @contextmanager
    def open(self, artifact):
        import pandas as pd
        import pyterrier as pt

        index = pt.terrier.IndexFactory.of(str(artifact / "index" / "data.properties"))

        def search(query, depth):
            retriever.controls["end"] = str(depth - 1)
            # Dataset text must not activate Terrier query operators or controls.
            tokens = dict(Counter(retriever.tokeniser.getTokens(query)))
            if not tokens:
                return []
            result = retriever.transform(pd.DataFrame([{"qid": "query", "query_toks": tokens}]))
            if not all(math.isfinite(value) for value in result["score"]):
                raise ValueError("Terrier returned nonfinite scores.")
            # Preserve deterministic TREC score ordering rather than implementation tie order.
            result = result.sort_values(["score", "docno"], ascending=[False, False])
            return [identifier(value) for value in result.head(depth)["docno"]]

        try:
            retriever = pt.terrier.Retriever(
                index,
                wmodel=self.config.get("wmodel", "BM25"),
                controls=self.config.get("controls", {}),
                num_results=10000,
                tokeniser=self.config.get("tokeniser", "english"),
            )
            yield search
        finally:
            index.close()


class Pipeline:
    boundary = "one PyTerrier transform call in the current process"

    def __init__(self, config):
        if set(config) - {"factory", "options", "identity", "files"}:
            raise ValueError("Unknown PyTerrier pipeline configuration field.")
        if not isinstance(config.get("identity"), dict) or not config["identity"]:
            raise ValueError(
                "Provide pipeline identity with engine, model, and external index versions."
            )
        module, name = config["factory"].split(":", 1)
        self.factory = getattr(importlib.import_module(module), name)
        self.config = config

    def identity(self):
        source = inspect.getsourcefile(self.factory)
        return {
            "engine": "pyterrier-pipeline",
            "configuration": self.config,
            "factory_sha256": digest(Path(source)) if source else None,
            "pyterrier": importlib.metadata.version("pyterrier"),
            "files": {
                name: inventory(Path(name)) if Path(name).is_dir() else digest(Path(name))
                for name in self.config.get("files", [])
            },
        }

    def build(self, corpus, artifact):
        # The factory owns engine-specific indexing and persists it under the artifact directory.
        self.factory(corpus=corpus, artifact=artifact, options=self.config.get("options", {}))

    @contextmanager
    def open(self, artifact):
        import pandas as pd

        pipeline = self.factory(
            corpus=None, artifact=artifact, options=self.config.get("options", {})
        )

        def search(query, depth):
            frame = pipeline.transform(pd.DataFrame([{"qid": "query", "query": query}]))
            if not {"docno", "score"} <= set(frame.columns):
                raise ValueError("A retrieval pipeline must return docno and score columns.")
            if not all(math.isfinite(value) for value in frame["score"]):
                raise ValueError("The pipeline returned nonfinite scores.")
            frame = frame.sort_values(["score", "docno"], ascending=[False, False])
            return [identifier(value) for value in frame.head(depth)["docno"]]

        try:
            yield search
        finally:
            close = getattr(pipeline, "close", None)
            if close is not None:
                close()
