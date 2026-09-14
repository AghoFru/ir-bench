"""Access the ir_datasets catalog and preserve explicit text-field choices."""

import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

from .cache import cache_key, digest, inventory
from .dataset import identifier


def registry(work):
    # Set the package cache before its first import. Keep user configuration when provided.
    os.environ.setdefault("IR_DATASETS_HOME", str(Path(work).resolve() / "ir-datasets"))
    import ir_datasets

    return ir_datasets


def catalog(work, pattern=""):
    package = registry(work)
    for name in sorted(package.registry):
        if pattern in name:
            dataset = package.load(name)
            yield {
                "id": name,
                "docs": dataset.has_docs(),
                "queries": dataset.has_queries(),
                "qrels": dataset.has_qrels(),
            }


def text_fields(record, fields):
    if fields:
        values = [getattr(record, field) for field in fields]
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Selected text fields must contain strings.")
        return "\n".join(values)
    default = getattr(record, "default_text", None)
    value = default() if callable(default) else getattr(record, "text", None)
    if not isinstance(value, str):
        raise ValueError("This record has no default text. Select text fields explicitly.")
    return value


def prepare(source, work, doc_fields=None, query_fields=None):
    if not str(source).startswith("irds:"):
        path = Path(source).resolve()
        if not path.is_dir():
            raise ValueError(f"Dataset directory not found: {path}")
        return path
    package = registry(work)
    name = str(source)[5:]
    dataset = package.load(name)
    if not (dataset.has_docs() and dataset.has_queries() and dataset.has_qrels()):
        raise ValueError("Select a dataset split with documents, queries, and judgments.")
    identity = {
        "source": str(source),
        "ir_datasets": importlib.metadata.version("ir-datasets"),
        "doc_fields": doc_fields or ["default_text"],
        "query_fields": query_fields or ["default_text"],
        "exporter": digest(Path(__file__)),
    }
    parent = Path(work) / "datasets"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / cache_key(identity)
    if target.exists():
        saved = json.loads((target / "provenance.json").read_text())
        actual = inventory(target / "dataset")
        if saved["identity"] != identity or saved["files"] != actual:
            raise ValueError(f"Prepared dataset is damaged: {target}")
        return target / "dataset"
    with tempfile.TemporaryDirectory(prefix="export-", dir=parent) as temporary:
        stage = Path(temporary)
        output = stage / "dataset"
        output.mkdir()
        with (output / "corpus.jsonl").open("w", encoding="utf-8") as stream:
            for document in dataset.docs_iter():
                row = {
                    "_id": identifier(document.doc_id),
                    "text": text_fields(document, doc_fields),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output / "queries.jsonl").open("w", encoding="utf-8") as stream:
            for query in dataset.queries_iter():
                row = {"_id": identifier(query.query_id), "text": text_fields(query, query_fields)}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output / "qrels.txt").open("w", encoding="utf-8") as stream:
            for judgment in dataset.qrels_iter():
                iteration = identifier(getattr(judgment, "iteration", "0"))
                stream.write(
                    f"{identifier(judgment.query_id)} {iteration} "
                    f"{identifier(judgment.doc_id)} {judgment.relevance}\n"
                )
        saved = {"identity": identity, "files": inventory(output)}
        (stage / "provenance.json").write_text(json.dumps(saved, indent=2) + "\n")
        stage.rename(target)
    return target / "dataset"
