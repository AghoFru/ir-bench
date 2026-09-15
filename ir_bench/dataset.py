"""Read text collections, queries, and BEIR or TREC relevance judgments."""

import gzip
import json
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ir_measures import Qrel

MAX_LINE_BYTES = 8 * 1024 * 1024


def lines(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as stream:
        number = 0
        while raw := stream.readline(MAX_LINE_BYTES + 1):
            number += 1
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{number}: Row exceeds {MAX_LINE_BYTES} bytes.")
            text = raw.decode("utf-8").rstrip("\r\n")
            if text.strip():
                yield text


def json_rows(path):
    for line in lines(path):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}: Expected a JSON object.")
        yield row


def identifier(value):
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError("Expected a nonempty string or integer identifier.")
    result = str(value)
    if not result or any(char.isspace() or ord(char) < 32 for char in result):
        raise ValueError("Identifiers cannot contain whitespace or control characters.")
    return result


def documents(path):
    path = Path(path)
    if ".tsv" in path.suffixes:
        for line in lines(path):
            fields = line.split("\t")
            if len(fields) not in (2, 3):
                raise ValueError("Collection TSV rows must contain id, text, and optional title.")
            yield identifier(fields[0]), fields[2] if len(fields) == 3 else "", fields[1]
        return
    for row in json_rows(path):
        doc_id = identifier(row.get("_id", row.get("id")))
        text, title = row.get("text"), row.get("title", "")
        if not isinstance(text, str) or not isinstance(title, str):
            raise ValueError(f"Document {doc_id}: Expected text and title strings.")
        yield doc_id, title, text


def load_queries(path):
    queries = {}
    for query_id, _, text in documents(path):
        if query_id in queries:
            raise ValueError(f"Duplicate query identifier: {query_id}")
        queries[query_id] = text
    if not queries:
        raise ValueError("The query file is empty.")
    return queries


def load_qrels(path):
    result, seen = [], set()
    rows = iter(lines(path))
    first = next(rows, None)
    if first is None:
        raise ValueError("The relevance file is empty.")
    beir = first.split() == ["query-id", "corpus-id", "score"]
    import itertools

    for line in rows if beir else itertools.chain([first], rows):
        fields = line.split()
        if beir and len(fields) == 3:
            query_id, doc_id, grade = fields
            iteration = "0"
        elif not beir and len(fields) == 4:
            query_id, iteration, doc_id, grade = fields
        else:
            raise ValueError("Use BEIR qrels with a header or four-column TREC qrels.")
        query_id, doc_id, iteration = map(identifier, (query_id, doc_id, iteration))
        relevance = int(grade)
        if not -(2**31) <= relevance < 2**31:
            raise ValueError("Relevance must fit a signed 32-bit integer.")
        key = query_id, doc_id, iteration
        if key in seen:
            raise ValueError(f"Duplicate relevance judgment: {key}")
        seen.add(key)
        result.append(Qrel(query_id, doc_id, relevance, iteration))
    if not result:
        raise ValueError("The relevance file is empty.")
    return result


def dataset_paths(directory, split="test"):
    directory = Path(directory)
    identifier(split)
    if Path(split).name != split or split in (".", ".."):
        raise ValueError("The split must be a filename component.")

    def choose(names):
        for name in names:
            for suffix in ("", ".gz"):
                path = directory / (name + suffix)
                if path.is_file():
                    return path
        raise ValueError(f"No supported input found under {directory}: {names}")

    return {
        "corpus": choose(["corpus.jsonl", "corpus.tsv", "collection.tsv"]),
        "queries": choose(["queries.jsonl", "queries.tsv"]),
        "qrels": choose([f"qrels/{split}.tsv", f"qrels/{split}.txt", "qrels.txt", "qrels.tsv"]),
    }


@contextmanager
def validate_corpus(corpus, qrels, work, expected_missing_qrel_docs=()):
    """Keep document identity validation on disk so large corpora do not fill RAM."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="validate-", dir=work) as temporary:
        connection = sqlite3.connect(Path(temporary) / "ids.sqlite")
        try:
            connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
            count = 0
            for doc_id, _, _ in documents(corpus):
                try:
                    connection.execute("INSERT INTO ids VALUES (?)", (doc_id,))
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"Duplicate corpus identifier: {doc_id}") from error
                count += 1
            connection.commit()
            if not count:
                raise ValueError("The corpus is empty.")

            def exists(doc_id):
                return (
                    connection.execute("SELECT 1 FROM ids WHERE id=?", (doc_id,)).fetchone()
                    is not None
                )

            def check(ids):
                for doc_id in ids:
                    if not exists(doc_id):
                        raise ValueError(f"Unknown corpus document identifier: {doc_id}")

            missing = {doc_id for doc_id in {row.doc_id for row in qrels} if not exists(doc_id)}
            if missing != set(expected_missing_qrel_docs):
                raise ValueError(
                    "Missing judged documents differ from expected_missing_qrel_docs: "
                    f"{sorted(missing)[:10]}"
                )
            yield check, count
        finally:
            connection.close()
