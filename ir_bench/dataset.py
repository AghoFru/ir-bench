"""Read BEIR-shaped corpora, queries, and relevance judgments."""

import csv
import json
from pathlib import Path

MAX_LINE_BYTES = 8 * 1024 * 1024


def json_rows(path: Path):
    with path.open("rb") as stream:
        number = 0
        while raw := stream.readline(MAX_LINE_BYTES + 1):
            number += 1
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{number}: Row exceeds {MAX_LINE_BYTES} bytes.")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: Expected a JSON object.")
            yield row


def identifier(value):
    if not isinstance(value, (str, int)) or isinstance(value, bool) or str(value) == "":
        raise ValueError("Expected a nonempty string or integer identifier.")
    return str(value)


def documents(path: Path):
    for row in json_rows(path):
        doc_id = identifier(row.get("_id", row.get("id")))
        text, title = row.get("text"), row.get("title", "")
        if not isinstance(text, str) or not isinstance(title, str):
            raise ValueError(f"Document {doc_id}: Expected text and title strings.")
        yield doc_id, title, text


def load_queries(path: Path):
    queries = {}
    for query_id, _, text in documents(path):
        if query_id in queries:
            raise ValueError(f"Duplicate query identifier: {query_id}")
        queries[query_id] = text
    if not queries:
        raise ValueError("The query file is empty.")
    return queries


def load_qrels(path: Path):
    qrels = {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if rows.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Expected qrels columns: query-id, corpus-id, score.")
        for row in rows:
            query_id, doc_id = identifier(row["query-id"]), identifier(row["corpus-id"])
            relevance = int(row["score"])
            if not 0 <= relevance <= 30:
                raise ValueError("Relevance must be an integer from 0 through 30.")
            judged = qrels.setdefault(query_id, {})
            if doc_id in judged:
                raise ValueError(f"Duplicate relevance judgment: {query_id}, {doc_id}")
            judged[doc_id] = relevance
    if not qrels:
        raise ValueError("The relevance file is empty.")
    return qrels
