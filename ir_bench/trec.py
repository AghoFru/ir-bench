"""Read and write the common TREC run interchange format."""

import json
import math

from .dataset import identifier, lines


def read_run(path, depth=100):
    if str(path).endswith(".json"):
        # BEIR returns query -> document -> score dictionaries.
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=unique_object)
        scores = {}
        for query_id, found in raw.items():
            scores[identifier(query_id)] = {}
            for doc_id, score in found.items():
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (float, int))
                    or not math.isfinite(score)
                ):
                    raise ValueError("Run scores must be finite numbers.")
                scores[query_id][identifier(doc_id)] = score
        return ordered(scores, depth)
    scores, tags = {}, set()
    for line in lines(path):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError("TREC run rows require query_id, Q0, doc_id, rank, score, and tag.")
        query_id, _, doc_id, rank, score, tag = fields
        query_id, doc_id = identifier(query_id), identifier(doc_id)
        if int(rank) < 0 or not math.isfinite(float(score)):
            raise ValueError("Run ranks must be nonnegative and scores must be finite.")
        tags.add(identifier(tag))
        if len(tags) > 1:
            raise ValueError("A run file must contain one system tag.")
        found = scores.setdefault(query_id, {})
        if doc_id in found:
            raise ValueError(f"Duplicate run document: {query_id}, {doc_id}")
        found[doc_id] = float(score)
    # trec_eval sorts scores, not the supplied rank column. Document IDs break score ties.
    return ordered(scores, depth)


def ordered(scores, depth):
    return {
        query_id: sorted(hits, key=lambda doc: (hits[doc], doc), reverse=True)[:depth]
        for query_id, hits in scores.items()
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def write_run(path, rankings, tag="ir-bench"):
    identifier(tag)
    with path.open("w", encoding="utf-8") as stream:
        for query_id, hits in rankings.items():
            for rank, doc_id in enumerate(hits, 1):
                stream.write(
                    f"{identifier(query_id)} Q0 {identifier(doc_id)} {rank} "
                    f"{len(hits) - rank + 1} {tag}\n"
                )
