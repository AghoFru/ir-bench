"""Calculate ranking metrics against the complete relevance judgments."""

import math


def evaluate(hits: list[str], judgments: dict[str, int], depth: int = 100):
    if not 10 <= depth <= 10000:
        raise ValueError("Retrieval depth must be from 10 through 10000.")
    if len(hits) > depth or any(not isinstance(doc_id, str) for doc_id in hits):
        raise ValueError("The engine returned invalid document identifiers or too many hits.")
    if len(set(hits)) != len(hits):
        raise ValueError("The engine returned duplicate document identifiers.")
    if any(type(value) is not int or not 0 <= value <= 30 for value in judgments.values()):
        raise ValueError("Relevance must be an integer from 0 through 30.")
    ranked = [judgments.get(doc_id, 0) for doc_id in hits]
    ideal = sorted(judgments.values(), reverse=True)[:10]
    dcg = sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(ranked[:10]))
    idcg = sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(ideal))
    relevant_count = sum(value > 0 for value in judgments.values())
    return {
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "mrr_at_10": next((1 / rank for rank, value in enumerate(ranked[:10], 1) if value), 0.0),
        f"recall_at_{depth}": sum(value > 0 for value in ranked) / relevant_count
        if relevant_count
        else 0.0,
    }
