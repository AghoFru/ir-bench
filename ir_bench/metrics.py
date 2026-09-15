"""Evaluate complete query populations with standard IR metric providers."""

import importlib.metadata
import math

import ir_measures

from .dataset import identifier


def validate_ranking(hits, depth):
    if not isinstance(hits, list) or len(hits) > depth:
        raise ValueError("The engine must return a ranked list within the requested depth.")
    if any(not isinstance(hit, str) or identifier(hit) != hit for hit in hits):
        raise ValueError("The engine must return string document identifiers.")
    if len(set(hits)) != len(hits):
        raise ValueError("The engine returned duplicate document identifiers.")


def remove_self_matches(rankings):
    return {query: [doc for doc in hits if doc != query] for query, hits in rankings.items()}


def evaluate(rankings, qrels, depth=100, measures=None, provider=None):
    if not 1 <= depth <= 10000:
        raise ValueError("Retrieval depth must be from 1 through 10000.")
    names = (
        measures
        if measures is not None
        else ["nDCG@10", "RR@10", f"AP@{depth}", f"R@{depth}", "P@10", "Judged@10"]
    )
    if not names or len(names) > 64:
        raise ValueError("Select from 1 through 64 measures.")
    selected = [ir_measures.parse_measure(name) for name in names]
    if len(set(selected)) != len(selected):
        raise ValueError("Measure definitions must be unique.")
    for measure in selected:
        cutoff = measure.params.get("cutoff")
        if cutoff is not None and cutoff > depth:
            raise ValueError(f"{measure} exceeds the retrieval depth {depth}.")
    for row in qrels:
        identifier(row.query_id)
        identifier(row.doc_id)
        if type(row.relevance) is not int or not -(2**31) <= row.relevance < 2**31:
            raise ValueError("Relevance must be a signed 32-bit integer.")
    pairs = [(row.query_id, row.doc_id) for row in qrels]
    if len(set(pairs)) != len(pairs):
        diversity = {"alpha_nDCG", "alpha_DCG", "ERR_IA", "NRBP", "nNRBP", "P_IA", "StRecall"}
        if any(measure.NAME not in diversity for measure in selected):
            raise ValueError(
                "Subtopic judgments require explicit diversity measures. Do not flatten them."
            )
    judged_queries = {row.query_id for row in qrels}
    if not judged_queries:
        raise ValueError("No relevance judgments were provided.")
    scored = []
    for query_id, hits in rankings.items():
        identifier(query_id)
        validate_ranking(hits, depth)
        if query_id in judged_queries:
            scored.extend(
                ir_measures.ScoredDoc(query_id, doc_id, float(len(hits) - rank))
                for rank, doc_id in enumerate(hits)
            )
    if provider is None:
        evaluator = ir_measures.evaluator(selected, qrels)
    else:
        if provider not in ir_measures.providers.registry:
            raise ValueError(f"Unknown metric provider: {provider}")
        evaluator = ir_measures.providers.registry[provider].evaluator(selected, qrels)
    per_query = {query_id: {} for query_id in sorted(judged_queries)}
    calculated = evaluator.calc(scored)
    if any(not math.isfinite(value) for value in calculated.aggregated.values()):
        raise ValueError("The metric provider returned a nonfinite aggregate.")
    for metric in calculated.per_query:
        if not math.isfinite(metric.value):
            raise ValueError(f"The metric provider returned a nonfinite value: {metric}")
        per_query[metric.query_id][str(metric.measure)] = metric.value
    expected = {str(measure) for measure in selected}
    if any(set(values) != expected for values in per_query.values()):
        raise ValueError("The metric provider omitted a query or measure.")
    # Record the actual evaluator classes, including any provider fallbacks.
    evaluators = getattr(evaluator, "evaluators", [evaluator])
    implementations = sorted(
        {type(value).__module__ + "." + type(value).__name__ for value in evaluators}
    )
    return {
        "metrics": {str(measure): value for measure, value in calculated.aggregated.items()},
        "per_query": per_query,
        "evaluation": {
            "library": "ir-measures",
            "version": importlib.metadata.version("ir-measures"),
            "provider": provider or "default",
            "implementations": implementations,
            "measures": [str(measure) for measure in selected],
            "query_population": "all query IDs in qrels, including missing and empty results",
            "missing_result_queries": sorted(judged_queries - rankings.keys()),
            "unjudged_run_queries": len(rankings.keys() - judged_queries),
            "gain_convention": "provider definition, default nDCG gains are linear",
            "provider_versions": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"].lower()
                in {
                    "ir-measures",
                    "pytrec-eval-terrier",
                    "pytrec-eval",
                    "pyndeval",
                    "gdeval",
                    "ranx",
                    "trectools",
                    "cwl-eval",
                }
            },
        },
        "query_count": len(judged_queries),
    }
