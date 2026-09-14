"""Example PyTerrier factory around the existing SQLite adapter."""

import pandas as pd
import pyterrier as pt

from ir_bench.adapters import SQLiteFTS5


def create(corpus, artifact, options):
    engine = SQLiteFTS5({})
    if corpus is not None:
        engine.build(corpus, artifact)
        return None

    def transform(queries):
        rows = []
        with engine.open(artifact) as search:
            for row in queries.itertuples(index=False):
                found = search(row.query, options.get("depth", 100))
                rows.extend(
                    {"qid": row.qid, "docno": doc_id, "score": len(found) - rank}
                    for rank, doc_id in enumerate(found)
                )
        return pd.DataFrame(rows, columns=["qid", "docno", "score"])

    return pt.apply.generic(transform)
