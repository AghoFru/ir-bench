# IR Bench

IR Bench compares retrieval engines through one evaluator. It uses the Python standard library.
The included adapters support Sift and SQLite FTS5. No engine source checkout is required.

## Run a comparison

Run the SQLite baseline from this repository:

```sh
python3 -m ir_bench.run --adapter ir_bench.adapters:SQLiteFTS5 \
  --dataset examples/tiny --output work/sqlite.json
```

For Sift, create `work/sift.json` with paths to a binary and a local static model:

```json
{
  "binary": "/path/to/sift",
  "model": "/path/to/model-directory",
  "build_args": ["--stop-df", "1.0"],
  "query_params": {}
}
```

```sh
python3 -m ir_bench.run --adapter ir_bench.adapters:Sift --config work/sift.json \
  --dataset examples/tiny --output work/sift-result.json
```

The model directory must contain `tokenizer.json` and `model.safetensors`. The adapter hashes
both files. It requires local model files to prevent a changing remote model from reusing an
old index. The benchmark does not download models or build the Sift binary.

## Dataset contract

Each dataset directory contains:

- `corpus.jsonl`: Objects with `_id` or `id`, `text`, and an optional `title`.
- `queries.jsonl`: Objects with `_id` or `id` and `text`.
- `qrels/test.tsv`: Tab-separated `query-id`, `corpus-id`, and `score` columns with a header.

Identifiers must be unique nonempty strings or integers. Relevance grades range from 0 through
30. Convert other dataset formats into these files. Dataset contents never select executable code.

## Engine contract

An adapter is a Python factory selected explicitly with `--adapter module:factory`. The factory
receives a configuration dictionary. It returns an object with these methods:

| Method | Contract |
|---|---|
| `identity()` | Return JSON-compatible engine, model, and build identities. Include external resource content hashes. |
| `build(corpus, artifact)` | Build into the supplied empty directory. Raise an error on failure. |
| `open(artifact)` | Return a context manager that yields `search(query, depth)`. Release resources on exit. |
| `search(query, depth)` | Return unique string document identifiers in descending relevance order. |

The two included adapters use this contract. A new adapter does not require changes to the
evaluator. Install trusted adapter modules into the active Python environment before use.

## Measurement and cache rules

nDCG@10 uses ideal gains from all judgments for each query. MRR@10 uses the first relevant hit.
Recall uses the requested retrieval depth, which defaults to 100. Queries with no relevant
judgments count as zero. The report averages over all judged queries. A failed query fails the
run and produces no new aggregate report. Reports include rankings for independent verification.

Latency covers each adapter call. Sift includes HTTP transport. SQLite uses an in-process call.
These measurements describe integration cost and must not be presented as equivalent engine-only
timings. Runs are sequential, include the first query, and disable Sift's result cache.
SQLite uses its `unicode61` tokenizer, OR queries, and BM25 with title weight 2 and body weight 1.
The Sift HTTP adapter rejects retrieval depths above its API limit of 200.
Its tokenization and BM25 parameters differ from Sift. This is an independent product baseline.
See the [SQLite FTS5 documentation](https://sqlite.org/fts5.html) for ranking behavior.

Cache identities include corpus content, adapter source, engine content, model content, and build
settings. Query settings can reuse an index. Reports record query and judgment content hashes.
Cached builds retain their original build duration, with an explicit `cache_hit` field.
Every cache hit verifies artifact content hashes. A failed build removes its staging directory.
Concurrent builds for the same key fail at an exclusive directory lock. If a process crashes,
verify that it has stopped before you remove its remaining lock directory.

All default cache and report paths stay under `work/`. The Sift adapter limits builds to 3600
seconds and requests to 60 seconds. Change these implementation limits only for measured needs.
Large suites should use separate runs. There is no scheduler or distributed runner.

## Verification

```sh
python3 -m unittest discover -s tests -v
SIFT_BIN=/path/to/sift SIFT_MODEL=/path/to/model \
  python3 -m unittest discover -s tests -v
```

The first command tests real SQLite search, metrics, input failures, cache invalidation, and
artifact corruption. The second also runs the native Sift adapter. Keep native integration
coverage explicit when the binary or model is absent.

The extracted runner passed all 13 checks with both native adapters. The package also built
as a wheel and source distribution. These checks use the small included corpus and do not
establish retrieval quality or performance on a production workload.

Historical Sift results predate the corrected nDCG calculation. They are retained as research
records and are not validated baselines for this runner.

## License

Apache License 2.0. Extracted Sift research files retain their original license.
