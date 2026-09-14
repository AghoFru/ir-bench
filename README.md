# IR Bench

Compare retrieval systems on shared datasets, metrics, and saved results.
Indexes are cached so unchanged builds can be reused.

## Run a benchmark

From this checkout, with Python 3.10 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/ir-bench run --adapter sqlite --dataset examples/tiny --output work/result.json
```

To compare engines across datasets, install Terrier support (requires Java 11 or later):

```sh
.venv/bin/python -m pip install -e '.[terrier]'
.venv/bin/ir-bench suite examples/matrix.json --output work/comparison
```

Edit [matrix.json](examples/matrix.json) to select datasets, engines, and metrics.
Use a new output directory for each suite. Downloads and cached indexes stay under `work/`.
Use `.venv/bin/ir-bench <command> --help` for options.

## Datasets

Pass a local directory or an [ir_datasets](https://ir-datasets.com/) identifier:

```sh
.venv/bin/ir-bench datasets --match beir
.venv/bin/ir-bench prepare --dataset irds:beir/scifact/test
```

Live retrieval requires text documents, queries, and judgments. Some catalog entries require
manual downloads. Use `--doc-fields` and `--query-fields` to select nondefault text fields.

Local datasets accept JSONL or TSV, including gzip files:

| File | Content |
|---|---|
| `corpus.jsonl` | Objects with `id` or `_id`, `text`, and optional `title`. |
| `queries.jsonl` | Objects with `id` or `_id` and `text`. |
| `qrels/test.tsv` | BEIR judgments with a header, or four-column TREC judgments. |

TSV alternatives are `corpus.tsv` (or `collection.tsv`) and `queries.tsv`, with
`id`, `text`, and optional `title` columns, without a header. `qrels.txt` also works.
Use `--split NAME` for another judgment split. Identifiers must be unique and contain no whitespace.

## Engines

Use `run --adapter NAME --config FILE` with one of these integrations:

| Adapter | Configuration |
|---|---|
| `sqlite` | None. |
| `sift` | `{"binary":"/path/to/sift","model":"/path/to/model"}`. |
| `terrier` | [BM25](examples/terrier-bm25.json), [DPH](examples/terrier-dph.json), [PL2](examples/terrier-pl2.json). |
| `command` | [Build/search commands](examples/command.json) and [working wrapper](examples/sqlite_command.py). |
| `pyterrier` | [Pipeline configuration](examples/pyterrier-pipeline.json) and [factory](examples/sqlite_pipeline.py). |
| `http` | Existing-index templates: [Elasticsearch](examples/elasticsearch.json), [Meilisearch](examples/meilisearch.json), [Typesense](examples/typesense.json). |

Sift needs local `tokenizer.json` and `model.safetensors` files. Custom commands and pipelines
must declare model and script files for cache invalidation. HTTP templates require your endpoint,
index revision, and corpus SHA-256.
Use `headers_env` for credentials, because reports contain configuration.

For a Python adapter, use `--adapter module:factory`. The factory accepts a configuration dictionary
and returns an object with `identity()`, `build(corpus, artifact)`, and `open(artifact)`.
`open` is a context manager that yields `search(query, depth)`, returning unique ranked document IDs.
Optional `build_identity()` separates index settings from query settings.
See the [included adapters](ir_bench/adapters.py).

## Evaluate saved results

Any system can supply a TREC run or BEIR JSON scores:

```sh
.venv/bin/ir-bench evaluate --run work/results.trec --qrels work/qrels.txt \
  --depth 1000 --measure nDCG@10 --measure R@1000 --output work/evaluation.json
```

TREC rows contain `query-id Q0 document-id rank score run-tag`.
BEIR JSON maps query IDs to document-score mappings. Evaluation orders by descending score,
then descending document ID. Export live rankings with `run --run-output PATH`.

## Read the results

Reports include rankings, per-query metrics, and latency.

Metrics use [ir_measures definitions](https://ir-measur.es/en/latest/measures.html).
Subtopic judgments require diversity metrics.

Latency includes adapter overhead, such as HTTP transport or process startup.
Saved results provide quality metrics only. HTTP runs cannot measure remote build time or index size.
Queries, judgments, and rankings must fit in memory. Maximum retrieval depth is 10,000.

Apache-2.0. See [LICENSE](LICENSE).
