# IR Bench

IR Bench evaluates retrieval systems through one dataset, adapter, and metric contract.
It imports datasets from `ir_datasets` and calculates metrics with `ir_measures`.
An engine can use a Python adapter, a PyTerrier pipeline, a command, an HTTP endpoint, or saved results.

## Install and run

Use a project virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/ir-bench run --adapter sqlite --dataset examples/tiny --output work/sqlite.json
```

For Terrier and PyTerrier, install the optional dependencies. Terrier also requires Java 11 or later.
The verified environment uses Python 3.12 and Java 17.

```sh
.venv/bin/python -m pip install -e '.[terrier]'
.venv/bin/ir-bench run --adapter terrier --config examples/terrier-bm25.json \
  --dataset irds:cranfield --warmup 1 --repeats 3 --output work/cranfield-bm25.json
```

Downloads, exported datasets, and indexes default to `work/`. Set `--work PATH` before the command
to select another workspace. Existing `IR_DATASETS_HOME` and `PYTERRIER_HOME` settings take precedence.

## Datasets

List catalog entries and their available components:

```sh
.venv/bin/ir-bench datasets --match beir
.venv/bin/ir-bench prepare --dataset irds:beir/scifact/test
```

The installed `ir_datasets` 0.6.3 catalog exposes 790 entries, including collections and splits.
These are not 790 independent, complete benchmarks. Live retrieval requires documents, queries,
and judgments. Some collections require credentials, licenses, or manual downloads.
See the [dataset catalog API](https://ir-datasets.com/python.html) for source-specific requirements.

A catalog export uses each record's default text. Select fields explicitly when a task needs them:

```sh
.venv/bin/ir-bench prepare --dataset irds:beir/scifact/test \
  --doc-fields title text --query-fields text
```

Exports record the source identifier, library version, selected fields, and file hashes.
An unchanged export is reused after its content hashes pass validation.

Local directories can contain these formats, with optional gzip compression:

| Component | Names and content |
|---|---|
| Documents | `corpus.jsonl`, `corpus.tsv`, or `collection.tsv` |
| Queries | `queries.jsonl` or `queries.tsv` |
| Judgments | `qrels/test.tsv`, `qrels/test.txt`, `qrels.txt`, or `qrels.tsv` |

JSONL records require `_id` or `id` and `text`. Documents can also contain `title`.
TSV records contain `id`, `text`, and an optional `title`, without a header.
Judgments use BEIR TSV with a header or TREC columns: `query-id iteration document-id relevance`.
Use `--split NAME` for another judgment split under `qrels/`.

Identifiers must be unique and cannot contain whitespace. Relevance grades are signed 32-bit
integers. Negative judgments are retained for the metric provider, as required by datasets such as
[Cranfield](https://ir-datasets.com/cranfield.html). Subtopic judgments require explicit diversity
metrics. The evaluator rejects ambiguous use with ordinary document relevance metrics.

Live catalog retrieval currently uses text records. For image, audio, or other retrieval tasks,
use saved rankings with compatible document identifiers and judgments. Source-specific conversion
or an engine adapter can still be necessary. Catalog availability does not guarantee task compatibility.

## Engines

| Integration | Use |
|---|---|
| `sqlite` | SQLite FTS5, OR queries, title weight 2 and body weight 1 |
| `sift` | A supplied Sift binary and local static model |
| `terrier` | Terrier weighting models, including BM25, DPH, and PL2 |
| `pyterrier` | A supplied PyTerrier pipeline factory |
| `command` | Build and search programs selected with explicit argument lists |
| `http` | JSON requests to an existing, versioned remote index |
| `module:factory` | An installed Python adapter using the contract below |
| Saved run | TREC or BEIR JSON rankings from any retrieval system |

These paths separate engine integration from dataset and metric support. They do not require
another benchmark implementation for each engine and dataset pair.

For Sift, create a configuration with local paths:

```json
{
  "binary": "/path/to/sift",
  "model": "/path/to/model-directory",
  "build_args": ["--stop-df", "1.0"],
  "query_params": {}
}
```

```sh
.venv/bin/ir-bench run --adapter sift --config work/sift.json \
  --dataset irds:cranfield --depth 100 --output work/sift-result.json
```

The model directory requires `tokenizer.json` and `model.safetensors`. The adapter hashes both
files and the binary. It disables the Sift result cache and pages requests above 200 results.

The command example wraps real SQLite search through JSON input and output:

```sh
.venv/bin/ir-bench run --adapter command --config examples/command.json \
  --dataset examples/tiny --output work/command.json
.venv/bin/ir-bench run --adapter pyterrier --config examples/pyterrier-pipeline.json \
  --dataset examples/tiny --output work/pipeline.json
```

Run repository examples from the repository root. Command placeholders are complete argument
values: `{corpus}` and `{artifact}`. Search receives `{"query": "text", "depth": 100}` on standard
input and returns `{"hits": ["document-id"]}` on standard output. Programs must accept the selected
portable corpus format. List all scripts, models, and configuration resources in `files` so their
content changes invalidate the cache. Argument lists never execute through a shell.

The PyTerrier factory receives `corpus`, `artifact`, and `options` keyword arguments.
When `corpus` is a path, build the index under `artifact` and return `None`.
When `corpus` is `None`, return a transformer that accepts `qid` and `query` columns.
Its results must contain `docno` and finite `score` columns. Configure pipeline retrieval depth
in its options. The adapter truncates results to the benchmark depth and calls `close()` when available.
See the [PyTerrier documentation](https://pyterrier.readthedocs.io/en/latest/experiments.html) for pipeline composition.

### Existing HTTP services

The HTTP adapter sends JSON POST requests. Request values `{query}` and `{depth}` are replaced
with query text and an integer depth. Dotted response paths can select nested arrays and identifiers.
The examples include templates for
[Elasticsearch](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-search),
[Meilisearch](https://www.meilisearch.com/docs/reference/api/search/search-with-post), and
[Typesense](https://typesense.org/docs/29.0/api/federated-multi-search.html).
These templates have not been tested against those deployed services.

Before use, configure the endpoint, collection fields, engine version, index revision, and corpus
SHA-256. The declared corpus hash must match the benchmark corpus file. Remote index contents remain
the caller's responsibility. The adapter cannot verify that a service actually indexed that corpus.
Supply credentials through the environment variables named by `headers_env`.
Do not put credentials in the URL or configuration file, because reports record configuration.

HTTP runs do not build or replace remote indexes. Their reports leave build time and index size empty.

### Adapter contract

| Method | Contract |
|---|---|
| `identity()` | Return a JSON-compatible identity with engine, resource, and query settings. |
| `build_identity()` (optional) | Return only settings that affect the persisted index. The default is `identity()`. |
| `build(corpus, artifact)` | Build into the supplied empty directory. Raise an error on failure. |
| `open(artifact)` | Return a context manager that yields `search(query, depth)`. Release resources on exit. |
| `search(query, depth)` | Return unique string document identifiers in descending relevance order, within the requested depth. |

Only explicitly selected, trusted modules execute code. Dataset contents do not select adapters.

## Evaluate results from any system

Saved results avoid a live integration requirement:

```sh
.venv/bin/ir-bench evaluate --run work/results.trec --qrels work/qrels.txt \
  --depth 1000 --measure nDCG@10 --measure AP@1000 --measure R@1000 \
  --output work/evaluation.json
```

TREC run rows contain `query-id Q0 document-id rank score run-tag`.
BEIR JSON maps query identifiers to document-score mappings.
The evaluator sorts by score descending, then document identifier descending, matching `trec_eval`.
It ignores the supplied rank column. Duplicate documents, mixed run tags, and nonfinite scores fail validation.
Use `run --run-output PATH` to export live rankings in TREC format.
Saved results provide quality metrics only. They cannot establish latency, build time, or index size.

## Metric and measurement rules

Default metrics are `nDCG@10`, `RR@10`, `AP@depth`, `R@depth`, `P@10`, and `Judged@10`.
Select other definitions with repeated `--measure` options and an optional `--provider`.
A metric cutoff cannot exceed retrieval depth. Additional measures can require optional providers.
See the [metric definitions](https://ir-measur.es/en/latest/measures.html).

Schema version 2 uses `ir_measures` provider definitions. Default nDCG uses linear relevance gains,
consistent with `trec_eval`. Version 0.1 used exponential gains, so its graded nDCG values are not
interchangeable. Older extracted Sift research has other evaluator defects and is not a current baseline.

Reports retain every judged query, including queries with empty or missing results. Unjudged run
queries are excluded and counted. Reports contain per-query metrics, rankings, exact measure names,
provider implementations and versions, input hashes, and the complete configuration.
Each measure uses its provider-defined aggregation rule, including sums for count metrics.
A failed live query fails the run. A provider that omits a query or produces nonfinite values also fails.

Live runs validate document identifiers against a temporary disk index. Warmup passes precede
measured passes. Query order uses a recorded random seed. Reports retain each measured latency,
mean, standard deviation, and nearest-rank p50, p95, and p99. Rankings must remain stable across passes.

Latency covers the adapter call, including its integration cost. Sift and HTTP include transport.
Commands include process startup. PyTerrier includes its transform call. SQLite runs in process.
These boundaries are recorded and are not equivalent engine-only measurements. Remote cache policy,
hardware, concurrent workloads, and pipeline depth must match the intended comparison.
The harness does not infer statistical significance or production capacity from a small run.

## Dataset and engine matrices

```sh
.venv/bin/ir-bench suite examples/matrix.json --output work/comparison
```

The suite evaluates every configured dataset and engine pair. Each pair produces a separate report.
`suite.json` retains successful and failed pairs. Any failure returns a nonzero exit status.
Use an empty output directory for each suite. An engine entry can supply a `runs` mapping from dataset
names to saved result paths instead of an `adapter` and `config`.

Cache keys include corpus content, harness and adapter source, engine resources, and build settings.
Sift query settings and Terrier weighting models reuse their unchanged indexes. Every cache hit checks
artifact content hashes. Reports retain the original build duration and explicitly mark cache hits.
Artifact mutation during retrieval fails the run. Failed builds remove their staging directories.
Concurrent builds for the same key fail at an exclusive directory lock. After a process crash,
confirm that it stopped before removing its stale lock directory.

Corpus import and validation stream documents. Queries, judgments, and result rankings remain in
memory, which limits very large evaluations. Retrieval depth is bounded at 10,000, measured passes
at 100, warmup passes at 10, and suite pairs at 10,000. Input lines are bounded at 8 MiB.
Command and HTTP result bodies are bounded at 16 MiB. This runner is sequential and does not schedule
distributed workloads. Use a source-specific exporter or adapter when a collection exceeds these limits.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -v
IR_BENCH_LIVE=1 SIFT_BIN=/path/to/sift SIFT_MODEL=/path/to/model \
  .venv/bin/python -m unittest discover -s tests -v
```

The suite checks direct agreement with `pytrec_eval` for graded, negative, tied, and missing results.
It also checks malformed input, compressed TSV, TREC and BEIR results, cache invalidation, failed
matrix entries, real SQLite command and HTTP integrations, and atomic reports.
Optional live checks exercise catalog import, Terrier models, PyTerrier pipelines, and native Sift.

The verified comparison covers the included tiny corpus, Cranfield, and BEIR SciFact test.
It runs Sift, SQLite FTS5, and Terrier BM25, DPH, and PL2 across all three datasets.
This coverage establishes working integrations. It does not establish universal dataset compatibility
or production quality. Catalog entries and untested service templates are not counted as verified runs.

## License

Apache License 2.0. Extracted Sift research files retain their original license.
