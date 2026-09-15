# CPU retrieval on a BEIR subset

Medians across SciFact, NFCorpus, and ArguAna, with each dataset weighted equally.
Apple M1 Ultra, CPU only. Higher nDCG@10 means better relevance. Lower times are better.

| System | Median nDCG@10 | Median query latency | Median ingestion |
|---|---:|---:|---:|
| Sift | 0.449 | 0.83 ms | 2.72 s |
| BM25 (Terrier) | 0.491 | 4.05 ms | 1.48 s |
| BGE-small | 0.603 | 14.09 ms | 175.42 s |
| SPLADE | 0.508 | 36.48 ms | 379.67 s |
| Weaviate hybrid (E5 + BM25) | 0.478 | 20.77 ms | 181.27 s |

The nDCG@10 summary is the median of each dataset's mean query score.
Query latency is the median of each dataset's median query latency.
Ingestion is the median of the full index build times for the three datasets.

## Per-dataset results

### SciFact

5,183 documents and 300 judged queries.

| System | nDCG@10 | p50 latency | p95 latency | Ingestion |
|---|---:|---:|---:|---:|
| Sift | 0.696 | 0.83 ms | 1.14 ms | 2.72 s |
| BM25 (Terrier) | 0.684 | 4.05 ms | 4.83 ms | 0.98 s |
| BGE-small | 0.713 | 14.09 ms | 16.46 ms | 175.42 s |
| SPLADE | 0.708 | 36.48 ms | 41.72 ms | 379.67 s |
| Weaviate hybrid (E5 + BM25) | 0.723 | 20.77 ms | 24.16 ms | 181.27 s |

### NFCorpus

3,633 documents and 323 judged queries.

| System | nDCG@10 | p50 latency | p95 latency | Ingestion |
|---|---:|---:|---:|---:|
| Sift | 0.332 | 0.56 ms | 0.75 ms | 2.36 s |
| BM25 (Terrier) | 0.328 | 3.45 ms | 4.48 ms | 1.48 s |
| BGE-small | 0.343 | 12.57 ms | 14.75 ms | 131.56 s |
| SPLADE | 0.352 | 35.36 ms | 39.78 ms | 273.68 s |
| Weaviate hybrid (E5 + BM25) | 0.340 | 18.62 ms | 20.84 ms | 123.17 s |

### ArguAna

8,674 documents and 1,406 judged queries.

| System | nDCG@10 | p50 latency | p95 latency | Ingestion |
|---|---:|---:|---:|---:|
| Sift | 0.449 | 2.03 ms | 3.27 ms | 4.77 s |
| BM25 (Terrier) | 0.491 | 7.72 ms | 10.98 ms | 1.89 s |
| BGE-small | 0.603 | 27.78 ms | 47.94 ms | 244.53 s |
| SPLADE | 0.508 | 68.75 ms | 108.13 ms | 528.81 s |
| Weaviate hybrid (E5 + BM25) | 0.478 | 41.32 ms | 66.73 ms | 249.41 s |

## Test settings

ArguAna has five judged documents absent from its corpus. Their judgments remain in the
scores, and all 1,406 queries are evaluated. The exact missing IDs are declared in the matrix
and recorded in each report.

- Apple M1 Ultra, 20 CPU cores, 128 GiB RAM, macOS ARM64. Neural encoders used four CPU threads.
  Weaviate 1.39.3 ran in Docker Desktop with 20 virtual CPUs and about 8 GB RAM.
- Each query retrieved up to 100 documents, with result caching disabled and one query at a time.
  One warmup pass preceded two measured passes. Query order was shuffled with seed 0.
- Relevance uses the first measured pass and all judged queries, including empty results.
  Both measured passes contribute to query latency. Matching query and document IDs are excluded,
  as in the [BEIR evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py).
- Sift used default settings with `minishlab/potion-base-8M`. BGE used exact cosine retrieval.
  SPLADE used exact sparse retrieval. Weaviate combined E5 vectors with BM25 using hybrid search.
  Model revisions and settings are in [matrix.json](matrix.json).
  SciFact and NFCorpus also informed [Sift's default settings](../../research/SIFT_RESULTS.md).
- Query latency includes encoding and adapter overhead. Sift and Weaviate include local HTTP
  transport. Terrier includes Python and Java overhead. BGE and SPLADE run in process.
- Ingestion includes document encoding and index writes, with dataset and model files already
  local. Reused indexes retain their original build times.

These results cover three small English datasets. They are not a full BEIR evaluation or
an enterprise-scale throughput test. No failed comparison is omitted from an aggregate.

## Reproduce

The recorded harness revisions are `8393c15` and `acfba18`. Sift uses the source at `a17c16f`.
Use the engine and library versions recorded in the reports.
Use the [installation instructions](../../README.md) with the `neural` and `terrier` extras.
Build Sift in release mode and set its binary path in [matrix.json](matrix.json).

From the IR Bench checkout:

```sh
.venv/bin/hf download minishlab/potion-base-8M \
  --revision bf8b056651a2c21b8d2565580b8569da283cab23 \
  --include tokenizer.json model.safetensors --local-dir work/model
docker compose -f examples/weaviate.yaml up -d
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/ir-bench \
  suite results/beir-cpu/matrix.json --output work/beir-comparison
```

The suite report includes `dataset_medians` after every comparison succeeds.

## Evidence

[CSV results](results.csv) contain unrounded per-dataset measurements.
[Reports and TREC rankings](reports.zip) contain all 15 comparisons, dataset medians,
per-query timings, model and input hashes, and configurations. Local paths use workspace placeholders.
Every TREC export reproduces its report's relevance metrics with the same metric settings.
