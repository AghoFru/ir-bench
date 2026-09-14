# CPU retrieval comparison

Sift had the lowest measured query latency on both datasets.
Weaviate hybrid had the highest nDCG@10.

Higher nDCG@10 and Recall@100 mean better relevance. Lower latency is better.
The p50 column shows median latency. The p95 column covers 95% of measured queries.

## Cranfield

1,400 documents and 225 queries.

| System | nDCG@10 | Recall@100 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|
| Sift | 0.349 | 0.739 | 0.80 ms | 1.00 ms |
| BM25 (Terrier) | 0.343 | 0.745 | 4.25 ms | 5.20 ms |
| E5-small-v2 | 0.344 | 0.769 | 13.95 ms | 16.28 ms |
| BGE-small-en-v1.5 | 0.359 | 0.776 | 14.42 ms | 16.85 ms |
| SPLADE | 0.359 | 0.775 | 37.50 ms | 43.08 ms |
| Weaviate vector (E5) | 0.344 | 0.769 | 19.78 ms | 22.75 ms |
| Weaviate hybrid (E5 + BM25) | 0.370 | 0.768 | 20.46 ms | 23.63 ms |
| Weaviate BM25 | 0.322 | 0.705 | 3.46 ms | 4.45 ms |

## SciFact

5,183 documents and 300 queries.

| System | nDCG@10 | Recall@100 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|
| Sift | 0.696 | 0.926 | 0.75 ms | 1.01 ms |
| BM25 (Terrier) | 0.684 | 0.926 | 4.20 ms | 4.90 ms |
| E5-small-v2 | 0.687 | 0.928 | 14.46 ms | 17.05 ms |
| BGE-small-en-v1.5 | 0.713 | 0.942 | 15.17 ms | 17.33 ms |
| SPLADE | 0.708 | 0.949 | 37.41 ms | 42.95 ms |
| Weaviate vector (E5) | 0.688 | 0.928 | 20.34 ms | 23.10 ms |
| Weaviate hybrid (E5 + BM25) | 0.723 | 0.955 | 20.82 ms | 23.58 ms |
| Weaviate BM25 | 0.667 | 0.883 | 3.49 ms | 5.20 ms |

## Test settings

- Apple M1 Ultra, 20 CPU cores, 128 GiB RAM, macOS ARM64. No GPU use.
- Weaviate 1.39.3 ran in Docker Desktop, with 20 virtual CPUs and about 8 GB RAM.
- Each query retrieved up to 100 documents. One warmup pass preceded two measured passes,
  with one query at a time and shuffled query order.
- Relevance uses every judged query and the first measured pass. The reports also contain
  the second pass. Latency combines both measured passes, with result caching disabled.
- Sift used its default settings and `minishlab/potion-base-8M`, with semantic weight 0.5.
  E5 and BGE used exact cosine retrieval in FAISS. SPLADE used exact sparse dot products.
  Weaviate used HNSW, with E5 vectors shared across all three modes.
- Neural encoders used four CPU threads. Sift used its default thread settings.
  Model names, revisions, prompts, and index settings are in [matrix.json](matrix.json).

Latency includes query encoding and each adapter call. Sift and Weaviate include local
HTTP transport. Terrier includes Python and Java overhead. FAISS and SPLADE run in process.
These are application-level measurements on small English datasets, with no concurrency
or large-corpus scaling tests.

## Reproduce

Measured with IR Bench at `b17e147` and Sift at `19ea72a`. Install the `neural` and `terrier`
extras from the [setup instructions](../../README.md). Build Sift in release mode.
Set its binary path in [matrix.json](matrix.json) to your local build.

From the IR Bench checkout:

```sh
.venv/bin/hf download minishlab/potion-base-8M \
  --revision bf8b056651a2c21b8d2565580b8569da283cab23 \
  --include tokenizer.json model.safetensors --local-dir work/model
docker compose -f examples/weaviate.yaml up -d
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/ir-bench \
  suite results/sift-cpu/matrix.json --output work/cpu-comparison
```

## Evidence

[CSV results](results.csv) contain unrounded values.
[Reports and TREC rankings](reports.zip) contain all 16 comparisons, per-query timings,
model and input hashes, configurations, and metrics. Local paths use workspace placeholders.
Every TREC export reproduces its report's relevance metrics with the same metric settings.
