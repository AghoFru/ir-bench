# IR Bench

Compare search quality and speed across retrieval systems and datasets.
Run benchmarks, evaluate saved rankings, and reuse indexes between comparisons.

## Get started

Requires Python 3.10 or later. From this checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[neural]'
.venv/bin/ir-bench run --adapter dense --config examples/dense-e5.json \
  --dataset examples/tiny --output work/result.json
```

The report contains relevance metrics, query latency, index size, and build time.
See the [CPU comparison of Sift, BM25, dense models, SPLADE, and Weaviate](results/sift-cpu).

## Compare systems

Compare BM25, dense retrieval with E5 or BGE, SPLADE, and Weaviate vector or
hybrid search. Use datasets from [ir_datasets](https://ir-datasets.com/) or
supply your own using the [example format](examples/tiny).

Choose engines, datasets, and metrics in [matrix.json](examples/matrix.json).
The BM25 baseline requires Java 11 or later:

```sh
.venv/bin/python -m pip install -e '.[neural,terrier]'
.venv/bin/ir-bench suite examples/matrix.json --output work/comparison
```

Results go into the output directory. Use a new directory for each comparison.
See [example configurations](examples/) for other integrations.

## Benchmark Weaviate

Start a local Weaviate server and run hybrid search:

```sh
docker compose -f examples/weaviate.yaml up -d
.venv/bin/ir-bench run --adapter weaviate --config examples/weaviate-hybrid.json \
  --dataset irds:cranfield --output work/weaviate.json
```

## Evaluate saved rankings

Evaluate TREC runs or BEIR JSON results from an existing system:

```sh
.venv/bin/ir-bench evaluate --run results.trec --qrels qrels.txt \
  --depth 1000 --measure nDCG@10 --measure R@1000 --output work/evaluation.json
```

Use `.venv/bin/ir-bench --help` to see available commands.

[Apache-2.0](LICENSE)
