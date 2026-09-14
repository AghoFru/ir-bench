# IR Bench

Compare search quality and speed across retrieval systems and datasets.
Run benchmarks, evaluate saved rankings, and reuse indexes between comparisons.

## Get started

Requires Python 3.10 or later. From this checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/ir-bench run --adapter sqlite --dataset examples/tiny --output work/result.json
```

The report contains relevance metrics, query latency, index size, and build time.

## Compare systems

IR Bench supports Sift, SQLite, Terrier, PyTerrier pipelines, command-line tools,
and HTTP services. Use datasets from [ir_datasets](https://ir-datasets.com/) or
supply your own using the [example format](examples/tiny).

Choose engines, datasets, and metrics in [matrix.json](examples/matrix.json).
The example uses Terrier and requires Java 11 or later:

```sh
.venv/bin/python -m pip install -e '.[terrier]'
.venv/bin/ir-bench suite examples/matrix.json --output work/comparison
```

Results go into the output directory. Use a new directory for each comparison.
See [example configurations](examples/) for other integrations.

## Evaluate saved rankings

Evaluate TREC runs or BEIR JSON results from an existing system:

```sh
.venv/bin/ir-bench evaluate --run results.trec --qrels qrels.txt \
  --depth 1000 --measure nDCG@10 --measure R@1000 --output work/evaluation.json
```

Use `.venv/bin/ir-bench --help` to see available commands.

[Apache-2.0](LICENSE)
