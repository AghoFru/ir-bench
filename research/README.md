# Historical Sift research

These experiments are not validated baselines. Older evaluators used incorrect
ideal DCG or omitted failed queries. Remeasure results with the current runner.

Run scripts from the repository root. Keep datasets, models, and outputs under
`work/`. Training dependencies are separate from the benchmark installation.
The composition experiments use `SIFT_BIN`. The contextual experiment also
accepts `SIFT_MODEL` and `SIFT_CROSS_ENCODER`.
