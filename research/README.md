# Historical Sift research

These records and the `../reranker` training scripts came from Sift revision
`df4310b99f736f2c254028a4013b625316b85fd9`. The original Sift Git history retains the old
evaluation scripts and their invocation details.

`SIFT_RESULTS.md` and `legacy_regression_baseline.json` are historical records. The old
evaluators computed ideal DCG from retrieved documents and could omit failed queries. Do not
use their quality values as acceptance thresholds for the corrected runner. Remeasure each
configuration before making a retrieval-quality claim.

Run training scripts from the IR Bench repository root. Supply dataset paths explicitly with
the supported command arguments or `SIFT_REGRESSION_DATA`. Keep datasets and generated models
under `work/`. Training dependencies are optional and are not required by the benchmark runner.
The Sift product loads the resulting model through its public server interface.
