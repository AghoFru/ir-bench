# Full BEIR CPU comparison

The comparison is in progress. No full-suite score is available yet.
BioASQ needs the complete corpus. Its available local copy is incomplete.

The [matrix](../../examples/beir18.json) covers all 18 tasks in the
[BEIR leaderboard](https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475).
It compares Sift, BM25, BGE-small, SPLADE, and Weaviate hybrid on an Apple M1 Ultra, CPU only.
CQADupStack contains 12 subsets, so the matrix has 29 corpora and 145 comparisons.

## Metrics

Mean nDCG@10 and Recall@100 give each task equal weight.
CQADupStack contributes the mean of its 12 subset scores.
All judged queries contribute to each subset score. No failed or missing comparison enters a full-suite mean.
Query times include encoding and adapter overhead. Ingestion includes document encoding and index writes.
Within CQADupStack, latency uses all query observations and ingestion sums all 12 index builds.
The suite reports both means and medians across tasks.

## Run

Install IR Bench with the `neural` and `terrier` extras. Build Sift in release mode.
Set the Sift binary and model paths in the matrix. Start Weaviate with the example Compose file.

Place the complete BioASQ, Robust04, Signal-1M, and TREC-NEWS datasets under
`work/beir18-source/<dataset>/dataset/`, with corpus, queries, and test judgments.
Use the [BEIR dataset instructions](https://github.com/beir-cellar/beir/tree/main/examples/dataset)
to obtain these corpora. IR Bench downloads the other datasets through `ir_datasets`.

```sh
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false .venv/bin/ir-bench \
  suite examples/beir18.json --output work/beir18-results
```

On macOS ARM64, use one OpenMP runtime for PyTorch and FAISS to prevent native crashes or deadlocks.
Set this variable before the suite command:

```sh
export DYLD_LIBRARY_PATH="$(.venv/bin/python -c \
  'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
```

Add `--resume` to the same command after an interruption. Completed reports must match their checksums,
input files, engine settings, and harness source before reuse.
The matrix checks full corpus sizes and uses all judged queries.

Touché-2020 uses the corrected v2 judgments. TREC-NEWS has 57 judged document IDs absent from its corpus.
Their judgments remain in the evaluation, including three positive judgments. ArguAna retains its five absent judged documents.

[Existing three-dataset measurements](../beir-cpu) remain available separately.
