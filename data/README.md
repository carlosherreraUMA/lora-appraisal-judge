# Data

`prepared/` is committed to this repo: it is small (about 20 MB) and its licence
permits redistribution with attribution (see `../NOTICE.md`). It is what
`notebooks/kaggle_train.ipynb` trains and evaluates on — no separate download is
needed to run the notebook.

| File | Rows | Content |
|---|---|---|
| `prepared/train.jsonl` | 5,594 | `{trajectory_id, instance_id, resolved, n_messages, prompt, response}` |
| `prepared/val.jsonl` | 817 | same shape |
| `prepared/test.jsonl` | 1,717 | same shape |
| `prepared/test_pairs.json` | 106 pairs | `[trajectory_id_resolved, trajectory_id_unresolved]` for within-task pairs in the test split |
| `prepared/manifest.json` | — | provenance: source revision, how many raw records were scanned, split sizes |
| `e3/test.jsonl.gz` | 12,691 | same shape as `prepared/test.jsonl`: every eligible attempt in the **full** corpus at a task in the test bucket (E3). Pairs are rebuilt from it with `splits.build_pairs` (6,426) |
| `e3/manifest.json` | — | provenance and the two checks `scripts/prepare_e3.py` runs: no overlap with train/val, all E2b test examples present with identical prompts |

`e3/` was built with `python scripts/prepare_e3.py --raw-path trajectories.parquet`
(whole corpus, about 25 minutes). Gzip with a fixed timestamp, so a rebuild from the
same file is byte-identical.

## Regenerating or extending it

`prepared/` was built by `scripts/prepare_dataset.py` from a local copy of the raw
corpus, scanning only the first 9,000 of 67,074 trajectories (see `EXPERIMENTS.md`
for why: speed, not necessity). To rebuild it, or to use the full corpus:

```
# 1. Get a local copy of the raw corpus (2.08 GB, CC BY 4.0). Either:
python -c "from datasets import load_dataset; load_dataset('nebius/SWE-rebench-openhands-trajectories', split='train').to_parquet('trajectories.parquet')"
# or reuse an existing copy, e.g. from situated-appraisal-in-coding-agent-trajectories's
# own data/raw/trajectories.parquet if you have that repo checked out with `make data` run.

# 2. Extract, split and write the prepared JSONL files:
python scripts/prepare_dataset.py --raw-path trajectories.parquet --out-dir data/prepared
# add --max-scan N to cap it, or omit for the whole corpus (~25 minutes, per
# situated-appraisal's own notes on the same file).
```

The raw corpus itself is never committed here — only the small derived files above.
