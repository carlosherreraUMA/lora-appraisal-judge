# Experiment log

Format follows `creatures/EXPERIMENTS.md`: one entry per run, numbers as they came
out, negative results kept.

## E1 — Baselines on the prepared split (28 sep 2026, no GPU)

**Setup.** `scripts/prepare_dataset.py` against a local copy of
`nebius/SWE-rebench-openhands-trajectories` (revision `35455389ab...`), scanning the
first 9,000 raw trajectories (of 67,074) and keeping the 8,128 eligible
(`exit_status == "submit"` and a non-empty closing message). Split by `instance_id`
(`splits.split_by_instance`, default 70/10/20): 5,594 train / 817 val / 1,717 test,
with 106 within-task (resolved, unresolved) pairs in the test split
(`splits.build_pairs`). Base rate: 49.3% resolved in the test split.

**Results** (`baselines.py` + `metrics.py`, exact numbers from this run):

| Model | Accuracy | Balanced accuracy | AUC (overall) | Paired separation (within-task) |
|---|---|---|---|---|
| Majority class | 0.493 | 0.500 | 0.500 | 0.500 |
| Length only (`n_messages`, logistic regression) | 0.588 | 0.590 | 0.638 | **0.491** |

**Reading.** The length baseline looks informative in the aggregate (AUC 0.638) but
that separation **collapses under the within-task paired comparison** (0.491 — no
better than chance, ties split evenly). This is the same shape of confound
`situated-appraisal`'s own decision D24 controls for: harder tasks tend to produce
both longer trajectories and more failures, so a length-outcome correlation appears
in an unpaired comparison without trajectory length actually carrying information
once task difficulty is held fixed by pairing. This is the honest floor a LoRA
result has to clear on the **paired** column, not the aggregate one — see the
project's `README.md` for why the paired statistic is the one that matters here.

**Caveats.**
- 106 pairs is a small denominator; a paired-separation figure at this size moves by
  about ±0.05 for a single pair flipping. Read the qualitative conclusion (collapses
  towards 0.5), not the third decimal.
- Only 9,000 of 67,074 raw trajectories were scanned, for speed. Re-running
  `prepare_dataset.py` without `--max-scan` would use the whole corpus; not done for
  this entry.
- `n_messages` (message count) is this project's own proxy for trajectory length, not
  `situated-appraisal`'s `n_steps` (agent-step count) — see the note in
  `baselines.py`. The two should correlate but are not identical.

## E2 — QLoRA fine-tune, `Qwen/Qwen2.5-1.5B-Instruct` (Kaggle T4) — not yet run

Planned: `notebooks/kaggle_train.ipynb`. Fill in after a real run, with the same four
metrics for the `lora` row, and the honest reading of whether `paired_separation`
clears 0.491 (E1) or not — either result gets reported.

| Model | Accuracy | Balanced accuracy | AUC (overall) | Paired separation (within-task) |
|---|---|---|---|---|
| Majority class | 0.493 | 0.500 | 0.500 | 0.500 |
| Length only | 0.588 | 0.590 | 0.638 | 0.491 |
| LoRA (Qwen2.5-1.5B-Instruct) | — | — | — | — |
