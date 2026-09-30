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

## E2a — QLoRA fine-tune attempt, 4-bit (Kaggle T4) — abandoned, no usable result (29-30 sep 2026)

**What happened.** First attempt used `bitsandbytes` 4-bit quantization (NF4) plus
`trl`'s `SFTTrainer` with a completion-only data collator. Neither survived contact
with the real environment:

1. `trl==1.14.1` (the version actually installed on Kaggle) no longer exports
   `DataCollatorForCompletionOnlyLM` — removed since whatever older version this
   project's first draft was written against. Switched to plain
   `transformers.Trainer` with hand-masked labels (`training.py`,
   `tokenize_example` / `CausalLMLabelCollator`).
2. Selecting Kaggle's "GPU T4 x2" accelerator (two GPUs) made `Trainer`
   automatically wrap the model in `torch.nn.DataParallel`. Combined with 4-bit
   quantization, this corrupted the quantized weights' state and crashed with `CUDA
   error: an illegal memory access`. Fixed by restricting to one GPU
   (`CUDA_VISIBLE_DEVICES=0`, `device_map={"": 0}`).
3. Training then ran, but at **~0.02 it/s — roughly 20-50x slower than a 1.5B model
   on a T4 should need.** `bitsandbytes` logged one `UserWarning` early on
   (`CUBLAS_STATUS_EXECUTION_FAILED when calling cublasLtMatmul ... Will attempt to
   recover by calling unfused cublas path`) and then went silent — Python's default
   warning filter shows a given `UserWarning` only once per call site, so the
   absence of further warnings is not evidence the slow fallback path stopped being
   used. The likeliest read: every forward pass silently took the slow path for the
   rest of the run.
4. The run was left going unattended for **~8 hours** (531/700 steps, loss trending
   0.24 → 0.13 — the fine-tune itself looked like it was working) with
   `save_strategy="no"`, on the assumption training would take minutes. It did not
   finish before the Kaggle session's hard time limit, and **no checkpoint had ever
   been written, so the entire run was unrecoverable** once the kernel state was
   lost. Nothing to report from this attempt except the cause and the fix.

**Reading.** Two independent, compounding mistakes: (a) not verifying the actual
`it/s` in the first few minutes before committing to an unattended multi-hour run,
and (b) not checkpointing periodically as a matter of course, regardless of how fast
a run is expected to be. Both are process failures, not modelling ones — recorded
here in the same spirit as the rest of this log, because a negative result about the
*pipeline* is still a result.

**Fix for the retry (E2b):** dropped `bitsandbytes`/4-bit entirely — a 1.5B model
fits a T4 in plain bf16 with room to spare, so quantization was buying memory
headroom this model size does not need, at the cost of a fused kernel that failed
silently. Also dropped gradient checkpointing (`prepare_model_for_kbit_training`),
the other plausible slowdown contributor, since bf16 activations fit without it.
Added checkpointing every 50 steps with automatic resume
(`transformers.trainer_utils.get_last_checkpoint`), so a future interruption loses
at most 50 steps.

## E2b — LoRA fine-tune, bf16, `Qwen/Qwen2.5-1.5B-Instruct` (Kaggle T4) — not yet run

Planned: `notebooks/kaggle_train.ipynb`, after the fixes in E2a above. Fill in after
a real run completes, with the same four metrics for the `lora` row, and the honest
reading of whether `paired_separation` clears 0.491 (E1) or not — either result gets
reported.

| Model | Accuracy | Balanced accuracy | AUC (overall) | Paired separation (within-task) |
|---|---|---|---|---|
| Majority class | 0.493 | 0.500 | 0.500 | 0.500 |
| Length only | 0.588 | 0.590 | 0.638 | 0.491 |
| LoRA (Qwen2.5-1.5B-Instruct, bf16) | — | — | — | — |
