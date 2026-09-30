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
3. Training then ran, but at **~0.02 it/s (about 50 s per optimizer step)**, when a
   rough estimate for a 1.5B model at 16 sequences of ~600 tokens per step on a T4 in
   fp16 is a few seconds per step: roughly **10x too slow**. The cause is in the one
   warning the run printed: `gemm_and_bias error: CUBLAS_STATUS_EXECUTION_FAILED
   when calling cublasLtMatmul ... abType 14 cType 14 computeType 68 ... Will
   attempt to recover by calling unfused cublas path`. In cuBLAS's `cudaDataType`,
   14 is `CUDA_R_16BF`: **bfloat16**. The run used bf16 compute
   (`bnb_4bit_compute_dtype=torch.bfloat16`, `bf16=True`) on a **T4, which is Turing,
   compute capability 7.5, with no native bf16 support** (Ampere, 8.0, introduced
   it). The fused matmul was rejected, and every forward pass fell back to a slow
   path. Python's default warning filter shows a given `UserWarning` once per call
   site, so the silence after the first warning was not evidence the fallback had
   stopped.

   **Correction, 30 sep 2026:** this entry first blamed `bitsandbytes`' 4-bit kernel
   and "fixed" it by switching to unquantized LoRA in *bf16*, which would have hit
   the same slow path. The dtype code in the warning says bf16 was the problem, not
   quantization. The first diagnosis was not checked against that evidence.
4. The run was left going unattended for **~8 hours** (531/700 steps, loss trending
   0.24 → 0.13 — the fine-tune itself looked like it was working) with
   `save_strategy="no"`, on the assumption training would take minutes. It did not
   finish before the Kaggle session's hard time limit, and **no checkpoint had ever
   been written, so the entire run was unrecoverable** once the kernel state was
   lost. Nothing to report from this attempt except the cause and the fix.

**Reading.** Three process failures, not modelling ones: (a) no measurement of the
actual speed before committing to an unattended multi-hour run; (b) no periodic
checkpointing; (c) the training logic lived in notebook cells, which Kaggle copies once
at import. Three times, a fix pushed to the repository did not reach the cells being
run, and each edit to the notebook file itself, which had no cell ids, drifted and
scrambled cell types. Recorded here in the same spirit as the rest of this log: a
negative result about the *pipeline* is still a result.

**Fixes for the retry (E2b), one per root cause:**
- **Speed:** the dtype is chosen from the GPU (`training.choose_dtype_name`: fp16 below
  compute capability 8.0, bf16 from 8.0) and printed at start. No 4-bit quantization
  (a 1.5B model is ~3 GB in 16-bit and fits a T4); no gradient checkpointing.
- **Blind launches:** `scripts/train_and_evaluate.py --smoke` trains 20 steps, scores
  20 test examples, and reports measured s/step, the projected full-run time and a
  verdict. The full run refuses to start (`--after-smoke`) unless that verdict found
  no problems (NaN loss, NaN scores, projection over 3 h).
- **Lost work:** checkpoints every 50 steps with automatic resume; each scored test
  example is appended to `test_scores.jsonl` as it goes, so evaluation resumes too.
- **Notebook drift:** all logic moved into `scripts/train_and_evaluate.py`; the
  notebook is a 9-cell launcher that clones or resets the repo to the latest commit
  and runs the script, rebuilt with an explicit id per cell.

## E2b — LoRA fine-tune, fp16 on T4, `Qwen/Qwen2.5-1.5B-Instruct` — not yet run

Planned: `notebooks/kaggle_train.ipynb` → `scripts/train_and_evaluate.py`, smoke run
first.

Setup note (30 sep 2026): on Kaggle's image, `get_peft_model` raised `ImportError:
Found an incompatible version of torchao. Found version 0.10.0, but only versions
above 0.16.0 are supported`. This came from peft's torchao dispatcher, which runs
even for plain LoRA. `scripts/kaggle_setup.sh` uninstalls torchao, which this project
does not use. `train_and_evaluate.py` checks for the problem before downloading the
model.

**Smoke run (30 sep 2026, Kaggle Tesla T4, compute 7.5, dtype float16 chosen
automatically):** 6.1 s/step, projected 700 steps ≈ 71 min; evaluation 0.93
s/example, projected 1,717 examples ≈ 27 min. Logged losses over 20 steps: 0.434,
0.199, 0.170, 0.164, all finite, so no fp16 overflow. Verdict: OK.

Against E2a's ~50 s/step this is about 8x faster, consistent with the bf16-on-T4
diagnosis. It does not isolate it: three things changed at once (bf16 → fp16, no
4-bit quantization, no gradient checkpointing). Attributing the speedup to one of
them would need a run changing one at a time, which is not worth the GPU quota here. Fill in after a real run completes with the same four metrics for the `lora`
row, the smoke report's measured speed, and the honest reading of whether
`paired_separation` clears 0.491 (E1). Either result gets reported.

| Model | Accuracy | Balanced accuracy | AUC (overall) | Paired separation (within-task) |
|---|---|---|---|---|
| Majority class | 0.493 | 0.500 | 0.500 | 0.500 |
| Length only | 0.588 | 0.590 | 0.638 | 0.491 |
| LoRA (Qwen2.5-1.5B-Instruct) | — | — | — | — |
