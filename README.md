# lora-appraisal-judge

LoRA fine-tuning of a small open model to judge a coding agent's own closing
message — built to get real, evaluated LoRA/Hugging Face experience for
research-engineer applications, and to extend
[`situated-appraisal-in-coding-agent-trajectories`](https://github.com/carlosherreraUMA/situated-appraisal-in-coding-agent-trajectories)'s
finding that the agent's own claim of success carries no information.

## The question

That project's finding 3: about 98% of both failed and successful agent runs close
with a first-person claim of having solved the task ("I have successfully
implemented / fixed / resolved …"). The claim itself is uninformative. This project
asks a narrower, honest follow-up: **reading the same closing message, not just its
surface claim, can a small model fine-tuned for the job do any better** — and does
that survive a **within-task paired comparison**, which controls for task difficulty
the way the source project's own design (decision D11) does?

The comparison that matters is the *paired* one, not plain accuracy on an unpaired
test set. `EXPERIMENTS.md` (E1) already shows why: a trajectory-length baseline
looks informative in the aggregate (AUC 0.638) but that separation collapses to
chance (0.491) once within-task pairing controls for the fact that harder tasks
produce both longer trajectories and more failures — the same shape of confound
D24 identifies in the source project, showing up again in a fresh derived task.

## What's here

| | |
|---|---|
| `src/lora_appraisal_judge/extract.py` | Closing-message extraction from a raw trajectory record. Pure functions, no GPU. |
| `src/lora_appraisal_judge/prompts.py` | Prompt/response format for the fine-tuning task. |
| `src/lora_appraisal_judge/splits.py` | Train/val/test split by task, and within-task pairing. |
| `src/lora_appraisal_judge/baselines.py` | Majority-class and trajectory-length baselines (scikit-learn, no GPU). |
| `src/lora_appraisal_judge/metrics.py` | Accuracy, balanced accuracy, AUC, and the within-task paired-separation statistic. |
| `src/lora_appraisal_judge/training.py` | Dtype choice per GPU, model loading (16-bit, no quantization), LoRA, prompt-masked tokenization, continuous scoring. The model-facing parts need `torch`/`transformers`/`peft` (the `train` extra) and a GPU; the pure helpers are tested without them. |
| `scripts/prepare_dataset.py` | Builds `data/prepared/` from a local copy of the raw corpus. |
| `scripts/train_and_evaluate.py` | The whole GPU run: baselines, LoRA training (checkpointed, resumable), evaluation (resumable), `results.json`. `--smoke` measures speed and projects the full run's time before committing to it. |
| `notebooks/kaggle_train.ipynb` | A launcher only: clones or resets this repo to the latest commit and runs the script, smoke test first. Nothing in it needs editing when the code changes. |
| `data/prepared/` | The train/val/test split used by the notebook — committed, small, CC BY 4.0 (see `NOTICE.md`). |
| `EXPERIMENTS.md` | One entry per run, numbers as they came out, negative results kept. |

## Why a T4 on Kaggle, and why a 1.5B model

No local GPU. Kaggle Notebooks give a free T4 (16 GB) with a weekly quota, which fits
a small instruct model in 16-bit without quantization, but not a large one. This is
fine-tuning at the scale a single free GPU affords, not a claim of experience at
production LLM training scale. The T4 has no native bf16, so the script picks fp16
there. `EXPERIMENTS.md` (E2a) records what running bf16 on it cost.

## Reproducing

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                     # all of src/ except training.py — no GPU needed, seconds
python scripts/prepare_dataset.py --raw-path <path-to-corpus> --out-dir data/prepared
```

Then import `notebooks/kaggle_train.ipynb` into Kaggle (GPU T4 accelerator, Internet
on) and follow its first cell: smoke test, read the verdict, then the full run. On
any machine with a GPU, the same thing without the notebook:

```
pip install -e ".[train]"
python -u scripts/train_and_evaluate.py --smoke --out out/smoke
python -u scripts/train_and_evaluate.py --out out/run --after-smoke out/smoke
```

## Licence and attribution

Code: MIT. Documentation and the prepared dataset: CC BY 4.0. See `NOTICE.md` for
full attribution to the source corpus (`nebius/SWE-rebench-openhands-trajectories`,
Nebius) and to `situated-appraisal-in-coding-agent-trajectories`, whose
closing-message definition and pairing methodology this project reuses.

Author: Carlos Herrera Pérez.
