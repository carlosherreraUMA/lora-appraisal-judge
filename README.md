# lora-appraisal-judge

QLoRA fine-tuning of a small open model to judge a coding agent's own closing
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
| `src/lora_appraisal_judge/training.py` | QLoRA loading, LoRA config, SFT dataset formatting, and continuous scoring — needs `torch`/`transformers`/`peft`/`bitsandbytes` (the `train` extra), so it only runs on a GPU machine. |
| `scripts/prepare_dataset.py` | Builds `data/prepared/` from a local copy of the raw corpus. |
| `notebooks/kaggle_train.ipynb` | The actual fine-tuning run: clone this repo, install the training extras, load the model in 4-bit, attach a LoRA adapter, fine-tune, evaluate. Written for a single Kaggle T4. |
| `data/prepared/` | The train/val/test split used by the notebook — committed, small, CC BY 4.0 (see `NOTICE.md`). |
| `EXPERIMENTS.md` | One entry per run, numbers as they came out, negative results kept. |

## Why a T4 on Kaggle, and why a 1.5B model

No local GPU. Kaggle Notebooks give a free T4 (16 GB) with a generous weekly quota,
which fits a small instruct model in 4-bit (QLoRA) comfortably but not a large one —
said so plainly rather than implied: this is fine-tuning at the scale a single free
GPU affords, not a claim of experience at production LLM training scale.

## Reproducing

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                     # all of src/ except training.py — no GPU needed, seconds
python scripts/prepare_dataset.py --raw-path <path-to-corpus> --out-dir data/prepared
```

Then open `notebooks/kaggle_train.ipynb` on Kaggle (GPU T4 accelerator on), set
`GITHUB_REPO_URL` in its second cell to this repo's URL, and run it top to bottom.

## Licence and attribution

Code: MIT. Documentation and the prepared dataset: CC BY 4.0. See `NOTICE.md` for
full attribution to the source corpus (`nebius/SWE-rebench-openhands-trajectories`,
Nebius) and to `situated-appraisal-in-coding-agent-trajectories`, whose
closing-message definition and pairing methodology this project reuses.

Author: Carlos Herrera Pérez.
