"""Generate the Kaggle launcher notebooks, with an explicit id per cell.

- notebooks/kaggle_train.ipynb: training and evaluation (E2b).
- notebooks/kaggle_e3_vllm.ipynb: the E2b adapter scored with vLLM on the
  full-corpus test set (E3).

Edit this file and re-run it (python scripts/build_notebook.py) to change the
notebooks; do not edit the .ipynb files by hand.

The previous notebook had no `id` fields (mandatory in nbformat 4.5), so cell edits
were resolved by position and drifted after an insert, scrambling cell types.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(cell_id, src):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(cell_id, src):
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


train_cells = [
    md("intro", """
# lora-appraisal-judge — Kaggle launcher

This notebook only launches. All the logic (baselines, LoRA training, evaluation,
checkpointing, resume) lives in
[`scripts/train_and_evaluate.py`](https://github.com/carlosherreraUMA/lora-appraisal-judge/blob/master/scripts/train_and_evaluate.py)
in the repository. The next cell clones the repo, or resets it to the latest commit,
so a fix pushed there is picked up by re-running this notebook. No cell here should
ever need editing by hand.

**Settings (right-hand panel):** Accelerator → **GPU T4** (x2 is fine, the script
uses one) · Internet → **On**.

**Order:** run steps 0 and 1 interactively and read the smoke verdict. If it says OK,
run step 2. For a long unattended run, use **Save Version → Save & Run All
(Commit)**: it runs in the background with the browser closed, and whatever is in
`/kaggle/working` shows up in that version's **Output** tab.
"""),
    code("clone", """
import os
import subprocess

REPO_URL = "https://github.com/carlosherreraUMA/lora-appraisal-judge"
REPO_DIR = "/kaggle/working/repo"

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "-q", "origin"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "-q", "--hard", "origin/master"], check=True)
else:
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)

# The commit being run: compare it with the latest one on GitHub if in doubt.
subprocess.run(["git", "-C", REPO_DIR, "log", "-1", "--format=%h %cd %s", "--date=short"], check=True)
"""),
    code("install", """
# Installs what the image lacks and removes what breaks it. The steps live in the
# repo (scripts/kaggle_setup.sh), so an environment fix arrives by re-cloning.
!bash {REPO_DIR}/scripts/kaggle_setup.sh
"""),
    md("smoke-md", """
## Step 1 — smoke test (about 5 minutes)

Trains 20 steps and scores 20 test examples, then prints the GPU, the dtype chosen
for it, the measured seconds per step, the projected time of the full run and a
**VERDICT**. Do not skip it: it is what would have caught the first attempt, which
ran eight hours at a tenth of the expected speed.
"""),
    code("smoke", """
!python -u {REPO_DIR}/scripts/train_and_evaluate.py --smoke --out /kaggle/working/smoke
"""),
    md("full-md", """
## Step 2 — full run

Refuses to start unless the smoke report above found no problems. Checkpoints every
50 steps and writes each scored test example as it goes: if the session dies, run the
notebook again and it resumes from where it stopped.
"""),
    code("full", """
!python -u {REPO_DIR}/scripts/train_and_evaluate.py --out /kaggle/working/run --after-smoke /kaggle/working/smoke
"""),
    md("results-md", """
## Results

`/kaggle/working/run/results.json`: the four metrics for majority, length and LoRA,
plus the environment (GPU, dtype, library versions) and training timings. Paste it
back to Claude to fill in `EXPERIMENTS.md`.
"""),
    code("results", """
print(open("/kaggle/working/run/results.json").read())
"""),
]

clone_cell = next(c for c in train_cells if c["id"] == "clone")

e3_cells = [
    md("intro", """
# lora-appraisal-judge — E3 launcher (vLLM)

Scores the adapter trained in E2b, unchanged, on the test set rebuilt from the full
corpus (`data/e3/`: 12,691 examples, 319 tasks with both outcomes, 6,426 within-task
pairs, against 63 tasks and 106 pairs in E2b), with vLLM instead of one-by-one
Hugging Face generation. All the logic is in
[`scripts/evaluate_vllm.py`](https://github.com/carlosherreraUMA/lora-appraisal-judge/blob/master/scripts/evaluate_vllm.py).

**Settings (right-hand panel):** Accelerator → **GPU T4** · Internet → **On**.

**Input:** Add Input → Your Work → Notebooks → the **E2b training notebook**, the
version whose Output has `run/adapter/` and `run/test_scores.jsonl`. The script finds
both under `/kaggle/input` by itself.

**Order:** steps 0 and 1 interactively, read the smoke verdict, then step 2 (or
**Save & Run All** for an unattended run; `/kaggle/working` lands in the Output tab).
"""),
    clone_cell,
    code("install", """
# vLLM pins its own torch, so this replaces the image's torch (a few minutes).
!bash {REPO_DIR}/scripts/kaggle_setup_vllm.sh
"""),
    md("smoke-md", """
## Step 1 — smoke test (about 10 minutes, mostly loading)

Checks that vLLM reproduces E2b's scores on 200 of E2b's own test examples (if it
does not, nothing else counts), times 500 E3 examples and projects the full run. It
tries vLLM's LoRA path first and, if that fails on the T4, the adapter merged into
the base weights, each in a fresh process. Ends with a **VERDICT**.
"""),
    code("smoke", """
!python -u {REPO_DIR}/scripts/evaluate_vllm.py --smoke --out /kaggle/working/e3-smoke
"""),
    md("full-md", """
## Step 2 — full run

Refuses to start unless the smoke verdict was OK, and uses the mode that passed.
Agreement with E2b on all 1,717 examples first, then the E3 set; scores are appended
chunk by chunk, so re-running after an interruption resumes.
"""),
    code("full", """
!python -u {REPO_DIR}/scripts/evaluate_vllm.py --out /kaggle/working/e3 --after-smoke /kaggle/working/e3-smoke
"""),
    md("results-md", """
## Results

`/kaggle/working/e3/results.json`: agreement with E2b, the E3 metrics for majority,
length and LoRA, the generated-label distribution and task-clustered bootstrap
intervals. Paste it back to Claude to fill in `EXPERIMENTS.md`.
"""),
    code("results", """
print(open("/kaggle/working/e3/results.json").read())
"""),
]


def write(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                       "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    with open(path, "w") as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {path}: {len(cells)} cells")


write("kaggle_train.ipynb", train_cells)
write("kaggle_e3_vllm.ipynb", e3_cells)
