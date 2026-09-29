# Licences and attribution

| What | Licence |
|---|---|
| Source code (`src/`, `tests/`, `scripts/`, `notebooks/`, `Makefile`, `pyproject.toml`) | MIT, see `LICENSE` |
| Documentation (`README.md`, `EXPERIMENTS.md`, `data/README.md`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Prepared dataset (`data/prepared/`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), inherited from the source corpus below |

## Data attribution

`data/prepared/` is derived from **`nebius/SWE-rebench-openhands-trajectories`**
(Nebius), CC BY 4.0. 67,074 trajectories of OpenHands v0.54.0 with
Qwen3-Coder-480B-A35B-Instruct on real GitHub issues, revision
`35455389ab51bf5e2306bfd436ef72d0f98bf882`.
https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories

**Changes made**, as CC BY 4.0 requires stating: a subsample (9,000 of 67,074
trajectories, see `EXPERIMENTS.md`) was filtered to self-terminated runs with a
non-empty closing message, reduced to `{trajectory_id, instance_id, resolved,
n_messages, closing message text}`, formatted as an instruction/response pair for
supervised fine-tuning, and split into train/val/test by task (`instance_id`).
Unlike `situated-appraisal-in-coding-agent-trajectories`'s own derived tables, the
closing-message *text* is kept here rather than reduced to indicators, because the
task is to fine-tune on it.

## Method attribution

The closing-message extraction logic (`src/lora_appraisal_judge/extract.py`) and the
within-task pairing methodology (`src/lora_appraisal_judge/splits.py`,
`metrics.paired_separation`) are adapted from
[`situated-appraisal-in-coding-agent-trajectories`](https://github.com/carlosherreraUMA/situated-appraisal-in-coding-agent-trajectories)
(same author, MIT-licensed), specifically `src/analysis/terminal_judgement.py` and
its within-task paired comparison (design decision D11), so that both projects use
the same definitions rather than two independently-drifting ones.

The trajectories quote source code and issue text from third-party open-source
repositories, which keep their own licences.
