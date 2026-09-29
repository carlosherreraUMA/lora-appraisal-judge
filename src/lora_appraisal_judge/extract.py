"""Turn a raw OpenHands trajectory record into one training example.

The corpus (`nebius/SWE-rebench-openhands-trajectories`, CC BY 4.0) stores each
trajectory as an ordered list of chat messages. What this project needs from that
list is the agent's closing account of what it did — the same "final message" that
`situated-appraisal-in-coding-agent-trajectories` computes indicators from, but here
the *text* is kept, because the task is to fine-tune on it, not just count it.

The closing-message logic (last assistant turn's prose, plus its `finish` tool-call
message, falling back to the last three assistant turns if the last one carries no
`finish` call) is adapted from that repository's own
`src/analysis/terminal_judgement.py` (same author, MIT-licensed) rather than
reimplemented blind, so the definition of "closing message" stays identical across
both projects.

Nothing here needs a GPU or the heavy training stack (torch, transformers, peft):
this module only needs `pyarrow` to stream the local corpus file, and plain dicts
otherwise, so it can be unit-tested without any of the Kaggle-only dependencies.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

#: How many trailing assistant turns to search for a `finish` call if the very last
#: one does not carry one. Matches situated-appraisal's terminal_judgement.py.
_FINISH_SEARCH_WINDOW = 3


def _finish_message(message: dict) -> str:
    """The `message` argument of a `finish` tool call in one assistant message, or ''."""
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") != "finish":
            continue
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (ValueError, TypeError):
            return ""
        if isinstance(args, dict) and isinstance(args.get("message"), str):
            return args["message"]
    return ""


def closing_message(trajectory: list[dict]) -> str:
    """The agent's closing account: last turn's prose plus its `finish` message.

    Taking only the prose undercounts badly — the agent often puts its whole summary
    in the `finish` tool argument and leaves the message body empty, or vice versa.
    """
    assistant = [m for m in trajectory if m.get("role") == "assistant"]
    if not assistant:
        return ""

    last = assistant[-1]
    finish = _finish_message(last)
    if not finish:
        for m in reversed(assistant[-_FINISH_SEARCH_WINDOW:]):
            finish = _finish_message(m)
            if finish:
                break
    prose = last.get("content") or ""

    parts = [p for p in (prose.strip(), finish.strip()) if p]
    return "\n\n".join(parts)


#: Only trajectories the agent ended by its own decision carry a meaningful closing
#: message — cut off by the iteration ceiling usually leaves none. Matches the scope
#: note in situated-appraisal's DATA_CARD.md.
_ELIGIBLE_EXIT_STATUS = "submit"


def extract_example(record: dict) -> dict | None:
    """One training example from one raw trajectory record, or None if ineligible.

    `record` is expected to carry the corpus's own field names: `trajectory_id`,
    `instance_id`, `exit_status`, `resolved`, `trajectory`. Works the same whether
    `record` came from the local parquet file (`iter_local_parquet`) or from a
    `datasets.load_dataset(..., streaming=True)` row on Kaggle — both expose the same
    schema, since they are the same corpus.
    """
    if record.get("exit_status") != _ELIGIBLE_EXIT_STATUS:
        return None
    message = closing_message(record.get("trajectory") or [])
    if not message:
        return None
    resolved = record.get("resolved")
    if resolved is None:
        return None
    return {
        "trajectory_id": record.get("trajectory_id", ""),
        "instance_id": record.get("instance_id", ""),
        "resolved": bool(int(resolved)),
        "closing_message": message,
        # A rough, honestly-named proxy for trajectory length: the message count, not
        # the agent-step count situated-appraisal's own `n_steps` uses. Good enough for
        # the length-control baseline in `baselines.py`; not claimed to be more.
        "n_messages": len(record.get("trajectory") or []),
    }


def iter_local_parquet(path: str, batch_size: int = 128) -> Iterator[dict]:
    """Stream raw records from a local copy of the corpus parquet file.

    Reads in small batches rather than loading the file whole — it is about 2 GB, and
    situated-appraisal's own notes record that a naive whole-row-group read exhausted
    memory on a small machine. Only used for local dataset preparation
    (`scripts/prepare_dataset.py`); the Kaggle notebook pulls the corpus straight from
    Hugging Face instead, so this function is not on that path and pyarrow is not
    needed there.
    """
    import pyarrow.parquet as pq

    columns = ["trajectory_id", "instance_id", "exit_status", "resolved", "trajectory"]
    f = pq.ParquetFile(path)
    for batch in f.iter_batches(batch_size=batch_size, columns=columns):
        cols = {c: batch.column(c).to_pylist() for c in columns if c != "trajectory"}
        traj_col = batch.column("trajectory")
        for i in range(batch.num_rows):
            rec = {c: cols[c][i] for c in cols}
            rec["trajectory"] = traj_col[i].as_py()
            yield rec
