"""Prompt and label format for the fine-tuning task.

Task: given only the agent's own closing message from a coding-agent trajectory,
predict whether the task was actually resolved. The paper this project extends
(`situated-appraisal-in-coding-agent-trajectories`, finding 3) found that the
message's own *claim* of success is uninformative — about 98% of both failed and
successful runs claim success. The question here is narrower and honest about that:
not "does the agent's claim say so" (it is known not to), but "does a fine-tuned
small model, reading the same text, do any better than that claim, or than the
trajectory-length control (`report/terminal_judgement.md`, finding 2)?"

Kept deliberately simple: one instruction, the closing message, one of two labels.
No chain-of-thought target, so the comparison against the length-only baseline
(`baselines.py`) is about what the message text carries, not about reasoning ability.
"""
from __future__ import annotations

import re

INSTRUCTION = (
    "You are reviewing a coding agent's own closing report on a software "
    "engineering task. Based only on this closing message, decide whether the "
    "agent actually resolved the task. Agents often claim success whether or not "
    "the task was actually solved, so read for concrete evidence, not just the "
    "claim. Respond with exactly one word: RESOLVED or UNRESOLVED."
)

RESOLVED_LABEL = "RESOLVED"
UNRESOLVED_LABEL = "UNRESOLVED"

#: Closing messages run up to ~28k characters in the corpus (median ~101, p90 ~566
#: per situated-appraisal's data/SOURCES.md profiling). Truncated to keep sequences
#: short enough to train cheaply on a single T4; truncation itself is a design choice
#: worth stating plainly, not hiding in a default argument deep in training code.
DEFAULT_MAX_CHARS = 2_000


def truncate_message(message: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The closing message, cut to `max_chars` from the end.

    Cutting from the end rather than the start keeps the `finish` call's explicit
    message (appended last by `extract.closing_message`), which is usually where any
    hedge or admission of failure would be, at the cost of dropping earlier prose on
    long messages.
    """
    message = message.strip()
    if len(message) <= max_chars:
        return message
    return "…" + message[-max_chars:]


def build_prompt(closing_message: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The user-turn text: instruction plus the (possibly truncated) closing message."""
    text = truncate_message(closing_message, max_chars=max_chars)
    return f"{INSTRUCTION}\n\nClosing message:\n\"\"\"\n{text}\n\"\"\""


def build_target(resolved: bool) -> str:
    """The single-word label the model is trained to produce."""
    return RESOLVED_LABEL if resolved else UNRESOLVED_LABEL


def format_example(example: dict, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """One `extract.extract_example` output as a `{prompt, response}` SFT pair.

    Chat-template formatting (system/user/assistant turns, special tokens) is left to
    `training.py`, which has the specific tokenizer in hand on Kaggle — this stays a
    plain-string function so it is testable without transformers installed.
    """
    return {
        "trajectory_id": example.get("trajectory_id", ""),
        "instance_id": example.get("instance_id", ""),
        "resolved": bool(example["resolved"]),
        # Carried through, not used by the prompt itself, so `baselines.py` can read
        # the same prepared JSONL files instead of needing a second, separate export.
        "n_messages": example.get("n_messages", 0),
        "prompt": build_prompt(example["closing_message"], max_chars=max_chars),
        "response": build_target(bool(example["resolved"])),
    }


#: Word-boundary matching, not plain substring containment: "RESOLVED" is a
#: substring of "UNRESOLVED", so `"RESOLVED" in text` would also match on the
#: negative label and every response would look ambiguous.
_RESOLVED_RE = re.compile(rf"\b{RESOLVED_LABEL}\b")
_UNRESOLVED_RE = re.compile(rf"\b{UNRESOLVED_LABEL}\b")


def parse_prediction(generated_text: str) -> bool | None:
    """RESOLVED -> True, UNRESOLVED -> False, anything else -> None (no verdict).

    A model that does not answer in the trained format is a result, not a crash —
    `metrics.py` counts `None` as a miss rather than guessing a direction for it.
    """
    text = generated_text.strip().upper()
    has_resolved = bool(_RESOLVED_RE.search(text))
    has_unresolved = bool(_UNRESOLVED_RE.search(text))
    if has_resolved and not has_unresolved:
        return True
    if has_unresolved and not has_resolved:
        return False
    return None
