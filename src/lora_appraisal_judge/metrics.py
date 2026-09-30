"""Evaluation metrics, including the within-task paired statistic.

`paired_separation` is the same shape of statistic as the six signals reported in
`situated-appraisal`'s `report/terminal_judgement.md`: for each (resolved,
unresolved) pair sharing a task, does the score rank the resolved attempt above the
unresolved one? 0.5 means no separation; that report found all six of its own
signals land between 0.47 and 0.51. This project's honest headline number is this
one, not plain accuracy, for the same reason that report gives: plain accuracy on an
imbalanced, non-paired set rewards a classifier for learning the base rate, not for
reading the message.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def accuracy(y_true: list[bool], y_pred: list[bool | None]) -> float:
    """Fraction of exact matches. A `None` prediction (see `prompts.parse_prediction`)
    always counts as wrong — a model that does not answer is not given credit for it."""
    if not y_true:
        return float("nan")
    return sum(1 for t, p in zip(y_true, y_pred) if p is not None and t == p) / len(y_true)


def balanced_accuracy(y_true: list[bool], y_pred: list[bool | None]) -> float:
    """`accuracy`, but averaged per class rather than per example.

    Substitutes the *opposite* label for a `None` prediction (guaranteeing it is
    scored wrong) rather than dropping it, so a model that refuses to answer on one
    class entirely cannot be flattered by exclusion.
    """
    if not y_true:
        return float("nan")
    filled = [p if p is not None else (not t) for t, p in zip(y_true, y_pred)]
    return balanced_accuracy_score(y_true, filled)


def auc(y_true: list[bool], scores: np.ndarray) -> float:
    """Overall AUC of a continuous score against the true label.

    Undefined (returns NaN rather than raising) when the split has only one class —
    happens on small test slices and should be visible as a gap in a results table,
    not a stack trace.
    """
    if len(set(y_true)) < 2:
        return float("nan")
    if not np.all(np.isfinite(np.asarray(scores, dtype=float))):
        # A NaN score (fp16 overflow) makes roc_auc_score raise; report the gap
        # instead of losing every other number in the results at the last step.
        return float("nan")
    return roc_auc_score(y_true, scores)


def paired_separation(
    pairs: list[tuple[dict, dict]],
    score_by_trajectory_id: dict[str, float],
) -> float:
    """Fraction of within-task (resolved, unresolved) pairs the score ranks correctly.

    `pairs` is `splits.build_pairs` output. `score_by_trajectory_id` maps a
    `trajectory_id` to a continuous score (higher = more confident the model or
    baseline is that the task was resolved) — from a baseline's `.scores()`, or from
    a fine-tuned model's P(RESOLVED) over its first generated token, computed in the
    Kaggle notebook where the tokenizer is available.

    Ties are split evenly (0.5 credit), matching the standard paired-comparison /
    Mann-Whitney convention and situated-appraisal's own treatment of ties in its
    six-signal comparison.
    """
    if not pairs:
        return float("nan")
    total = 0.0
    for resolved_ex, unresolved_ex in pairs:
        r = score_by_trajectory_id[resolved_ex["trajectory_id"]]
        u = score_by_trajectory_id[unresolved_ex["trajectory_id"]]
        if r > u:
            total += 1.0
        elif r == u:
            total += 0.5
    return total / len(pairs)


def summary(
    test_examples: list[dict],
    y_pred: list[bool | None],
    scores: list[float],
    pairs: list[tuple[dict, dict]],
) -> dict[str, float]:
    """The four reported metrics for one model, as plain floats (JSON-serialisable).

    One function for every row of the results table (majority, length, LoRA), so
    the rows cannot drift apart by being computed with slightly different code.
    """
    y_true = [ex["resolved"] for ex in test_examples]
    score_by_id = {ex["trajectory_id"]: float(s) for ex, s in zip(test_examples, scores)}
    return {
        "accuracy": float(accuracy(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy(y_true, y_pred)),
        "auc": float(auc(y_true, list(scores))),
        "paired_separation": float(paired_separation(pairs, score_by_id)),
    }
