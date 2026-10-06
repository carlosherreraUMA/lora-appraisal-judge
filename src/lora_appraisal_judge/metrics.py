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


def task_bootstrap(
    examples: list[dict],
    scores: list[float],
    n_resamples: int = 5000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Percentile intervals for AUC and paired separation, resampling whole tasks.

    Attempts at the same task are not independent: they share the issue text and its
    difficulty. Resampling examples would treat them as independent and give
    intervals that are too narrow. Here a task is drawn with all its attempts and,
    for the paired statistic, all its pairs (`splits.build_pairs` semantics: every
    resolved × unresolved combination, ties at 0.5).

    E1's interval for the length baseline (30 sep 2026) was computed the same way
    outside the repository; this is the in-repo version.
    """
    by_task: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_task.setdefault(ex["instance_id"], []).append(i)
    tasks = [np.asarray(idx) for idx in by_task.values()]
    y = np.array([bool(ex["resolved"]) for ex in examples])
    s = np.asarray(scores, dtype=float)

    # Per task: credited pairs (wins + half the ties) and number of pairs.
    wins = np.zeros(len(tasks))
    n_pairs = np.zeros(len(tasks))
    for t, idx in enumerate(tasks):
        r = s[idx[y[idx]]]
        u = s[idx[~y[idx]]]
        if len(r) and len(u):
            diff = r[:, None] - u[None, :]
            wins[t] = (diff > 0).sum() + 0.5 * (diff == 0).sum()
            n_pairs[t] = diff.size

    rng = np.random.default_rng(seed)
    aucs = np.empty(n_resamples)
    paired = np.empty(n_resamples)
    for b in range(n_resamples):
        draw = rng.integers(0, len(tasks), size=len(tasks))
        idx = np.concatenate([tasks[t] for t in draw])
        aucs[b] = auc(list(y[idx]), s[idx])
        n = n_pairs[draw].sum()
        paired[b] = wins[draw].sum() / n if n else np.nan

    alpha = (1.0 - level) / 2.0
    pairs_total = n_pairs.sum()

    def interval(point: float, draws: np.ndarray) -> dict[str, float]:
        lo, hi = np.nanquantile(draws, [alpha, 1.0 - alpha])
        return {"point": float(point), "lo": float(lo), "hi": float(hi)}

    return {
        "auc": interval(auc(list(y), s), aucs),
        "paired_separation": interval(
            wins.sum() / pairs_total if pairs_total else float("nan"), paired
        ),
        "n_tasks": len(tasks),
        "n_tasks_with_pairs": int((n_pairs > 0).sum()),
        "n_pairs": int(pairs_total),
        "n_resamples": n_resamples,
        "level": level,
    }


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
