"""Train/val/test splitting and within-task pairing.

Several trajectories in the corpus share an `instance_id` (the same GitHub issue,
attempted more than once — a mean of 10.6 rollouts per instance across the corpus,
per situated-appraisal's `data/SOURCES.md`). Splitting by trajectory would let the
same task appear in both train and test, which would let a model memorise
task-specific vocabulary rather than learn anything about closing messages in
general. Splitting is by `instance_id` instead, so a task is wholly in one split.

`build_pairs` recreates, on whatever split it is given, the within-task paired
comparison that is the load-bearing methodology of
`situated-appraisal-in-coding-agent-trajectories` (design decision D11): pairing a
resolved and an unresolved attempt at the *same* task controls for task difficulty
by construction, because the two attempts share the model, scaffold and issue text,
and differ only in the random seed.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict


def _bucket(instance_id: str, n_buckets: int = 1000) -> int:
    """A stable, seed-free hash bucket in [0, n_buckets) for one instance_id.

    Stable across processes and Python versions (unlike the built-in `hash`, which is
    salted per-process for strings) so a train/test split is reproducible without
    having to persist the split itself.
    """
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def split_by_instance(
    examples: list[dict],
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> dict[str, list[dict]]:
    """Partition `examples` into train/val/test, grouped by `instance_id`.

    Deterministic: the same `instance_id` always lands in the same split, for any
    input list, as long as the fractions are unchanged — there is no shuffling and no
    seed to lose track of.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must leave room for a train split")

    val_cut = int(1000 * val_frac)
    test_cut = val_cut + int(1000 * test_frac)

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for ex in examples:
        b = _bucket(ex["instance_id"])
        if b < val_cut:
            out["val"].append(ex)
        elif b < test_cut:
            out["test"].append(ex)
        else:
            out["train"].append(ex)
    return out


def build_pairs(examples: list[dict]) -> list[tuple[dict, dict]]:
    """Every (resolved, unresolved) pair sharing an `instance_id`.

    Pairs every resolved attempt at a task with every unresolved attempt at the same
    task (not just one-to-one), matching how situated-appraisal's own paired analysis
    is described (§ within-task pairing) — a task with 3 resolved and 2 unresolved
    attempts contributes 6 pairs. Tasks with only one outcome contribute none, since
    there is nothing to control for within them.
    """
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_instance[ex["instance_id"]].append(ex)

    pairs: list[tuple[dict, dict]] = []
    for group in by_instance.values():
        resolved = [e for e in group if e["resolved"]]
        unresolved = [e for e in group if not e["resolved"]]
        for r in resolved:
            for u in unresolved:
                pairs.append((r, u))
    return pairs
