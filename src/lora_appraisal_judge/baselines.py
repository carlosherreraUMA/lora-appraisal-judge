"""Cheap baselines the fine-tuned model has to beat.

The point of this module is not accuracy, it is honesty: `situated-appraisal`'s own
finding 2 is that trajectory length is a control every statistic must beat, and its
finding 3 is that the closing message's own claim of success is uninformative
(~98% of both outcomes claim success). Any LoRA result gets compared against both,
so "the model learned something about the text" and "the model learned to be a
longer-context length counter" stay distinguishable.

Only `scikit-learn`, already a project dependency elsewhere in this CV's repos, and
no GPU — these run on a laptop in milliseconds.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression


def majority_baseline(train_examples: list[dict]) -> bool:
    """The single most common label in `train_examples`.

    A classifier this simple can still score above 0.5 accuracy if the split is
    imbalanced — reported alongside the others precisely so a headline accuracy
    number is never read without it.
    """
    counts = Counter(ex["resolved"] for ex in train_examples)
    return counts.most_common(1)[0][0]


def majority_scores(train_examples: list[dict], test_examples: list[dict]) -> np.ndarray:
    """Constant score per test example: 1.0 if the majority class is resolved, else 0.0.

    A constant score gives an AUC of exactly 0.5 by construction, which is the point:
    it is the "no separation" reference the paired statistic in `metrics.py` is
    measured against.
    """
    label = majority_baseline(train_examples)
    return np.full(len(test_examples), 1.0 if label else 0.0)


class LengthBaseline:
    """Logistic regression on `n_messages` alone.

    Deliberately the simplest model that could plausibly work, on the same feature
    situated-appraisal's own length control (finding 2, decision D24) uses in spirit
    — not the identical `n_steps` feature (this project only extracts message count,
    see `extract.py`), so the analogy is named as a proxy, not claimed as a replica.
    """

    def __init__(self) -> None:
        self._model = LogisticRegression()

    def fit(self, train_examples: list[dict]) -> "LengthBaseline":
        X = np.array([[ex["n_messages"]] for ex in train_examples])
        y = np.array([int(ex["resolved"]) for ex in train_examples])
        if len(set(y.tolist())) < 2:
            # A constant training label makes sklearn's fit degenerate; fall back to
            # the majority baseline's constant-score behaviour rather than crash.
            self._model = None
            self._constant = float(y[0])
        else:
            self._model = self._model.fit(X, y)
            self._constant = None
        return self

    def scores(self, examples: list[dict]) -> np.ndarray:
        if self._model is None:
            return np.full(len(examples), self._constant)
        X = np.array([[ex["n_messages"]] for ex in examples])
        return self._model.predict_proba(X)[:, 1]
