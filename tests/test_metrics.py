import math

from lora_appraisal_judge.metrics import (
    accuracy,
    auc,
    balanced_accuracy,
    paired_separation,
    summary,
)


def test_accuracy_counts_none_predictions_as_wrong():
    y_true = [True, True, False, False]
    y_pred = [True, None, False, True]
    assert accuracy(y_true, y_pred) == 0.5  # 2 of 4 correct


def test_accuracy_empty_is_nan():
    assert math.isnan(accuracy([], []))


def test_balanced_accuracy_all_correct_is_one():
    y_true = [True, True, False, False]
    y_pred = [True, True, False, False]
    assert balanced_accuracy(y_true, y_pred) == 1.0


def test_balanced_accuracy_none_prediction_counts_as_the_opposite_label():
    y_true = [True, False]
    y_pred = [None, None]
    # None -> not(True) = False (wrong for the first), None -> not(False) = True (wrong for the second)
    assert balanced_accuracy(y_true, y_pred) == 0.0


def test_auc_perfect_separation():
    y_true = [False, False, True, True]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auc(y_true, scores) == 1.0


def test_auc_single_class_is_nan():
    assert math.isnan(auc([True, True], [0.5, 0.6]))


def _pair(resolved_id, unresolved_id):
    return (
        {"trajectory_id": resolved_id},
        {"trajectory_id": unresolved_id},
    )


def test_paired_separation_all_correct():
    pairs = [_pair("r1", "u1"), _pair("r2", "u2")]
    scores = {"r1": 0.9, "u1": 0.1, "r2": 0.7, "u2": 0.3}
    assert paired_separation(pairs, scores) == 1.0


def test_paired_separation_matches_the_papers_no_separation_reading():
    # Constant score for everyone: exactly what a majority baseline produces.
    pairs = [_pair("r1", "u1"), _pair("r2", "u2")]
    scores = {"r1": 0.5, "u1": 0.5, "r2": 0.5, "u2": 0.5}
    assert paired_separation(pairs, scores) == 0.5  # ties split evenly


def test_paired_separation_empty_is_nan():
    assert math.isnan(paired_separation([], {}))


def test_auc_with_a_nan_score_is_nan_not_a_crash():
    assert math.isnan(auc([False, True], [0.1, float("nan")]))


def test_summary_rows_are_plain_floats_and_consistent():
    test = [
        {"trajectory_id": "r1", "resolved": True},
        {"trajectory_id": "u1", "resolved": False},
    ]
    pairs = [(test[0], test[1])]
    out = summary(test, [True, False], [0.9, 0.1], pairs)
    assert out == {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "auc": 1.0,
        "paired_separation": 1.0,
    }
    assert all(type(v) is float for v in out.values())  # JSON-serialisable, no numpy
