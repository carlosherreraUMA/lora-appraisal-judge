import numpy as np

from lora_appraisal_judge.baselines import LengthBaseline, majority_baseline, majority_scores


def _ex(resolved, n_messages):
    return {"resolved": resolved, "n_messages": n_messages}


def test_majority_baseline_picks_the_more_common_label():
    train = [_ex(True, 5)] * 3 + [_ex(False, 5)] * 7
    assert majority_baseline(train) is False


def test_majority_scores_is_constant_and_matches_majority_label():
    train = [_ex(True, 5)] * 8 + [_ex(False, 5)] * 2
    test = [_ex(True, 5), _ex(False, 5), _ex(True, 5)]
    scores = majority_scores(train, test)
    assert (scores == 1.0).all()


def test_length_baseline_separates_when_length_is_perfectly_informative():
    # Short trajectories always unresolved, long ones always resolved: a trivially
    # separable synthetic case, just to confirm the plumbing (fit -> scores) works.
    train = [_ex(False, n) for n in range(1, 21)] + [_ex(True, n) for n in range(50, 70)]
    test = [_ex(False, 3), _ex(True, 60)]
    model = LengthBaseline().fit(train)
    scores = model.scores(test)
    assert scores[1] > scores[0]


def test_length_baseline_handles_constant_training_label_without_crashing():
    train = [_ex(True, n) for n in range(1, 10)]
    test = [_ex(True, 3), _ex(False, 30)]
    model = LengthBaseline().fit(train)
    scores = model.scores(test)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2
    assert scores[0] == scores[1] == 1.0
