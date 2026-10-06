import math
from types import SimpleNamespace

import pytest

from lora_appraisal_judge.vllm_scoring import (
    agreement,
    build_request,
    completion_logprob,
    label_distribution,
)


class FakeTokenizer:
    """One token per character; enough to check which positions get scored."""

    eos_token = "$"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_build_request_matches_sequence_logprob_layout():
    req = build_request(FakeTokenizer(), {"trajectory_id": "t1", "prompt": "hi"})
    prompt_text = "<user>hi<assistant>"
    assert req.prompt_ids == [ord(c) for c in prompt_text]
    assert req.n_prompt_tokens == len(prompt_text)
    assert req.resolved_ids == [ord(c) for c in prompt_text + "RESOLVED$"]
    assert req.unresolved_ids == [ord(c) for c in prompt_text + "UNRESOLVED$"]
    # The completion is everything after the prompt, eos included, as in training.
    assert req.resolved_ids[req.n_prompt_tokens:] == [ord(c) for c in "RESOLVED$"]


def _logprob(x):
    return SimpleNamespace(logprob=x)


def test_completion_logprob_sums_only_the_completion_positions():
    token_ids = [10, 11, 12, 13]
    prompt_logprobs = [
        None,
        {11: _logprob(-5.0)},              # prompt token: not counted
        {12: _logprob(-0.5), 99: _logprob(-0.1)},
        {13: _logprob(-0.25)},
    ]
    assert completion_logprob(prompt_logprobs, token_ids, start=2) == -0.75


def test_completion_logprob_accepts_plain_floats():
    assert completion_logprob([None, {7: -1.5}], [6, 7], start=1) == -1.5


def test_completion_logprob_missing_token_raises_rather_than_skipping():
    with pytest.raises(KeyError):
        completion_logprob([None, {99: _logprob(-0.1)}], [6, 7], start=1)


def test_agreement_identical_scores():
    recs = {
        "a": {"score": 1.0, "generated": "RESOLVED"},
        "b": {"score": -2.0, "generated": "UNRESOLVED"},
        "c": {"score": 0.5, "generated": "RESOLVED"},
    }
    out = agreement(recs, recs, ["a", "b", "c"])
    assert math.isclose(out["spearman"], 1.0)
    assert out["max_abs_diff"] == 0.0
    assert out["sign_agreement"] == 1.0
    assert out["generated_agreement"] == 1.0


def test_agreement_detects_a_flipped_sign_and_label():
    ref = {"a": {"score": 0.1, "generated": "RESOLVED"},
           "b": {"score": -1.0, "generated": "UNRESOLVED"},
           "c": {"score": 2.0, "generated": "RESOLVED"}}
    new = {"a": {"score": -0.1, "generated": "UNRESOLVED"},
           "b": {"score": -1.0, "generated": "UNRESOLVED"},
           "c": {"score": 2.0, "generated": "RESOLVED"}}
    out = agreement(ref, new, ["a", "b", "c"])
    assert math.isclose(out["max_abs_diff"], 0.2)
    assert math.isclose(out["sign_agreement"], 2 / 3)
    assert math.isclose(out["generated_agreement"], 2 / 3)
    assert math.isclose(out["spearman"], 1.0)  # ranking unchanged


def test_label_distribution_counts_by_true_label():
    examples = [{"resolved": True}, {"resolved": True}, {"resolved": False}]
    out = label_distribution(examples, [True, None, True])
    assert out == {"resolved": {"RESOLVED": 1, "unparsed": 1},
                   "unresolved": {"RESOLVED": 1}}
