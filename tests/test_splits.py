from lora_appraisal_judge.splits import build_pairs, split_by_instance


def _ex(trajectory_id, instance_id, resolved, n_messages=5):
    return {
        "trajectory_id": trajectory_id,
        "instance_id": instance_id,
        "resolved": resolved,
        "closing_message": "message",
        "n_messages": n_messages,
    }


def test_split_by_instance_keeps_one_instance_in_one_split():
    examples = [_ex(f"t{i}", f"inst{i % 20}", i % 2 == 0) for i in range(200)]
    splits = split_by_instance(examples, val_frac=0.1, test_frac=0.2)

    instance_to_splits = {}
    for split_name, exs in splits.items():
        for ex in exs:
            instance_to_splits.setdefault(ex["instance_id"], set()).add(split_name)

    assert all(len(s) == 1 for s in instance_to_splits.values())
    # Every input example lands in exactly one split.
    assert sum(len(v) for v in splits.values()) == len(examples)


def test_split_by_instance_is_deterministic_across_calls():
    examples = [_ex(f"t{i}", f"inst{i % 20}", True) for i in range(50)]
    a = split_by_instance(examples)
    b = split_by_instance(examples)
    assert {k: [e["trajectory_id"] for e in v] for k, v in a.items()} == {
        k: [e["trajectory_id"] for e in v] for k, v in b.items()
    }


def test_split_by_instance_rejects_fractions_that_dont_leave_a_train_split():
    import pytest

    with pytest.raises(ValueError):
        split_by_instance([], val_frac=0.5, test_frac=0.6)


def test_build_pairs_pairs_resolved_with_unresolved_same_instance():
    examples = [
        _ex("t1", "inst1", True),
        _ex("t2", "inst1", False),
        _ex("t3", "inst2", True),  # no unresolved partner: contributes no pair
    ]
    pairs = build_pairs(examples)
    assert len(pairs) == 1
    resolved_ex, unresolved_ex = pairs[0]
    assert resolved_ex["trajectory_id"] == "t1"
    assert unresolved_ex["trajectory_id"] == "t2"


def test_build_pairs_is_cross_product_within_instance():
    examples = [
        _ex("r1", "inst1", True),
        _ex("r2", "inst1", True),
        _ex("u1", "inst1", False),
    ]
    pairs = build_pairs(examples)
    assert len(pairs) == 2  # 2 resolved x 1 unresolved


def test_build_pairs_empty_when_all_same_outcome():
    examples = [_ex("t1", "inst1", True), _ex("t2", "inst1", True)]
    assert build_pairs(examples) == []
