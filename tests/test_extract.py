from lora_appraisal_judge.extract import closing_message, extract_example


def _assistant(content="", tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


def _finish_call(message):
    import json

    return {"function": {"name": "finish", "arguments": json.dumps({"message": message})}}


def test_closing_message_combines_prose_and_finish_call():
    trajectory = [
        {"role": "user", "content": "please fix the bug"},
        _assistant(content="Looking at the code."),
        _assistant(content="Done.", tool_calls=[_finish_call("Fixed the off-by-one error.")]),
    ]
    assert closing_message(trajectory) == "Done.\n\nFixed the off-by-one error."


def test_closing_message_falls_back_to_earlier_assistant_turn():
    # The last assistant turn carries no finish call; the one before it does.
    trajectory = [
        _assistant(content="Investigating.", tool_calls=[_finish_call("I fixed it.")]),
        _assistant(content="", tool_calls=[{"function": {"name": "execute_bash", "arguments": "{}"}}]),
    ]
    assert closing_message(trajectory) == "I fixed it."


def test_closing_message_empty_trajectory():
    assert closing_message([]) == ""


def test_closing_message_no_assistant_turns():
    assert closing_message([{"role": "user", "content": "hello"}]) == ""


def test_closing_message_prose_only_no_finish_call():
    trajectory = [_assistant(content="I believe this is resolved now.")]
    assert closing_message(trajectory) == "I believe this is resolved now."


def _record(exit_status="submit", resolved=1, trajectory=None):
    return {
        "trajectory_id": "t1",
        "instance_id": "i1",
        "exit_status": exit_status,
        "resolved": resolved,
        "trajectory": trajectory if trajectory is not None else [
            _assistant(content="Done.", tool_calls=[_finish_call("Fixed it.")]),
        ],
    }


def test_extract_example_happy_path():
    ex = extract_example(_record())
    assert ex == {
        "trajectory_id": "t1",
        "instance_id": "i1",
        "resolved": True,
        "closing_message": "Done.\n\nFixed it.",
        "n_messages": 1,
    }


def test_extract_example_resolved_is_boolean_not_int():
    ex = extract_example(_record(resolved=0))
    assert ex["resolved"] is False


def test_extract_example_skips_non_submit_exit_status():
    assert extract_example(_record(exit_status="iteration_ceiling")) is None


def test_extract_example_skips_empty_closing_message():
    assert extract_example(_record(trajectory=[{"role": "user", "content": "hi"}])) is None


def test_extract_example_skips_missing_resolved():
    rec = _record()
    rec["resolved"] = None
    assert extract_example(rec) is None
