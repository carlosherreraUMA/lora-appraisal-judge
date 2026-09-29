from lora_appraisal_judge.prompts import (
    build_prompt,
    build_target,
    format_example,
    parse_prediction,
    truncate_message,
)


def test_truncate_message_keeps_short_message_unchanged():
    assert truncate_message("short message") == "short message"


def test_truncate_message_cuts_from_the_start_keeping_the_end():
    long_message = "x" * 10 + "IMPORTANT_TAIL"
    truncated = truncate_message(long_message, max_chars=15)
    assert truncated.endswith("IMPORTANT_TAIL")
    assert truncated.startswith("…")
    assert len(truncated) == 16  # 1 ellipsis char + 15 kept chars


def test_build_prompt_contains_instruction_and_message():
    prompt = build_prompt("Fixed the bug.")
    assert "RESOLVED or UNRESOLVED" in prompt
    assert "Fixed the bug." in prompt


def test_build_target():
    assert build_target(True) == "RESOLVED"
    assert build_target(False) == "UNRESOLVED"


def test_format_example_round_trip():
    example = {
        "trajectory_id": "t1",
        "instance_id": "i1",
        "resolved": True,
        "closing_message": "All tests pass now.",
        "n_messages": 12,
    }
    out = format_example(example)
    assert out["trajectory_id"] == "t1"
    assert out["response"] == "RESOLVED"
    assert out["n_messages"] == 12
    assert "All tests pass now." in out["prompt"]


def test_parse_prediction_resolved():
    assert parse_prediction("RESOLVED") is True
    assert parse_prediction("  resolved\n") is True


def test_parse_prediction_unresolved():
    assert parse_prediction("UNRESOLVED") is False


def test_parse_prediction_ambiguous_or_empty_is_none():
    assert parse_prediction("") is None
    assert parse_prediction("I'm not sure") is None
    # Contains both words: genuinely ambiguous, not a crash.
    assert parse_prediction("RESOLVED, not UNRESOLVED") is None
