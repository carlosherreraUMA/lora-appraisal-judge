import json
import math
from dataclasses import replace

from lora_appraisal_judge.training import (
    LoraTrainingConfig,
    choose_dtype_name,
    full_run_steps,
    load_scored,
)


def test_choose_dtype_t4_is_float16():
    # Tesla T4 is Turing, compute capability 7.5: no native bf16.
    assert choose_dtype_name((7, 5)) == "float16"


def test_choose_dtype_ampere_and_newer_is_bfloat16():
    assert choose_dtype_name((8, 0)) == "bfloat16"  # A100
    assert choose_dtype_name((8, 6)) == "bfloat16"  # RTX 30xx / A10
    assert choose_dtype_name((9, 0)) == "bfloat16"  # H100


def test_choose_dtype_older_gpus_are_float16():
    assert choose_dtype_name((6, 0)) == "float16"  # P100


def test_full_run_steps_matches_what_trainer_reported_on_the_first_run():
    # E2a: 5,594 train examples, batch 4, accumulation 4, 2 epochs -> Trainer said 700.
    first_run = replace(
        LoraTrainingConfig(), per_device_train_batch_size=4, gradient_accumulation_steps=4
    )
    assert full_run_steps(5594, first_run) == 700


def test_full_run_steps_current_config_keeps_the_same_effective_batch():
    # Batch 2 x accumulation 8 is the same 16 examples per step as 4 x 4.
    assert full_run_steps(5594, LoraTrainingConfig()) == 700


def test_load_scored_missing_file_is_empty(tmp_path):
    assert load_scored(tmp_path / "nope.jsonl") == {}


def test_load_scored_skips_a_truncated_last_line(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps({"trajectory_id": "a", "score": 1.0, "generated": "RESOLVED"}) + "\n"
        + json.dumps({"trajectory_id": "b", "score": -2.0, "generated": "UNRESOLVED"}) + "\n"
        + '{"trajectory_id": "c", "sco'  # process died mid-write
    )
    done = load_scored(path)
    assert set(done) == {"a", "b"}
    assert done["b"]["score"] == -2.0


def test_load_scored_reads_back_nan_scores(tmp_path):
    # json.dumps writes NaN for float('nan'); it must round-trip, not be dropped.
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps({"trajectory_id": "a", "score": float("nan"), "generated": ""}) + "\n")
    assert math.isnan(load_scored(path)["a"]["score"])
