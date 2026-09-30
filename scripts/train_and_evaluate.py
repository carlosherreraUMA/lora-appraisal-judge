#!/usr/bin/env python3
"""Train the LoRA adapter and evaluate it against the baselines, end to end.

Everything the Kaggle notebook used to do in cells lives here, so the notebook is
only a launcher: it clones or updates this repository and runs this script. A fix
pushed to the repository is picked up by re-running the notebook; no cell has to be
copied by hand. Each run is also a fresh process, so CUDA state left by an earlier
crash cannot leak into the next attempt.

Two modes:

    python -u scripts/train_and_evaluate.py --smoke --out /kaggle/working/smoke
    python -u scripts/train_and_evaluate.py --out /kaggle/working/run

--smoke trains 20 steps and scores 20 test examples, then prints the measured
seconds per step, the projected time of the full run and a verdict. Run it first,
every time: the first full attempt (EXPERIMENTS.md, E2a) ran unattended for eight
hours at a tenth of the expected speed before anyone looked.

The full run checkpoints every 50 steps and writes each scored test example to an
append-only file; re-run with the same --out after an interruption and it resumes
both.
"""
from __future__ import annotations

import os

# Before anything imports torch: one GPU only (see training.load_model_and_tokenizer).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lora_appraisal_judge.baselines import (  # noqa: E402
    LengthBaseline,
    majority_baseline,
    majority_scores,
)
from lora_appraisal_judge.metrics import summary  # noqa: E402
from lora_appraisal_judge.prompts import parse_prediction  # noqa: E402
from lora_appraisal_judge.training import (  # noqa: E402
    CausalLMLabelCollator,
    LoraTrainingConfig,
    build_lora_model,
    build_tokenized_dataset,
    choose_dtype_name,
    full_run_steps,
    generate_prediction,
    load_model_and_tokenizer,
    load_scored,
    resolved_score,
)

SMOKE_TRAIN_STEPS = 20
SMOKE_EVAL_EXAMPLES = 20
#: A full run projected past this is not launched blind: the smoke verdict says STOP.
MAX_PROJECTED_HOURS = 3.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory: checkpoints, adapter, scores, results.")
    p.add_argument("--data-dir", type=Path, default=REPO / "data" / "prepared")
    p.add_argument("--smoke", action="store_true",
                   help="Short timing and sanity run (see the module docstring).")
    p.add_argument("--after-smoke", type=Path, default=None,
                   help="Smoke output directory; refuse to start the full run unless its "
                        "report exists and found no problems. Makes an unattended "
                        "run-everything launch safe.")
    return p.parse_args()


def _require_smoke_ok(smoke_dir: Path) -> None:
    report_path = smoke_dir / "smoke_report.json"
    if not report_path.exists():
        sys.exit(f"Refusing the full run: no smoke report at {report_path}. Run --smoke first.")
    problems = json.loads(report_path.read_text()).get("problems", [])
    if problems:
        sys.exit("Refusing the full run; the smoke run found: " + "; ".join(problems))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _environment(dtype_name: str) -> dict:
    import peft
    import torch
    import transformers

    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "dtype": dtype_name,
        "visible_gpus": torch.cuda.device_count(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def _eta_callback():
    """A `TrainerCallback` that prints measured seconds per step and time left."""
    from transformers import TrainerCallback

    class EtaCallback(TrainerCallback):
        def __init__(self):
            self.t0 = None
            self.s0 = 0
            self.seconds_per_step = None

        def on_step_end(self, args, state, control, **kwargs):
            if self.t0 is None:
                # Timing starts after the first step, so warm-up (and the steps a
                # resumed run skips) do not distort the rate.
                self.t0 = time.time()
                self.s0 = state.global_step
                return
            self.seconds_per_step = (time.time() - self.t0) / (state.global_step - self.s0)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.seconds_per_step is None:
                return
            left = (state.max_steps - state.global_step) * self.seconds_per_step
            print(f"[eta] step {state.global_step}/{state.max_steps}  "
                  f"{self.seconds_per_step:.1f} s/step  ~{left / 60:.0f} min left", flush=True)

    return EtaCallback()


def _train(model, tokenizer, train, config, dtype_name, out_dir, max_steps, callback):
    """Train in place. `max_steps=None` is the full, checkpointed, resumable run."""
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    smoke = max_steps is not None
    ckpt_dir = out_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=max_steps if smoke else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=5 if smoke else 20,
        save_strategy="no" if smoke else "steps",
        save_steps=50,
        save_total_limit=3,
        fp16=dtype_name == "float16",
        bf16=dtype_name == "bfloat16",
        report_to=[],
        disable_tqdm=True,  # plain log lines stream cleanly through a notebook's `!`
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=build_tokenized_dataset(tokenizer, train, config.max_seq_length),
        data_collator=CausalLMLabelCollator(tokenizer.pad_token_id),
        callbacks=[callback],
    )
    last = None
    if not smoke and ckpt_dir.is_dir():
        last = get_last_checkpoint(str(ckpt_dir))
        if last:
            print(f"Resuming training from {last}", flush=True)
    output = trainer.train(resume_from_checkpoint=last)
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    return dict(output.metrics), losses


def _score(model, tokenizer, examples, dtype, scored_path=None):
    """Score and generate for each example; resumable through `scored_path`."""
    import torch

    done = load_scored(scored_path) if scored_path else {}
    todo = [ex for ex in examples if ex["trajectory_id"] not in done]
    if done:
        print(f"Resuming evaluation: {len(done)} already scored, {len(todo)} to go", flush=True)

    model.eval()
    model.config.use_cache = True
    fh = scored_path.open("a") if scored_path else None
    t0 = time.time()
    try:
        for i, ex in enumerate(todo, start=1):
            with torch.autocast("cuda", dtype=dtype):
                score = resolved_score(model, tokenizer, ex["prompt"])
                generated = generate_prediction(model, tokenizer, ex["prompt"])
            rec = {"trajectory_id": ex["trajectory_id"], "score": score, "generated": generated}
            done[ex["trajectory_id"]] = rec
            if fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if i % 50 == 0:
                per = (time.time() - t0) / i
                print(f"[eval] {i}/{len(todo)}  {per:.2f} s/example  "
                      f"~{(len(todo) - i) * per / 60:.0f} min left", flush=True)
    finally:
        if fh:
            fh.close()
    seconds_per_example = (time.time() - t0) / len(todo) if todo else None
    return done, seconds_per_example


def _smoke(config, dtype_name, dtype, env, train, test, out):
    model, tokenizer = load_model_and_tokenizer(config, dtype_name)
    model = build_lora_model(model, config)
    model.print_trainable_parameters()

    eta = _eta_callback()
    train_metrics, losses = _train(
        model, tokenizer, train, config, dtype_name, out, SMOKE_TRAIN_STEPS, eta
    )
    scored, sec_per_example = _score(model, tokenizer, test[:SMOKE_EVAL_EXAMPLES], dtype)

    sec_per_step = eta.seconds_per_step or train_metrics["train_runtime"] / SMOKE_TRAIN_STEPS
    steps = full_run_steps(len(train), config)
    train_hours = steps * sec_per_step / 3600
    eval_hours = len(test) * sec_per_example / 3600

    problems = []
    if not losses or not all(math.isfinite(x) for x in losses):
        problems.append("training loss is NaN/inf (fp16 overflow?)")
    if not all(math.isfinite(r["score"]) for r in scored.values()):
        problems.append("some test scores are NaN/inf")
    if train_hours + eval_hours > MAX_PROJECTED_HOURS:
        problems.append(f"projected {train_hours + eval_hours:.1f} h exceeds "
                        f"{MAX_PROJECTED_HOURS:.0f} h — something is slow; do not launch blind")

    report = {
        "environment": env,
        "seconds_per_step": sec_per_step,
        "full_run_steps": steps,
        "projected_train_hours": train_hours,
        "seconds_per_test_example": sec_per_example,
        "projected_eval_hours": eval_hours,
        "losses": losses,
        "sample_generations": [r["generated"] for r in list(scored.values())[:5]],
        "problems": problems,
    }
    _write_json(out / "smoke_report.json", report)

    print("\n=== SMOKE REPORT ===")
    print(f"GPU {env['gpu']} (compute {env['compute_capability']}), dtype {dtype_name}")
    print(f"train: {sec_per_step:.1f} s/step x {steps} steps = ~{train_hours * 60:.0f} min")
    print(f"eval:  {sec_per_example:.2f} s/example x {len(test)} = ~{eval_hours * 60:.0f} min")
    print(f"losses: {[round(x, 3) for x in losses]}")
    if problems:
        print("VERDICT: STOP — " + "; ".join(problems))
    else:
        print("VERDICT: OK to launch the full run")


def _full(config, dtype_name, dtype, env, train, test, pairs, out):
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results.update({
        "environment": env,
        "config": asdict(config),
        "n_train": len(train),
        "n_test": len(test),
        "n_test_pairs": len(pairs),
    })

    # Baselines first: seconds on CPU, and they make results.json worth reading even
    # if the GPU part fails later.
    maj = majority_baseline(train)
    results["majority"] = summary(
        test, [bool(maj)] * len(test), majority_scores(train, test).tolist(), pairs
    )
    length_scores = LengthBaseline().fit(train).scores(test).tolist()
    results["length"] = summary(
        test, [s >= 0.5 for s in length_scores], length_scores, pairs
    )
    _write_json(results_path, results)

    adapter_dir = out / "adapter"
    base, tokenizer = load_model_and_tokenizer(config, dtype_name)
    if (adapter_dir / "adapter_config.json").exists():
        from peft import PeftModel

        print(f"Training already finished; loading the adapter from {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        model = build_lora_model(base, config)
        model.print_trainable_parameters()
        train_metrics, losses = _train(
            model, tokenizer, train, config, dtype_name, out, None, _eta_callback()
        )
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        results["train"] = {
            **train_metrics,
            "loss_first_logged": losses[0] if losses else None,
            "loss_last_logged": losses[-1] if losses else None,
        }
        _write_json(results_path, results)

    scored, _ = _score(model, tokenizer, test, dtype, out / "test_scores.jsonl")
    preds = [parse_prediction(scored[ex["trajectory_id"]]["generated"]) for ex in test]
    scores = [scored[ex["trajectory_id"]]["score"] for ex in test]
    results["lora"] = summary(test, preds, scores, pairs)
    results["lora"]["n_unparsed_predictions"] = sum(1 for p in preds if p is None)
    _write_json(results_path, results)

    print("\n=== RESULTS ===")
    print(json.dumps({k: results[k] for k in ("majority", "length", "lora")}, indent=2))
    print(f"\nFull results: {results_path}")


def main() -> None:
    args = _parse_args()
    if args.after_smoke and not args.smoke:
        _require_smoke_ok(args.after_smoke)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        sys.exit("No GPU visible. On Kaggle: Settings -> Accelerator -> GPU T4.")
    dtype_name = choose_dtype_name(torch.cuda.get_device_capability(0))
    dtype = getattr(torch, dtype_name)
    env = _environment(dtype_name)
    print("environment:", json.dumps(env), flush=True)

    config = LoraTrainingConfig()
    train = _load_jsonl(args.data_dir / "train.jsonl")
    test = _load_jsonl(args.data_dir / "test.jsonl")
    pair_ids = json.loads((args.data_dir / "test_pairs.json").read_text())
    by_id = {ex["trajectory_id"]: ex for ex in test}
    pairs = [(by_id[r], by_id[u]) for r, u in pair_ids]

    if args.smoke:
        _smoke(config, dtype_name, dtype, env, train, test, args.out)
    else:
        _full(config, dtype_name, dtype, env, train, test, pairs, args.out)


if __name__ == "__main__":
    main()
