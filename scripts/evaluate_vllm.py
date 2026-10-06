#!/usr/bin/env python3
"""E3: score the E2b adapter, unchanged, on the full-corpus test set, with vLLM.

    python -u scripts/evaluate_vllm.py --smoke --out /kaggle/working/e3-smoke
    python -u scripts/evaluate_vllm.py --out /kaggle/working/e3 --after-smoke /kaggle/working/e3-smoke

Inputs found automatically under /kaggle/input when E2b's notebook output is attached
to the notebook (or given with --adapter / --reference-scores):
- the adapter: `.../run/adapter/` (not a `checkpoint-*` directory);
- E2b's per-example scores: `.../run/test_scores.jsonl`, computed with Hugging Face.

Before anything else, both modes check that vLLM reproduces E2b's scores on E2b's
own test examples. A faster path that gives different numbers would make E3
incomparable with E2b; that check is what makes the speed usable.

--smoke: agreement on 200 E2b examples, measured throughput on 500 E3 examples, the
projected time of the full run, and a verdict. It tries vLLM's LoRA path first; if
that fails on this GPU, it retries with the adapter merged into the base weights,
each attempt in its own process so a failed CUDA context cannot leak. The full run
uses whichever mode the smoke report says passed.

Full run: agreement on all 1,717 E2b examples, then the E3 set (resumable: scores
are appended to `e3_scores.jsonl` chunk by chunk), baselines on the same set, and
task-clustered bootstrap intervals, all in `results.json`.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# FlashInfer's kernels are compiled at run time and that fails on Kaggle (see
# vllm_scoring.ATTENTION_BACKEND); keep its sampler out too. Greedy decoding would
# not use it, but this removes the question.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lora_appraisal_judge.baselines import (  # noqa: E402
    LengthBaseline,
    majority_baseline,
    majority_scores,
)
from lora_appraisal_judge.metrics import summary, task_bootstrap  # noqa: E402
from lora_appraisal_judge.prompts import parse_prediction  # noqa: E402
from lora_appraisal_judge.splits import build_pairs  # noqa: E402
from lora_appraisal_judge.training import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    choose_dtype_name,
    load_scored,
)
from lora_appraisal_judge.vllm_scoring import (  # noqa: E402
    ATTENTION_BACKEND,
    agreement,
    build_request,
    label_distribution,
)

SMOKE_AGREEMENT_EXAMPLES = 200
SMOKE_TIMING_EXAMPLES = 500
CHUNK = 1024
#: Loading a 1.5B model and scoring 700 examples takes minutes; past this, something
#: is compiling or hanging and the attempt is stopped rather than left to eat quota.
SMOKE_ATTEMPT_MINUTES = 20
MAX_PROJECTED_HOURS = 3.0
#: vLLM must rank E2b's examples as Hugging Face did. Kernels differ, so scores are
#: not bit-identical in fp16; a rank correlation below this means something other
#: than rounding (wrong tokens, wrong adapter, adapter not applied).
MIN_SPEARMAN = 0.99
KAGGLE_INPUT = Path("/kaggle/input")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--after-smoke", type=Path, default=None)
    p.add_argument("--mode", choices=["auto", "lora", "merged"], default="auto",
                   help="auto (smoke only): try lora, fall back to merged.")
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--reference-scores", type=Path, default=None)
    p.add_argument("--prepared-dir", type=Path, default=REPO / "data" / "prepared")
    p.add_argument("--e3-data", type=Path, default=REPO / "data" / "e3" / "test.jsonl.gz")
    return p.parse_args()


def _find_one(pattern: str, what: str, flag: str) -> Path:
    hits = sorted(KAGGLE_INPUT.glob(pattern)) if KAGGLE_INPUT.is_dir() else []
    if len(hits) != 1:
        found = "\n  ".join(str(h) for h in hits) or "(nothing)"
        sys.exit(f"Need exactly one {what} under {KAGGLE_INPUT} ({pattern}); found:\n  {found}\n"
                 f"Attach E2b's notebook output as an input, or pass {flag}.")
    return hits[0]


def _load_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _environment(dtype_name: str, mode: str) -> dict:
    import torch
    import transformers
    import vllm

    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "dtype": dtype_name,
        "mode": mode,
        "attention_backend": ATTENTION_BACKEND,
        "enforce_eager": True,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


class Scorer:
    """The engine plus the adapter, in one of two modes, behind one `score` call."""

    def __init__(self, mode: str, adapter: Path, dtype_name: str, max_model_len: int, work: Path):
        from lora_appraisal_judge.vllm_scoring import (
            generation_repetition_penalty,
            load_llm,
            merge_adapter,
        )

        cfg = json.loads((adapter / "adapter_config.json").read_text())
        base = cfg.get("base_model_name_or_path") or DEFAULT_BASE_MODEL
        self.repetition_penalty = generation_repetition_penalty(base)
        print(f"greedy generation with repetition_penalty={self.repetition_penalty} "
              f"(from {base}'s generation_config, as HF generate applied it in E2b)", flush=True)
        self.lora_request = None
        if mode == "lora":
            from vllm.lora.request import LoRARequest

            self.llm = load_llm(base, dtype_name, int(cfg["r"]), max_model_len)
            self.lora_request = LoRARequest("e2b", 1, str(adapter))
        else:
            merged = work / "merged"
            if not (merged / "config.json").exists():
                print(f"Merging the adapter into {base} -> {merged}", flush=True)
                merge_adapter(base, str(adapter), str(merged), dtype_name)
            self.llm = load_llm(str(merged), dtype_name, None, max_model_len)
        self.tokenizer = self.llm.get_tokenizer()

    def requests(self, examples):
        return [build_request(self.tokenizer, ex) for ex in examples]

    def score(self, requests):
        from lora_appraisal_judge.vllm_scoring import score_batch

        return score_batch(self.llm, requests, self.lora_request,
                           repetition_penalty=self.repetition_penalty)


def _max_model_len(examples: list[dict]) -> int:
    """Longest teacher-forced sequence in the data, plus room for generation."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    longest = max(len(build_request(tok, ex).unresolved_ids) for ex in examples)
    return longest + 16


def _agreement_block(scorer, examples, reference, pairs=None) -> dict:
    t0 = time.time()
    new = {r["trajectory_id"]: r for r in scorer.score(scorer.requests(examples))}
    block = {"n": len(examples), "seconds": time.time() - t0,
             **agreement(reference, new, [ex["trajectory_id"] for ex in examples])}
    if pairs is not None:
        def row(recs):
            preds = [parse_prediction(recs[ex["trajectory_id"]]["generated"]) for ex in examples]
            scores = [recs[ex["trajectory_id"]]["score"] for ex in examples]
            return summary(examples, preds, scores, pairs)
        block["metrics_reference_hf"] = row(reference)
        block["metrics_vllm"] = row(new)
    return block


def _run_smoke_mode(args, mode: str) -> None:
    """One smoke attempt in one mode, in this process. Writes smoke_<mode>.json."""
    import torch

    dtype_name = choose_dtype_name(torch.cuda.get_device_capability(0))
    adapter, reference_path = _inputs(args)
    reference = load_scored(reference_path)
    e2b = _load_jsonl(args.prepared_dir / "test.jsonl")[:SMOKE_AGREEMENT_EXAMPLES]
    e3 = _load_jsonl(args.e3_data)[:SMOKE_TIMING_EXAMPLES]

    report = {"mode": mode, "problems": []}
    try:
        scorer = Scorer(mode, adapter, dtype_name, _max_model_len(e2b + e3), args.out)
        report["environment"] = _environment(dtype_name, mode)
        report["agreement"] = _agreement_block(scorer, e2b, reference)
        reqs = scorer.requests(e3)
        t0 = time.time()
        recs = scorer.score(reqs)
        sec = (time.time() - t0) / len(recs)
        n_full = sum(1 for _ in gzip.open(args.e3_data, "rt"))
        report["seconds_per_example"] = sec
        report["projected_hours"] = sec * (n_full + 1717) / 3600
        report["sample_generations"] = [r["generated"] for r in recs[:5]]
    except Exception as exc:  # the point of the smoke run is to catch these
        report["problems"].append(f"{mode} path failed: {type(exc).__name__}: {exc}")
        # The message alone hid the cause in the first E3 smoke run; keep the trace.
        report["traceback"] = traceback.format_exc().splitlines()[-25:]
        _write_json(args.out / f"smoke_{mode}.json", report)
        return

    a = report["agreement"]
    if not math.isfinite(a["spearman"]) or a["spearman"] < MIN_SPEARMAN:
        report["problems"].append(
            f"scores disagree with E2b (Spearman {a['spearman']:.4f} < {MIN_SPEARMAN})")
    if not all(math.isfinite(r["score"]) for r in recs):
        report["problems"].append("some scores are NaN/inf")
    if report["projected_hours"] > MAX_PROJECTED_HOURS:
        report["problems"].append(
            f"projected {report['projected_hours']:.1f} h exceeds {MAX_PROJECTED_HOURS:.0f} h")
    _write_json(args.out / f"smoke_{mode}.json", report)


def _smoke(args) -> None:
    """Try each mode in a fresh process; keep the first that passes."""
    modes = ["lora", "merged"] if args.mode == "auto" else [args.mode]
    attempts = []
    for mode in modes:
        cmd = [sys.executable, "-u", __file__, "--smoke", "--mode", mode, "--out", str(args.out),
               "--prepared-dir", str(args.prepared_dir), "--e3-data", str(args.e3_data)]
        if args.adapter:
            cmd += ["--adapter", str(args.adapter)]
        if args.reference_scores:
            cmd += ["--reference-scores", str(args.reference_scores)]
        print(f"\n--- smoke attempt: {mode} ---", flush=True)
        path = args.out / f"smoke_{mode}.json"
        path.unlink(missing_ok=True)  # a report left by an earlier session is not this one
        # Own process group: vLLM runs its engine in a child process, which would
        # keep holding GPU memory if only the direct child were killed.
        proc = subprocess.Popen(cmd, start_new_session=True)
        try:
            proc.wait(timeout=SMOKE_ATTEMPT_MINUTES * 60)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True
        if timed_out:
            attempt = {"mode": mode, "problems": [
                f"{mode} attempt killed after {SMOKE_ATTEMPT_MINUTES} min "
                "(the second smoke run spent 26 min compiling kernels before failing)"]}
        elif path.exists():
            attempt = json.loads(path.read_text())
        else:
            attempt = {"mode": mode, "problems": [f"{mode} attempt crashed before reporting"]}
        attempts.append(attempt)
        if not attempt["problems"]:
            break

    chosen = next((a for a in attempts if not a["problems"]), None)
    report = {"chosen_mode": chosen["mode"] if chosen else None,
              "problems": [] if chosen else [p for a in attempts for p in a["problems"]],
              "attempts": attempts}
    _write_json(args.out / "smoke_report.json", report)

    print("\n=== SMOKE REPORT ===")
    for a in attempts:
        print(f"[{a['mode']}] problems: {a['problems'] or 'none'}")
        if "traceback" in a:
            print("  " + "\n  ".join(a["traceback"]))
        if "agreement" in a:
            g = a["agreement"]
            print(f"  agreement with E2b on {g['n']}: Spearman {g['spearman']:.4f}, "
                  f"max |diff| {g['max_abs_diff']:.4f}, same sign {g['sign_agreement']:.3f}, "
                  f"same generated label {g['generated_agreement']:.3f}")
        if "seconds_per_example" in a:
            print(f"  {a['seconds_per_example'] * 1000:.0f} ms/example, "
                  f"full run ~{a['projected_hours'] * 60:.0f} min")
    print(f"VERDICT: OK, full run will use mode '{chosen['mode']}'" if chosen
          else "VERDICT: STOP — see problems above")


def _inputs(args) -> tuple[Path, Path]:
    adapter = args.adapter or _find_one("**/adapter/adapter_config.json", "adapter",
                                        "--adapter").parent
    ref = args.reference_scores or _find_one("**/test_scores.jsonl", "E2b test_scores.jsonl",
                                             "--reference-scores")
    return adapter, ref


def _full(args) -> None:
    import torch

    smoke = json.loads((args.after_smoke / "smoke_report.json").read_text())
    mode = smoke["chosen_mode"]
    dtype_name = choose_dtype_name(torch.cuda.get_device_capability(0))
    adapter, reference_path = _inputs(args)
    reference = load_scored(reference_path)

    train = _load_jsonl(args.prepared_dir / "train.jsonl")
    e2b = _load_jsonl(args.prepared_dir / "test.jsonl")
    e2b_by_id = {ex["trajectory_id"]: ex for ex in e2b}
    e2b_pairs = [(e2b_by_id[r], e2b_by_id[u])
                 for r, u in json.loads((args.prepared_dir / "test_pairs.json").read_text())]
    e3 = _load_jsonl(args.e3_data)
    e3_pairs = build_pairs(e3)

    results_path = args.out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    scorer = Scorer(mode, adapter, dtype_name, _max_model_len(e2b + e3), args.after_smoke)
    results["environment"] = _environment(dtype_name, mode)
    results["adapter"] = str(adapter)

    # 1. Same examples, same adapter, two engines: does vLLM reproduce E2b?
    if "agreement_e2b" not in results:
        print("Agreement with E2b on all its test examples...", flush=True)
        results["agreement_e2b"] = _agreement_block(scorer, e2b, reference, e2b_pairs)
        _write_json(results_path, results)
    g = results["agreement_e2b"]
    if g["spearman"] < MIN_SPEARMAN:
        sys.exit(f"STOP: vLLM disagrees with E2b on its own examples (Spearman "
                 f"{g['spearman']:.4f}); E3 numbers would not be comparable. See {results_path}")

    # 2. E3, chunk by chunk, resumable.
    scores_path = args.out / "e3_scores.jsonl"
    done = load_scored(scores_path)
    todo = [ex for ex in e3 if ex["trajectory_id"] not in done]
    print(f"E3: {len(e3)} examples, {len(done)} already scored, {len(todo)} to go", flush=True)
    t0 = time.time()
    with scores_path.open("a") as fh:
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            for rec in scorer.score(scorer.requests(chunk)):
                done[rec["trajectory_id"]] = rec
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n = start + len(chunk)
            per = (time.time() - t0) / n
            print(f"[e3] {n}/{len(todo)}  {per * 1000:.0f} ms/example  "
                  f"~{(len(todo) - n) * per / 60:.0f} min left", flush=True)
    if todo:
        results["e3_seconds_per_example"] = (time.time() - t0) / len(todo)

    # 3. Metrics on E3: baselines (fit on E2b's train split, as in E2b) and the LoRA.
    preds = [parse_prediction(done[ex["trajectory_id"]]["generated"]) for ex in e3]
    scores = [done[ex["trajectory_id"]]["score"] for ex in e3]
    maj = majority_baseline(train)
    maj_scores = majority_scores(train, e3).tolist()
    length_scores = LengthBaseline().fit(train).scores(e3).tolist()

    results["e3"] = {
        "n_test": len(e3),
        "n_resolved": sum(1 for ex in e3 if ex["resolved"]),
        "n_pairs": len(e3_pairs),
        "majority": summary(e3, [bool(maj)] * len(e3), maj_scores, e3_pairs),
        "length": summary(e3, [s >= 0.5 for s in length_scores], length_scores, e3_pairs),
        "lora": summary(e3, preds, scores, e3_pairs),
    }
    results["e3"]["lora"]["n_unparsed_predictions"] = sum(1 for p in preds if p is None)
    results["e3"]["lora_label_distribution"] = label_distribution(e3, preds)
    print("Bootstrap over tasks (length, lora)...", flush=True)
    results["e3"]["bootstrap"] = {
        "length": task_bootstrap(e3, length_scores),
        "lora": task_bootstrap(e3, scores),
    }
    _write_json(results_path, results)

    print("\n=== E3 RESULTS ===")
    print(json.dumps({k: results["e3"][k] for k in ("majority", "length", "lora",
                                                    "lora_label_distribution", "bootstrap")},
                     indent=2))
    print(f"\nFull results: {results_path}")


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        sys.exit("No GPU visible. On Kaggle: Settings -> Accelerator -> GPU T4.")

    if args.smoke:
        if args.mode == "auto":
            _smoke(args)
        else:
            _run_smoke_mode(args, args.mode)
        return

    if not args.after_smoke:
        sys.exit("The full run needs --after-smoke <smoke dir>: it reads the mode that passed.")
    report_path = args.after_smoke / "smoke_report.json"
    if not report_path.exists():
        sys.exit(f"Refusing the full run: no smoke report at {report_path}. Run --smoke first.")
    problems = json.loads(report_path.read_text()).get("problems", [])
    if problems:
        sys.exit("Refusing the full run; the smoke run found: " + "; ".join(problems))
    _full(args)


if __name__ == "__main__":
    main()
