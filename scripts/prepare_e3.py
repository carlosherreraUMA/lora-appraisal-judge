#!/usr/bin/env python3
"""Build the pair-rich E3 test set from the full corpus (EXPERIMENTS.md, E3).

E2b's test split came from the first 9,000 of 67,074 raw trajectories, where
attempts at the same task are spread thin: 106 within-task pairs from 63 tasks.
`splits.split_by_instance` hashes `instance_id`, so every eligible attempt anywhere
in the corpus at a task in the test bucket is a test example by construction,
disjoint from training. This script collects all of them.

    python scripts/prepare_e3.py --raw-path /path/to/trajectories.parquet

Writes `data/e3/test.jsonl.gz` (same row format as `data/prepared/test.jsonl`) and
`data/e3/manifest.json`. Pairs are not written: `splits.build_pairs` rebuilds them
from the examples, and with every resolved × unresolved combination per task the
list is far larger than the examples themselves.

Two checks run before anything is written, and abort on failure:
- no test instance occurs in `data/prepared/train.jsonl` or `val.jsonl`;
- every E2b test example is present here with an identical prompt, which confirms
  the same corpus revision and the same extraction.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lora_appraisal_judge.extract import extract_example, iter_local_parquet  # noqa: E402
from lora_appraisal_judge.prompts import format_example  # noqa: E402
from lora_appraisal_judge.splits import build_pairs, split_by_instance  # noqa: E402

from prepare_dataset import ATTRIBUTION  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-path", required=True, help="Local path to trajectories.parquet")
    p.add_argument("--prepared-dir", type=Path, default=REPO / "data" / "prepared")
    p.add_argument("--out-dir", type=Path, default=REPO / "data" / "e3")
    args = p.parse_args()

    examples = []
    n_scanned = 0
    for record in iter_local_parquet(args.raw_path):
        n_scanned += 1
        ex = extract_example(record)
        if ex is not None:
            examples.append(ex)
        if n_scanned % 5000 == 0:
            print(f"  scanned {n_scanned:,}, eligible {len(examples):,}", file=sys.stderr)
    print(f"scanned {n_scanned:,} raw records, {len(examples):,} eligible", file=sys.stderr)

    # Same default fractions as prepare_dataset.py: the bucket of each instance_id
    # is what E2b trained and tested on.
    test = [format_example(ex) for ex in split_by_instance(examples)["test"]]

    seen = set()
    for split in ("train", "val"):
        seen |= {ex["instance_id"] for ex in _load_jsonl(args.prepared_dir / f"{split}.jsonl")}
    leaked = {ex["instance_id"] for ex in test} & seen
    if leaked:
        sys.exit(f"ABORT: {len(leaked)} test instances also occur in train/val, "
                 f"e.g. {sorted(leaked)[:3]}")

    by_id = {ex["trajectory_id"]: ex for ex in test}
    e2b_test = _load_jsonl(args.prepared_dir / "test.jsonl")
    missing = [ex["trajectory_id"] for ex in e2b_test if ex["trajectory_id"] not in by_id]
    differ = [ex["trajectory_id"] for ex in e2b_test
              if ex["trajectory_id"] in by_id and by_id[ex["trajectory_id"]]["prompt"] != ex["prompt"]]
    if missing or differ:
        sys.exit(f"ABORT: E2b test examples missing ({len(missing)}) or with a different "
                 f"prompt ({len(differ)}) — not the same corpus revision or extraction")

    by_task = defaultdict(list)
    for ex in test:
        by_task[ex["instance_id"]].append(ex["resolved"])
    paired_tasks = sum(1 for v in by_task.values() if any(v) and not all(v))
    n_pairs = len(build_pairs(test))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "test.jsonl.gz"
    # mtime=0: the same input gives a byte-identical file, so a rebuild shows no diff.
    with gzip.GzipFile(out, "wb", mtime=0) as fh:
        for ex in test:
            fh.write((json.dumps(ex) + "\n").encode("utf-8"))

    manifest = {
        "attribution": ATTRIBUTION,
        "n_raw_scanned": n_scanned,
        "n_eligible": len(examples),
        "n_test": len(test),
        "n_test_resolved": sum(1 for ex in test if ex["resolved"]),
        "n_test_tasks": len(by_task),
        "n_tasks_with_both_outcomes": paired_tasks,
        "n_test_pairs": n_pairs,
        "checks": {
            "test_instances_in_train_or_val": 0,
            "e2b_test_examples_present_with_identical_prompt": len(e2b_test),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
