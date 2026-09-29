#!/usr/bin/env python3
"""Build the train/val/test JSONL files from a local copy of the raw corpus.

Run once, locally, against a downloaded copy of
`nebius/SWE-rebench-openhands-trajectories` (see `data/README.md` for how to get
one). Output is small enough to commit or to upload as a Kaggle Dataset — the raw
2 GB corpus itself is neither.

    python scripts/prepare_dataset.py \\
        --raw-path /path/to/trajectories.parquet \\
        --out-dir data/prepared

Deterministic given the same raw file and the same --max-scan / --*-size
arguments: `splits.split_by_instance` hashes on `instance_id`, so re-running this
script reproduces the same split without needing to persist or pass a seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lora_appraisal_judge.extract import extract_example, iter_local_parquet  # noqa: E402
from lora_appraisal_judge.prompts import format_example  # noqa: E402
from lora_appraisal_judge.splits import build_pairs, split_by_instance  # noqa: E402

#: CC BY 4.0 requires stating what was changed. This is the whole change: a subset
#: of eligible trajectories, reduced to (closing message, resolved label, message
#: count), formatted as an instruction/response pair. See NOTICE.md.
ATTRIBUTION = (
    "Derived from nebius/SWE-rebench-openhands-trajectories (Nebius, CC BY 4.0), "
    "revision 35455389ab51bf5e2306bfd436ef72d0f98bf882. "
    "https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-path", required=True, help="Local path to trajectories.parquet")
    p.add_argument("--out-dir", default="data/prepared")
    p.add_argument(
        "--max-scan", type=int, default=None,
        help="Stop after scanning this many raw records (before eligibility filtering). "
             "Omit to scan the whole file (about 25 minutes per situated-appraisal's own notes).",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    n_scanned = 0
    for record in iter_local_parquet(args.raw_path, batch_size=args.batch_size):
        n_scanned += 1
        ex = extract_example(record)
        if ex is not None:
            examples.append(ex)
        if args.max_scan is not None and n_scanned >= args.max_scan:
            break
        if n_scanned % 5000 == 0:
            print(f"  scanned {n_scanned:,}, eligible {len(examples):,}", file=sys.stderr)

    print(f"scanned {n_scanned:,} raw records, {len(examples):,} eligible", file=sys.stderr)

    splits = split_by_instance(examples, val_frac=args.val_frac, test_frac=args.test_frac)
    pairs = build_pairs(splits["test"])

    for name, split_examples in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as fh:
            for ex in split_examples:
                fh.write(json.dumps(format_example(ex)) + "\n")
        n_resolved = sum(1 for ex in split_examples if ex["resolved"])
        print(
            f"{name}: {len(split_examples):,} examples "
            f"({n_resolved:,} resolved, {len(split_examples) - n_resolved:,} unresolved) "
            f"-> {path}",
            file=sys.stderr,
        )

    pairs_path = out_dir / "test_pairs.json"
    pairs_path.write_text(json.dumps(
        [[r["trajectory_id"], u["trajectory_id"]] for r, u in pairs], indent=2
    ))
    print(f"test pairs (within-task, resolved vs unresolved): {len(pairs):,} -> {pairs_path}",
          file=sys.stderr)

    manifest = {
        "attribution": ATTRIBUTION,
        "n_raw_scanned": n_scanned,
        "n_eligible": len(examples),
        "split_sizes": {name: len(exs) for name, exs in splits.items()},
        "n_test_pairs": len(pairs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out_dir / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
