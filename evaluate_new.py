#!/usr/bin/env python3
"""Run the TinyBenchmark-style TinyPerson test evaluation for a saved YOLO run.

This evaluates corner-window predictions, translates them back to original-image
coordinates, applies the official one-class NMS merge, and writes the AP50/AP25/
AP75 size-bucket metrics produced by TinyBenchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing weights/best.pt")
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted TinyPerson dataset root")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    if not (args.run_dir / "weights/best.pt").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.run_dir / 'weights/best.pt'}")

    # Reuse the validated preprocessing, coordinate translation, NMS, and
    # TinyBenchmark adapter from the training workflow.
    sys.path.insert(0, str(ROOT))
    import train_all_tinyperson as workflow

    test_out_dir = workflow.prepare_test_set(args.data_root, args.dataset_root)
    metrics = workflow.evaluate_merged_test(args.run_dir, test_out_dir, args.data_root, args)
    output = args.output or (args.run_dir / "evaluation" / "evaluation_metrics_new.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "TinyBenchmark official corner-window merged-original evaluation",
        "checkpoint": str(args.run_dir / "weights/best.pt"),
        "test_crops": str(test_out_dir),
        "data_root": str(args.data_root),
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "device": args.device,
        "workers": args.workers,
        "metrics": metrics,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
