#!/usr/bin/env python3
"""Backfill split-qualified native size-bucket metrics for one completed run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

from evaluate_test.size_bucket_evaluator import evaluate_native_size_buckets
from utils.marimo_ops import ensure_hf_repo, require_training_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--remote-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_training_context(hf_repo_id=args.hf_repo_id)
    repo_id = ensure_hf_repo(args.hf_repo_id)
    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "evaluation_metrics.json"
    manifest_path = run_dir / "experiment_manifest.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for split in ("val", "test"):
        metrics.update(
            evaluate_native_size_buckets(
                run_dir,
                args.dataset_yaml.resolve(),
                split=split,
                imgsz=args.imgsz,
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
            )
        )
    metrics["val_size/source_artifact"] = "evaluation/val_size_predictions.json"
    metrics["test_size/source_artifact"] = "evaluation/test_size_predictions.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "val_size_protocol": metrics["val_size/protocol"],
            "test_size_protocol": metrics["test_size/protocol"],
            "val_size_source_artifact": "evaluation/val_size_predictions.json",
            "test_size_source_artifact": "evaluation/test_size_predictions.json",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    marker = {
        "repo_id": repo_id,
        "remote_prefix": args.remote_prefix,
        "verified": [
            f"{args.remote_prefix}/evaluation_metrics.json",
            f"{args.remote_prefix}/experiment_manifest.json",
            f"{args.remote_prefix}/evaluation/val_size_ground_truth.json",
            f"{args.remote_prefix}/evaluation/val_size_predictions.json",
            f"{args.remote_prefix}/evaluation/test_size_ground_truth.json",
            f"{args.remote_prefix}/evaluation/test_size_predictions.json",
        ],
    }
    marker_path = run_dir / "size_metrics_complete.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    api = HfApi(token=os.environ["HF_TOKEN"])
    for local, remote in (
        (metrics_path, "evaluation_metrics.json"),
        (manifest_path, "experiment_manifest.json"),
        (run_dir / "evaluation/val_size_ground_truth.json", "evaluation/val_size_ground_truth.json"),
        (run_dir / "evaluation/val_size_predictions.json", "evaluation/val_size_predictions.json"),
        (run_dir / "evaluation/test_size_ground_truth.json", "evaluation/test_size_ground_truth.json"),
        (run_dir / "evaluation/test_size_predictions.json", "evaluation/test_size_predictions.json"),
        (marker_path, "size_metrics_complete.json"),
    ):
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"{args.remote_prefix}/{remote}",
            repo_id=repo_id,
            repo_type="dataset",
        )

    selected = {
        key: metrics[key]
        for key in sorted(metrics)
        if key.startswith("val_size/AP50-") or key.startswith("test_size/AP50-")
    }
    print(json.dumps({"remote_prefix": args.remote_prefix, "metrics": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
