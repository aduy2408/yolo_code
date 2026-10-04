#!/usr/bin/env python3
"""Evaluate historical YOLO augmentation checkpoints without retraining.

Supports source repositories whose checkpoint layout is not the current 38-run
matrix layout, such as ``mosaic-only-yolo-runs``. It writes split-qualified
core metrics and native size-bucket metrics, then uploads a separate artifact
family to a task-specific Hugging Face dataset repository.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from evaluate_test.evaluate_augmentation_size_metrics import (
    evaluate_size_metrics,
    evaluate_standard_metrics,
    prepare_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--data-root-levir", type=Path, required=True)
    p.add_argument("--data-root-varroa", type=Path, required=True)
    p.add_argument("--data-root-tinyperson", type=Path, required=True)
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    remote_files = set(api.list_repo_files(repo_id=args.hf_repo_id, repo_type="dataset"))
    base = SimpleNamespace(
        data_root_levir=args.data_root_levir,
        data_root_varroa=args.data_root_varroa,
        data_root_tinyperson=args.data_root_tinyperson,
        dataset_root=args.dataset_root,
        split_seed=args.split_seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
    prepared = {}
    for dataset in sorted({job.split("/")[1] for job in args.jobs}):
        prepared[dataset] = prepare_dataset(dataset, base, "mosaic")
    print(json.dumps({"jobs": len(args.jobs), "output_repo": args.hf_repo_id}, sort_keys=True), flush=True)
    for source_prefix in args.jobs:
        parts = source_prefix.split("/")
        if len(parts) < 4 or parts[0] != "runs":
            raise RuntimeError(f"Unexpected historical prefix: {source_prefix}")
        dataset = parts[1]
        variant = parts[2]
        seed_name = parts[3]
        out_prefix = f"historical_mosaic/{dataset}/{variant}/{seed_name}"
        marker_remote = f"{out_prefix}/size_metrics_complete.json"
        if not args.force and marker_remote in remote_files:
            print(f"SKIP_SIZE_VERIFIED {source_prefix}", flush=True)
            continue
        out_dir = args.project / out_prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = out_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            downloaded = hf_hub_download(
                repo_id=args.source_repo,
                filename=f"{source_prefix}/weights/best.pt",
                repo_type="dataset",
                local_dir=str(args.project / "hf_cache"),
            )
            shutil.copy2(downloaded, checkpoint)
        data_yaml, test_root = prepared[dataset]
        eval_args = SimpleNamespace(
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
            workers=args.workers,
        )
        source_metrics = {}
        try:
            downloaded = hf_hub_download(
                repo_id=args.source_repo,
                filename=f"{source_prefix}/evaluation_metrics.json",
                repo_type="dataset",
                local_dir=str(out_dir / "source_metrics"),
            )
            source_metrics = json.loads(Path(downloaded).read_text())
        except Exception:
            pass
        size_metrics, artifacts, protocol = evaluate_size_metrics(
            out_dir, dataset, data_yaml, test_root, base.data_root_levir if dataset == "levir" else base.data_root_varroa if dataset == "varroa" else base.data_root_tinyperson, eval_args
        )
        metrics = {**source_metrics, **size_metrics, "source_repo": args.source_repo, "source_prefix": source_prefix, "output_repo": args.hf_repo_id, "output_prefix": out_prefix, "dataset": dataset, "variant": variant, "seed": int(seed_name.removeprefix("seed_")), "split_seed": args.split_seed, "nms_iou": 0.5, "test_protocol": "TinyPerson official corner-window merged evaluator" if dataset == "tinyperson" else "native held-out test split"}
        for key in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95"):
            if key not in metrics:
                metrics.update(evaluate_standard_metrics(out_dir, data_yaml, dataset, eval_args))
                break
        metrics_path = out_dir / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        manifest = {"source_repo": args.source_repo, "source_prefix": source_prefix, "output_repo": args.hf_repo_id, "output_prefix": out_prefix, "dataset": dataset, "variant": variant, "seed": metrics["seed"], "split_seed": args.split_seed, "nms_iou": 0.5, "test_protocol": metrics["test_protocol"], "size_metric_keys": sorted(k for k in size_metrics if k.startswith(("val_size/", "test_size/", "test_merged/"))), "size_metric_source_artifacts": [str(p.relative_to(out_dir)) for p in artifacts if p.is_file()]}
        manifest_path = out_dir / "size_metrics_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        marker = out_dir / "size_metrics_complete.json"
        marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_prefix": out_prefix, "verified": True}, indent=2) + "\n")
        uploads = [(metrics_path, f"{out_prefix}/evaluation_metrics.json"), (manifest_path, f"{out_prefix}/size_metrics_manifest.json"), (marker, marker_remote)]
        uploads.extend((p, f"{out_prefix}/{p.relative_to(out_dir)}") for p in artifacts if p.is_file())
        api.create_commit(repo_id=args.hf_repo_id, repo_type="dataset", operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in uploads], commit_message=f"Add historical Mosaic metrics {variant}")
        remote_files.update(remote for _, remote in uploads)
        print(json.dumps({"source_prefix": source_prefix, "output_prefix": out_prefix, "metrics": {k: metrics.get(k) for k in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95", "val_size/AP75", "val_size/AP50-Small", "test_size/AP75", "test_size/AP50-Small")}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
