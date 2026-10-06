#!/usr/bin/env python3
"""Evaluate current augmentation-matrix checkpoints with native size metrics."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--data-root-levir", type=Path, required=True)
    p.add_argument("--data-root-varroa", type=Path, required=True)
    p.add_argument("--data-root-tinyperson", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-yaml", default="source checkpoint weights/best.pt")
    p.add_argument("--data-root", default="matrix")
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
    args.project.mkdir(parents=True, exist_ok=True)
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
    pairs = {(j.split("::", 1)[1].split("/")[1], j.split("::", 1)[1].split("/")[2]) for j in args.jobs}
    prepared = {pair: prepare_dataset(pair[0], base, pair[1]) for pair in sorted(pairs)}
    remote_files = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
    print(json.dumps({"jobs": len(args.jobs), "output_repo": args.hf_repo_id}), flush=True)
    for item in args.jobs:
        source_repo, prefix = item.split("::", 1)
        parts = prefix.split("/")
        if len(parts) != 6:
            raise RuntimeError(f"Expected model/dataset/method/variant/mosaic/seed prefix: {prefix}")
        model, dataset, method, variant, mosaic_mode, seed_name = parts
        source_slug = source_repo.split("/", 1)[1].replace("/", "_")
        out_prefix = f"{source_slug}/{prefix}"
        marker_remote = f"{out_prefix}/size_metrics_complete.json"
        if marker_remote in remote_files and not args.force:
            print(json.dumps({"prefix": prefix, "status": "SKIP_SIZE_VERIFIED"}), flush=True)
            continue
        out_dir = args.project / out_prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = out_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            downloaded = hf_hub_download(repo_id=source_repo, filename=f"{prefix}/weights/best.pt", repo_type="dataset", local_dir=str(args.project / "hf_cache" / source_slug))
            shutil.copy2(downloaded, checkpoint)
        data_yaml, test_root = prepared[dataset, method]
        eval_args = SimpleNamespace(batch_size=args.batch_size, image_size=args.image_size, device=args.device, workers=args.workers)
        size_metrics, artifacts, protocol = evaluate_size_metrics(out_dir, dataset, data_yaml, test_root, base.data_root_levir if dataset == "levir" else base.data_root_varroa if dataset == "varroa" else base.data_root_tinyperson, eval_args)
        source_metrics = {}
        try:
            source_json = hf_hub_download(repo_id=source_repo, filename=f"{prefix}/evaluation_metrics.json", repo_type="dataset", local_dir=str(out_dir / "source_metrics"))
            source_metrics = json.loads(Path(source_json).read_text())
        except Exception:
            pass
        metrics = {**source_metrics, **size_metrics}
        standard = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")
        if any(k not in metrics for k in standard):
            metrics.update(evaluate_standard_metrics(out_dir, data_yaml, dataset, eval_args))
        metrics.update({"source_repo": source_repo, "source_prefix": prefix, "output_repo": args.hf_repo_id, "output_prefix": out_prefix, "model": model, "dataset": dataset, "method": method, "variant": variant, "mosaic_mode": mosaic_mode, "seed": int(seed_name.removeprefix("seed_")), "split_seed": args.split_seed, "nms_iou": 0.5, "test_protocol": "TinyPerson official corner-window merged evaluator" if dataset == "tinyperson" else ("LEVIR-Ship standard held-out test split" if dataset == "levir" else "Varroa standard held-out test split"), "size_metric_protocol": protocol})
        metrics_path = out_dir / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        manifest_path = out_dir / "size_metrics_manifest.json"
        manifest_path.write_text(json.dumps({"source_repo": source_repo, "source_prefix": prefix, "output_repo": args.hf_repo_id, "output_prefix": out_prefix, "model": model, "dataset": dataset, "method": method, "variant": variant, "mosaic_mode": mosaic_mode, "seed": metrics["seed"], "split_seed": args.split_seed, "nms_iou": 0.5, "size_metric_keys": sorted(k for k in size_metrics if k.startswith(("val_size/", "test_size/", "test_merged/"))), "size_metric_source_artifacts": [str(p.relative_to(out_dir)) for p in artifacts if p.is_file()]}, indent=2, sort_keys=True) + "\n")
        marker_path = out_dir / "size_metrics_complete.json"
        marker_path.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_prefix": out_prefix, "verified": True}, indent=2, sort_keys=True) + "\n")
        uploads = [(metrics_path, f"{out_prefix}/evaluation_metrics.json"), (manifest_path, f"{out_prefix}/size_metrics_manifest.json"), (marker_path, marker_remote)]
        uploads.extend((p, f"{out_prefix}/{p.relative_to(out_dir)}") for p in artifacts if p.is_file())
        api.create_commit(repo_id=args.hf_repo_id, repo_type="dataset", operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in uploads], commit_message=f"Add AP50-small metrics for {prefix}")
        remote_files.update(remote for _, remote in uploads)
        print(json.dumps({"prefix": prefix, "status": "verified", **{k: metrics.get(k) for k in (*standard, "val_size/AP50-small", "test_size/AP50-small")}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
