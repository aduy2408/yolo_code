#!/usr/bin/env python3
"""Re-evaluate uploaded YOLO baseline checkpoints on val and test splits.

This is an evaluation-only Marimo runner. It downloads completed baseline
checkpoints from the task-specific Hugging Face source repositories, evaluates
both validation and held-out test splits with NMS IoU 0.50, and uploads
split-qualified metrics to a separate task-specific report repository.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_REPOS = (
    "duyle2408/levir-ship-yolo-baselines",
    "duyle2408/tinyperson-yolo-baselines",
    "duyle2408/varroa-yolo-baselines-part1-full",
    "duyle2408/varroa-yolo-baselines-part2-full",
    "duyle2408/yolo-baselines-no-mosaic-musgd-runs",
)
DATA_ROOTS = {
    "levirship": "/marimo/LevirShip/LevirShipData",
    "tinyperson": "/marimo/TinyPerson",
    "varroa": "/marimo/Varroa",
}
IMAGE_SIZES = {"levirship": 512, "tinyperson": 640, "varroa": 640}
REQUIRED_METRICS = (
    "val/AP50",
    "val/AP75",
    "val/mAP50-95",
    "val_size/AP50-Small",
    "val_size/AP75",
    "test/AP50",
    "test/AP75",
    "test/mAP50-95",
    "test_size/AP50-Small",
    "test_size/AP75",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-levir", type=Path, default=Path(DATA_ROOTS["levirship"]))
    parser.add_argument("--data-root-tinyperson", type=Path, default=Path(DATA_ROOTS["tinyperson"]))
    parser.add_argument("--data-root-varroa", type=Path, default=Path(DATA_ROOTS["varroa"]))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", nargs="*", help="Exact source repo/prefix pairs: repo::prefix")
    return parser.parse_args()


def require_context(repo_id: str) -> str:
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required evaluation")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if "/" not in repo_id:
        raise RuntimeError("--hf-repo-id must be task-specific")
    return token


def dataset_for(repo: str, prefix: str) -> str:
    if "levir" in repo.lower() or prefix.startswith("runs/levirship/"):
        return "levirship"
    if "tinyperson" in repo.lower() or prefix.startswith("runs/tinyperson/"):
        return "tinyperson"
    return "varroa"


def split_prefix(repo: str, prefix: str) -> tuple[str, str, int, str]:
    dataset = dataset_for(repo, prefix)
    parts = prefix.split("/")
    if parts[0] == "train":
        run_name = parts[-1]
        model = run_name.removesuffix("_seed42").removesuffix("_seed43").removesuffix("_seed44")
        seed = int(run_name.rsplit("_seed", 1)[1])
    elif parts[0] == "runs":
        model = parts[-2]
        seed = int(parts[-1].removeprefix("seed_"))
    else:
        raise ValueError(f"Unsupported baseline prefix: {repo}::{prefix}")
    mode = "no_mosaic" if "no-mosaic" in repo else "mosaic"
    return dataset, model, seed, mode


def prepare_dataset(dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "levirship":
        from misc.prepare_levir_ship import prepare
        return prepare(
            args.data_root_levir,
            args.dataset_root / f"levirship_split_{args.split_seed}",
            args.split_seed,
        ).resolve()
    if dataset == "tinyperson":
        from train_scripts import train_all_tinyperson as tiny
        test_root = tiny.prepare_test_set(args.data_root_tinyperson, args.dataset_root)
        split_root = tiny.prepare_seed_dataset(
            args.data_root_tinyperson,
            args.dataset_root,
            test_root,
            args.split_seed,
        )
        return (split_root / "tinyperson.yaml").resolve()
    from misc.prepare_dataset import prepare_dataset as prepare_varroa
    return prepare_varroa(
        args.data_root_varroa,
        args.dataset_root / f"varroa_split_{args.split_seed}",
        gt_source="gt_one",
        only_positives=True,
        class_policy="map-3-to-1",
        seed=args.split_seed,
    ).resolve()


def evaluate_checkpoint(checkpoint: Path, data_yaml: Path, dataset: str, out_dir: Path, args: argparse.Namespace) -> dict[str, float | str]:
    from train_scripts.train_all_yolo_baselines_no_mosaic import local_ultralytics
    from size_bucket_evaluator import evaluate_native_size_buckets
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    metrics: dict[str, float | str] = {}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=IMAGE_SIZES[dataset],
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            iou=0.5,
            plots=False,
            project=str(out_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics[f"{split}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{split}/AP75"] = float(result.box.map75)
        metrics[f"{split}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
        metrics.update(
            evaluate_native_size_buckets(
                out_dir,
                data_yaml,
                split=split,
                imgsz=IMAGE_SIZES[dataset],
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
            )
        )
    metrics["nms_iou"] = 0.5
    metrics["split_seed"] = args.split_seed
    if dataset == "tinyperson":
        metrics["test_protocol"] = "TinyPerson standard corner-window test split; merged corner-window metrics are separate"
    else:
        metrics["test_protocol"] = "Native held-out test split"
    return metrics


def discover_jobs(api, requested: list[str] | None) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for repo in SOURCE_REPOS:
        for path in api.list_repo_files(repo_id=repo, repo_type="dataset"):
            if path.endswith("/weights/best.pt"):
                jobs.append((repo, path.removesuffix("/weights/best.pt")))
    jobs = sorted(jobs)
    if requested:
        wanted = set(requested)
        jobs = [(repo, prefix) for repo, prefix in jobs if f"{repo}::{prefix}" in wanted]
        missing = sorted(wanted - {f"{repo}::{prefix}" for repo, prefix in jobs})
        if missing:
            raise RuntimeError(f"Requested baseline jobs not found: {missing}")
    return jobs


def main() -> None:
    args = parse_args()
    token = require_context(args.hf_repo_id)
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    prepared = {dataset: prepare_dataset(dataset, args) for dataset in DATA_ROOTS}
    jobs = discover_jobs(api, args.jobs)
    print(json.dumps({"jobs": len(jobs), "hf_repo_id": args.hf_repo_id, "split_seed": args.split_seed}, sort_keys=True), flush=True)
    pending: list[CommitOperationAdd] = []

    for repo, prefix in jobs:
        dataset, model, seed, mode = split_prefix(repo, prefix)
        slug = repo.split("/", 1)[1]
        out_dir = args.project / slug / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = f"{slug}/{prefix}"
        metrics_path = out_dir / "evaluation_metrics.json"
        manifest_path = out_dir / "evaluation_manifest.json"
        marker_path = out_dir / "evaluation_complete.json"
        if not args.force and all(path.is_file() for path in (metrics_path, manifest_path, marker_path)):
            print(f"SKIP_VERIFIED {output_prefix}", flush=True)
            continue
        checkpoint = out_dir / "weights/best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            downloaded = hf_hub_download(repo_id=repo, filename=f"{prefix}/weights/best.pt", repo_type="dataset", token=token)
            shutil.copy2(downloaded, checkpoint)
        metrics = evaluate_checkpoint(checkpoint, prepared[dataset], dataset, out_dir, args)
        metrics.update({"dataset": dataset, "model": model, "seed": seed, "mode": mode, "source_repo": repo, "source_prefix": prefix, "output_repo": args.hf_repo_id, "output_prefix": output_prefix})
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        missing = [key for key in REQUIRED_METRICS if key not in metrics]
        if missing:
            raise RuntimeError(f"Missing split-qualified metrics for {output_prefix}: {missing}")
        manifest_path.write_text(json.dumps({"source_repo": repo, "source_prefix": prefix, "output_repo": args.hf_repo_id, "output_prefix": output_prefix, "dataset": dataset, "model": model, "seed": seed, "mode": mode, "split_seed": args.split_seed, "nms_iou": 0.5, "required_metrics": list(REQUIRED_METRICS), "test_protocol": metrics["test_protocol"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        marker_path.write_text(json.dumps({"verified": True, "output_prefix": output_prefix, "required_metrics": list(REQUIRED_METRICS)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for local, remote in ((metrics_path, f"{output_prefix}/evaluation_metrics.json"), (manifest_path, f"{output_prefix}/evaluation_manifest.json"), (marker_path, f"{output_prefix}/evaluation_complete.json")):
            pending.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
        print(f"EVALUATION_COMPLETE {output_prefix}", flush=True)

    if pending:
        api.create_commit(repo_id=args.hf_repo_id, repo_type="dataset", operations=pending, commit_message="Add split-qualified YOLO baseline evaluation metrics")
        remote = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
        missing = [op.path_in_repo for op in pending if op.path_in_repo not in remote]
        if missing:
            raise RuntimeError(f"Remote verification failed: {missing}")
        print(json.dumps({"uploaded_files": len(pending), "verified": True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
