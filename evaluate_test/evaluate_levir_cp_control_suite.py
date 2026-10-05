#!/usr/bin/env python3
"""Evaluate a small LEVIR control suite at matched image sizes.

The suite is intended to separate three questions without retraining:

1. Is the reported baseline checkpoint actually the same baseline as the
   augmentation matrix control?
2. Does the Copy-Paste checkpoint lose standard AP50 at the same image size?
3. How much of the apparent small-object gap is caused by evaluating the same
   checkpoint at 512 versus 640?

Each job is supplied explicitly as ``repo_id::prefix::label``. The script uses
one project Ultralytics runtime for every checkpoint, evaluates both val and
test at every requested image size, writes a comparison summary, and uploads
only to the task-specific output repository.

Real remote evaluation must be launched through ``python -m utils.marimo_ops
launch``. This file does not train and does not create detached processes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ULTRALYTICS = ROOT / "models_related" / "ultralytics"
REQUIRED_METRICS = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        nargs="+",
        required=True,
        metavar="REPO::PREFIX::LABEL",
        help="Explicit HF dataset checkpoint jobs. Include one canonical baseline and CP variants.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-repo", required=False, help="Task-specific HF dataset repository")
    parser.add_argument("--hf-repo-id", dest="hf_repo_id", help="Contract-compatible alias for --output-repo")
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[512, 640])
    parser.add_argument("--image-size", type=int, default=512, help="Contract metadata for the primary evaluation size")
    parser.add_argument("--epochs", type=int, default=1, help="Contract metadata; evaluation does not train")
    parser.add_argument("--patience", type=int, default=0, help="Contract metadata; evaluation does not train")
    parser.add_argument("--seed", type=int, default=42, help="Contract metadata for this slot")
    parser.add_argument("--model-yaml", default="source checkpoint weights/best.pt", help="Contract metadata")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_context(output_repo: str) -> str:
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required evaluation")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if "/" not in output_repo:
        raise RuntimeError("--output-repo must be task-specific")
    return token


def parse_job(raw: str) -> tuple[str, str, str]:
    parts = raw.split("::")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Expected REPO::PREFIX::LABEL, got {raw!r}")
    repo, prefix, label = parts
    if "/" not in repo or prefix.startswith("/") or ".." in Path(prefix).parts:
        raise ValueError(f"Unsafe or invalid job specification: {raw!r}")
    return repo, prefix, label


def project_runtime() -> None:
    project = str(PROJECT_ULTRALYTICS)
    sys.path[:] = [entry for entry in sys.path if "vendor/ultralytics_upstream" not in str(Path(entry).resolve())]
    if project in sys.path:
        sys.path.remove(project)
    sys.path.insert(0, project)


def prepare_dataset(data_root: Path, dataset_root: Path, split_seed: int) -> Path:
    from misc.prepare_levir_ship import prepare

    return prepare(data_root, dataset_root / f"levir_cp_control_split_{split_seed}", split_seed).resolve()


def split_metrics(result, split: str) -> dict[str, float]:
    values = {key: float(value) for key, value in result.results_dict.items()}
    return {
        f"{split}/AP50": values["metrics/mAP50(B)"],
        f"{split}/mAP50-95": values["metrics/mAP50-95(B)"],
        f"{split}/AP75": float(result.box.map75),
    }


def evaluate_checkpoint(checkpoint: Path, data_yaml: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, float | int | str]:
    project_runtime()
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    metrics: dict[str, float | int | str] = {
        "split_seed": args.split_seed,
        "nms_iou": args.nms_iou,
        "runtime": "models_related/ultralytics",
    }
    for image_size in args.image_sizes:
        for split in ("val", "test"):
            result = model.val(
                data=str(data_yaml),
                split=split,
                imgsz=image_size,
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
                plots=False,
                iou=args.nms_iou,
                project=str(out_dir / f"imgsz_{image_size}"),
                name=split,
                exist_ok=True,
            )
            values = split_metrics(result, split)
            for key, value in values.items():
                metrics[f"imgsz{image_size}/{key}"] = value
    return metrics


def download_optional_json(hf_hub_download, repo: str, filename: str, destination: Path) -> dict:
    try:
        downloaded = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=str(destination.parent))
    except Exception:
        return {}
    try:
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    args = parse_args()
    if args.output_repo is None:
        args.output_repo = args.hf_repo_id
    elif args.hf_repo_id is not None and args.output_repo != args.hf_repo_id:
        raise ValueError("--output-repo and --hf-repo-id must match")
    if not args.output_repo:
        raise ValueError("Provide --output-repo or --hf-repo-id")
    token = require_context(args.output_repo)
    if args.split_seed != 42 or args.batch_size != 8 or args.workers != 8 or args.nms_iou != 0.5:
        raise ValueError("This control suite requires split_seed=42, batch_size=8, workers=8, and nms_iou=0.5")
    if args.image_size not in args.image_sizes:
        raise ValueError("--image-size must be included in --image-sizes")
    if sorted(set(args.image_sizes)) != sorted(args.image_sizes) or not args.image_sizes:
        raise ValueError("--image-sizes must be non-empty and duplicate-free")

    jobs = [parse_job(raw) for raw in args.jobs]
    labels = [label for _, _, label in jobs]
    if len(labels) != len(set(labels)):
        raise ValueError("Job labels must be unique")
    if not any(label.lower().startswith("baseline") for label in labels):
        raise ValueError("Include an explicitly labelled baseline job")

    from huggingface_hub import HfApi, hf_hub_download

    args.project = args.project.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    data_yaml = prepare_dataset(args.data_root.resolve(), args.dataset_root, args.split_seed)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.output_repo, repo_type="dataset", exist_ok=True)
    remote_files = set(api.list_repo_files(repo_id=args.output_repo, repo_type="dataset"))
    rows: list[dict[str, object]] = []
    all_metrics: dict[str, dict] = {}

    for repo, prefix, label in jobs:
        out_dir = args.project / label
        checkpoint = out_dir / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        remote_base = f"control_suite/{label}"
        summary_remote = f"{remote_base}/evaluation_metrics.json"
        if not args.force and summary_remote in remote_files:
            print(f"SKIP_VERIFIED {label}", flush=True)
            continue
        if not checkpoint.is_file():
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=f"{prefix}/weights/best.pt",
                repo_type="dataset",
                token=token,
                local_dir=str(args.project / "hf_cache" / label),
            )
            shutil.copy2(downloaded, checkpoint)
        source_manifest = download_optional_json(
            hf_hub_download, repo, f"{prefix}/experiment_manifest.json", out_dir / "source_manifest"
        )
        metrics = evaluate_checkpoint(checkpoint, data_yaml, out_dir, args)
        metrics.update({"label": label, "source_repo": repo, "source_prefix": prefix, "checkpoint": f"{prefix}/weights/best.pt"})
        all_metrics[label] = metrics
        for image_size in args.image_sizes:
            row = {"label": label, "source_repo": repo, "source_prefix": prefix, "image_size": image_size}
            row.update({key: value for key, value in metrics.items() if key.startswith(f"imgsz{image_size}/")})
            rows.append(row)
        metrics_path = out_dir / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path = out_dir / "evaluation_manifest.json"
        manifest_path.write_text(json.dumps({
            "suite": "levir_copy_paste_control",
            "label": label,
            "source_repo": repo,
            "source_prefix": prefix,
            "data_root": str(args.data_root.resolve()),
            "dataset_yaml": str(data_yaml),
            "split_seed": args.split_seed,
            "image_sizes": args.image_sizes,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "nms_iou": args.nms_iou,
            "runtime": "models_related/ultralytics",
            "source_manifest": source_manifest,
            "required_metrics": [f"imgsz{size}/{key}" for size in args.image_sizes for key in REQUIRED_METRICS],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"label": label, "metrics": metrics}, sort_keys=True), flush=True)

    baseline = all_metrics.get(next(label for label in labels if label.lower().startswith("baseline")))
    if baseline is None:
        raise RuntimeError("Baseline metrics were not evaluated in this invocation")
    for row in rows:
        size = int(row["image_size"])
        for key in REQUIRED_METRICS:
            current = row.get(f"imgsz{size}/{key}")
            reference = baseline.get(f"imgsz{size}/{key}")
            row[f"delta_vs_baseline/{key}"] = float(current) - float(reference) if current is not None and reference is not None else None

    summary_dir = args.project / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summary_dir / "comparison.json"
    summary_json.write_text(json.dumps({"jobs": all_metrics, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_csv = summary_dir / "comparison.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    uploads = [(summary_json, "control_suite/comparison.json"), (summary_csv, "control_suite/comparison.csv")]
    for label in all_metrics:
        out_dir = args.project / label
        uploads.extend([
            (out_dir / "evaluation_metrics.json", f"control_suite/{label}/evaluation_metrics.json"),
            (out_dir / "evaluation_manifest.json", f"control_suite/{label}/evaluation_manifest.json"),
        ])
    from huggingface_hub import CommitOperationAdd

    api.create_commit(
        repo_id=args.output_repo,
        repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in uploads if local.is_file()],
        commit_message="Add matched LEVIR baseline and Copy-Paste control evaluation",
    )
    remote = set(api.list_repo_files(repo_id=args.output_repo, repo_type="dataset"))
    missing = [remote_path for _, remote_path in uploads if remote_path not in remote]
    if missing:
        raise RuntimeError(f"Remote verification failed: {missing}")
    print(json.dumps({"jobs": len(all_metrics), "rows": len(rows), "output_repo": args.output_repo, "verified": True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
