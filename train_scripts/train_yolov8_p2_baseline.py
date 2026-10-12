#!/usr/bin/env python3
"""Train, evaluate, and upload one strict YOLOv8 P2 baseline run.

The runner is dataset-specific at invocation time so the declared dataset YAML,
P2 topology, split protocol, and upload prefix remain immutable and auditable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ULTRALYTICS = ROOT / "models_related/ultralytics"
MODEL_DEFAULTS = {
    "levirship": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml",
    "tinyperson": ROOT / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3_plain.yaml",
}
IMAGE_SIZES = {"levirship": 512, "tinyperson": 640}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "config.yaml",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)
REQUIRED_METRICS = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")


def local_ultralytics() -> None:
    if str(PROJECT_ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(PROJECT_ULTRALYTICS))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(args: argparse.Namespace) -> Path:
    if args.dataset_name == "levirship":
        from misc.prepare_levir_ship import prepare

        return prepare(
            args.data_root,
            args.dataset_root / f"levirship_split_{args.split_seed}",
            args.split_seed,
        ).resolve()
    import train_scripts.train_all_tinyperson as tiny

    test_dir = tiny.prepare_test_set(args.data_root, args.dataset_root)
    seed_dir = tiny.prepare_seed_dataset(
        args.data_root,
        args.dataset_root,
        test_dir,
        args.split_seed,
    )
    return (seed_dir / "tinyperson.yaml").resolve()


def training_complete(run_dir: Path) -> bool:
    results = run_dir / "results.csv"
    return (
        all((run_dir / item).is_file() for item in ("weights/best.pt", "weights/last.pt"))
        and results.is_file()
        and sum(1 for _ in results.open(encoding="utf-8")) > 1
    )


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, object] = {"checkpoint": "weights/best.pt", "nms_iou": 0.5}
    model = YOLO(str(run_dir / "weights/best.pt"))
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            iou=0.5,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics[f"{split}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{split}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
        for key, value in result.results_dict.items():
            metrics[f"{split}/{key}"] = float(value)
        metrics[f"{split}/protocol"] = "native YOLO split evaluation"
    if args.dataset_name == "tinyperson":
        import train_scripts.train_all_tinyperson as tiny

        test_dir = args.dataset_root / "tinyperson_test_corner_sw640_sh512"
        tiny_args = argparse.Namespace(
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device,
            workers=args.workers,
            dataset_root=args.dataset_root,
        )
        merged = tiny.evaluate_merged_test(run_dir, test_dir, args.data_root, tiny_args)
        metrics.update(merged)
        metrics["test/protocol"] = "official TinyPerson corner-window merged test"
        metrics["test/source_artifact"] = str(test_dir / "corner_manifest.json")
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise RuntimeError(f"Missing split-qualified metrics: {missing}")
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def upload(run_dir: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi
    from utils.marimo_ops import ensure_hf_repo

    repo_id = ensure_hf_repo(args.hf_repo_id)
    api = HfApi(token=os.environ["HF_TOKEN"])
    remote = f"{args.dataset_name}/p2_baseline/seed_{args.seed}"
    api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="dataset",
    )
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    expected = {f"{remote}/{item}" for item in REQUIRED}
    missing = sorted(expected - remote_files)
    if missing:
        raise RuntimeError(f"HF upload verification failed: {missing}")
    marker = {
        "repo_id": repo_id,
        "remote_prefix": remote,
        "verified": sorted(expected),
    }
    marker_path = run_dir / "upload_complete.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(marker_path),
        path_in_repo=f"{remote}/upload_complete.json",
        repo_id=repo_id,
        repo_type="dataset",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", choices=tuple(MODEL_DEFAULTS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-yaml", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from utils.marimo_ops import require_training_context

    if args.split_seed != 42:
        raise ValueError("split seed must remain fixed at 42")
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.model_yaml = (args.model_yaml or MODEL_DEFAULTS[args.dataset_name]).resolve()
    args.imgsz = args.imgsz or IMAGE_SIZES[args.dataset_name]
    if not args.model_yaml.is_file():
        raise FileNotFoundError(args.model_yaml)
    require_training_context(hf_repo_id=args.hf_repo_id)
    data_yaml = prepare_dataset(args)
    run_dir = args.project / "p2_baseline" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "yolov8_p2_baseline",
        "dataset": args.dataset_name,
        "data_root": str(args.data_root),
        "dataset_yaml": str(data_yaml),
        "model_yaml": str(args.model_yaml),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "image_size": args.imgsz,
        "workers": args.workers,
        "nms_iou": 0.5,
        "mosaic": 0.0,
        "hf_repo_id": args.hf_repo_id,
        "remote_prefix": f"{args.dataset_name}/p2_baseline/seed_{args.seed}",
        "upload_required": True,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "test_protocol": "official TinyPerson corner-window merged test" if args.dataset_name == "tinyperson" else "native LEVIR-Ship test split",
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not training_complete(run_dir):
        seed_everything(args.seed)
        local_ultralytics()
        from ultralytics import YOLO

        model = YOLO(str(args.model_yaml))
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            amp=True,
            mosaic=0.0,
            close_mosaic=0,
            plots=False,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            optimizer="auto",
        )
    if not training_complete(run_dir):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    metrics = evaluate(run_dir, data_yaml, args)
    manifest["metrics"] = {key: metrics[key] for key in REQUIRED_METRICS}
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upload(run_dir, args)
    print(json.dumps({"run_dir": str(run_dir), **{key: metrics[key] for key in REQUIRED_METRICS}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
