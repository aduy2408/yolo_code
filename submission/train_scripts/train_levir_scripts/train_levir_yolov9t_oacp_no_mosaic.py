#!/usr/bin/env python3
"""Train, evaluate, and upload official single-head YOLOv9t + OACP without Mosaic on LEVIR-Ship."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
EXPERIMENT = "levir_yolov9t_oacp_no_mosaic"
VARIANT = "yolov9t_oacp_no_mosaic_seed42"
HF_REPO = "duyle2408/levir-yolov9t-oacp-no-mosaic-seed42"
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv")
COMPLETE = (*REQUIRED, "evaluation_metrics.json", "manifest.json")


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_files(root: Path, names: tuple[str, ...]) -> bool:
    return all((root / name).is_file() for name in names)


def evaluate(run: Path, data: Path, args: argparse.Namespace) -> dict[str, float]:
    if (run / "evaluation_metrics.json").is_file():
        return json.loads((run / "evaluation_metrics.json").read_text())
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, float] = {"nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run / "weights/best.pt").val(
            data=str(data), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/AP75"] = float(result.box.map75)
    (run / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def upload(run: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    prefix = f"runs/{VARIANT}"
    api.upload_folder(folder_path=str(run), path_in_repo=prefix, repo_id=repo_id, repo_type="dataset")
    remote = {
        item.rfilename
        for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True)
        if hasattr(item, "rfilename")
    }
    missing = {f"{prefix}/{name}" for name in COMPLETE} - remote
    if missing:
        raise RuntimeError(f"Upload verification failed: {sorted(missing)}")
    (run / "upload_complete.json").write_text(json.dumps({
        "repo_id": repo_id, "remote_prefix": prefix, "remote_verified": True,
    }, indent=2) + "\n")
    api.upload_file(
        path_or_fileobj=str(run / "upload_complete.json"),
        path_in_repo=f"{prefix}/upload_complete.json",
        repo_id=repo_id,
        repo_type="dataset",
    )


def run(args: argparse.Namespace) -> None:
    from utils.marimo_ops import require_training_context
    from misc.prepare_levir_ship import prepare

    require_training_context(hf_repo_id=args.hf_repo_id)
    os.environ["YOLO_CONTEXT_AUG"] = "oacp"
    local_ultralytics()
    from ultralytics import YOLO

    data = prepare(args.data_root, args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}", args.split_seed)
    run_dir = args.project / VARIANT
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if not has_files(run_dir, REQUIRED):
        model = YOLO("yolov9t.pt")
        model.train(
            data=str(data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
            batch=args.batch_size, device=args.device, workers=args.workers,
            seed=args.seed, deterministic=False, amp=args.amp, plots=False, mosaic=0.0, close_mosaic=0,
            project=str(args.project), name=VARIANT, exist_ok=True,
        )
    if not has_files(run_dir, REQUIRED):
        raise FileNotFoundError(f"Training ended without required artifacts: {run_dir}")
    metrics = evaluate(run_dir, data, args)
    (run_dir / "manifest.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "variant": VARIANT, "architecture": "official YOLOv9t, single Detect(P3,P4,P5)",
        "augmentation": "oacp", "model_source": "yolov9t.pt", "commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(), "seed": args.seed, "split_seed": args.split_seed, "epochs": args.epochs,
        "patience": args.patience, "imgsz": args.imgsz, "batch_size": args.batch_size,
        "nms_iou": 0.5, "hf_repo_id": args.hf_repo_id, "metrics": metrics,
    }, indent=2, sort_keys=True) + "\n")
    if not has_files(run_dir, COMPLETE):
        raise FileNotFoundError(f"Incomplete completion artifacts: {run_dir}")
    upload(run_dir, args.hf_repo_id)
    print(f"COMPLETE {VARIANT}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
