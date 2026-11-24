#!/usr/bin/env python3
"""Train, evaluate, and upload H1 direct enhancement with canonical FTAL."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from misc.prepare_levir_ship import prepare
from utils.marimo_ops import require_training_context

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_h1_object_aware.yaml"
TRAIN_REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv")
COMPLETE_REQUIRED = (*TRAIN_REQUIRED, "evaluation_metrics.json")
FTAL = {
    "factorized_tal_target": True,
    "factorized_tal_mode": "legacy",
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
    "factorized_tal_s_max": 32.0,
    "factorized_tal_warmup_start": 5,
    "factorized_tal_warmup_end": 15,
    "factorized_tal_p2_only": True,
}


def local_ultralytics() -> None:
    sys.path.insert(0, str(ULTRALYTICS))


def has_files(path: Path, files: tuple[str, ...]) -> bool:
    return all((path / item).is_file() for item in files)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(run_dir: Path, data: Path, args: argparse.Namespace) -> None:
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = model.val(
            data=str(data), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in (result.results_dict or {}).items()})
        metrics[f"{split}/AP75"] = float(result.box.map75)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def upload_and_verify(run_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    prefix = "runs/H1_FTAL"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=prefix, repo_id=repo_id, repo_type="dataset")
    remote_files = {
        item.rfilename for item in api.list_repo_tree(
            repo_id=repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True
        ) if hasattr(item, "rfilename")
    }
    missing = {f"{prefix}/{item}" for item in COMPLETE_REQUIRED} - remote_files
    if missing:
        raise RuntimeError(f"Remote upload verification failed: {sorted(missing)}")
    (run_dir / "upload_complete.json").write_text(json.dumps({"repo_id": repo_id, "remote_prefix": prefix}, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    require_training_context(hf_repo_id=args.hf_repo_id)
    data = prepare(args.data_root, args.dataset_root / f"levir_ship_yolo_seed{args.seed}", args.seed)
    run_dir = args.project / "H1_FTAL"
    if not has_files(run_dir, TRAIN_REQUIRED):
        local_ultralytics()
        from ultralytics import YOLO

        seed_everything(args.seed)
        model = YOLO(str(CONFIG))
        model.load("yolov8n.pt", smart_transfer=True)
        model.train(
            data=str(data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
            batch=args.batch_size, device=args.device, workers=args.workers, amp=args.amp,
            seed=args.seed, deterministic=True, project=str(args.project), name="H1_FTAL", exist_ok=True,
            **FTAL,
        )
    if not has_files(run_dir, TRAIN_REQUIRED):
        raise FileNotFoundError(f"Incomplete training artifacts: {run_dir}")
    if not (run_dir / "evaluation_metrics.json").is_file():
        evaluate(run_dir, data, args)
    if not has_files(run_dir, COMPLETE_REQUIRED):
        raise FileNotFoundError(f"Incomplete evaluation artifacts: {run_dir}")
    upload_and_verify(run_dir, args.hf_repo_id)
    print("COMPLETE H1_FTAL", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
