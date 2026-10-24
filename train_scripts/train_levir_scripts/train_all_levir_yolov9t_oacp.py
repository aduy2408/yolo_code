#!/usr/bin/env python3
"""Train, evaluate, and upload official single-head YOLOv9t + OACP on LEVIR-Ship."""

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
EXPERIMENT = "levir_yolov9t_oacp_official"
VARIANT_PREFIX = "yolov9t_oacp"
HF_REPO = "duyle2408/levir-yolov9t-oacp-official-runs"
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


def upload(run: Path, repo_id: str, variant: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    prefix = f"runs/{variant}"
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


def run(args: argparse.Namespace) -> None:
    from utils.marimo_ops import require_training_context
    from misc.prepare_levir_ship import prepare

    require_training_context(hf_repo_id=args.hf_repo_id)
    os.environ["YOLO_CONTEXT_AUG"] = "oacp"
    local_ultralytics()
    from ultralytics import YOLO

    data = prepare(args.data_root, args.dataset_root / f"levir_ship_yolo_split_seed{args.split_seed}", args.split_seed)
    variant = f"{VARIANT_PREFIX}_seed{args.seed}"
    run_dir = args.project / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if not has_files(run_dir, REQUIRED):
        model = YOLO("yolov9t.pt")
        model.train(
            data=str(data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
            batch=args.batch_size, device=args.device, workers=args.workers,
            seed=args.seed, deterministic=True, amp=args.amp, plots=False,
            project=str(args.project), name=variant, exist_ok=True,
        )
    if not has_files(run_dir, REQUIRED):
        raise FileNotFoundError(f"Training ended without required artifacts: {run_dir}")
    metrics = evaluate(run_dir, data, args)
    (run_dir / "manifest.json").write_text(json.dumps({
        "experiment": EXPERIMENT, "variant": variant, "architecture": "official YOLOv9t, single Detect(P3,P4,P5)",
        "augmentation": "oacp", "model_source": "yolov9t.pt", "commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(), "seed": args.seed, "split_seed": args.split_seed, "epochs": args.epochs,
        "patience": args.patience, "imgsz": args.imgsz, "batch_size": args.batch_size,
        "nms_iou": 0.5, "hf_repo_id": args.hf_repo_id, "metrics": metrics,
    }, indent=2, sort_keys=True) + "\n")
    if not has_files(run_dir, COMPLETE):
        raise FileNotFoundError(f"Incomplete completion artifacts: {run_dir}")
    upload(run_dir, args.hf_repo_id, variant)
    print(f"COMPLETE {variant}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", dest="seeds", type=int, help="single training seed")
    seed_group.add_argument("--seeds", dest="seed_list", type=int, nargs="+", help="training seeds")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.seeds = [args.seeds] if args.seeds is not None else (args.seed_list or [43])
    del args.seed_list
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.split_seed != 42 or any(seed not in (43, 44) for seed in args.seeds):
        raise ValueError("requires --split-seed 42 and training seeds limited to 43/44")
    for seed in args.seeds:
        args.seed = seed
        run(args)
