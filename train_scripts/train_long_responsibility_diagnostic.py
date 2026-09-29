"""Contract-compliant single-run driver for long B0/B4/B6 diagnostics.

The Marimo supervisor launches one dataset/variant at a time. This keeps the
run contract explicit while the outer queue verifies artifacts before starting
the next item.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "models_related/ultralytics"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tinyperson", "visdrone", "levirship"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--model-yaml", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--responsibility-mode", choices=("off", "residual_tiny", "residual_curriculum", "kl_tiny", "kl_curriculum"), required=True)
    parser.add_argument("--responsibility-lambda-max", type=float, default=0.25)
    parser.add_argument("--responsibility-eta-max", type=float, default=0.25)
    parser.add_argument("--responsibility-delta-clip", type=float, default=0.5)
    parser.add_argument("--responsibility-warmup-epochs", type=int, default=20)
    parser.add_argument("--responsibility-ramp-epochs", type=int, default=20)
    parser.add_argument("--responsibility-tiny-max-dim", type=float, default=16.0)
    parser.add_argument("--responsibility-consistency-tau", type=float, default=5.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required training")
    if args.split_seed != 42:
        raise ValueError("This matched diagnostic matrix requires split-seed=42")
    for key, value in {
        "RESPONSIBILITY_MODE": args.responsibility_mode,
        "RESPONSIBILITY_LAMBDA_MAX": args.responsibility_lambda_max,
        "RESPONSIBILITY_ETA_MAX": args.responsibility_eta_max,
        "RESPONSIBILITY_DELTA_CLIP": args.responsibility_delta_clip,
        "RESPONSIBILITY_WARMUP_EPOCHS": args.responsibility_warmup_epochs,
        "RESPONSIBILITY_RAMP_EPOCHS": args.responsibility_ramp_epochs,
        "RESPONSIBILITY_TINY_MAX_DIM": args.responsibility_tiny_max_dim,
        "RESPONSIBILITY_CONSISTENCY_TAU": args.responsibility_consistency_tau,
    }.items():
        os.environ[key] = str(value)
    seed_everything(args.seed)
    from ultralytics import YOLO
    model = YOLO(str(args.model_yaml))
    model.load(args.pretrained, smart_transfer=True)
    args.project.mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(args.dataset_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch_size, device=args.device, workers=args.workers,
        patience=args.patience, seed=args.seed, deterministic=True, amp=True,
        optimizer="auto", mosaic=0.0, close_mosaic=0, save_period=10,
        plots=False, project=str(args.project), name="seed_42", exist_ok=True,
    )
    run_dir = args.project / "seed_42"
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Missing checkpoint: {best}")
    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": args.nms_iou}
    for split in ("val", "test"):
        result = YOLO(str(best)).val(
            data=str(args.dataset_yaml), split=split, imgsz=args.imgsz,
            batch=args.batch_size, device=args.device, workers=args.workers,
            iou=args.nms_iou, plots=False, project=str(run_dir / "evaluation"),
            name=split, exist_ok=True,
        )
        metrics[f"{split}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{split}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
    manifest = {
        "dataset": args.dataset, "data_root": str(args.data_root), "dataset_yaml": str(args.dataset_yaml),
        "model_yaml": str(args.model_yaml), "pretrained": args.pretrained, "seed": args.seed,
        "split_seed": args.split_seed, "epochs": args.epochs, "patience": args.patience,
        "imgsz": args.imgsz, "batch_size": args.batch_size, "workers": args.workers,
        "nms_iou": args.nms_iou, "responsibility_mode": args.responsibility_mode,
        "responsibility_lambda_max": args.responsibility_lambda_max,
        "responsibility_eta_max": args.responsibility_eta_max,
        "responsibility_delta_clip": args.responsibility_delta_clip,
        "responsibility_warmup_epochs": args.responsibility_warmup_epochs,
        "responsibility_ramp_epochs": args.responsibility_ramp_epochs,
        "responsibility_tiny_max_dim": args.responsibility_tiny_max_dim,
        "responsibility_consistency_tau": args.responsibility_consistency_tau,
        "test_protocol": "dataset YAML native test split",
        "hf_repo_id": args.hf_repo_id,
        **metrics,
    }
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    from utils.marimo_ops import ensure_hf_repo
    from huggingface_hub import HfApi
    repo_id = ensure_hf_repo(args.hf_repo_id)
    api = HfApi(token=os.environ["HF_TOKEN"])
    remote = f"runs/{args.dataset}/{args.responsibility_mode}/seed_{args.seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    required = ["weights/best.pt", "weights/last.pt", "results.csv", "evaluation_metrics.json", "experiment_manifest.json"]
    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = [f"{remote}/{item}" for item in required]
    if not all(item in files for item in expected):
        raise RuntimeError("Remote artifact verification failed")
    marker = {"repo_id": repo_id, "remote_prefix": remote, "verified": expected}
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2) + "\n")
    api.upload_file(path_or_fileobj=str(run_dir / "upload_complete.json"), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")
    print(json.dumps({"run_dir": str(run_dir), "remote_prefix": remote, **metrics}, indent=2))


if __name__ == "__main__":
    main()
