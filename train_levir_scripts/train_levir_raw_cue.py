#!/usr/bin/env python3
"""Train Raw Color & Multi-Cue Evidence Fusion variants on LEVIR-Ship seed 42."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc.prepare_levir_ship import prepare


def local_ultralytics() -> None:
    local = ROOT / "models_related/ultralytics"
    if (local / "ultralytics/__init__.py").is_file() and str(local) not in sys.path:
        sys.path.insert(0, str(local))


VARIANTS = {
    "yolov8n_p2_color_slots": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_color_slots.yaml",
    "yolov8n_p2_color_formation": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_color_formation.yaml",
    "yolov8n_p2_multicue": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_multicue.yaml",
}

SEEDS = [42]


def completed(run_dir: Path) -> bool:
    return all((run_dir / relative).is_file() for relative in ("weights/best.pt", "weights/last.pt", "results.csv", "evaluation_metrics.json"))


def upload_run(run_dir: Path, repo_id: str | None, token: str) -> str:
    from huggingface_hub import HfApi

    if not completed(run_dir):
        raise FileNotFoundError(f"Refusing to upload incomplete run: {run_dir}")
    api = HfApi(token=token)
    if repo_id is None:
        repo_id = f"{api.whoami()['name']}/levir-yolo-raw-cue-fusion"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"train/{run_dir.name}",
    )
    print(f"Uploaded {run_dir.name} to https://huggingface.co/datasets/{repo_id}")
    return repo_id


def evaluate_run(model, data_yaml: str, run_dir: Path) -> dict:
    best_pt = run_dir / "weights/best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Missing best.pt weights in {run_dir}")

    # Explicit NMS IoU 0.5 protocol per AGENTS.md
    val_res = model.val(data=data_yaml, split="val", iou=0.5, save=False, verbose=False)
    test_res = model.val(data=data_yaml, split="test", iou=0.5, save=False, verbose=False)

    metrics = {
        "nms_iou": 0.5,
        "val": {
            "mp": float(val_res.results_dict.get("metrics/precision(B)", 0.0)),
            "mr": float(val_res.results_dict.get("metrics/recall(B)", 0.0)),
            "map50": float(val_res.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map75": float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0)),  # mAP50-95 proxy
        },
        "test": {
            "mp": float(test_res.results_dict.get("metrics/precision(B)", 0.0)),
            "mr": float(test_res.results_dict.get("metrics/recall(B)", 0.0)),
            "map50": float(test_res.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map75": float(test_res.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        },
    }

    metrics_file = run_dir / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run(args: argparse.Namespace) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN is not set. Uploads to HuggingFace will be skipped if unauthenticated.")

    local_ultralytics()
    from ultralytics import YOLO

    repo_id = args.hf_repo_id

    for seed in SEEDS:
        ds_dir = args.dataset_root / f"levir_ship_yolo_seed{seed}"
        data_yaml = ds_dir / "levir_ship.yaml"
        if not data_yaml.is_file():
            data_root = args.data_root if args.data_root.exists() else Path("/marimo/LevirShipData")
            data_yaml = prepare(data_root, ds_dir, seed)
        for variant_name, yaml_cfg in VARIANTS.items():
            run_name = f"{variant_name}_seed{seed}"
            run_dir = args.project / run_name
            if completed(run_dir):
                print(f"Reusing completed run: {run_name}")
                continue

            last = run_dir / "weights/last.pt"
            if last.is_file():
                model = YOLO(str(last))
                model.train(resume=True)
            else:
                model = YOLO(str(yaml_cfg))
                model.train(
                    data=str(data_yaml),
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch_size,
                    device=args.device,
                    workers=args.workers,
                    patience=args.patience,
                    seed=seed,
                    deterministic=True,
                    project=str(args.project),
                    name=run_name,
                    exist_ok=True,
                )

            # Load best checkpoint for evaluation
            best_model = YOLO(str(run_dir / "weights/best.pt"))
            print(f"Evaluating {run_name} with explicit nms_iou=0.5...")
            metrics = evaluate_run(best_model, str(data_yaml), run_dir)
            print(f"Run {run_name} Metrics: {json.dumps(metrics, indent=2)}")

            if token:
                repo_id = upload_run(run_dir, repo_id, token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_raw_cue_fusion")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--hf-repo-id")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
