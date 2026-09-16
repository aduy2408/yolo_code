#!/usr/bin/env python3
"""Run matched YOLOv8n P2/P3/P4 hard-negative Mosaic experiments.

This is an upload-required Marimo runner. It trains one dataset at a time,
evaluates val and test splits, then verifies the task-specific HF upload before
continuing. The detector YAMLs are explicit P2/P3/P4 plain-neck configs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ULTRA = ROOT / "models_related/ultralytics"
CONFIGS = {
    "levir": ROOT / "models_related/models_config/yolov8/levir/baseline_controls/yolov8n_p2p3p4_levir_plain.yaml",
    "tinyperson": ROOT / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3p4_plain.yaml",
}
REQUIRED = (
    "weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
    "config.yaml", "experiment_manifest.json", "evaluation_metrics.json",
    "hard_negative_bank.json",
)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(dataset: str, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if dataset == "levir":
        from misc.prepare_levir_ship import prepare
        output = args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}"
        data_yaml = prepare(args.levir_data_root, output, args.split_seed)
        return data_yaml, output / "images/train", output
    import train_all_tinyperson as workflow
    test_root = workflow.prepare_test_set(args.tinyperson_data_root, args.dataset_root)
    split_root = workflow.prepare_seed_dataset(
        args.tinyperson_data_root, args.dataset_root, test_root, args.split_seed
    )
    return split_root / "tinyperson.yaml", split_root / "images/train", test_root


def mine_bank(images: Path, labels: Path, output: Path, weights: str) -> None:
    if output.is_file():
        bank = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(bank, list) and bank:
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ULTRA}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    subprocess.run([
        sys.executable, "tools/mine_mosaic_hard_negatives.py",
        "--weights", weights, "--images", str(images), "--labels", str(labels),
        "--output", str(output), "--conf-min", "0.20", "--gt-iou-max", "0.10",
        "--context-expand", "4.0", "--min-crop", "96",
    ], cwd=ROOT, env=env, check=True)
    bank = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(bank, list) or not bank:
        raise RuntimeError(f"hard-negative miner produced no crops: {output}")


def manifest(dataset: str, data_yaml: Path, bank: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = CONFIGS[dataset]
    return {
        "experiment": f"{dataset}_yolov8n_p2p3p4_hard_negative_mosaic",
        "dataset": dataset,
        "detector": "YOLOv8n explicit P2/P3/P4 plain neck",
        "model_config": str(config),
        "model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "augmentation": {
            "mosaic": 1.0, "close_mosaic": 10, "mosaic_policy": "hard_negative",
            "hard_negative_tile": True, "hardneg_mosaic_prob": 0.30,
            "mixup": 0.0, "copy_paste": 0.0,
        },
        "hard_negative_bank": str(bank),
        "data_yaml": str(data_yaml), "seed": args.seed, "split_seed": args.split_seed,
        "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz[dataset],
        "batch_size": args.batch_size, "workers": args.workers, "device": args.device,
        "nms_iou": 0.5, "hf_repo_id": args.hf_repo_id[dataset],
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "test_protocol": "LEVIR standard held-out test split" if dataset == "levir" else "TinyPerson standard test plus merged corner-window evaluator",
    }


def train_one(dataset: str, data_yaml: Path, train_images: Path, dataset_root: Path, args: argparse.Namespace) -> Path:
    config = CONFIGS[dataset]
    run_dir = args.project / f"{dataset}_yolov8n_p2p3p4_hard_negative_mosaic" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = train_images.parent.parent / "labels/train"
    bank = run_dir / "hard_negative_bank.json"
    mine_bank(train_images, labels, bank, args.pretrained)
    (run_dir / "config.yaml").write_bytes(config.read_bytes())
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest(dataset, data_yaml, bank, args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if all((run_dir / p).is_file() for p in REQUIRED if p not in {"evaluation_metrics.json", "hard_negative_bank.json"}):
        return run_dir
    if str(ULTRA) not in sys.path:
        sys.path.insert(0, str(ULTRA))
    from ultralytics import YOLO
    seed_everything(args.seed)
    model = YOLO(str(config), task="detect")
    model.load(args.pretrained, smart_transfer=True)
    model.train(
        data=str(data_yaml), epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz[dataset], batch=args.batch_size, device=args.device,
        workers=args.workers, amp=args.amp, seed=args.seed, deterministic=True,
        plots=False, project=str(run_dir.parent), name=f"seed_{args.seed}", exist_ok=True,
        mosaic=1.0, close_mosaic=10, mosaic_policy="hard_negative",
        hard_negative_tile=True, hardneg_mosaic_prob=0.30,
        hard_negative_bank=str(bank), mosaic_postprocess_enabled=False,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        mixup=0.0, copy_paste=0.0,
    )
    if not all((run_dir / p).is_file() for p in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        raise RuntimeError(f"incomplete training artifacts: {run_dir}")
    return run_dir


def evaluate(dataset: str, run_dir: Path, data_yaml: Path, test_root: Path, args: argparse.Namespace) -> None:
    if str(ULTRA) not in sys.path:
        sys.path.insert(0, str(ULTRA))
    from ultralytics import YOLO
    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, Any] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz[dataset], batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        values = {k: float(v) for k, v in result.results_dict.items()}
        metrics.update({f"{split}/{k}": v for k, v in values.items()})
        metrics[f"{split}/AP50"] = values["metrics/mAP50(B)"]
        metrics[f"{split}/mAP50-95"] = values["metrics/mAP50-95(B)"]
        metrics[f"{split}/mAP75"] = float(result.box.map75)
    if dataset == "tinyperson":
        import train_all_tinyperson as workflow
        custom = argparse.Namespace(imgsz=args.imgsz[dataset], batch_size=args.batch_size, device=args.device, workers=args.workers)
        merged = workflow.evaluate_merged_test(run_dir, test_root, args.tinyperson_data_root, custom)
        metrics.update({k: float(v) for k, v in merged.items() if isinstance(v, (int, float))})
        metrics["test_protocol"] = "TinyPerson standard test plus merged corner-window evaluator"
        metrics["test_protocol_source"] = str(args.tinyperson_data_root / workflow.TEST_MERGED_JSON)
    else:
        metrics["test_protocol"] = "LEVIR-Ship standard held-out test split"
        metrics["test_protocol_source"] = str(data_yaml)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def upload_verify(dataset: str, run_dir: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi
    missing = [p for p in REQUIRED if not (run_dir / p).is_file()]
    if missing:
        raise RuntimeError(f"refusing incomplete upload: {missing}")
    metrics = json.loads((run_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    for key in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95"):
        if key not in metrics:
            raise RuntimeError(f"missing split-qualified metric: {key}")
    repo = args.hf_repo_id[dataset]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    remote = f"runs/{dataset}/seed_{args.seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo, repo_type="dataset")
    files = set(api.list_repo_files(repo, repo_type="dataset"))
    expected = {f"{remote}/{p}" for p in REQUIRED}
    missing_remote = sorted(expected - files)
    if missing_remote:
        raise RuntimeError(f"remote verification failed: {missing_remote}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": repo, "remote_prefix": remote, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo, repo_type="dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levir-data-root", type=Path, required=True)
    p.add_argument("--tinyperson-data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hf-repo-levir", required=True)
    p.add_argument("--hf-repo-tinyperson", required=True)
    p.add_argument("--pretrained", default="yolov8n.pt")
    p.add_argument("--datasets", nargs="+", choices=("levir", "tinyperson"), default=("levir", "tinyperson"))
    p.add_argument("--seed", type=int, default=42); p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda"); p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--imgsz-levir", type=int, default=512); p.add_argument("--imgsz-tinyperson", type=int, default=640)
    p.add_argument("--confirm-settings", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from utils.marimo_ops import require_training_context
    args = parse_args(argv)
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train without --confirm-settings")
    if (args.seed, args.split_seed, args.epochs, args.patience, args.workers) != (42, 42, 100, 0, 8):
        raise ValueError("requires seed=42, split-seed=42, epochs=100, patience=0, workers=8")
    require_training_context(hf_repo_id=args.hf_repo_levir)
    args.levir_data_root = args.levir_data_root.resolve(); args.tinyperson_data_root = args.tinyperson_data_root.resolve()
    args.dataset_root = args.dataset_root.resolve(); args.project = args.project.resolve()
    args.hf_repo_id = {"levir": args.hf_repo_levir, "tinyperson": args.hf_repo_tinyperson}
    args.imgsz = {"levir": args.imgsz_levir, "tinyperson": args.imgsz_tinyperson}
    for dataset in args.datasets:
        data_yaml, train_images, test_root = prepare_dataset(dataset, args)
        run_dir = train_one(dataset, data_yaml, train_images, test_root, args)
        evaluate(dataset, run_dir, data_yaml, test_root, args)
        upload_verify(dataset, run_dir, args)
        print(f"COMPLETE {dataset}/seed_{args.seed}", flush=True)


if __name__ == "__main__":
    main()
