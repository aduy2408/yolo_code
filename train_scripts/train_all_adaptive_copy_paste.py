#!/usr/bin/env python3
"""Run the matched six-way Adaptive Copy-Paste matrix on LEVIR-Ship.

The matrix is intentionally no-Mosaic by default.  This keeps the adaptive
budget conditioned on the destination scene rather than a raw single-image
distribution.  Enabling Mosaic requires an explicit post-Mosaic target-count
file, and the public builder will fail closed if it is omitted.

Run this file through ``python -m utils.marimo_ops launch``.  It never starts
detached processes itself.  Each variant/seed is trained, evaluated on both
``val`` and ``test`` with ``iou=0.5``, uploaded, and remotely verified before
the next matrix cell starts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Keep direct execution compatible with the repository-root imports.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ULTRA = ROOT / "models_related" / "ultralytics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))

from misc.prepare_levir_ship import prepare
from utils.marimo_ops import ensure_hf_repo, require_training_context

CANONICAL_MODEL_YAML = ULTRA / "ultralytics/cfg/models/v8/yolov8.yaml"
VARIANTS = (
    "control",
    "adaptive_scale_conditioned",
    "adaptive_scale_deficit",
    "adaptive_cluster",
    "online_negative",
    "online_negative_scale_matched",
)
ADAPTIVE_VARIANTS = set(VARIANTS[1:])
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "experiment_manifest.json",
    "upload_complete.json",
)
REQUIRED_METRICS = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _split_metrics(result: Any, split: str) -> dict[str, float]:
    values = {key: float(value) for key, value in result.results_dict.items()}
    return {
        **{f"{split}/{key}": value for key, value in values.items()},
        f"{split}/AP50": values["metrics/mAP50(B)"],
        f"{split}/mAP50-95": values["metrics/mAP50-95(B)"],
    }


def _resolve_split_path(data_yaml: Path, value: str) -> Path:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(config.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _target_counts(data_yaml: Path) -> list[int]:
    """Read the fixed train split counts used by no-Mosaic adaptive runs."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    train = _resolve_split_path(data_yaml, str(config["train"]))
    image_paths = (
        [Path(line.strip()) for line in train.read_text().splitlines() if line.strip()]
        if train.is_file()
        else sorted(path for path in train.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    )
    counts: list[int] = []
    for image_path in image_paths:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_path = Path(*parts).with_suffix(".txt")
        else:
            label_path = image_path.with_suffix(".txt")
        count = 0
        if label_path.is_file():
            count = sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
        counts.append(count)
    if not counts:
        raise RuntimeError(f"No training labels found for adaptive target counts: {data_yaml}")
    return counts


def _load_post_mosaic_counts(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("target_counts", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values or not all(isinstance(item, int) for item in values):
        raise ValueError("--adaptive-target-counts-file must contain a non-empty integer list or {'target_counts': [...]} ")
    return [max(0, int(item)) for item in values]


def _settings(variant: str, *, target_counts: list[int], bank_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "mosaic": float(args.mosaic),
        "close_mosaic": int(args.close_mosaic),
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste_enabled": variant != "control",
        "copy_paste_p": 1.0,
        "copy_paste_max_trials": 30,
        "copy_paste_allow_empty_target": True,
        "copy_paste_allow_same_source": True,
        "adaptive_cp_target_counts": target_counts,
        "adaptive_cp_max_objects": 4,
        "adaptive_cp_factor_min": 0.4,
        "adaptive_cp_factor_max": 1.0,
        "adaptive_cp_deficit_gamma": 0.5,
        "adaptive_cp_max_weight_ratio": 3.0,
        "online_negcp_bank_path": str(bank_path),
        "online_negcp_max_objects": 3,
        "online_negcp_conf_threshold": 0.25,
        "online_negcp_max_candidates": 3,
        "negcp": 0.30,
        "negcp_max_gt_ioa": 0.05,
    }
    if variant.startswith("adaptive_"):
        settings["copy_paste_mode"] = variant
    elif variant.startswith("online_negative"):
        settings["copy_paste_mode"] = variant
    return settings


def _complete(run_dir: Path, repo_id: str, prefix: str) -> bool:
    if not all((run_dir / item).is_file() for item in REQUIRED):
        return False
    try:
        marker = json.loads((run_dir / "upload_complete.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("repo_id") == repo_id and marker.get("remote_prefix") == prefix and all(
        item in marker.get("verified", []) for item in REQUIRED
    )


def _upload_and_verify(run_dir: Path, repo_id: str, prefix: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(run_dir), path_in_repo=prefix, repo_id=repo_id, repo_type="dataset")
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    required_remote = [f"{prefix}/{item}" for item in REQUIRED[:-1]]
    missing = [item for item in required_remote if item not in remote_files]
    if missing:
        raise RuntimeError(f"HF upload verification failed for {prefix}: {missing}")
    marker = {
        "repo_id": repo_id,
        "remote_prefix": prefix,
        "verified": list(REQUIRED),
    }
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(run_dir / "upload_complete.json"),
        path_in_repo=f"{prefix}/upload_complete.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    if f"{prefix}/upload_complete.json" not in set(api.list_repo_files(repo_id=repo_id, repo_type="dataset")):
        raise RuntimeError(f"HF upload marker verification failed for {prefix}")


def _train_one(args: argparse.Namespace, data_yaml: Path, repo_id: str, variant: str, seed: int, target_counts: list[int]) -> Path:
    from ultralytics import YOLO

    bank_path = args.project / "online_banks" / f"seed_{seed}"
    run_dir = args.project / variant / f"seed_{seed}"
    prefix = f"adaptive_copy_paste/levir/{variant}/seed_{seed}"
    if _complete(run_dir, repo_id, prefix):
        print(f"SKIP verified {variant}/seed_{seed}", flush=True)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = _settings(variant, target_counts=target_counts, bank_path=bank_path, args=args)
    manifest = {
        "experiment": "adaptive_copy_paste_matrix",
        "dataset": "levir",
        "data_root": str(args.data_root),
        "dataset_yaml": str(data_yaml),
        "model_yaml": str(args.model_yaml),
        "pretrained": args.pretrained,
        "variant": variant,
        "seed": seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "optimizer": args.optimizer,
        "nms_iou": 0.5,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "augmentation": settings,
        "commit_sha": _git_sha(),
        "command": sys.argv,
        "hf_repo_id": repo_id,
        "remote_prefix": prefix,
        "upload_required": True,
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training_files = (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv")
    if not all(path.is_file() for path in training_files):
        model = YOLO(str(args.model_yaml))
        model.load(args.pretrained, smart_transfer=True)
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, patience=args.patience, seed=seed,
            deterministic=True, amp=True, plots=False, project=str(run_dir.parent),
            name=run_dir.name, exist_ok=True, val=True, iou=args.nms_iou,
            optimizer=args.optimizer, **settings,
        )
    if not all(path.is_file() for path in training_files):
        raise RuntimeError(f"Training artifacts incomplete: {run_dir}")
    model = YOLO(str(run_dir / "weights/best.pt"))
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=args.nms_iou,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update(_split_metrics(result, split))
    missing_metrics = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing_metrics:
        raise RuntimeError(f"Evaluation missing split-qualified metrics: {missing_metrics}")
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["metrics"] = metrics
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _upload_and_verify(run_dir, repo_id, prefix)
    print(json.dumps({"variant": variant, "seed": seed, **{key: metrics[key] for key in REQUIRED_METRICS}}, sort_keys=True), flush=True)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--model-yaml", type=Path, default=CANONICAL_MODEL_YAML)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", choices=("auto", "SGD", "Adam", "AdamW"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Single-seed alias required by the Marimo launch contract")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--adaptive-target-counts-file", type=Path, default=None)
    parser.add_argument("--print-effective-config", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs != 100:
        raise ValueError("The full adaptive matrix requires --epochs 100")
    if args.workers != 8:
        raise ValueError("Matched adaptive runs require --workers 8")
    if args.seed is not None and args.seeds is not None:
        raise ValueError("Use either --seed or --seeds, not both")
    args.seeds = [args.seed] if args.seed is not None else (args.seeds or [42])
    if args.nms_iou != 0.5:
        raise ValueError("The adaptive matrix requires --nms-iou 0.5")
    if args.model_yaml.resolve() != CANONICAL_MODEL_YAML.resolve():
        raise ValueError("The adaptive matrix is fixed to the canonical YOLOv8 P3/P4/P5 YAML")
    args.data_root, args.dataset_root, args.project, args.model_yaml = (
        path.resolve() for path in (args.data_root, args.dataset_root, args.project, args.model_yaml)
    )
    if not args.model_yaml.is_file():
        raise FileNotFoundError(args.model_yaml)
    post_mosaic_counts = _load_post_mosaic_counts(args.adaptive_target_counts_file)
    if args.mosaic > 0 and ADAPTIVE_VARIANTS.intersection(args.variants) and post_mosaic_counts is None:
        raise ValueError("Adaptive Mosaic runs require --adaptive-target-counts-file with post-Mosaic counts")
    if args.print_effective_config:
        print(json.dumps({"variants": args.variants, "seeds": args.seeds, "model_yaml": str(args.model_yaml), "optimizer": args.optimizer, "mosaic": args.mosaic, "close_mosaic": args.close_mosaic, "split_seed": args.split_seed}, indent=2, sort_keys=True))
        return
    require_training_context(hf_repo_id=args.hf_repo_id)
    repo_id = ensure_hf_repo(args.hf_repo_id)
    data_yaml = prepare(args.data_root, args.dataset_root / f"levir_adaptive_copy_paste_split_{args.split_seed}", args.split_seed)
    counts = post_mosaic_counts if args.mosaic > 0 else _target_counts(data_yaml)
    args.project.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for variant in args.variants:
            _train_one(args, data_yaml, repo_id, variant, seed, counts)
    (args.project / "matrix_complete.json").write_text(
        json.dumps({"experiment": "adaptive_copy_paste_matrix", "repo_id": repo_id, "variants": args.variants, "seeds": args.seeds, "split_seed": args.split_seed, "commit_sha": _git_sha()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
