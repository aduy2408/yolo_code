#!/usr/bin/env python3
"""Run the matched LEVIR-Ship/TinyPerson Mosaic policy matrix.

The runner is intentionally sequential: each run is evaluated, uploaded, and
verified before the next run starts. It is designed for ``utils.marimo_ops
launch`` and refuses direct upload-required invocation.
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

from utils.marimo_ops import require_training_context

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = ROOT / "models_related/ultralytics"
if str(ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS))
POLICIES = {
    "M0_standard": "standard",
    "M1_visibility": "visibility",
    "M2_occupancy": "occupancy_match",
    "M3_context": "context_contrast",
}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "args.yaml",
    "experiment_manifest.json",
)
TRAIN_REQUIRED = tuple(path for path in REQUIRED if path != "evaluation_metrics.json")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.dataset == "levir":
        from misc.prepare_levir_ship import prepare

        output = args.dataset_root / "levir_ship_yolo_seed42"
        data_yaml = prepare(args.data_root, output, args.split_seed)
        image_root = output / "images/train"
        config = ROOT / "models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml"
        return data_yaml, image_root, config

    from train_all_tinyperson import prepare_seed_dataset, prepare_test_set

    test_root = prepare_test_set(args.data_root, args.dataset_root)
    split_root = prepare_seed_dataset(args.data_root, args.dataset_root, test_root, args.split_seed)
    data_yaml = split_root / "tinyperson.yaml"
    image_root = split_root / "images/train"
    config = ROOT / "models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml"
    return data_yaml, image_root, config


def build_context_cache(image_root: Path, output: Path) -> Path:
    if not output.exists():
        subprocess.run(
            [sys.executable, "tools/build_mosaic_context_cache.py", "--image-dir", str(image_root), "--output", str(output)],
            cwd=ROOT,
            check=True,
        )
    return output


def ftal_kwargs() -> dict[str, object]:
    return {
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


def train(run_dir: Path, data_yaml: Path, config: Path, cache: Path, args: argparse.Namespace, policy: str) -> None:
    from ultralytics import YOLO

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "policy_name": policy,
        "mosaic_policy": POLICIES[policy],
        "seed": args.seed,
        "split_seed": args.split_seed,
        "data_yaml": str(data_yaml),
        "model_config": str(config),
        "model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "nms_iou": 0.5,
        "mosaic_postprocess_enabled": True,
        "mosaic_postprocess_mode": "resize",
        "post_mosaic_geometry": {
            "mode": "full_canvas_resize",
            "random_perspective": False,
            "translate": 0.0,
            "scale": 1.0,
            "shear": 0.0,
            "perspective": 0.0,
        },
        "copy_paste": 0.0,
        "mixup": 0.0,
        "oacp": False,
        "context_cache": str(cache),
        "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "hf_repo_id": args.hf_repo_id,
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if all((run_dir / path).is_file() for path in TRAIN_REQUIRED):
        return

    seed_everything(args.seed)
    model = YOLO(str(config), task="detect")
    model.load(args.pretrained, smart_transfer=True)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        seed=args.seed,
        deterministic=True,
        plots=False,
        project=str(args.project),
        name=f"{policy}/seed_{args.seed}",
        exist_ok=True,
        mosaic=1.0,
        close_mosaic=args.close_mosaic,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic_policy=POLICIES[policy],
        mosaic_policy_candidates=16,
        mosaic_policy_topk=4,
        mosaic_visibility_thresh=0.70,
        mosaic_visibility_lambda=1.0,
        mosaic_occupancy_mode="dataset",
        mosaic_occupancy_tolerance=1.0,
        mosaic_context_candidates=32,
        mosaic_context_occupancy_tolerance=1.0,
        mosaic_context_cache=str(cache),
        mosaic_postprocess_enabled=True,
        mosaic_postprocess_mode="resize",
        mixup=0.0,
        copy_paste=0.0,
        **ftal_kwargs(),
    )


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/mAP75"] = float(result.box.map75)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def upload_and_verify(run_dir: Path, policy: str, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    required = list(REQUIRED)
    if POLICIES[policy] != "standard":
        required.append("mosaic_policy_diagnostics.json")
    missing = [path for path in required if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"{policy}: incomplete local artifacts: {missing}")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    remote = f"runs/{args.dataset}/{policy}/seed_{args.seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=args.hf_repo_id, repo_type="dataset")
    remote_files = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
    expected = {f"{remote}/{path}" for path in required}
    missing_remote = sorted(expected - remote_files)
    if missing_remote:
        raise RuntimeError(f"{policy}: remote verification failed: {missing_remote}")
    marker = {"repo_id": args.hf_repo_id, "remote_prefix": remote, "verified": sorted(expected)}
    marker_path = run_dir / "upload_complete.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    api.upload_file(path_or_fileobj=str(marker_path), path_in_repo=f"{remote}/upload_complete.json", repo_id=args.hf_repo_id, repo_type="dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("levir", "tinyperson"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES), default=list(POLICIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--confirm-settings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    require_training_context(hf_repo_id=args.hf_repo_id)
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train without --confirm-settings")
    if args.seed != 42 or args.split_seed != 42 or args.epochs != 100 or args.patience != 0:
        raise ValueError("This matrix requires seed=42, split-seed=42, epochs=100, patience=0")
    args.imgsz = args.imgsz or (512 if args.dataset == "levir" else 640)
    args.data_root, args.dataset_root, args.project = args.data_root.resolve(), args.dataset_root.resolve(), args.project.resolve()
    data_yaml, image_root, config = prepare_dataset(args)
    cache = build_context_cache(image_root, args.dataset_root / f"{args.dataset}_mosaic_context_seed{args.split_seed}.npz")
    for policy in args.policies:
        run_dir = args.project / policy / f"seed_{args.seed}"
        train(run_dir, data_yaml, config, cache, args, policy)
        evaluate(run_dir, data_yaml, args)
        upload_and_verify(run_dir, policy, args)
        print(f"COMPLETE {args.dataset}/{policy}/seed_{args.seed}", flush=True)


if __name__ == "__main__":
    main()
