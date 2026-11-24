#!/usr/bin/env python3
"""Matched LEVIR default-YOLOv8 OACP/FTAL/DMM/SAMC matrix."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml"
DMM_CFG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_dmm_lite.yaml"
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
RUNS = {
    "default_oacp_ftal_no_mosaic": {
        "config": DEFAULT_CFG, "mosaic": 0.0, "close_mosaic": 0,
        "ftal": True, "oacp": "current", "oacp_params": {},
    },
    "dmm_lite_strong_oacp_no_mosaic": {
        "config": DMM_CFG, "mosaic": 0.0, "close_mosaic": 0,
        "ftal": False, "oacp": "current",
        "oacp_params": {"OACP_P": "0.20", "OACP_PROTECTED_EXPAND": "3.0",
                        "OACP_STRENGTH_MIN": "0.40", "OACP_STRENGTH_MAX": "0.65",
                        "OACP_SCALE_MIN": "0.45", "OACP_SCALE_MAX": "0.70"},
    },
    "default_oacp_samc_mosaic": {
        "config": DEFAULT_CFG, "mosaic": 1.0, "close_mosaic": 10,
        "ftal": False, "oacp": "current", "oacp_params": {},
    },
    "dmm_lite_strong_oacp_samc_mosaic": {
        "config": DMM_CFG, "mosaic": 1.0, "close_mosaic": 10,
        "ftal": False, "oacp": "current",
        "oacp_params": {"OACP_P": "0.20", "OACP_PROTECTED_EXPAND": "3.0",
                        "OACP_STRENGTH_MIN": "0.40", "OACP_STRENGTH_MAX": "0.65",
                        "OACP_SCALE_MIN": "0.45", "OACP_SCALE_MAX": "0.70"},
    },
}
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv", "evaluation_metrics.json", "experiment_manifest.json")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_oacp(spec: dict) -> None:
    os.environ["YOLO_CONTEXT_AUG"] = "oacp"
    os.environ["OACP_VARIANT"] = spec["oacp"]
    os.environ["YOLO_LEGACY_DOUBLE_OACP"] = "0"
    for key in ("OACP_P", "OACP_PROTECTED_EXPAND", "OACP_STRENGTH_MIN", "OACP_STRENGTH_MAX", "OACP_SCALE_MIN", "OACP_SCALE_MAX"):
        os.environ.pop(key, None)
    os.environ.update(spec["oacp_params"])


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    from project_ultralytics.parser import load_project_model, project_parser, project_runtime
    model = load_project_model(run_dir / "weights/best.pt", verbose=False)
    metrics = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
                           device=args.device, workers=args.workers, plots=False, iou=0.5,
                           project=str(run_dir / "evaluation"), name=split, exist_ok=True)
        metrics.update({f"{split}/{k}": float(v) for k, v in result.results_dict.items()})
        metrics[f"{split}/mAP75"] = float(result.box.map75)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def upload_verify(run_dir: Path, run_name: str, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    prefix = f"runs/{run_name}/seed_{args.seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=prefix, repo_id=args.hf_repo_id, repo_type="dataset")
    remote = {x.rfilename for x in api.list_repo_tree(repo_id=args.hf_repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True) if hasattr(x, "rfilename")}
    missing = {f"{prefix}/{item}" for item in REQUIRED} - remote
    if missing:
        raise RuntimeError(f"HF verification failed: {sorted(missing)}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_prefix": prefix, "verified": sorted(missing | ({f'{prefix}/{item}' for item in REQUIRED} - missing))}, indent=2) + "\n")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{prefix}/upload_complete.json", repo_id=args.hf_repo_id, repo_type="dataset")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    from utils.marimo_ops import require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    if (args.seed, args.split_seed, args.epochs, args.patience) != (42, 42, 100, 0):
        raise ValueError("matrix requires seed=42, split-seed=42, epochs=100, patience=0")
    from misc.prepare_levir_ship import prepare
    data_yaml = prepare(args.data_root, args.dataset_root / "levir_default_oacp_split_42", args.split_seed)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    from project_ultralytics.parser import load_project_model, project_parser, project_runtime
    for name, spec in RUNS.items():
        run_dir = args.project / name / f"seed_{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"experiment": name, "model_config": str(spec["config"]), "commit_sha": sha,
                    "dataset": "levir", "data_yaml": str(data_yaml), "seed": args.seed,
                    "split_seed": args.split_seed, "epochs": args.epochs, "patience": args.patience,
                    "imgsz": args.imgsz, "batch_size": args.batch_size, "workers": args.workers,
                    "nms_iou": 0.5, "mosaic": spec["mosaic"], "close_mosaic": spec["close_mosaic"],
                    "mosaic_policy": "scale_adaptive" if spec["mosaic"] else "none",
                    "oacp_variant": spec["oacp"], "oacp_params": spec["oacp_params"],
                    "ftal": spec["ftal"], "hf_repo_id": args.hf_repo_id}
        (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if not all((run_dir / item).is_file() for item in REQUIRED):
            seed_everything(args.seed)
            configure_oacp(spec)
            model = load_project_model(spec["config"], verbose=False)
            model.load("yolov8n.pt", smart_transfer=True)
            kwargs = dict(data=str(data_yaml), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                          batch=args.batch_size, device=args.device, workers=args.workers, amp=args.amp,
                          seed=args.seed, deterministic=True, plots=False, project=str(args.project),
                          name=f"{name}/seed_{args.seed}", exist_ok=True, mosaic=spec["mosaic"],
                          close_mosaic=spec["close_mosaic"], iou=0.5)
            if spec["mosaic"]:
                kwargs.update(mosaic_policy="scale_adaptive", mosaic_scale_quantile=0.05, mosaic_scale_modes=[4, 2])
            if spec["ftal"]:
                kwargs.update(FTAL)
            # Ultralytics rebuilds the trainer model from YAML inside train().
            # Keep the project parser/runtime installed for custom DMM layers.
            from ultralytics.nn import tasks
            from project_ultralytics.registry import install_custom_modules
            # DetectionModel.parse_model is resolved from the tasks module during
            # trainer reconstruction.  The temporary parser bridge handles the
            # project-specific layer rules, while this namespace installation
            # makes custom YAML symbols available to any upstream parse path.
            install_custom_modules(vars(tasks))
            with project_parser(tasks), project_runtime():
                model.train(**kwargs)
        evaluate(run_dir, data_yaml, args)
        upload_verify(run_dir, name, args)
        print(f"COMPLETE {name}", flush=True)


if __name__ == "__main__":
    main()
