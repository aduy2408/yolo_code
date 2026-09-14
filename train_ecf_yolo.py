#!/usr/bin/env python3
"""Train the upstream ECF-YOLO implementation on Varroa, LEVIR-Ship, or TinyPerson.

The defaults intentionally mirror ``ECF-YOLO/train.py``. The only schedule changes
are ``epochs=100`` and ``patience=0`` so a requested run reaches all 100 epochs.
Full training is Marimo-workflow gated because completed artifacts are uploaded and
verified before the run is considered complete.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from utils.marimo_ops import MarimoOpsError, require_training_context

ROOT = Path(__file__).resolve().parent
ECF_ROOT = ROOT / "ECF-YOLO"
ECF_MODEL = ECF_ROOT / "ultralytics/cfg/models/11/yolo11-EPAN.yaml"
DATASET_DEFAULTS = {
    "varroa": ROOT / "datasets/varroa_yolo/varroa.yaml",
    "levirship": ROOT / "datasets/levir_ship_yolo_scene_seed42/levir_ship.yaml",
    "tinyperson": ROOT / "datasets/tinyperson_seed42/tinyperson.yaml",
}
DATA_ROOT_DEFAULTS = {
    "varroa": ROOT.parent,
    "levirship": ROOT / "LevirShipData",
    "tinyperson": ROOT.parent / "TinyPerson/tiny_set",
}
REQUIRED_ARTIFACTS = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)
UPLOAD_MARKER = "upload_complete.json"


def git_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def ecf_sha() -> str:
    return git_sha(ECF_ROOT)


def resolve_data_yaml(args: argparse.Namespace) -> Path:
    data_yaml = args.data_yaml or DATASET_DEFAULTS[args.dataset]
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Missing {args.dataset} data YAML: {data_yaml}. "
            f"Use --prepare with --data-root/--dataset-root or pass --data-yaml explicitly."
        )
    return data_yaml


def prepare_dataset(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    if args.dataset == "varroa":
        from misc.prepare_dataset import prepare_dataset as prepare_varroa

        return prepare_varroa(args.data_root, dataset_root / "varroa_yolo", seed=args.split_seed)
    if args.dataset == "levirship":
        from misc.prepare_levir_ship import prepare

        return prepare(
            args.data_root,
            dataset_root / f"levir_ship_yolo_scene_seed{args.split_seed}",
            seed=args.split_seed,
        )
    if args.dataset == "tinyperson":
        import train_all_tinyperson as tinyperson

        test_dir = tinyperson.prepare_test_set(args.data_root, dataset_root)
        seed_dir = tinyperson.prepare_seed_dataset(
            args.data_root, dataset_root, test_dir, args.split_seed
        )
        return seed_dir / "tinyperson.yaml"
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def effective_config(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    return {
        "experiment_id": f"ecf_yolo_{args.dataset}_seed_{args.seed}",
        "control_config_or_baseline_config": "ECF-YOLO/train.py at upstream commit",
        "variant_config_or_explicit_change": {
            "model": str(args.model),
            "epochs": args.epochs,
            "patience": args.patience,
            "dataset": args.dataset,
        },
        "source_commit": git_sha(ROOT),
        "ecf_source_commit": ecf_sha(),
        "runner": "train_ecf_yolo.py",
        "python_executable": sys.executable,
        "dataset_root": str(data_yaml),
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "model_backbone_pretrained_source": {
            "model_yaml": str(args.model),
            "pretrained": False,
            "resolved_components": "YOLO11 EPAN neck with Fusion(fusion_mode=bifpn), C3k2 backbone, C2PSA, Detect(P3/P4/P5)",
        },
        "image_size": args.imgsz,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "amp": args.amp,
        "nms_iou": 0.5,
        "upstream_defaults": {
            "cache": False,
            "close_mosaic": 0,
            "workers": 4,
            "optimizer": "SGD",
            "lr0": 0.01,
            "momentum": 0.949,
        },
        "hf_repo": args.hf_repo_id,
        "remote_prefix": f"ecf_yolo/{args.dataset}/seed_{args.seed}",
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "upload_required": True,
    }


def remote_paths_verified(run_dir: Path, args: argparse.Namespace, prefix: str) -> bool:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    remote_files = set(api.list_repo_files(repo_id=args.hf_repo_id, repo_type="dataset"))
    expected = [*REQUIRED_ARTIFACTS, UPLOAD_MARKER]
    return all(f"{prefix}/{name}" in remote_files for name in expected)


def upload_and_verify(run_dir: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    prefix = f"ecf_yolo/{args.dataset}/seed_{args.seed}"
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=prefix,
        repo_id=args.hf_repo_id,
        repo_type="dataset",
    )
    remote_files = set(api.list_repo_files(repo_id=args.hf_repo_id, repo_type="dataset"))
    missing = [f"{prefix}/{name}" for name in REQUIRED_ARTIFACTS if f"{prefix}/{name}" not in remote_files]
    if missing:
        raise RuntimeError("Hugging Face upload verification failed:\n" + "\n".join(missing))
    marker = {
        "repo_id": args.hf_repo_id,
        "remote_prefix": prefix,
        "verified_files": [f"{prefix}/{name}" for name in REQUIRED_ARTIFACTS],
    }
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(run_dir / "upload_complete.json"),
        path_in_repo=f"{prefix}/upload_complete.json",
        repo_id=args.hf_repo_id,
        repo_type="dataset",
    )
    if f"{prefix}/upload_complete.json" not in set(api.list_repo_files(repo_id=args.hf_repo_id, repo_type="dataset")):
        raise RuntimeError("Hugging Face upload marker verification failed")


def run(args: argparse.Namespace) -> Path:
    if args.epochs != 100:
        raise ValueError("ECF-YOLO runs are fixed to epochs=100")
    if args.patience != 0:
        raise ValueError("ECF-YOLO runs use patience=0 to ensure all 100 epochs run")
    if not ECF_ROOT.is_dir() or not (ECF_ROOT / ".git").exists():
        raise FileNotFoundError(f"Expected ECF-YOLO checkout at {ECF_ROOT}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {args.model}")
    if not args.hf_repo_id:
        raise ValueError("--hf-repo-id is required for upload-required training")
    require_training_context(hf_repo_id=args.hf_repo_id)
    data_yaml = resolve_data_yaml(args)
    run_dir = args.project.resolve() / args.dataset / f"seed_{args.seed}"
    prefix = f"ecf_yolo/{args.dataset}/seed_{args.seed}"
    if all((run_dir / name).is_file() for name in (*REQUIRED_ARTIFACTS, UPLOAD_MARKER)):
        if remote_paths_verified(run_dir, args, prefix):
            print(f"Verified completed run already exists: {run_dir}")
            return run_dir
        raise RuntimeError(
            f"Local completion marker exists but remote artifacts are incomplete: {run_dir}. "
            "Stop and inspect the Hugging Face repository before retraining."
        )
    sys.path.insert(0, str(ECF_ROOT))
    from ultralytics import YOLO

    run_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    model.train(
        data=str(data_yaml),
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch_size,
        close_mosaic=0,
        workers=4,
        device=args.device,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.949,
        patience=0,
        seed=args.seed,
        deterministic=True,
        amp=args.amp,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        val=True,
        iou=0.5,
    )
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training returned without {best}")
    metrics = YOLO(str(best)).val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=4,
        plots=False,
        iou=0.5,
    )
    metrics_dict = {key: float(value) for key, value in metrics.results_dict.items()}
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics_dict, indent=2, sort_keys=True) + "\n")
    manifest = effective_config(args, data_yaml)
    manifest["metrics"] = metrics_dict
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError("Local artifact contract failed:\n" + "\n".join(missing))
    upload_and_verify(run_dir, args)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--prepare", action="store_true", help="Prepare the fixed split before printing or training")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/ecf_yolo")
    parser.add_argument("--model", type=Path, default=ECF_MODEL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--print-effective-config", action="store_true")
    args = parser.parse_args(argv)
    args.data_root = (args.data_root or DATA_ROOT_DEFAULTS[args.dataset]).resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.model = args.model.resolve()
    args.project = args.project.resolve()
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare:
        args.data_yaml = prepare_dataset(args)
    data_yaml = resolve_data_yaml(args)
    config = effective_config(args, data_yaml)
    if args.print_effective_config:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    run(args)


if __name__ == "__main__":
    try:
        main()
    except MarimoOpsError as exc:
        raise SystemExit(str(exc)) from exc
