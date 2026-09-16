#!/usr/bin/env python3
"""Run the matched LEVIR Copy-Paste matrix on canonical and P2/P3/P4 YOLOv8.

The matrix keeps detector, split, seed, schedule, and no-Mosaic protocol explicit.
Each detector/variant is trained, evaluated on val and test, and uploaded before
moving to the next run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from copy_paste_protocol import VARIANTS, variant_overrides
from misc.prepare_levir_ship import prepare
from utils.marimo_ops import ensure_hf_repo, require_training_context

ULTRA = ROOT / "models_related" / "ultralytics"
CANONICAL_YAML = ULTRA / "ultralytics/cfg/models/v8/yolov8.yaml"
P2P3P4_YAML = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2p3p4_levir_plain.yaml"
DETECTORS = {
    "canonical_p3p4p5": CANONICAL_YAML,
    "p2p3p4": P2P3P4_YAML,
}
VARIANT_ORDER = (
    "cp0",
    "cp1_single1",
    "cp2_single2",
    "cp3_cluster1",
    "crowd_mild",
    "crowd_moderate",
    "scale_matched",
    "negcp_offline",
)
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _complete(run_dir: Path) -> bool:
    return all((run_dir / item).is_file() for item in REQUIRED)


def _upload_verified(run_dir: Path, repo_id: str, prefix: str) -> bool:
    marker = run_dir / "upload_complete.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return False
    return payload.get("repo_id") == repo_id and payload.get("remote_prefix") == prefix


def _upload(run_dir: Path, repo_id: str, prefix: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=prefix,
        repo_id=repo_id,
        repo_type="dataset",
    )
    remote = {
        item.rfilename
        for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=prefix, recursive=True)
        if hasattr(item, "rfilename")
    }
    missing = [f"{prefix}/{item}" for item in REQUIRED if f"{prefix}/{item}" not in remote]
    if missing:
        raise RuntimeError(f"HF upload verification failed: {missing}")
    marker = {"repo_id": repo_id, "remote_prefix": prefix}
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2) + "\n")
    api.upload_file(
        path_or_fileobj=str(run_dir / "upload_complete.json"),
        path_in_repo=f"{prefix}/upload_complete.json",
        repo_id=repo_id,
        repo_type="dataset",
    )


def _evaluate(model_path: Path, data_yaml: Path, run_dir: Path, args: argparse.Namespace) -> dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        values = {key: float(value) for key, value in result.results_dict.items()}
        metrics.update({f"{split}/{key}": value for key, value in values.items()})
        metrics[f"{split}/AP50"] = values["metrics/mAP50(B)"]
        metrics[f"{split}/mAP50-95"] = values["metrics/mAP50-95(B)"]
    return metrics


def _mine_negative_bank(data_yaml: Path, checkpoint: Path, output: Path, args: argparse.Namespace) -> Path:
    from train_copy_paste import _mine_negcp_bank

    helper_args = SimpleNamespace(
        dataset="levir", data_root=args.data_root, dataset_root=args.dataset_root,
        device=args.device,
    )
    return _mine_negcp_bank(helper_args, data_yaml, checkpoint, output)


def _train_one(
    args: argparse.Namespace,
    data_yaml: Path,
    detector_name: str,
    model_yaml: Path,
    variant: str,
    repo_id: str,
    bank_path: Path | None,
) -> Path:
    from ultralytics import YOLO

    run_dir = args.project / detector_name / variant / "seed_42"
    prefix = f"copy_paste_matrix/levir/{detector_name}/{variant}/seed_42"
    if _complete(run_dir) and _upload_verified(run_dir, repo_id, prefix):
        print(f"SKIP verified {detector_name}/{variant}", flush=True)
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    settings = variant_overrides(variant)
    if variant == "negcp_offline":
        if bank_path is None:
            raise RuntimeError("Negative Copy-Paste requires a detector-specific bank")
        settings["negcp_bank_path"] = str(bank_path)
    manifest = {
        "experiment": "levir_copy_paste_detector_matrix",
        "detector": detector_name,
        "model_yaml": str(model_yaml),
        "pretrained": args.pretrained,
        "dataset": "levir",
        "data_yaml": str(data_yaml),
        "variant": variant,
        "seed": 42,
        "split_seed": 42,
        "epochs": args.epochs,
        "patience": 0,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "nms_iou": 0.5,
        "augmentation": settings,
        "commit_sha": _git_sha(),
        "hf_repo_id": repo_id,
        "remote_prefix": prefix,
        "upload_required": True,
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not (run_dir / "weights/best.pt").is_file() or not (run_dir / "weights/last.pt").is_file() or not (run_dir / "results.csv").is_file():
        model = YOLO(str(model_yaml))
        model.load(args.pretrained, smart_transfer=True)
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, patience=0, seed=42,
            deterministic=True, amp=True, plots=False, project=str(run_dir.parent),
            name=run_dir.name, exist_ok=True, val=True, iou=0.5, **settings,
        )
    if not all((run_dir / item).is_file() for item in REQUIRED[:3]):
        raise RuntimeError(f"Training artifacts incomplete: {run_dir}")

    metrics = _evaluate(run_dir / "weights/best.pt", data_yaml, run_dir, args)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest["metrics"] = metrics
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _upload(run_dir, repo_id, prefix)
    print(f"COMPLETE {detector_name}/{variant}", flush=True)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detectors", nargs="+", choices=list(DETECTORS), default=list(DETECTORS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANT_ORDER))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs != 100:
        raise ValueError("Full matrix requires epochs=100")
    if args.workers != 8:
        raise ValueError("Matched matrix requires workers=8")
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        require_training_context(hf_repo_id=args.hf_repo_id)
    repo_id = ensure_hf_repo(args.hf_repo_id)
    args.data_root, args.dataset_root, args.project = (
        path.resolve() for path in (args.data_root, args.dataset_root, args.project)
    )
    data_yaml = prepare(args.data_root, args.dataset_root / "levir_copy_paste_matrix_split_42", 42)
    for detector_name in args.detectors:
        model_yaml = DETECTORS[detector_name]
        if not model_yaml.is_file():
            raise FileNotFoundError(model_yaml)
        detector_bank = None
        if "negcp_offline" in args.variants:
            detector_bank = args.project / detector_name / "negcp_banks" / "seed_42.json"
            cp0_checkpoint = args.project / detector_name / "cp0" / "seed_42" / "weights/best.pt"
            if not detector_bank.is_file():
                if not cp0_checkpoint.is_file():
                    _train_one(args, data_yaml, detector_name, model_yaml, "cp0", repo_id, None)
                detector_bank = _mine_negative_bank(data_yaml, cp0_checkpoint, detector_bank, args)
        for variant in args.variants:
            _train_one(args, data_yaml, detector_name, model_yaml, variant, repo_id, detector_bank)


if __name__ == "__main__":
    main()
