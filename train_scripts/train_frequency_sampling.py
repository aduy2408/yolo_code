#!/usr/bin/env python3
"""Train and upload one fixed-Haar frequency-sampling run.

This runner is intentionally one dataset/seed at a time so every completed run
is evaluated and uploaded before the next launch.  LEVIR-Ship uses no Mosaic;
TinyPerson enables the canonical Mosaic policy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from utils.marimo_ops import ensure_hf_repo, require_training_context

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "levir_p3p5": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_freq_pair_v1.yaml",
    "levir_p2_only": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_freq_pair_v1.yaml",
    "tinyperson_p3p5": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p3p5_freq_pair_v1.yaml",
    "tinyperson_p2p4": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p4_freq_pair_v1.yaml",
    "levir_p3p5_v2": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_freq_pair_v2.yaml",
    "levir_p2_only_v2": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_freq_pair_v2.yaml",
    "tinyperson_p3p5_v2": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p3p5_freq_pair_v2.yaml",
    "tinyperson_p2p4_v2": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p4_freq_pair_v2.yaml",
    "levir_p3p5_ibs": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_ibs_v1.yaml",
    "levir_p2_only_ibs": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_ibs_v1.yaml",
    "levir_p2_only_frfdet_down": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_frfdet_down.yaml",
    "tinyperson_p3p5_ibs": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p3p5_ibs_v1.yaml",
    "tinyperson_p2p4_ibs": ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p4_ibs_v1.yaml",
}
CONTEXT_AUGS = ("none", "oacp")
VARIANT_DATASET = {
    "levir_p3p5": "levirship",
    "levir_p2_only": "levirship",
    "tinyperson_p3p5": "tinyperson",
    "tinyperson_p2p4": "tinyperson",
    "levir_p3p5_v2": "levirship",
    "levir_p2_only_v2": "levirship",
    "tinyperson_p3p5_v2": "tinyperson",
    "tinyperson_p2p4_v2": "tinyperson",
    "levir_p3p5_ibs": "levirship",
    "levir_p2_only_ibs": "levirship",
    "levir_p2_only_frfdet_down": "levirship",
    "tinyperson_p3p5_ibs": "tinyperson",
    "tinyperson_p2p4_ibs": "tinyperson",
}
DATASET_DEFAULTS = {
    "levirship": {"imgsz": 512, "batch": 8, "mosaic": 0.0, "data_root": ROOT / "LevirShipData"},
    "tinyperson": {"imgsz": 640, "batch": 8, "mosaic": 1.0, "data_root": ROOT / "TinyPerson"},
}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "experiment_manifest.json",
    "config.yaml",
)


def prepare_data(dataset: str, data_root: Path, dataset_root: Path, split_seed: int) -> Path:
    if dataset == "levirship":
        from misc.prepare_levir_ship import prepare

        return prepare(data_root, dataset_root / f"levir_ship_frequency_split_{split_seed}", split_seed)
    from train_scripts.train_all_tinyperson import prepare_seed_dataset, prepare_test_set

    test_root = prepare_test_set(data_root, dataset_root)
    split_root = prepare_seed_dataset(data_root, dataset_root, test_root, split_seed)
    return split_root / "tinyperson.yaml"


def complete(run_dir: Path) -> bool:
    return all((run_dir / path).is_file() for path in REQUIRED)


def train(args: argparse.Namespace, data_yaml: Path, run_dir: Path) -> None:
    from project_ultralytics import load_project_model
    from project_ultralytics.parser import project_parser, project_runtime
    from ultralytics.nn import tasks

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    model = load_project_model(str(CONFIGS[args.variant]), task="detect", verbose=False)
    if args.pretrained:
        model.load(args.pretrained)
    with project_parser(tasks), project_runtime():
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            amp=True,
            optimizer="AdamW",
            plots=False,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            mosaic=args.mosaic,
            close_mosaic=10 if args.mosaic else 0,
            mixup=0.0,
        )


def evaluate(args: argparse.Namespace, data_yaml: Path, run_dir: Path) -> dict[str, object]:
    from project_ultralytics import load_project_model
    from project_ultralytics.parser import project_parser, project_runtime
    from ultralytics.nn import tasks

    model = load_project_model(str(run_dir / "weights/best.pt"), task="detect", verbose=False)
    metrics: dict[str, object] = {"checkpoint": "weights/best.pt", "nms_iou": 0.5}
    with project_parser(tasks), project_runtime():
        for split in ("val", "test"):
            result = model.val(
                data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device=args.device, plots=False, iou=0.5,
                project=str(run_dir / "evaluation"), name=split, exist_ok=True,
            )
            metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
            metrics[f"{split}/mAP50-75"] = float(result.box.map)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def upload(args: argparse.Namespace, run_dir: Path) -> None:
    from huggingface_hub import HfApi

    repo_id = ensure_hf_repo(args.hf_repo_id)
    api = HfApi(token=os.environ["HF_TOKEN"])
    remote = f"{run_dir.parent.name}/seed_{args.seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    expected = {f"{remote}/{path}" for path in REQUIRED}
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    missing = sorted(expected - remote_files)
    if missing:
        raise RuntimeError(f"Remote upload verification failed: {missing}")
    marker = {"repo_id": repo_id, "remote_prefix": remote, "verified": sorted(expected)}
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    api.upload_file(path_or_fileobj=str(run_dir / "upload_complete.json"), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("levirship", "tinyperson"), required=True)
    parser.add_argument("--variant", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/frequency_sampling_v1")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--mosaic", type=float)
    parser.add_argument("--context-aug", choices=CONTEXT_AUGS, default="none")
    parser.add_argument("--oacp-variant", default="current")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if VARIANT_DATASET[args.variant] != args.dataset:
        raise ValueError(f"variant {args.variant!r} belongs to {VARIANT_DATASET[args.variant]!r}, not {args.dataset!r}")
    defaults = DATASET_DEFAULTS[args.dataset]
    args.data_root = (args.data_root or defaults["data_root"]).resolve()
    args.imgsz = args.imgsz or defaults["imgsz"]
    args.batch = args.batch or defaults["batch"]
    args.mosaic = defaults["mosaic"] if args.mosaic is None else args.mosaic
    if args.dataset == "levirship" and args.mosaic != 0.0:
        raise ValueError("LEVIR-Ship frequency V1 requires mosaic=0.0")
    if args.dataset == "tinyperson" and args.mosaic != 1.0:
        raise ValueError("TinyPerson frequency V1 requires mosaic=1.0")
    if args.context_aug == "oacp":
        os.environ["YOLO_CONTEXT_AUG"] = "oacp"
        os.environ["OACP_VARIANT"] = args.oacp_variant
    else:
        os.environ.pop("YOLO_CONTEXT_AUG", None)
        os.environ.pop("OACP_VARIANT", None)
    require_training_context(hf_repo_id=args.hf_repo_id)
    data_yaml = prepare_data(args.dataset, args.data_root, args.dataset_root.resolve(), args.split_seed)
    run_variant = args.variant if args.context_aug == "none" else f"{args.variant}_{args.context_aug}"
    run_dir = args.project.resolve() / run_variant / f"seed_{args.seed}"
    manifest = {
        "dataset": args.dataset, "variant": args.variant, "run_variant": run_variant,
        "data_yaml": str(data_yaml), "model_yaml": str(CONFIGS[args.variant]),
        "seed": args.seed, "split_seed": args.split_seed, "epochs": args.epochs, "patience": args.patience,
        "workers": args.workers, "imgsz": args.imgsz, "batch": args.batch, "mosaic": args.mosaic,
        "close_mosaic": 10 if args.mosaic else 0, "nms_iou": 0.5, "upload_required": True,
        "context_aug": args.context_aug, "oacp_variant": args.oacp_variant if args.context_aug == "oacp" else None,
        "hf_repo_id": args.hf_repo_id,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.copy2(CONFIGS[args.variant], run_dir / "config.yaml")
    if args.prepare_only:
        print(data_yaml)
        return
    if not complete(run_dir):
        train(args, data_yaml, run_dir)
    if not (run_dir / "evaluation_metrics.json").is_file():
        evaluate(args, data_yaml, run_dir)
    if not (run_dir / "upload_complete.json").is_file():
        upload(args, run_dir)
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
