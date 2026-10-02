#!/usr/bin/env python3
"""Train the selected OACP, Mosaic, and Copy-Paste augmentation matrix.

Protocol
--------
* OACP: ``load_adaptive``, ``spacing_adaptive``, ``mass_adaptive``.
* Mosaic: ``standard``, ``M2_cluster_preserving``, ``M3_post_scale_constrained``,
  ``M4_adaptive_geometry``, ``M5_hard_negative``.
* Copy-Paste: ``cp1_single1``, ``cp3_cluster1``, ``negative_canvas_r1``,
  ``negative_canvas_r4``.

Dataset-specific Mosaic policy follows the requested baseline convention:
LEVIR-Ship runs use no Mosaic, TinyPerson runs can be selected with and without
Mosaic for OACP/Copy-Paste, and Varroa runs use Mosaic. Mosaic-policy jobs are
only emitted for datasets where Mosaic is enabled by the selected protocol.

Real training must be launched through ``python -m utils.marimo_ops launch``.
This file does not create detached processes itself.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ULTRALYTICS = ROOT / "models_related" / "ultralytics"
CANONICAL_CONFIG = ULTRALYTICS / "ultralytics/cfg/models/v8/yolov8.yaml"

from copy_paste_protocol import variant_overrides
from train_scripts.train_augmentation_baselines_seed42 import COMMON_AUGMENTATION
from train_scripts.train_copy_paste import _split_metrics

OACP_VARIANTS = ("load_adaptive", "spacing_adaptive", "mass_adaptive")
MOSAIC_VARIANTS = (
    "standard",
    "M2_cluster_preserving",
    "M3_post_scale_constrained",
    "M4_adaptive_geometry",
    "M5_hard_negative",
)
COPY_PASTE_VARIANTS = (
    "cp1_single1",
    "cp3_cluster1",
    "negative_canvas_r1",
    "negative_canvas_r4",
)
METHODS = ("oacp", "mosaic", "copy_paste")
DATASETS = ("levir", "tinyperson", "varroa")
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "experiment_manifest.json",
    "upload_complete.json",
)
REQUIRED_METRICS = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")

MOSAIC_POLICY = {
    "standard": {"mosaic_policy": "standard"},
    "M2_cluster_preserving": {
        "mosaic_policy": "cluster_preserve",
        "cluster_preserve": True,
    },
    "M3_post_scale_constrained": {
        "mosaic_policy": "post_scale",
        "post_scale_constraint": True,
    },
    "M4_adaptive_geometry": {
        "mosaic_policy": "adaptive_geometry",
        "adaptive_geometry": True,
    },
    "M5_hard_negative": {
        "mosaic_policy": "hard_negative",
        "hard_negative_tile": True,
        "hardneg_mosaic_prob": 0.30,
    },
}


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _clear_aug_env() -> None:
    for key in (
        "YOLO_CONTEXT_AUG",
        "YOLO_LEGACY_DOUBLE_OACP",
        "OACP_VARIANT",
        "OACP_PLACEMENT",
        "OACP_PROB_POLICY",
        "OACP_EFFECT_POLICY",
        "OACP_STRENGTH_POLICY",
        "OACP_SCALE_POLICY",
        "OACP_PROTECTION_POLICY",
    ):
        os.environ.pop(key, None)


def _dataset_mosaic_modes(dataset: str, method: str, tiny_modes: tuple[str, ...]) -> tuple[str, ...]:
    if method == "mosaic":
        if dataset == "levir":
            return ()
        return ("mosaic",)
    if dataset == "levir":
        return ("no_mosaic",)
    if dataset == "varroa":
        return ("mosaic",)
    return tiny_modes


def job_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    variants = {
        "oacp": args.oacp_variants,
        "mosaic": args.mosaic_variants,
        "copy_paste": args.copy_paste_variants,
    }
    for dataset in args.datasets:
        for method in args.methods:
            for variant in variants[method]:
                for mosaic_mode in _dataset_mosaic_modes(dataset, method, tuple(args.tinyperson_mosaic_modes)):
                    specs.append({"dataset": dataset, "method": method, "variant": variant, "mosaic_mode": mosaic_mode})
    return specs


def _prepare_dataset(dataset: str, method: str, args: argparse.Namespace) -> tuple[Path, Path]:
    if dataset == "levir":
        from misc.prepare_levir_ship import prepare

        output = args.dataset_root / f"levir_ship_augmentation_matrix_split_{args.split_seed}"
        return prepare(args.data_roots[dataset], output, args.split_seed).resolve(), output.resolve()
    if dataset == "tinyperson":
        from train_scripts import train_all_tinyperson as workflow

        test_root = workflow.prepare_test_set(args.data_roots[dataset], args.dataset_root)
        split_root = workflow.prepare_seed_dataset(
            args.data_roots[dataset], args.dataset_root, test_root, args.split_seed
        )
        return (split_root / "tinyperson.yaml").resolve(), test_root.resolve()
    from misc.prepare_dataset import prepare_dataset

    only_positives = method != "copy_paste"
    suffix = "copy_paste" if method == "copy_paste" else "augmentation"
    output = args.dataset_root / f"varroa_{suffix}_matrix_split_{args.split_seed}"
    data_yaml = prepare_dataset(
        args.data_roots[dataset], output, gt_source="gt_one", only_positives=only_positives,
        class_policy="map-3-to-1", seed=args.split_seed,
    )
    return data_yaml.resolve(), output.resolve()


def _mosaic_settings(spec: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    if spec["mosaic_mode"] == "no_mosaic":
        return {"mosaic": 0.0, "close_mosaic": 0, "mosaic_policy": "none", "mosaic_postprocess_enabled": False}
    settings: dict[str, Any] = {
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mosaic_postprocess_enabled": False,
        "mosaic_policy_candidates": 16,
        "mosaic_policy_topk": 4,
        "mosaic_visibility_thresh": 0.7,
        "mosaic_visibility_lambda": 1.0,
    }
    if spec["method"] == "mosaic":
        settings.update(MOSAIC_POLICY[spec["variant"]])
    else:
        settings["mosaic_policy"] = "standard"
    if spec["variant"] == "M3_post_scale_constrained":
        settings["mosaic_scale_statistics"] = str(args.scale_statistics[spec["dataset"]])
    if spec["variant"] == "M5_hard_negative":
        settings["hard_negative_bank"] = str(args.hard_negative_banks[spec["dataset"]])
    return settings


def _settings(spec: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {**COMMON_AUGMENTATION, **_mosaic_settings(spec, args), "mixup": 0.0, "cutmix": 0.0}
    if spec["method"] == "oacp":
        settings.update({"copy_paste": 0.0})
    elif spec["method"] == "copy_paste":
        settings.update(variant_overrides(spec["variant"]))
        settings.update(_mosaic_settings(spec, args))
    else:
        settings.update({"copy_paste": 0.0})
    return settings


def _configure_env(spec: dict[str, str]) -> None:
    _clear_aug_env()
    if spec["method"] == "oacp":
        os.environ.update({
            "YOLO_CONTEXT_AUG": "oacp",
            "YOLO_LEGACY_DOUBLE_OACP": "0",
            "OACP_VARIANT": spec["variant"],
            "OACP_PLACEMENT": "pre_transform",
        })


def _complete(run_dir: Path, repo_id: str, remote_prefix: str) -> bool:
    if not all((run_dir / path).is_file() for path in REQUIRED):
        return False
    try:
        marker = json.loads((run_dir / "upload_complete.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("repo_id") == repo_id and marker.get("remote_prefix") == remote_prefix


def _evaluate(run_dir: Path, data_yaml: Path, spec: dict[str, str], args: argparse.Namespace, test_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ULTRALYTICS))
    from ultralytics import YOLO

    if spec["dataset"] == "tinyperson":
        from train_scripts.train_copy_paste import _evaluate_run

        eval_args = argparse.Namespace(
            dataset="tinyperson",
            data_root=args.data_roots["tinyperson"],
            dataset_root=args.dataset_root,
            imgsz=args.imgsz["tinyperson"],
            batch_size=args.batch_size,
            device=args.device,
            workers=args.workers,
            nms_iou=args.nms_iou,
        )
        return _evaluate_run(run_dir, data_yaml, eval_args)

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, Any] = {"test_protocol": "LEVIR-Ship standard held-out test split" if spec["dataset"] == "levir" else "Varroa standard held-out test split"}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz[spec["dataset"]], batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=args.nms_iou,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update(_split_metrics(result, split))
    return metrics


def _upload(run_dir: Path, repo_id: str, remote_prefix: str) -> None:
    from huggingface_hub import HfApi

    missing = [path for path in REQUIRED[:-1] if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload for {run_dir}: {missing}")
    metrics = json.loads((run_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    missing_metrics = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing_metrics:
        raise RuntimeError(f"Missing split-qualified metrics for {run_dir}: {missing_metrics}")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote_prefix, repo_id=repo_id, repo_type="dataset")
    marker = {"repo_id": repo_id, "remote_prefix": remote_prefix, "verified": list(REQUIRED)}
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(run_dir / "upload_complete.json"), path_in_repo=f"{remote_prefix}/upload_complete.json", repo_id=repo_id, repo_type="dataset")
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    expected = [f"{remote_prefix}/{path}" for path in REQUIRED]
    missing_remote = [path for path in expected if path not in remote]
    if missing_remote:
        raise RuntimeError(f"HF upload verification failed for {remote_prefix}: {missing_remote}")


def _train_one(spec: dict[str, str], data_yaml: Path, test_root: Path, args: argparse.Namespace) -> Path:
    _configure_env(spec)
    repo_id = args.hf_repos[spec["method"]]
    remote_prefix = "/".join((spec["dataset"], spec["method"], spec["variant"], spec["mosaic_mode"], f"seed_{args.seed}"))
    run_dir = args.project / spec["dataset"] / spec["method"] / spec["variant"] / spec["mosaic_mode"] / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if _complete(run_dir, repo_id, remote_prefix):
        print(f"SKIP verified {remote_prefix}", flush=True)
        return run_dir

    settings = _settings(spec, args)
    manifest = {
        "experiment": "augmentation_matrix",
        "dataset": spec["dataset"],
        "method": spec["method"],
        "variant": spec["variant"],
        "mosaic_mode": spec["mosaic_mode"],
        "model_config": str(CANONICAL_CONFIG),
        "detector": "canonical YOLOv8 P3/P4/P5",
        "data_root": str(args.data_roots[spec["dataset"]]),
        "dataset_yaml": str(data_yaml),
        "dataset_label_policy": "all_records_including_negative_images" if spec["method"] == "copy_paste" and spec["dataset"] == "varroa" else "positive_only",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz[spec["dataset"]],
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "nms_iou": args.nms_iou,
        "augmentation": settings,
        "oacp_env": {key: os.environ[key] for key in ("YOLO_CONTEXT_AUG", "YOLO_LEGACY_DOUBLE_OACP", "OACP_VARIANT") if key in os.environ},
        "hf_repo_id": repo_id,
        "remote_prefix": remote_prefix,
        "commit_sha": _git_sha(),
        "upload_required": True,
        "test_protocol": "TinyPerson standard test plus merged corner-window evaluator" if spec["dataset"] == "tinyperson" else manifest_test_protocol(spec["dataset"]),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    training_files = (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv")
    if not all(path.is_file() for path in training_files):
        sys.path.insert(0, str(ULTRALYTICS))
        from ultralytics import YOLO

        model = YOLO(str(CANONICAL_CONFIG), task="detect")
        model.load(args.pretrained, smart_transfer=True)
        model.train(
            data=str(data_yaml), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz[spec["dataset"]],
            batch=args.batch_size, device=args.device, workers=args.workers, amp=args.amp, seed=args.seed,
            deterministic=True, plots=False, project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
            val=True, iou=args.nms_iou, optimizer="auto", **settings,
        )
    if not all(path.is_file() for path in training_files):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    metrics = _evaluate(run_dir, data_yaml, spec, args, test_root)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["metrics"] = metrics
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _upload(run_dir, repo_id, remote_prefix)
    return run_dir


def manifest_test_protocol(dataset: str) -> str:
    return "LEVIR-Ship standard held-out test split" if dataset == "levir" else "Varroa standard held-out test split"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root-levir", type=Path, default=Path("/marimo/LevirShip/LevirShipData"))
    parser.add_argument("--data-root-tinyperson", type=Path, default=Path("/marimo/TinyPerson"))
    parser.add_argument("--data-root-varroa", type=Path, default=Path("/marimo/Varroa"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/augmentation_matrix")
    parser.add_argument("--hf-repo-oacp", default="duyle2408/augmentation-oacp-runs")
    parser.add_argument("--hf-repo-mosaic", default="duyle2408/augmentation-mosaic-runs")
    parser.add_argument("--hf-repo-copy-paste", default="duyle2408/augmentation-copy-paste-runs")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--oacp-variants", nargs="+", choices=OACP_VARIANTS, default=list(OACP_VARIANTS))
    parser.add_argument("--mosaic-variants", nargs="+", choices=MOSAIC_VARIANTS, default=list(MOSAIC_VARIANTS))
    parser.add_argument("--copy-paste-variants", nargs="+", choices=COPY_PASTE_VARIANTS, default=list(COPY_PASTE_VARIANTS))
    parser.add_argument("--tinyperson-mosaic-modes", nargs="+", choices=("mosaic", "no_mosaic"), default=["mosaic", "no_mosaic"])
    parser.add_argument("--scale-statistics-levir", type=Path)
    parser.add_argument("--scale-statistics-tinyperson", type=Path)
    parser.add_argument("--scale-statistics-varroa", type=Path)
    parser.add_argument("--hard-negative-bank-levir", type=Path)
    parser.add_argument("--hard-negative-bank-tinyperson", type=Path)
    parser.add_argument("--hard-negative-bank-varroa", type=Path)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imgsz-levir", type=int, default=512)
    parser.add_argument("--imgsz-tinyperson", type=int, default=640)
    parser.add_argument("--imgsz-varroa", type=int, default=640)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--confirm-settings", action="store_true")
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from utils.marimo_ops import ensure_hf_repo, require_training_context

    args = parse_args(argv)
    args.data_roots = {
        "levir": args.data_root_levir.resolve(),
        "tinyperson": args.data_root_tinyperson.resolve(),
        "varroa": args.data_root_varroa.resolve(),
    }
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.imgsz = {"levir": args.imgsz_levir, "tinyperson": args.imgsz_tinyperson, "varroa": args.imgsz_varroa}
    args.hf_repos = {"oacp": args.hf_repo_oacp, "mosaic": args.hf_repo_mosaic, "copy_paste": args.hf_repo_copy_paste}
    args.scale_statistics = {"levir": args.scale_statistics_levir, "tinyperson": args.scale_statistics_tinyperson, "varroa": args.scale_statistics_varroa}
    args.hard_negative_banks = {"levir": args.hard_negative_bank_levir, "tinyperson": args.hard_negative_bank_tinyperson, "varroa": args.hard_negative_bank_varroa}

    specs = job_specs(args)
    if not specs:
        raise ValueError("No jobs selected. LEVIR Mosaic-policy jobs are intentionally skipped because LEVIR uses no Mosaic.")
    if args.print_effective_config:
        print(json.dumps({"job_count": len(specs), "jobs": specs}, indent=2, sort_keys=True))
        return
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train without --confirm-settings")
    if (args.seed, args.split_seed, args.epochs, args.patience, args.workers) != (42, 42, 100, 0, 8):
        raise ValueError("Matrix requires seed=42, split-seed=42, epochs=100, patience=0, workers=8")
    if any(spec["variant"] == "M3_post_scale_constrained" and (args.scale_statistics[spec["dataset"]] is None or not args.scale_statistics[spec["dataset"]].is_file()) for spec in specs):
        raise ValueError("M3 requires an existing --scale-statistics-{levir,tinyperson,varroa} file for every selected dataset")
    if any(spec["variant"] == "M5_hard_negative" and (args.hard_negative_banks[spec["dataset"]] is None or not args.hard_negative_banks[spec["dataset"]].is_file()) for spec in specs):
        raise ValueError("M5 requires an existing --hard-negative-bank-{levir,tinyperson,varroa} file for every selected dataset")
    for repo_id in args.hf_repos.values():
        require_training_context(hf_repo_id=repo_id)
        ensure_hf_repo(repo_id)

    prepared: dict[tuple[str, str], tuple[Path, Path]] = {}
    for dataset in args.datasets:
        for method in args.methods:
            prepared[(dataset, method)] = _prepare_dataset(dataset, method, args)
    if args.prepare_only:
        print(json.dumps({f"{dataset}/{method}": {"dataset_yaml": str(values[0]), "test_root": str(values[1])} for (dataset, method), values in prepared.items()}, indent=2, sort_keys=True))
        return
    for spec in specs:
        print(f"START {spec['dataset']}/{spec['method']}/{spec['variant']}/{spec['mosaic_mode']}", flush=True)
        _train_one(spec, prepared[(spec["dataset"], spec["method"])][0], prepared[(spec["dataset"], spec["method"])][1], args)
    print(f"Completed {len(specs)} augmentation runs.", flush=True)


if __name__ == "__main__":
    main()
