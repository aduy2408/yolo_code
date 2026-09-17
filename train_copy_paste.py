#!/usr/bin/env python3
"""Run the matched CP0-CP3 screening matrix on LEVIR-Ship or TinyPerson.

This runner intentionally defaults to printing the effective protocol. Full
training requires the Marimo workflow marker and HF authentication so every
completed run is uploaded before the next one starts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from copy_paste_protocol import VARIANTS, effective_settings, variant_overrides

ROOT = Path(__file__).resolve().parent


def _retry_hf(operation, *, attempts: int = 6, delay: float = 5.0):
    """Retry transient Hugging Face failures without hiding permanent errors."""
    last_error = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # network/service errors are provider-specific
            last_error = exc
            if attempt + 1 == attempts:
                raise
            print(f"Hugging Face operation failed ({type(exc).__name__}); retrying {attempt + 1}/{attempts - 1}...", flush=True)
            time.sleep(delay * (attempt + 1))
    raise last_error  # pragma: no cover


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _prepare(dataset: str, data_root: Path, dataset_root: Path, split_seed: int) -> Path:
    if dataset == "levir":
        from misc.prepare_levir_ship import prepare

        return prepare(data_root, dataset_root / f"levir_ship_copy_paste_split_{split_seed}", split_seed)
    if dataset == "tinyperson":
        import train_all_tinyperson as workflow

        test_dir = workflow.prepare_test_set(data_root, dataset_root)
        seed_dir = workflow.prepare_seed_dataset(data_root, dataset_root, test_dir, split_seed)
        return seed_dir / "tinyperson.yaml"
    raise ValueError(f"unsupported dataset: {dataset}")


def _upload(run_dir: Path, repo_id: str, dataset: str, variant: str, seed: int) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    _retry_hf(lambda: api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True))
    required = [
        "weights/best.pt", "weights/last.pt", "results.csv",
        "evaluation_metrics.json", "experiment_manifest.json",
    ]
    missing = [path for path in required if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload for {run_dir}: {missing}")
    metrics = json.loads((run_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    required_metrics = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")
    missing_metrics = [key for key in required_metrics if key not in metrics]
    if missing_metrics:
        raise RuntimeError(f"Refusing upload without split-qualified metrics for {run_dir}: {missing_metrics}")
    remote_prefix = f"copy_paste/{dataset}/{variant}/seed_{seed}"
    _retry_hf(lambda: api.upload_folder(
        folder_path=str(run_dir), repo_id=repo_id, repo_type="dataset",
        path_in_repo=remote_prefix,
    ))
    remote_files = set(_retry_hf(lambda: api.list_repo_files(repo_id=repo_id, repo_type="dataset")))
    missing_remote = [f"{remote_prefix}/{path}" for path in required if f"{remote_prefix}/{path}" not in remote_files]
    if missing_remote:
        raise RuntimeError(f"Hugging Face upload verification failed: {missing_remote}")
    (run_dir / "upload_complete.json").write_text(
        json.dumps({"repo_id": repo_id, "dataset": dataset, "variant": variant, "seed": seed}, indent=2) + "\n",
        encoding="utf-8",
    )
    _retry_hf(lambda: api.upload_file(
        path_or_fileobj=str(run_dir / "upload_complete.json"),
        path_in_repo=f"{remote_prefix}/upload_complete.json",
        repo_id=repo_id,
        repo_type="dataset",
    ))


def _train_images_and_labels(data_yaml: Path) -> tuple[list[Path], list[Path]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train = Path(config["train"])
    train = train if train.is_absolute() else root / train
    if train.is_file():
        image_paths = [Path(line.strip()) for line in train.read_text().splitlines() if line.strip()]
    else:
        image_paths = sorted(path for path in train.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    label_paths = []
    for image_path in image_paths:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_path = Path(*parts).with_suffix(".txt")
        else:
            label_path = image_path.with_suffix(".txt")
        label_paths.append(label_path)
    return image_paths, label_paths


def _read_yolo_boxes(label_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            _, xc, yc, bw, bh = map(float, values[:5])
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _mine_negcp_bank(args: argparse.Namespace, data_yaml: Path, checkpoint: Path, output: Path) -> Path:
    """Mine one offline bank from the completed CP0 checkpoint."""
    if output.is_file():
        return output
    ultralytics_path = ROOT / "models_related" / "ultralytics"
    if str(ultralytics_path) not in sys.path:
        sys.path.insert(0, str(ultralytics_path))
    from ultralytics import YOLO
    from project_ultralytics.negative_copy_paste import HardNegativeMiner

    image_paths, label_paths = _train_images_and_labels(data_yaml)
    if not image_paths:
        raise RuntimeError(f"No training images found for NegCP bank: {data_yaml}")
    model = YOLO(str(checkpoint))
    predictions = []
    valid_gts = []
    for image_path, label_path, result in zip(
        image_paths,
        label_paths,
        model.predict(source=[str(path) for path in image_paths], conf=0.25, iou=0.5,
                      device=args.device, batch=args.negcp_mine_batch_size,
                      stream=True, verbose=False),
    ):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read training image while mining NegCP: {image_path}")
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            predictions.append(np.empty((0, 5), dtype=np.float32))
        else:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy().reshape(-1, 1)
            predictions.append(np.concatenate((xyxy, conf), axis=1))
        valid_gts.append(_read_yolo_boxes(label_path, image.shape[:2]))
    miner = HardNegativeMiner()
    bank = miner.mine(image_paths, predictions, valid_gts)
    output.parent.mkdir(parents=True, exist_ok=True)
    bank.save(output)
    (output.with_suffix(".stats.json")).write_text(json.dumps(miner.stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _find_copy_paste_diagnostics(obj):
    """Find the configured small-object transform in a nested Compose tree."""
    if obj is None:
        return None
    diagnostics = getattr(obj, "diagnostics", None)
    if callable(diagnostics):
        return diagnostics()
    for child in getattr(obj, "transforms", []) or []:
        found = _find_copy_paste_diagnostics(child)
        if found is not None:
            return found
    return None


def _split_metrics(result, split: str) -> dict[str, float]:
    """Persist stable split-qualified metric aliases alongside Ultralytics keys."""
    values = {key: float(value) for key, value in result.results_dict.items()}
    output = {f"{split}/{key}": value for key, value in values.items()}
    output[f"{split}/AP50"] = values["metrics/mAP50(B)"]
    output[f"{split}/mAP50-95"] = values["metrics/mAP50-95(B)"]
    return output


def _evaluate_run(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    """Evaluate both standard validation and test splits.

    TinyPerson additionally runs its merged corner-window evaluator, whose
    protocol-specific metrics remain in the same JSON artifact.
    """
    ultralytics_path = ROOT / "models_related" / "ultralytics"
    if str(ultralytics_path) not in sys.path:
        sys.path.insert(0, str(ultralytics_path))
    from ultralytics import YOLO

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update(_split_metrics(result, split))

    if args.dataset == "tinyperson":
        import train_all_tinyperson as workflow

        test_out_dir = workflow.prepare_test_set(args.data_root, args.dataset_root)
        if not (test_out_dir / "corner_manifest.json").is_file():
            shutil.rmtree(test_out_dir, ignore_errors=True)
            test_out_dir = workflow.prepare_test_set(args.data_root, args.dataset_root)
        custom_args = argparse.Namespace(
            imgsz=args.imgsz, batch_size=args.batch_size, device=args.device, workers=args.workers,
        )
        custom_metrics = workflow.evaluate(run_dir, data_yaml, test_out_dir, args.data_root, custom_args)
        metrics.update({key: float(value) for key, value in custom_metrics.items() if isinstance(value, (int, float))})
    return metrics


def _run_one(args: argparse.Namespace, data_yaml: Path, variant: str, seed: int) -> Path:
    if args.oacp_variant != "none":
        os.environ["YOLO_CONTEXT_AUG"] = "oacp"
        os.environ["OACP_VARIANT"] = args.oacp_variant
        os.environ["YOLO_LEGACY_DOUBLE_OACP"] = "0"
    ultralytics_path = ROOT / "models_related" / "ultralytics"
    if str(ultralytics_path) not in sys.path:
        sys.path.insert(0, str(ultralytics_path))
    from ultralytics import YOLO

    run_dir = args.project / args.dataset / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    training_artifacts = [run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv"]
    training_complete = all(path.is_file() for path in training_artifacts)
    settings = variant_overrides(variant)
    settings["copy_paste_policy"] = args.copy_paste_policy
    settings["copy_paste_stats_path"] = str(args.copy_paste_stats_path or "")
    if variant == "negcp_offline":
        if not args.negcp_bank_path:
            raise RuntimeError("negcp_offline requires --negcp-bank-path or an auto-mined CP0 bank")
        settings["negcp_bank_path"] = str(args.negcp_bank_path)
    if args.mosaic_interaction:
        settings["mosaic"] = args.mosaic
        settings["close_mosaic"] = args.close_mosaic
        settings["mosaic_policy"] = args.mosaic_policy
        settings["scene_compatible_mosaic"] = args.scene_compatible_mosaic
        settings["mosaic_scale_quantile"] = args.mosaic_scale_quantile
        settings["mosaic_scale_modes"] = [4, 2]
        settings["mosaic_policy_candidates"] = 16
        settings["mosaic_policy_topk"] = 4
        settings["mosaic_visibility_thresh"] = 0.7
        settings["mosaic_visibility_lambda"] = 1.0
    if not training_complete:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, patience=0, seed=seed,
            deterministic=True, amp=True, plots=False, project=str(run_dir.parent),
            name=run_dir.name, exist_ok=True, val=True, iou=0.5, **settings,
        )
        cp_diagnostics = _find_copy_paste_diagnostics(
            getattr(getattr(model, "trainer", None), "train_loader", None)
            and model.trainer.train_loader.dataset.transforms
        )
        if cp_diagnostics is not None:
            (run_dir / "copy_paste_diagnostics.json").write_text(
                json.dumps(cp_diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    metrics = _evaluate_run(run_dir, data_yaml, args)
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = effective_settings(
        args.dataset, variant, seed, args.split_seed, commit_sha=_git_sha(),
        model=args.model, data_yaml=str(data_yaml), epochs=args.epochs, patience=0, imgsz=args.imgsz,
        batch_size=args.batch_size, device=args.device, workers=args.workers,
        hf_repo_id=args.hf_repo_id, upload_required=True,
        augmentation=settings,
        mosaic_interaction=args.mosaic_interaction,
        oacp_variant=args.oacp_variant,
        oacp_legacy_double=False,
        metrics=metrics,
    )
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("levir", "tinyperson"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--mosaic-interaction", action="store_true",
                        help="Enable Mosaic while retaining the matched Copy-Paste settings")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mosaic-policy", choices=("standard", "visibility", "occupancy_match", "context_contrast", "scale_adaptive"), default="standard")
    parser.add_argument("--mosaic-scale-quantile", type=float, default=0.05)
    parser.add_argument("--scene-compatible-mosaic", action="store_true",
                        help="Soft-gate existing Mosaic using dataset-derived scene statistics")
    parser.add_argument("--oacp-variant", choices=("none", "current", "budget", "density", "mass_adaptive", "load_adaptive", "spacing_adaptive"), default="none")
    parser.add_argument("--copy-paste-policy", choices=("fixed", "load_adaptive", "layout_adaptive"), default="fixed")
    parser.add_argument("--copy-paste-stats-path", type=Path, default=None)
    parser.add_argument("--negcp-bank-path", type=Path, default=None)
    parser.add_argument("--negcp-mine-device", default="cpu",
                        help="Device used only for offline hard-negative mining")
    parser.add_argument("--negcp-mine-batch-size", type=int, default=4,
                        help="Bounded inference batch size used for offline hard-negative mining")
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one bounded smoke variant and upload its artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.scene_compatible_mosaic and not args.mosaic_interaction:
        raise ValueError("--scene-compatible-mosaic requires --mosaic-interaction")
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    configs = [effective_settings(args.dataset, variant, seed, args.split_seed, epochs=args.epochs, patience=0,
                                  imgsz=args.imgsz, batch_size=args.batch_size, device=args.device,
                                  workers=args.workers, hf_repo_id=args.hf_repo_id,
                                  augmentation={**variant_overrides(variant),
                                                **({"mosaic": args.mosaic, "close_mosaic": args.close_mosaic}
                                                   if args.mosaic_interaction else {}),
                                                **({"scene_compatible_mosaic": True}
                                                   if args.scene_compatible_mosaic else {}),
                                                **({"mosaic_scale_quantile": args.mosaic_scale_quantile,
                                                    "mosaic_scale_modes": [4, 2]}
                                                   if args.mosaic_policy == "scale_adaptive" else {}),
                                                **({"mosaic_policy": args.mosaic_policy,
                                                    "mosaic_policy_candidates": 16,
                                                    "mosaic_policy_topk": 4,
                                                    "mosaic_visibility_thresh": 0.7,
                                                    "mosaic_visibility_lambda": 1.0}
                                                   if args.mosaic_interaction else {}),
                                                "copy_paste_policy": args.copy_paste_policy,
                                                "copy_paste_stats_path": str(args.copy_paste_stats_path or "")},
                                  mosaic_interaction=args.mosaic_interaction,
                                  oacp_variant=args.oacp_variant,
                                  oacp_legacy_double=False)
               for seed in args.seeds for variant in args.variants]
    if args.print_effective_config:
        print(json.dumps({"runs": configs}, indent=2, sort_keys=True))
        return
    data_yaml = _prepare(args.dataset, args.data_root, args.dataset_root, args.split_seed)
    if args.prepare_only:
        print(data_yaml)
        return
    if args.epochs != 100 and not args.smoke:
        raise ValueError("full Copy-Paste screening requires epochs=100")
    if args.smoke and (len(args.seeds) != 1 or len(args.variants) != 1):
        raise ValueError("--smoke requires exactly one seed and one variant")
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("refusing to train outside the Marimo training workflow")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for upload-required Copy-Paste screening")
    if "negcp_offline" in args.variants and not args.negcp_bank_path and "cp0" not in args.variants:
        raise ValueError("negcp_offline without --negcp-bank-path requires cp0 in --variants for bank mining")
    ordered_variants = ["cp0"] + [variant for variant in args.variants if variant != "cp0"] if "cp0" in args.variants else list(args.variants)
    configured_bank_path = args.negcp_bank_path
    for seed in args.seeds:
        args.negcp_bank_path = configured_bank_path
        for variant in ordered_variants:
            if variant == "negcp_offline" and not configured_bank_path:
                mine_args = argparse.Namespace(**vars(args))
                mine_args.device = args.negcp_mine_device
                args.negcp_bank_path = _mine_negcp_bank(
                    mine_args,
                    data_yaml,
                    args.project / args.dataset / "cp0" / f"seed_{seed}" / "weights" / "best.pt",
                    args.project / args.dataset / "negcp_banks" / f"split_{args.split_seed}_seed_{seed}.json",
                )
            run_dir = _run_one(args, data_yaml, variant, seed)
            _upload(run_dir, args.hf_repo_id, args.dataset, variant, seed)


if __name__ == "__main__":
    main()
