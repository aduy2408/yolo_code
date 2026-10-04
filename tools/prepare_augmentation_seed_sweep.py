#!/usr/bin/env python3
"""Prepare fixed-split artifacts required by the augmentation seed sweep.

This creates train-only M3 scale statistics and static M5 hard-negative banks.
It never touches validation or test labels when deriving either artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _yaml_train_paths(data_yaml: Path) -> tuple[Path, Path]:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(payload.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    train = Path(payload["train"])
    if not train.is_absolute():
        train = base / train
    train = train.resolve()
    labels = Path(str(train).replace("/images/", "/labels/"))
    return train, labels


def _download_weights(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    urllib.request.urlretrieve(url, path)


def _prepare_splits(args: argparse.Namespace) -> dict[str, Path]:
    sys.path.insert(0, str(ROOT))
    from misc.prepare_dataset import prepare_dataset
    from misc.prepare_levir_ship import prepare as prepare_levir
    from train_scripts import train_all_tinyperson as tinyperson

    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    if "levir" in args.datasets:
        result["levir"] = prepare_levir(
            args.levir_root,
            args.output / f"levir_ship_augmentation_matrix_split_{args.split_seed}",
            args.split_seed,
        ).resolve()
    if "tinyperson" in args.datasets:
        test_root = tinyperson.prepare_test_set(args.tinyperson_root, args.output)
        split_root = tinyperson.prepare_seed_dataset(
            args.tinyperson_root, args.output, test_root, args.split_seed
        )
        result["tinyperson"] = (split_root / "tinyperson.yaml").resolve()
    if "varroa" in args.datasets:
        result["varroa"] = prepare_dataset(
            args.varroa_root,
            args.output / f"varroa_augmentation_matrix_split_{args.split_seed}",
            gt_source="gt_one",
            only_positives=True,
            class_policy="map-3-to-1",
            seed=args.split_seed,
        ).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levir-root", type=Path, default=Path("/marimo/LevirShip/LevirShipData"))
    parser.add_argument("--tinyperson-root", type=Path, default=Path("/marimo/TinyPerson"))
    parser.add_argument("--varroa-root", type=Path, default=Path("/marimo/Varroa"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=("levir", "tinyperson", "varroa"), default=("levir", "tinyperson", "varroa"))
    args = parser.parse_args()

    yamls = _prepare_splits(args)
    _download_weights(args.weights)
    artifacts = {}
    for dataset, data_yaml in yamls.items():
        train_images, train_labels = _yaml_train_paths(data_yaml)
        dataset_artifacts = args.output / "artifacts"
        scale_path = dataset_artifacts / f"{dataset}_mosaic_scale_statistics.json"
        bank_path = dataset_artifacts / f"{dataset}_hard_negative_bank.json"
        imgsz = {"levir": 512, "tinyperson": 640, "varroa": 640}[dataset]
        subprocess.run(
            [sys.executable, "tools/build_mosaic_scale_statistics.py", "--labels", str(train_labels), "--imgsz", str(imgsz), "--output", str(scale_path)],
            cwd=ROOT,
            check=True,
        )
        env = dict(os.environ)
        env["MOSAIC_MINER_ULTRALYTICS_ROOT"] = str(ROOT / "models_related" / "ultralytics")
        env["PYTHONPATH"] = f"{env['MOSAIC_MINER_ULTRALYTICS_ROOT']}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        subprocess.run(
            [sys.executable, "tools/mine_mosaic_hard_negatives.py", "--weights", str(args.weights), "--images", str(train_images), "--labels", str(train_labels), "--output", str(bank_path)],
            cwd=ROOT,
            env=env,
            check=True,
        )
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        if not bank:
            raise RuntimeError(f"Empty hard-negative bank for {dataset}: {bank_path}")
        artifacts[dataset] = {"dataset_yaml": str(data_yaml), "scale_statistics": str(scale_path), "hard_negative_bank": str(bank_path), "bank_count": len(bank)}
    print(json.dumps(artifacts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
