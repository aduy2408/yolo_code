#!/usr/bin/env python3
"""Retrain one strong seed for under-2M WIoU configs, then test with Soft-NMS."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import misc.train_under_2m_wiou as base


VERIFY_CONFIGS = (
    "yolov8_varroa_p2p3_local_detail_boundary_contrast.yaml",
    "tried/yolov8_varroa_p3_local_detail_api.yaml",
    "tried/yolov8_varroa_compare_baseline_p3_edge_pooling_p3only.yaml",
    "yolov8_varroa_p2p3_local_detail_api.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=69, help="Strong seed to verify for the top config.")
    parser.add_argument("--scale", default=base.MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=base.IMGSZ)
    parser.add_argument("--batch", type=int, default=base.BATCH)
    parser.add_argument("--workers", type=int, default=base.WORKERS)
    parser.add_argument("--device", default=base.DEVICE)
    parser.add_argument("--patience", type=int, default=base.PATIENCE)
    parser.add_argument("--train-project", default=str(base.PROJECT.parent / "train_verify_under_2m_wiou"))
    parser.add_argument("--test-project", default=str(base.TEST_PROJECT.parent / "test_verify_under_2m_wiou"))
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Instantiate selected configs and print parameter counts only.")
    return parser.parse_args()


def selected_configs(args: argparse.Namespace) -> list[tuple[Path, Path, int]]:
    configs: list[tuple[Path, Path, int]] = []
    for rel_path in VERIFY_CONFIGS:
        config = base.CONFIG_ROOT / rel_path
        if not config.is_file():
            raise FileNotFoundError(config)
        model_yaml = base.config_for_scale(config, args.scale)
        model = base.YOLO(str(model_yaml))
        params = sum(p.numel() for p in model.model.parameters())
        if params >= base.PARAM_LIMIT:
            raise ValueError(f"{config} has {params} params, expected < {base.PARAM_LIMIT}")
        configs.append((config, model_yaml, params))
        print(f"VERIFY {config.relative_to(base.CONFIG_ROOT)}: params={params}")
    return configs


def print_dataset_seed() -> None:
    if not base.DATA_PATH.is_file():
        return
    for line in base.DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("seed:"):
            print(line)
            return


def main() -> None:
    args = parse_args()
    args.seeds = [args.seed]
    args.num_machines = 1
    args.machine_index = 0
    args.test_only = False
    args.no_upload = True

    print("ultralytics path:", base.ultralytics.__file__)
    print("verify seed:", args.seed)
    print("train project:", args.train_project)
    print("test project:", args.test_project)

    if base.GENERATED_CONFIG_DIR.exists():
        shutil.rmtree(base.GENERATED_CONFIG_DIR)

    configs = selected_configs(args)
    if args.check_only:
        return

    for config, model_yaml, _ in configs:
        if not args.skip_prepare_data:
            base.prepare_data(seed=args.seed)
            print_dataset_seed()
        if not base.DATA_PATH.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {base.DATA_PATH}")
        base.train_config(config, model_yaml, args.seed, args)

    base.run_test_inference(configs, args)


if __name__ == "__main__":
    main()
