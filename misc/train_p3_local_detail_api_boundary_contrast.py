#!/usr/bin/env python3
"""Train the P3-only LocalDetail API + boundary contrast variant."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import misc.train_under_2m_wiou as base


CONFIG = "yolov8_varroa_p3_local_detail_api_boundary_contrast.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default=base.MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=base.IMGSZ)
    parser.add_argument("--batch", type=int, default=base.BATCH)
    parser.add_argument("--workers", type=int, default=base.WORKERS)
    parser.add_argument("--device", default=base.DEVICE)
    parser.add_argument("--patience", type=int, default=base.PATIENCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(base.SEEDS))
    parser.add_argument("--train-project", default=str(base.PROJECT.parent / "train_p3_local_detail_api_boundary_contrast"))
    parser.add_argument("--test-project", default=str(base.TEST_PROJECT.parent / "test_p3_local_detail_api_boundary_contrast"))
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
    parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def selected_config(args: argparse.Namespace) -> list[tuple[Path, Path, int]]:
    config = base.CONFIG_ROOT / CONFIG
    if not config.is_file():
        raise FileNotFoundError(config)
    model_yaml = base.config_for_scale(config, args.scale)
    model = base.YOLO(str(model_yaml))
    params = sum(p.numel() for p in model.model.parameters())
    if params >= base.PARAM_LIMIT:
        raise ValueError(f"{config} has {params} params, expected < {base.PARAM_LIMIT}")
    print(f"SELECT {config.relative_to(base.CONFIG_ROOT)}: params={params}")
    return [(config, model_yaml, params)]


def print_dataset_seed() -> None:
    if not base.DATA_PATH.is_file():
        return
    for line in base.DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("seed:"):
            print(line)
            return


def main() -> None:
    args = parse_args()
    args.num_machines = 1
    args.machine_index = 0
    args.test_only = False

    print("ultralytics path:", base.ultralytics.__file__)
    print("train project:", args.train_project)
    print("test project:", args.test_project)
    print("seeds:", args.seeds)

    if base.GENERATED_CONFIG_DIR.exists():
        shutil.rmtree(base.GENERATED_CONFIG_DIR)

    configs = selected_config(args)
    if args.check_only:
        return

    for config, model_yaml, _ in configs:
        for seed in args.seeds:
            if not args.skip_prepare_data:
                base.prepare_data(seed=seed)
                print_dataset_seed()
            if not base.DATA_PATH.is_file():
                raise FileNotFoundError(f"Dataset YAML not found: {base.DATA_PATH}")
            base.train_config(config, model_yaml, seed, args)

    if base.run_test_inference(configs, args) is not None:
        base.upload_runs_to_hf(args)


if __name__ == "__main__":
    main()
