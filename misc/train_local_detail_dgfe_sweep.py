#!/usr/bin/env python3
"""Train/test the LocalDetail P3-only DGFE strength sweep."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import misc.train_under_2m_wiou as base


CONFIGS = (
    "yolov8_varroa_local_detail_p3only_dgfe_strong.yaml",
    "yolov8_varroa_local_detail_p3only_dgfe_stronger.yaml",
    "yolov8_varroa_local_detail_p3only_dgfe_ultra.yaml",
    "yolov8_varroa_local_detail_p3only_dgfe_max.yaml",
    "yolov8_varroa_local_detail_p3only_dgfe_extreme.yaml",
    "yolov8_varroa_local_detail_p3only_dgfe_extreme_noapi.yaml",
    "yolov8_varroa_local_detail_p3only_dgfe_tight_apiplus.yaml",
    "yolov8_varroa_local_detail_p3only_dgfe_tight_warmup.yaml",
)
CONFIG_ALIASES = {Path(config).stem.rsplit("_", 1)[-1]: config for config in CONFIGS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default=base.MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--param-limit", type=int, default=base.PARAM_LIMIT)
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=base.IMGSZ)
    parser.add_argument("--batch", type=int, default=base.BATCH)
    parser.add_argument("--workers", type=int, default=base.WORKERS)
    parser.add_argument("--device", default=base.DEVICE)
    parser.add_argument("--patience", type=int, default=base.PATIENCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(base.SEEDS))
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIG_ALIASES),
        default=tuple(CONFIG_ALIASES),
        help="DGFE strength configs to run.",
    )
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--train-project", default=str(base.PROJECT.parent / "train_local_detail_dgfe_sweep"))
    parser.add_argument("--test-project", default=str(base.TEST_PROJECT.parent / "test_local_detail_dgfe_sweep"))
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
    parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def selected_configs(args: argparse.Namespace) -> list[tuple[Path, Path, int]]:
    configs = []
    for config_key in args.configs:
        config_name = CONFIG_ALIASES[config_key]
        config = base.CONFIG_ROOT / config_name
        if not config.is_file():
            raise FileNotFoundError(config)

        model_yaml = base.config_for_scale(config, args.scale)
        model = base.YOLO(str(model_yaml))
        params = sum(p.numel() for p in model.model.parameters())
        if params >= args.param_limit:
            raise ValueError(f"{config} has {params} params, expected < {args.param_limit}")

        print(f"SELECT {config.relative_to(base.CONFIG_ROOT)}: params={params}")
        configs.append((config, model_yaml, params))
    return base.selected_shard(configs, args)


def print_dataset_seed() -> None:
    if not base.DATA_PATH.is_file():
        return
    for line in base.DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("seed:"):
            print(line)
            return


def main() -> None:
    args = parse_args()

    print("ultralytics path:", base.ultralytics.__file__)
    print("train project:", args.train_project)
    print("test project:", args.test_project)
    print("configs:", ", ".join(args.configs))
    print("seeds:", args.seeds)

    if base.GENERATED_CONFIG_DIR.exists():
        shutil.rmtree(base.GENERATED_CONFIG_DIR)

    configs = selected_configs(args)
    if args.check_only:
        return

    if not args.test_only:
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
