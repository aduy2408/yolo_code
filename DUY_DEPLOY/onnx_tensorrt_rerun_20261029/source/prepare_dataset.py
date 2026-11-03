#!/usr/bin/env python3
"""Prepare exactly one dataset split and emit its Ultralytics YAML."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("levirship", "tinyperson"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.dataset == "levirship":
        from misc.prepare_levir_ship import prepare

        raw_root = args.raw_root / "LevirShip" / "LevirShipData" if (args.raw_root / "LevirShip" / "LevirShipData").is_dir() else args.raw_root
        dataset_root = args.output_root / "levir_ship"
        yaml_path = prepare(raw_root, dataset_root, seed=args.split_seed)
        if yaml_path != args.dataset_yaml:
            args.dataset_yaml.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
        result = {"dataset": args.dataset, "dataset_yaml": str(args.dataset_yaml), "split_seed": args.split_seed}
    else:
        from train_scripts.train_all_tinyperson import prepare_seed_dataset, prepare_test_set

        raw_root = args.raw_root / "TinyPerson" if (args.raw_root / "TinyPerson").is_dir() else args.raw_root
        test_root = prepare_test_set(raw_root, args.output_root)
        split_root = prepare_seed_dataset(raw_root, args.output_root, test_root, args.split_seed)
        args.dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
        args.dataset_yaml.write_text(
            f"path: {args.output_root.resolve()}\n"
            f"train: {split_root.relative_to(args.output_root)}/images/train\n"
            f"val: {split_root.relative_to(args.output_root)}/images/val\n"
            f"test: {test_root.relative_to(args.output_root)}/images\n\n"
            "names:\n  0: person\n",
            encoding="utf-8",
        )
        result = {"dataset": args.dataset, "dataset_yaml": str(args.dataset_yaml), "split_seed": args.split_seed}
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
