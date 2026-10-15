#!/usr/bin/env python3
"""Prepare the official TinyPerson split and merged corner-window test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_scripts.train_all_tinyperson import prepare_seed_dataset, prepare_test_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-yaml", default="dataset preparation")
    parser.add_argument("--hf-repo-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_root = prepare_test_set(args.data_root, args.dataset_root)
    split_root = prepare_seed_dataset(
        args.data_root,
        args.dataset_root,
        test_root,
        args.split_seed,
    )
    args.dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_yaml.write_text(
        "path: " + str(args.dataset_root) + "\n"
        + "train: " + str(split_root.relative_to(args.dataset_root)) + "/images/train\n"
        + "val: " + str(split_root.relative_to(args.dataset_root)) + "/images/val\n"
        + "test: " + str(test_root.relative_to(args.dataset_root)) + "/images\n\n"
        + "names:\n  0: person\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset_yaml": str(args.dataset_yaml),
        "train": str(split_root / "images/train"),
        "val": str(split_root / "images/val"),
        "test": str(test_root / "images"),
        "corner_manifest": str(test_root / "corner_manifest.json"),
        "split_seed": args.split_seed,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
