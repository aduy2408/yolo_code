#!/usr/bin/env python3
"""Prepare one STW-OACP dataset split and validate its exact YAML contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc.train_stw_yolo_marimo import prepare_dataset
from utils.marimo_ops import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("levirship", "tinyperson"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    runtime_root = args.runtime_root.resolve()
    dataset_yaml = prepare_dataset(args.dataset, runtime_root, data_root)
    validation = validate_dataset(data_root, dataset_yaml)
    print(json.dumps({"dataset": args.dataset, "data_root": str(data_root), "dataset_yaml": str(dataset_yaml.resolve()), "validation": validation}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
