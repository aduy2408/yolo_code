#!/usr/bin/env python3
"""Precompute train-only object scale statistics for post-scale Mosaic.

The input must be a YOLO training label directory. No validation or test files
should be included. Values are stored in pixels at ``--imgsz``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sqrt_area, min_side = [], []
    for path in sorted(args.labels.rglob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            _, _, _, width, height = map(float, fields[:5])
            w, h = width * args.imgsz, height * args.imgsz
            sqrt_area.append(float(np.sqrt(max(w * h, 0.0))))
            min_side.append(float(min(w, h)))
    if not sqrt_area:
        raise RuntimeError("No YOLO boxes found in the training label directory")
    quantiles = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    result = {
        "imgsz": args.imgsz,
        "source": str(args.labels.resolve()),
        "count": len(sqrt_area),
        "sqrt_area_quantiles": {str(q): float(np.quantile(sqrt_area, q)) for q in quantiles},
        "min_side_quantiles": {str(q): float(np.quantile(min_side, q)) for q in quantiles},
        "small_threshold": float(np.quantile(sqrt_area, 0.25)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
