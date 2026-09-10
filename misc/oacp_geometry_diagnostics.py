#!/usr/bin/env python3
"""Log OACP geometry diagnostics before training on LEVIR-Ship or TinyPerson.

The script is intentionally dataset-format agnostic: provide an image directory
and its matching YOLO label directory. It writes one JSON object per image and a
small aggregate JSON summary for each of the three planned ablations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from project_ultralytics.context_augment import oacp_diagnostics

VARIANTS = ("current", "budget", "density")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_boxes(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if path.is_file():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if len(values) < 5:
                raise ValueError(f"{path}:{line_number}: expected class + xywh")
            _, xc, yc, bw, bh = map(float, values[:5])
            boxes.append(((xc - bw / 2) * width, (yc - bh / 2) * height,
                          (xc + bw / 2) * width, (yc + bh / 2) * height))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name recorded in output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N images (0 means all)")
    parser.add_argument("--budget", type=float, default=0.45,
                        help="Budget fraction of valid background for budget/density")
    args = parser.parse_args()

    images = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        if args.limit < 0:
            raise SystemExit("--limit must be non-negative")
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.images}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {variant: {key: 0.0 for key in (
        "num_gt", "protected_area_ratio", "perturbable_area_ratio",
        "actual_perturbed_area_ratio", "mean_object_size", "protected_expand")}
               for variant in VARIANTS}
    counts = {variant: 0 for variant in VARIANTS}
    with args.output.open("w", encoding="utf-8") as stream:
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            label_path = args.labels / f"{image_path.stem}.txt"
            boxes = read_boxes(label_path, image.shape[1], image.shape[0])
            for variant in VARIANTS:
                row = oacp_diagnostics(image.shape, boxes, variant, args.budget)
                row.update({"dataset": args.dataset, "image": str(image_path), "label": str(label_path)})
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                counts[variant] += 1
                for key in summary[variant]:
                    summary[variant][key] += float(row[key])
    for variant in VARIANTS:
        if counts[variant]:
            for key in summary[variant]:
                summary[variant][key] /= counts[variant]
        summary[variant]["images"] = counts[variant]
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps({"dataset": args.dataset, "variants": summary}, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
