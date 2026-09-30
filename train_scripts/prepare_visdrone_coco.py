#!/usr/bin/env python3
"""Convert official VisDrone2019-DET annotations to split COCO JSON files.

The converter preserves the official train/val/test-dev images and only drops
ignored regions (score 0) and the non-detector ``others`` category.  Images
remain in their original directories, so the generated COCO JSON can be used
with absolute ``data_prefix`` paths by the MMDetection runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}

CATEGORIES = [
    {"id": index, "name": name}
    for index, name in enumerate(
        (
            "pedestrian",
            "people",
            "bicycle",
            "car",
            "van",
            "truck",
            "tricycle",
            "awning-tricycle",
            "bus",
            "motor",
        ),
        start=1,
    )
]


def convert_split(source: Path, destination: Path) -> None:
    images_dir = source / "images"
    annotations_dir = source / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and annotations/ under {source}")

    images = []
    annotations = []
    annotation_id = 1
    image_files = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_id, image_path in enumerate(image_files, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        label_path = annotations_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 6:
                continue
            x, y, box_width, box_height, score, category = map(int, fields[:6])
            if score == 0 or not 1 <= category <= 10:
                continue
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            box_width = max(0, min(box_width, width - x))
            box_height = max(0, min(box_height, height - y))
            if box_width == 0 or box_height == 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {"description": "VisDrone2019-DET official split"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")
    print(f"{destination}: {len(images)} images, {len(annotations)} boxes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    for split, folder in SPLITS.items():
        convert_split(args.data_root / folder, args.output_dir / "annotations" / f"{split}.json")
    print(json.dumps({"data_root": str(args.data_root), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
