#!/usr/bin/env python3
"""Export extra real LevirShip donor scenes with red annotation boxes."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "datasets/LevirShipData/All Images"
LABEL_DIR = ROOT / "datasets/LevirShipData/All Annotations"
OUT = ROOT / "docs/reports/augmentation_examples/drawio_levirship_resource_pack"
COUNT = 16


def read_boxes(label_path: Path, width: int, height: int):
    boxes = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, cx, cy, bw, bh = map(float, fields[:5])
        x1 = max(0, int(round((cx - bw / 2) * width)))
        y1 = max(0, int(round((cy - bh / 2) * height)))
        x2 = min(width - 1, int(round((cx + bw / 2) * width)))
        y2 = min(height - 1, int(round((cy + bh / 2) * height)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    selected = []
    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = read_boxes(label_path, width, height)
        if not boxes:
            continue
        selected.append((image_path, image, boxes))
        if len(selected) >= COUNT:
            break

    for index, (image_path, image, boxes) in enumerate(selected, start=1):
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        output_name = f"extra_donor_{index:02d}_{image_path.stem}.png"
        output_path = OUT / output_name
        cv2.imwrite(str(output_path), image)
        records.append({
            "output": output_name,
            "source_image": str(image_path),
            "source_annotation": str(LABEL_DIR / f"{image_path.stem}.txt"),
            "boxes_xyxy": boxes,
        })

    if len(records) < COUNT:
        raise RuntimeError(f"Only selected {len(records)} donor scenes")
    (OUT / "extra_donor_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"selected={len(records)}")
    for record in records:
        print(record["output"], len(record["boxes_xyxy"]))


if __name__ == "__main__":
    main()
