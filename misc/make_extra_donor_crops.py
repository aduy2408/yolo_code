#!/usr/bin/env python3
"""Create corresponding zoomed crops for every extra LevirShip donor object."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "docs/reports/augmentation_examples/drawio_levirship_resource_pack"
PAD = 16


def main():
    manifest = json.loads((PACK / "extra_donor_manifest.json").read_text())
    crop_records = []
    for donor in manifest:
        source = Path(donor["source_image"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        height, width = image.shape[:2]
        stem = Path(donor["output"]).stem
        for object_index, box in enumerate(donor["boxes_xyxy"], start=1):
            x1, y1, x2, y2 = [int(value) for value in box]
            left = max(0, x1 - PAD)
            top = max(0, y1 - PAD)
            right = min(width, x2 + PAD)
            bottom = min(height, y2 + PAD)
            crop = image[top:bottom, left:right].copy()
            local = (x1 - left, y1 - top, x2 - left, y2 - top)
            cv2.rectangle(crop, (local[0], local[1]), (local[2], local[3]), (0, 0, 255), 3)
            output = f"{stem}_object_{object_index:02d}_crop.png"
            cv2.imwrite(str(PACK / output), crop)
            crop_records.append({
                "output": output,
                "donor_output": donor["output"],
                "source_image": donor["source_image"],
                "source_box_xyxy": box,
                "crop_box_xyxy": [left, top, right, bottom],
                "local_box_xyxy": list(local),
            })
    (PACK / "extra_donor_crop_manifest.json").write_text(json.dumps(crop_records, indent=2) + "\n")
    print(f"crops={len(crop_records)}")
    for record in crop_records:
        print(record["output"])


if __name__ == "__main__":
    main()
