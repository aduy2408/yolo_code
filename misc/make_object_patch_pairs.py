#!/usr/bin/env python3
"""Create original and Gaussian-blurred object crop pairs."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "docs/reports/augmentation_examples/drawio_levirship_resource_pack"
PAIR_DIR = PACK / "object_patch_pairs"
PAIR_DIR.mkdir(parents=True, exist_ok=True)


def crop_from_source(source: Path, box, pad=12):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(width, x2 + pad)
    bottom = min(height, y2 + pad)
    crop = image[top:bottom, left:right].copy()
    local = (x1 - left, y1 - top, x2 - left, y2 - top)
    return crop, local


def pair(name, image, box, record):
    x1, y1, x2, y2 = box
    original = image.copy()
    blurred = image.copy()
    region = blurred[y1:y2, x1:x2]
    if region.size:
        blurred[y1:y2, x1:x2] = cv2.GaussianBlur(region, (0, 0), sigmaX=1.4)
    for canvas in (original, blurred):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
    original_name = f"{name}_original.png"
    blurred_name = f"{name}_gaussian_blur.png"
    cv2.imwrite(str(PAIR_DIR / original_name), original)
    cv2.imwrite(str(PAIR_DIR / blurred_name), blurred)
    record.update({
        "original": original_name,
        "gaussian_blur": blurred_name,
        "box_local_xyxy": list(box),
        "sigma_x": 1.4,
    })


def main():
    records = []
    provenance = json.loads((ROOT / "docs/reports/augmentation_examples/levir_cp_donor_provenance.json").read_text())
    for item in provenance["variants"]:
        image, local = crop_from_source(Path(item["donor_image"]), item["donor_bbox_xyxy"], pad=12)
        record = {"kind": "original_provenance", "source_image": item["donor_image"], "variant": item["name"]}
        pair(item["name"].lower().replace(" ", "_"), image, local, record)
        records.append(record)

    extra_manifest = json.loads((PACK / "extra_donor_crop_manifest.json").read_text())
    for item in extra_manifest:
        crop_path = PACK / item["output"]
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(crop_path)
        x1, y1, x2, y2 = [int(value) for value in item["local_box_xyxy"]]
        name = Path(item["output"]).stem
        record = {"kind": "extra_donor", "source_image": item["source_image"], "donor_output": item["donor_output"]}
        pair(name, image, (x1, y1, x2, y2), record)
        records.append(record)

    manifest = PAIR_DIR / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"pairs={len(records)}")
    print(f"pair_dir={PAIR_DIR}")


if __name__ == "__main__":
    main()
