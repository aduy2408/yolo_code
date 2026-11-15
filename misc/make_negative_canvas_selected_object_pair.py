#!/usr/bin/env python3
"""Paste the user-selected LevirShip object into a clean negative canvas."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "docs/reports/augmentation_examples/drawio_levirship_resource_pack"
OUT = ROOT / "docs/reports/augmentation_examples/negative_canvas_selected_object_pair"
CROP_NAME = "extra_donor_15_GF1_WFV1_E110.0_N17.9_20200703_L2A0004902374_9216_8192_object_01_crop.png"
TARGET = ROOT / "datasets/LevirShipData/All Images/GF1_WFV1_E110.0_N17.9_20200703_L2A0004902374_10752_10752.png"
DEST_X, DEST_Y = 360, 245
SCALE = 1.4
BLUR_SIGMA = 1.4


def main():
    manifest = json.loads((PACK / "extra_donor_crop_manifest.json").read_text())
    record = next(item for item in manifest if item["output"] == CROP_NAME)
    source = cv2.imread(record["source_image"], cv2.IMREAD_COLOR)
    target = cv2.imread(str(TARGET), cv2.IMREAD_COLOR)
    if source is None or target is None:
        raise RuntimeError("Could not load selected source or target")
    sx1, sy1, sx2, sy2 = [int(value) for value in record["source_box_xyxy"]]
    object_patch = source[sy1:sy2, sx1:sx2].copy()
    new_width = max(1, int(round(object_patch.shape[1] * SCALE)))
    new_height = max(1, int(round(object_patch.shape[0] * SCALE)))
    object_patch = cv2.resize(object_patch, (new_width, new_height), interpolation=cv2.INTER_AREA)
    blurred_patch = cv2.GaussianBlur(object_patch, (0, 0), sigmaX=BLUR_SIGMA)
    before = target.copy()
    after = target.copy()
    after[DEST_Y:DEST_Y + new_height, DEST_X:DEST_X + new_width] = blurred_patch
    cv2.rectangle(after, (DEST_X, DEST_Y), (DEST_X + new_width, DEST_Y + new_height), (0, 0, 255), 1)
    pad = 24
    left = max(0, DEST_X - pad)
    top = max(0, DEST_Y - pad)
    right = min(after.shape[1], DEST_X + new_width + pad)
    bottom = min(after.shape[0], DEST_Y + new_height + pad)
    zoom = after[top:bottom, left:right].copy()
    cv2.rectangle(zoom, (DEST_X - left, DEST_Y - top), (DEST_X + new_width - left, DEST_Y + new_height - top), (0, 0, 255), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "selected_object_source_crop.png"), object_patch)
    cv2.imwrite(str(OUT / "negative_canvas_selected_before.png"), before)
    cv2.imwrite(str(OUT / "negative_canvas_selected_after_paste.png"), after)
    cv2.imwrite(str(OUT / "negative_canvas_selected_after_paste_zoom.png"), zoom)
    cv2.imwrite(str(OUT / "negative_canvas_selected_before_after_pair.png"), cv2.hconcat([before, after]))
    (OUT / "README.txt").write_text(
        "Selected object pair generated from the requested crop provenance.\n"
        f"Requested crop: {PACK / CROP_NAME}\n"
        f"Source image: {record['source_image']}\n"
        f"Source bbox: {record['source_box_xyxy']}\n"
        f"Scale: {SCALE}\n"
        f"Gaussian blur sigma: {BLUR_SIGMA}\n"
        f"Destination bbox: {[DEST_X, DEST_Y, DEST_X + new_width, DEST_Y + new_height]}\n"
    )
    print(f"source={record['source_image']}")
    print(f"destination_box={[DEST_X, DEST_Y, DEST_X + new_width, DEST_Y + new_height]}")
    print(f"output_dir={OUT}")


if __name__ == "__main__":
    main()
