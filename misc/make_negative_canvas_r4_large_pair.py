#!/usr/bin/env python3
"""Create a clean R4-style negative-canvas pair with a larger pasted object."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from misc.plot_cp_donor_provenance import (
    TraceNegativeCanvasCopyPaste,
    boxes_from_labels,
    image_for,
    load_dataset,
    make_labels,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reports/augmentation_examples/negative_canvas_r4_large_pair"


def draw_boxes(image, boxes):
    output = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    return output


def main():
    dataset = load_dataset(ROOT / "datasets/LevirShipData")
    negative_indices = [i for i, label in enumerate(dataset.labels) if not len(label["bboxes"])]
    target_index = negative_indices[0]
    before = image_for(dataset.im_files[target_index])
    selected = None
    for seed in range(46, 146):
        transform = TraceNegativeCanvasCopyPaste(
            dataset,
            p=1.0,
            target_policy="deficit",
            donor_policy="larger",
            target_max_size=32.0,
            degradation="weak_blur",
            blur_sigma=1.4,
            rng=random.Random(seed),
            max_trials=100,
        )
        result = transform(make_labels(dataset, target_index))
        boxes = boxes_from_labels(result)
        if len(boxes):
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            if float(max(widths)) >= 24 or float(max(heights)) >= 24:
                selected = (seed, transform, result, boxes)
                break
    if selected is None:
        raise RuntimeError("Could not find a larger R4-style pasted object")
    seed, transform, result, boxes = selected
    after = draw_boxes(result["img"], boxes)
    x1, y1, x2, y2 = [int(round(value)) for value in boxes[0]]
    pad = 24
    height, width = after.shape[:2]
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    right, bottom = min(width, x2 + pad), min(height, y2 + pad)
    zoom = after[top:bottom, left:right].copy()
    cv2.rectangle(zoom, (x1 - left, y1 - top), (x2 - left, y2 - top), (0, 0, 255), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "negative_canvas_r4_large_before.png"), before)
    cv2.imwrite(str(OUT / "negative_canvas_r4_large_after_paste.png"), after)
    cv2.imwrite(str(OUT / "negative_canvas_r4_large_after_paste_zoom.png"), zoom)
    cv2.imwrite(str(OUT / "negative_canvas_r4_large_before_after_pair.png"), cv2.hconcat([before, after]))
    (OUT / "README.txt").write_text(
        "Clean R4-style negative-canvas pair with a larger pasted object.\n"
        f"Seed: {seed}\n"
        "target_max_size: 32 pixels\n"
        "gaussian_blur_sigma: 1.4\n"
        f"output_boxes_xyxy: {boxes.tolist()}\n"
        "The after image contains a thin red box around the newly pasted object.\n"
        "The zoomed object region is saved as negative_canvas_r4_large_after_paste_zoom.png.\n"
    )
    print(f"seed={seed}")
    print(f"boxes={boxes.tolist()}")
    print(f"output_dir={OUT}")


if __name__ == "__main__":
    main()
