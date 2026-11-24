#!/usr/bin/env python3
"""Create a before/after pair for the exact Negative Canvas R4 transform."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from misc.plot_cp_donor_provenance import boxes_from_labels, image_for, make_labels, load_dataset, transform_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reports/augmentation_examples/negative_canvas_r4_pair"


def draw_boxes(image, boxes):
    output = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    return output


def labeled(image, title):
    return image.copy()


def main():
    dataset = load_dataset(ROOT / "datasets/LevirShipData")
    negative_indices = [i for i, label in enumerate(dataset.labels) if not len(label["bboxes"])]
    if not negative_indices:
        raise RuntimeError("No negative canvas found")
    target_index = negative_indices[0]
    target_path = Path(dataset.im_files[target_index])
    before = image_for(target_path)
    transform = transform_for(dataset, "negative_canvas_r4", 45)
    result = transform(make_labels(dataset, target_index))
    if not transform.donor_records:
        raise RuntimeError("Negative Canvas R4 did not select a donor")
    output_boxes = boxes_from_labels(result)
    after = draw_boxes(result["img"], output_boxes)
    if len(output_boxes):
        x1, y1, x2, y2 = [int(round(value)) for value in output_boxes[0]]
        pad = 24
        height, width = after.shape[:2]
        left, top = max(0, x1 - pad), max(0, y1 - pad)
        right, bottom = min(width, x2 + pad), min(height, y2 + pad)
        zoom = after[top:bottom, left:right].copy()
        cv2.rectangle(zoom, (x1 - left, y1 - top), (x2 - left, y2 - top), (0, 0, 255), 5)
    else:
        zoom = after.copy()
    before_labeled = labeled(before, "Negative canvas before")
    after_labeled = labeled(after, "Negative Canvas R4 after paste")
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "negative_canvas_r4_before.png"), before_labeled)
    cv2.imwrite(str(OUT / "negative_canvas_r4_after_paste.png"), after_labeled)
    cv2.imwrite(str(OUT / "negative_canvas_r4_after_paste_zoom.png"), zoom)
    pair = cv2.hconcat([before_labeled, after_labeled])
    cv2.imwrite(str(OUT / "negative_canvas_r4_before_after_pair.png"), pair)
    (OUT / "README.txt").write_text(
        "Before/after pair generated with the exact project Negative Canvas R4 transform.\n"
        f"Target source: {target_path}\n"
        "Seed: 45\n"
        "The after image has a thick red box around the newly pasted object.\n"
        "A zoomed crop is saved as negative_canvas_r4_after_paste_zoom.png.\n"
    )
    print(f"target={target_path}")
    print(f"output_boxes={len(boxes_from_labels(result))}")
    print(f"output_dir={OUT}")


if __name__ == "__main__":
    main()
