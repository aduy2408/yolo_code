#!/usr/bin/env python3
"""Create standalone paper images for dataset examples."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from make_paper_dataset_samples import (
    OUT,
    Sample,
    choose_levir,
    choose_tiny,
    choose_varroa,
    clip_box,
    crop_context,
)


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def make_single(sample: Sample) -> dict:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(sample.image_path)
    h, w = image.shape[:2]
    selected = sample.boxes[sample.selected_index]
    stem = sample.name.lower().replace("-", "").replace(" ", "_")
    OUT.mkdir(parents=True, exist_ok=True)

    full = image.copy()
    for box in sample.boxes:
        draw_box(full, box, (0, 0, 255), 3)
    full_path = OUT / f"{stem}_full.png"
    cv2.imwrite(str(full_path), full)

    if sample.name == "Varroa":
        return {
            "dataset": sample.name,
            "source": str(sample.image_path),
            "full": str(full_path),
            "num_boxes": int(len(sample.boxes)),
            "selected_box_xyxy": selected.tolist(),
        }

    factor = 12.0 if sample.name == "TinyPerson" else 8.0
    crop, local_box, crop_bounds = crop_context(image, selected, factor)
    for box in sample.boxes:
        x1, y1, x2, y2 = crop_bounds
        local = np.asarray([box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1])
        if local[2] <= 0 or local[3] <= 0 or local[0] >= crop.shape[1] or local[1] >= crop.shape[0]:
            continue
        draw_box(crop, local, (0, 0, 255), 3)
    crop_path = OUT / f"{stem}_context_crop.png"
    cv2.imwrite(str(crop_path), crop)

    zx1, zy1, zx2, zy2 = clip_box(selected, w, h)
    pad_x = max(4, int((zx2 - zx1) * (2.5 if sample.name == "TinyPerson" else 1.8)))
    pad_y = max(4, int((zy2 - zy1) * (2.5 if sample.name == "TinyPerson" else 1.8)))
    zx1, zy1 = max(0, zx1 - pad_x), max(0, zy1 - pad_y)
    zx2, zy2 = min(w, zx2 + pad_x), min(h, zy2 + pad_y)
    zoom = image[zy1:zy2, zx1:zx2].copy()
    target = 768
    zoom = cv2.resize(zoom, (target, target), interpolation=cv2.INTER_CUBIC)
    local = np.asarray([(selected[0] - zx1) * target / max(zx2 - zx1, 1),
                        (selected[1] - zy1) * target / max(zy2 - zy1, 1),
                        (selected[2] - zx1) * target / max(zx2 - zx1, 1),
                        (selected[3] - zy1) * target / max(zy2 - zy1, 1)])
    draw_box(zoom, local, (0, 0, 255), 5)
    zoom_path = OUT / f"{stem}_object_zoom.png"
    cv2.imwrite(str(zoom_path), zoom)
    return {
        "dataset": sample.name,
        "source": str(sample.image_path),
        "full": str(full_path),
        "context_crop": str(crop_path),
        "object_zoom": str(zoom_path),
        "num_boxes": int(len(sample.boxes)),
        "selected_box_xyxy": selected.tolist(),
    }


def main() -> None:
    records = [make_single(sample) for sample in (choose_levir(), choose_varroa(), choose_tiny())]
    manifest = OUT / "single_images_manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        outputs = [record["full"]]
        if "context_crop" in record:
            outputs.extend([record["context_crop"], record["object_zoom"]])
        print(record["dataset"], *outputs)


if __name__ == "__main__":
    main()
