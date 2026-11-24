#!/usr/bin/env python3
"""Create a compact 3-row PDF: three representative samples per dataset."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
plt.rcParams["font.family"] = "Open Sans"
OPEN_SANS = FontProperties(family="Open Sans")
OPEN_SANS_BOLD = FontProperties(family="Open Sans", weight="bold")

from make_paper_dataset_samples import (
    ROOT,
    Sample,
    clip_box,
    read_tiny,
    read_yolo,
)

OUT = ROOT / "docs/reports/paper_samples/dataset_samples_3x3.pdf"
DATASETS = ("Varroa", "LEVIR-Ship", "TinyPerson")


def load_candidates(name: str) -> list[Sample]:
    if name == "Varroa":
        image_root = ROOT / "datasets/varroa_yolo/images/test"
        label_root = ROOT / "datasets/varroa_yolo/labels/test"
        candidates = []
        for path in sorted(image_root.glob("*.png")):
            boxes = read_yolo(path, label_root)
            if len(boxes):
                candidates.append(Sample(name, path, boxes, 0, "test split"))
    elif name == "TinyPerson":
        candidates = [Sample(name, path, boxes, 0, "test split") for path, boxes in read_tiny()[0]]
    else:
        image_root = ROOT / "datasets/LevirShipData/All Images"
        label_root = ROOT / "datasets/LevirShipData/All Annotations"
        candidates = []
        for path in sorted(image_root.glob("*.png")):
            boxes = read_yolo(path, label_root)
            if len(boxes):
                candidates.append(Sample(name, path, boxes, 0, "test scene"))
    if len(candidates) < 3:
        raise RuntimeError(f"Only {len(candidates)} usable {name} samples found")
    for sample in candidates:
        areas = (sample.boxes[:, 2] - sample.boxes[:, 0]) * (sample.boxes[:, 3] - sample.boxes[:, 1])
        sample.selected_index = int(np.argsort(areas)[len(areas) // 2])
    return candidates


def pick_three(name: str) -> list[Sample]:
    candidates = load_candidates(name)
    # Spread the examples by object count, then object scale; deterministic and reproducible.
    candidates.sort(key=lambda s: (len(s.boxes), float(np.median((s.boxes[:, 2] - s.boxes[:, 0]) * (s.boxes[:, 3] - s.boxes[:, 1]))), s.image_path.name))
    indices = [0, len(candidates) // 2, len(candidates) - 1]
    picked = []
    for index in indices:
        sample = candidates[index]
        if not any(sample.image_path == previous.image_path for previous in picked):
            picked.append(sample)
    return picked


def draw_boxes(ax, boxes: np.ndarray, selected: int | None = None) -> None:
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        linewidth = 1.4 if index == selected else 0.9
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#e31a1c", linewidth=linewidth))



def draw_sample(fig, cell, sample: Sample) -> None:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(sample.image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    box = sample.boxes[sample.selected_index]
    x1, y1, x2, y2 = clip_box(box, w, h)
    pad_x = max(4, int((x2 - x1) * (2.5 if sample.name == "TinyPerson" else 1.8)))
    pad_y = max(4, int((y2 - y1) * (2.5 if sample.name == "TinyPerson" else 1.8)))
    zx1, zy1, zx2, zy2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    zoom = rgb[zy1:zy2, zx1:zx2]
    zoom_box = np.asarray([box[0] - zx1, box[1] - zy1, box[2] - zx1, box[3] - zy1])
    crop_box = (zx1, zy1, zx2, zy2)

    position = cell.get_position()
    cell.remove()
    if sample.name == "TinyPerson":
        full_ax = fig.add_axes([position.x0, position.y0 + position.height * 0.33, position.width, position.height * 0.67])
        zoom_ax = fig.add_axes([position.x0, position.y0, position.width, position.height * 0.29])
        connectors = [
            ((zx1, zy2), (0, 0)),
            ((zx2, zy2), (zoom.shape[1], 0)),
        ]
    else:
        full_ax = fig.add_axes([position.x0, position.y0, position.width * 0.64, position.height])
        zoom_ax = fig.add_axes([position.x0 + position.width * 0.68, position.y0, position.width * 0.32, position.height])
        connectors = [
            ((zx2, zy1), (0, 0)),
            ((zx2, zy2), (0, zoom.shape[0])),
        ]
    full_ax.imshow(rgb)
    draw_boxes(full_ax, sample.boxes, sample.selected_index)
    full_ax.add_patch(Rectangle(
        (crop_box[0], crop_box[1]),
        crop_box[2] - crop_box[0],
        crop_box[3] - crop_box[1],
        fill=False,
        edgecolor="#e31a1c",
        linewidth=1.0,
        linestyle=(0, (3, 2)),
    ))
    zoom_ax.imshow(zoom)
    draw_boxes(zoom_ax, np.asarray([zoom_box]), 0)
    for full_point, zoom_point in connectors:
        fig.add_artist(ConnectionPatch(
            xyA=full_point,
            coordsA=full_ax.transData,
            xyB=zoom_point,
            coordsB=zoom_ax.transData,
            color="#e31a1c",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.9,
        ))
    for ax in (full_ax, zoom_ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#bbbbbb")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with PdfPages(OUT) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        left, right, top, bottom = 0.055, 0.985, 0.955, 0.035
        grid = fig.add_gridspec(3, 3, left=left, right=right, top=top, bottom=bottom, hspace=0.08, wspace=0.12)
        for col in range(3):
            x = left + (right - left) * (col + 0.5) / 3
            fig.text(x, 0.985, f"Sample {col + 1}", ha="center", va="center", fontsize=9, fontproperties=OPEN_SANS)
        for row, dataset in enumerate(DATASETS):
            samples = pick_three(dataset)
            row_y = bottom + (top - bottom) * (1 - (row + 0.5) / 3)
            fig.text(0.018, row_y, dataset, rotation=90, va="center", ha="center", fontsize=10, fontproperties=OPEN_SANS_BOLD)
            for col, sample in enumerate(samples):
                cell = fig.add_subplot(grid[row, col])
                draw_sample(fig, cell, sample)
                records.append({"dataset": dataset, "source": str(sample.image_path), "num_boxes": int(len(sample.boxes)), "selected_box_xyxy": sample.boxes[sample.selected_index].tolist()})
        pdf.savefig(fig, dpi=250, bbox_inches="tight")
        plt.close(fig)
    manifest = OUT.with_suffix(".json")
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
