#!/usr/bin/env python3
"""Create paper-ready dataset sample figures with context crops and object zooms."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reports/paper_samples"


@dataclass
class Sample:
    name: str
    image_path: Path
    boxes: np.ndarray
    selected_index: int
    note: str


def read_yolo(image_path: Path, label_root: Path) -> np.ndarray:
    label_path = label_root / f"{image_path.stem}.txt"
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    boxes = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, xc, yc, bw, bh = map(float, fields[:5])
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def read_tiny() -> tuple[list[tuple[Path, np.ndarray]], dict]:
    ann_path = ROOT / "datasets/TinyPerson/annotations/tiny_set_test.json"
    data = json.loads(ann_path.read_text())
    image_map = {item["id"]: item for item in data["images"]}
    grouped: dict[int, list[list[float]]] = {key: [] for key in image_map}
    for ann in data["annotations"]:
        if ann.get("ignore") or ann.get("uncertain"):
            continue
        x, y, w, h = ann["bbox"]
        if w > 0 and h > 0:
            grouped.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])
    output = []
    for image_id, raw_boxes in grouped.items():
        item = image_map[image_id]
        path = ROOT / "datasets/TinyPerson/test" / item["file_name"]
        if path.exists() and raw_boxes:
            output.append((path, np.asarray(raw_boxes, dtype=np.float32)))
    return output, data


def choose_tiny() -> Sample:
    candidates, _ = read_tiny()
    ranked = []
    for path, boxes in candidates:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        count = len(boxes)
        small = float(np.mean((boxes[:, 2] - boxes[:, 0]) < 32))
        score = (0 if 8 <= count <= 35 else -100) + min(count, 35) + 10 * small
        ranked.append((score, path, boxes, areas))
    _, path, boxes, areas = max(ranked, key=lambda item: item[0])
    selected = int(np.argmin(np.abs(areas - np.median(areas))))
    return Sample("TinyPerson", path, boxes, selected, "test image; non-uncertain annotations")


def choose_varroa() -> Sample:
    image_root = ROOT / "datasets/varroa_yolo/images/test"
    label_root = ROOT / "datasets/varroa_yolo/labels/test"
    ranked = []
    for path in sorted(image_root.glob("*.png")):
        boxes = read_yolo(path, label_root)
        if len(boxes) == 0:
            continue
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        score = min(len(boxes), 8) + float(np.clip(np.median(areas) / 1000, 0, 3))
        ranked.append((score, path, boxes, areas))
    _, path, boxes, areas = max(ranked, key=lambda item: item[0])
    selected = int(np.argmax(areas))
    return Sample("Varroa", path, boxes, selected, "test image; YOLO annotations")


def choose_levir() -> Sample:
    path = ROOT / "datasets/LevirShipData/All Images/GF1_WFV1_E110.0_N17.9_20200703_L2A0004902374_8192_8704.png"
    boxes = read_yolo(path, ROOT / "datasets/LevirShipData/All Annotations")
    if len(boxes) == 0:
        raise RuntimeError(f"No LEVIR annotations for {path}")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return Sample("LEVIR-Ship", path, boxes, int(np.argmin(areas)), "native 512x512 scene; all GT boxes")


def clip_box(box: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
            min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2))))


def crop_context(image: np.ndarray, box: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = max(bw * factor / 2, 24), max(bh * factor / 2, 24)
    crop_box = clip_box(np.asarray([cx - half_w, cy - half_h, cx + half_w, cy + half_h]), w, h)
    left, top, right, bottom = crop_box
    crop = image[top:bottom, left:right].copy()
    local = np.asarray([x1 - left, y1 - top, x2 - left, y2 - top], dtype=np.float32)
    return crop, local, crop_box


def draw_boxes(ax, boxes: np.ndarray, color: str = "#34d399", linewidth: float = 1.2, selected: int | None = None) -> None:
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        is_selected = selected == index
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor="#ef4444" if is_selected else color,
                               linewidth=linewidth + 0.8 if is_selected else linewidth))


def render(sample: Sample) -> Path:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(sample.image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    selected_box = sample.boxes[sample.selected_index]
    factor = 12.0 if sample.name == "TinyPerson" else 8.0 if sample.name == "LEVIR-Ship" else 5.0
    crop, local_box, crop_bounds = crop_context(image, selected_box, factor)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    zoom_size = 512
    zx1, zy1, zx2, zy2 = clip_box(selected_box, w, h)
    pad_x = max(4, int((zx2 - zx1) * (1.8 if sample.name != "TinyPerson" else 2.5)))
    pad_y = max(4, int((zy2 - zy1) * (1.8 if sample.name != "TinyPerson" else 2.5)))
    zx1, zy1 = max(0, zx1 - pad_x), max(0, zy1 - pad_y)
    zx2, zy2 = min(w, zx2 + pad_x), min(h, zy2 + pad_y)
    zoom = rgb[zy1:zy2, zx1:zx2]
    zoom = cv2.resize(zoom, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)
    zoom_local = np.asarray([(selected_box[0] - zx1) * zoom_size / max(zx2 - zx1, 1),
                             (selected_box[1] - zy1) * zoom_size / max(zy2 - zy1, 1),
                             (selected_box[2] - zx1) * zoom_size / max(zx2 - zx1, 1),
                             (selected_box[3] - zy1) * zoom_size / max(zy2 - zy1, 1)])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    panels = [(rgb, "Full image", sample.boxes, sample.selected_index),
              (crop_rgb, "Context crop", np.asarray([local_box]), 0),
              (zoom, "Object zoom", np.asarray([zoom_local]), 0)]
    for ax, (panel, title, boxes, selected) in zip(axes, panels):
        ax.imshow(panel)
        draw_boxes(ax, boxes, selected=selected)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"{sample.name}  |  {sample.note}", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015, f"Selected object: red box; other annotations: green boxes  |  source: {sample.image_path.name}",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"{sample.name.lower().replace('-', '').replace(' ', '_')}_sample.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    samples = [choose_levir(), choose_varroa(), choose_tiny()]
    outputs = [render(sample) for sample in samples]
    summary = []
    for sample, output in zip(samples, outputs):
        summary.append({
            "dataset": sample.name,
            "source": str(sample.image_path),
            "selected_index": sample.selected_index,
            "num_boxes": int(len(sample.boxes)),
            "selected_box_xyxy": sample.boxes[sample.selected_index].tolist(),
            "output": str(output),
        })
        print(f"{sample.name}: {sample.image_path.name} boxes={len(sample.boxes)} output={output}")
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
