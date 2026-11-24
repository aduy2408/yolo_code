#!/usr/bin/env python3
"""Plot Original vs OACP vs Mosaic examples from a YOLO-format dataset.

Example:
    python misc/plot_augmentation_examples.py \
        --dataset datasets/levir_ship_yolo --split val --num-samples 3

The output is a single contact sheet with one sample per row and three columns:
Original, OACP, and Mosaic. Bounding boxes are drawn to make label geometry
changes visible as well as the image appearance.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_boxes(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            _, xc, yc, bw, bh = map(float, values[:5])
            boxes.append([(xc - bw / 2) * width, (yc - bh / 2) * height,
                          (xc + bw / 2) * width, (yc + bh / 2) * height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def image_items(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    # Also accept the original LEVIR-Ship layout, which stores all files in
    # directories named "All Images" and "All Annotations".
    if not image_dir.is_dir():
        image_dir = dataset / "All Images"
        label_dir = dataset / "All Annotations"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {dataset / 'images' / split} or {image_dir}")
    items = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            items.append((image_path, label_path))
    return items


def apply_oacp(image: np.ndarray, boxes: np.ndarray, seed: int) -> np.ndarray:
    """Apply the project OACP transform deterministically with p=1 for preview."""
    os.environ["OACP_P"] = "1.0"
    from project_ultralytics.context_augment import OACP

    random.seed(seed)
    np.random.seed(seed)
    h, w = image.shape[:2]
    # OACP's fallback ``bboxes`` path expects normalized xywh, while this
    # plotting script keeps boxes as pixel-space xyxy for drawing.
    normalized = np.empty_like(boxes)
    if len(boxes):
        normalized[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
        normalized[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
        normalized[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w
        normalized[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h
    labels = {"img": image.copy(), "bboxes": normalized}
    return OACP(p=1.0)(labels)["img"]


def load_image(item: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
    image_path, label_path = item
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    return image, read_boxes(label_path, w, h)


def mosaic(items: list[tuple[Path, Path]], indices: list[int], size: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Build a simple 2x2, aspect-preserving Mosaic preview and transform boxes."""
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    out_boxes = []
    half = size // 2
    for slot, index in enumerate(indices[:4]):
        image, boxes = load_image(items[index])
        h, w = image.shape[:2]
        x0, y0 = (slot % 2) * half, (slot // 2) * half
        scale = min(half / w, half / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        dx, dy = x0 + (half - nw) // 2, y0 + (half - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        if len(boxes):
            transformed = boxes.copy()
            transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + dx
            transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + dy
            out_boxes.append(transformed)
    return canvas, np.concatenate(out_boxes) if out_boxes else np.empty((0, 4), dtype=np.float32)


def draw(ax, image: np.ndarray, boxes: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor="#ff3b30", linewidth=1.2))
    ax.set_title(title, fontsize=11, weight="bold", pad=6)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/levir_ship_yolo")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/reports/augmentation_examples/contact_sheet.png")
    args = parser.parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")

    items = image_items(args.dataset, args.split)
    if len(items) < args.num_samples + 3:
        raise RuntimeError(f"Need at least {args.num_samples + 3} annotated images, found {len(items)}")
    rng = random.Random(args.seed)
    chosen = rng.sample(range(len(items)), args.num_samples)

    fig, axes = plt.subplots(args.num_samples, 3, figsize=(12, 4 * args.num_samples), squeeze=False, dpi=160)
    for row, base_index in enumerate(chosen):
        original, boxes = load_image(items[base_index])
        oacp = apply_oacp(original, boxes, args.seed + row)
        mosaic_indices = [base_index] + rng.sample([i for i in range(len(items)) if i != base_index], 3)
        mosaic_image, mosaic_boxes = mosaic(items, mosaic_indices, args.imgsz)
        draw(axes[row, 0], original, boxes, "Original")
        draw(axes[row, 1], oacp, boxes, "OACP")
        draw(axes[row, 2], mosaic_image, mosaic_boxes, "Mosaic")
        axes[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=11, weight="bold", labelpad=12)

    fig.suptitle(f"Augmentation preview | {args.dataset.name} | split={args.split}", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2, w_pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.num_samples} x 3 contact sheet to {args.output}")
    print("Samples: " + ", ".join(items[i][0].name for i in chosen))


if __name__ == "__main__":
    main()
