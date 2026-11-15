#!/usr/bin/env python3
"""Render a 4x2 LevirShip preview for CP1, CP2, negative-canvas R2, and R4.

Each row contains the original target canvas and the transformed canvas. The
transforms are the project implementations, not hand-drawn illustrations.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from project_ultralytics.copy_paste import SmallObjectCopyPaste
from project_ultralytics.negative_canvas_copy_paste import NegativeCanvasCopyPaste
from ultralytics.utils.instance import Instances

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_boxes(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            _, xc, yc, bw, bh = map(float, values[:5])
            boxes.append([
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            ])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def load_dataset(root: Path):
    image_dir = root / "All Images"
    label_dir = root / "All Annotations"
    images = []
    labels = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or not label_path.exists():
            continue
        h, w = image.shape[:2]
        boxes = read_yolo_boxes(label_path, w, h)
        normalized = np.empty_like(boxes)
        if len(boxes):
            normalized[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
            normalized[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
            normalized[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w
            normalized[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h
        labels.append({
            "bboxes": normalized,
            "cls": np.zeros((len(normalized),), dtype=np.float32),
            "bbox_format": "xywh",
            "normalized": True,
            "shape": (h, w),
        })
        images.append(str(image_path))
    if not images:
        raise RuntimeError(f"No annotated images found under {root}")
    return type("LevirDataset", (), {"labels": labels, "im_files": images})()


def make_labels(dataset, index: int) -> dict:
    image = cv2.imread(dataset.im_files[index], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(dataset.im_files[index])
    source = dataset.labels[index]
    h, w = image.shape[:2]
    boxes = np.asarray(source["bboxes"], dtype=np.float32).reshape(-1, 4).copy()
    if len(boxes):
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        xc, yc, bw, bh = boxes.T
        boxes = np.stack((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2), axis=1)
    return {
        "img": image,
        "instances": Instances(
            boxes,
            np.zeros((len(boxes), 0, 2), dtype=np.float32),
            bbox_format="xyxy",
            normalized=False,
        ),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
        "im_file": dataset.im_files[index],
        "image_index": index,
    }


def boxes_from_labels(labels: dict) -> np.ndarray:
    instances = labels["instances"]
    if len(instances) == 0:
        return np.empty((0, 4), dtype=np.float32)
    instances.convert_bbox("xyxy")
    return np.asarray(instances.bboxes, dtype=np.float32).copy()


def choose_indices(dataset) -> tuple[int, int]:
    positive = []
    negative = []
    for index, label in enumerate(dataset.labels):
        if len(label["bboxes"]):
            positive.append(index)
        else:
            negative.append(index)
    if not positive or not negative:
        raise RuntimeError("LevirShip preview needs both positive and negative canvases")
    positive_index = max(positive, key=lambda i: float(np.max(dataset.labels[i]["bboxes"][:, 2] * dataset.labels[i]["bboxes"][:, 3])))
    negative_index = negative[0]
    return positive_index, negative_index


def draw(ax, image: np.ndarray, boxes: np.ndarray, title: str, color: str = "#ef4444") -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.3))
    ax.set_title(title, fontsize=11, weight="bold", pad=5)
    ax.axis("off")


def apply_variant(dataset, labels: dict, variant: str, seed: int) -> dict:
    rng = random.Random(seed)
    if variant == "cp1":
        transform = SmallObjectCopyPaste(
            dataset, p=1.0, unit="single", copies=1, placement="random",
            max_overlap=0.0, allow_empty_target=True, allow_same_source=True, rng=rng,
        )
    elif variant == "cp2":
        transform = SmallObjectCopyPaste(
            dataset, p=1.0, unit="single", copies=2, placement="random",
            max_overlap=0.0, allow_empty_target=True, allow_same_source=True, rng=rng,
        )
    elif variant == "negative_canvas_r2":
        transform = NegativeCanvasCopyPaste(
            dataset, p=1.0, target_policy="deficit", donor_policy="matched",
            target_max_size=20.0, degradation="none", rng=rng, max_trials=100,
        )
    elif variant == "negative_canvas_r4":
        transform = NegativeCanvasCopyPaste(
            dataset, p=1.0, target_policy="deficit", donor_policy="larger",
            target_max_size=20.0, degradation="weak_blur", blur_sigma=0.5,
            rng=rng, max_trials=100,
        )
    else:
        raise ValueError(variant)
    return transform(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/LevirShipData")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/augmentation_examples/levir_cp1_cp2_negative_r2_r4_2samples.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    negative_indices = [
        index for index, label in enumerate(dataset.labels) if not len(label["bboxes"])
    ]
    sample_indices = negative_indices[:2]
    if len(sample_indices) < 2:
        raise RuntimeError("Need at least two negative canvases for the two sample rows")

    columns = [
        ("Original", None),
        ("CP1", "cp1"),
        ("CP2", "cp2"),
        ("NC1 (R2)", "negative_canvas_r2"),
        ("NC2 (R4)", "negative_canvas_r4"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), dpi=180, squeeze=False)
    for row, sample_index in enumerate(sample_indices):
        for col, (display_name, variant) in enumerate(columns):
            original = make_labels(dataset, sample_index)
            result = original if variant is None else apply_variant(
                dataset, make_labels(dataset, sample_index), variant, args.seed + row * 10 + col
            )
            draw(axes[row, col], result["img"], boxes_from_labels(result), display_name if row == 0 else "")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, hspace=0.08, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")
    print("samples:", ", ".join(dataset.im_files[index] for index in sample_indices))


if __name__ == "__main__":
    main()
