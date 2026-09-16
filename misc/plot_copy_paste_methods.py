#!/usr/bin/env python3
"""Create a clean five-panel Copy-Paste augmentation illustration.

The figure is designed for papers/slides rather than debugging. It uses real
LEVIR-Ship images and YOLO boxes when available, then renders five hypotheses:
positive injection, cluster injection, controlled crowding, scale matching, and
hard-negative patches.

Example:
    conda run -n ml2 python misc/plot_copy_paste_methods.py \
        --dataset LevirShipData \
        --output docs/reports/augmentation_examples/copy_paste_methods.png
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Item:
    image: Path
    label: Path
    boxes: np.ndarray


def read_boxes(path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            _, xc, yc, bw, bh = map(float, values[:5])
            x1, y1 = (xc - bw / 2) * width, (yc - bh / 2) * height
            x2, y2 = (xc + bw / 2) * width, (yc + bh / 2) * height
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def load_items(dataset: Path) -> list[Item]:
    image_dir = dataset / "images"
    label_dir = dataset / "labels"
    if not image_dir.is_dir():
        image_dir, label_dir = dataset / "All Images", dataset / "All Annotations"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {dataset}/All Images and All Annotations")

    items = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or not label_path.exists():
            continue
        h, w = image.shape[:2]
        boxes = read_boxes(label_path, w, h)
        if len(boxes):
            items.append(Item(image_path, label_path, boxes))
    if len(items) < 4:
        raise RuntimeError(f"Need at least four annotated images, found {len(items)}")
    return items


def read_image(item: Item) -> np.ndarray:
    image = cv2.imread(str(item.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {item.image}")
    return image


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))


def padded_crop(image: np.ndarray, box: np.ndarray, padding: float = 0.12) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    crop_box = clip_box(np.array([x1 - dx, y1 - dy, x2 + dx, y2 + dy]), w, h)
    a, b, c, d = crop_box
    return image[b:d, a:c].copy(), crop_box


def alpha_mask(crop: np.ndarray) -> np.ndarray:
    """Estimate a soft object mask, with a safe rectangle fallback."""
    h, w = crop.shape[:2]
    if min(h, w) < 12:
        return np.full((h, w), 255, dtype=np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[:] = cv2.GC_BGD
    inset_x, inset_y = max(2, w // 14), max(2, h // 14)
    rect = (inset_x, inset_y, max(1, w - 2 * inset_x), max(1, h - 2 * inset_y))
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if int(alpha.sum()) < h * w * 0.04:
            alpha[:] = 255
    except cv2.error:
        alpha = np.full((h, w), 255, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2)
    return alpha


def paste(target: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x: int, y: int, scale: float = 1.0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if scale != 1.0:
        h, w = patch.shape[:2]
        nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
    h, w = patch.shape[:2]
    H, W = target.shape[:2]
    x, y = max(0, min(W - 1, x)), max(0, min(H - 1, y))
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x or y2 <= y:
        return target, (x, y, x, y)
    patch, alpha = patch[: y2 - y, : x2 - x], alpha[: y2 - y, : x2 - x]
    a = alpha.astype(np.float32)[..., None] / 255.0
    target[y:y2, x:x2] = (patch.astype(np.float32) * a + target[y:y2, x:x2].astype(np.float32) * (1 - a)).astype(np.uint8)
    return target, (x, y, x2, y2)


def choose(items: list[Item], predicate, rng: random.Random, fallback: int = 0) -> Item:
    candidates = [item for item in items if predicate(item)]
    return rng.choice(candidates or [items[fallback % len(items)]])


def object_patch(item: Item, index: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image = read_image(item)
    box = item.boxes[min(index, len(item.boxes) - 1)]
    patch, crop_box = padded_crop(image, box, 0.18)
    return patch, alpha_mask(patch), crop_box


def cluster_patch(item: Item) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = read_image(item)
    boxes = item.boxes[: min(4, len(item.boxes))]
    x1, y1 = boxes[:, :2].min(axis=0)
    x2, y2 = boxes[:, 2:].max(axis=0)
    patch, crop_box = padded_crop(image, np.array([x1, y1, x2, y2]), 0.16)
    return patch, crop_box


def target_with_boxes(item: Item) -> tuple[np.ndarray, np.ndarray]:
    return read_image(item), item.boxes.copy()


def add_boxes(ax, boxes: np.ndarray, color: str = "#20a464", dashed: bool = False, linewidth: float = 1.5) -> None:
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
                               linewidth=linewidth, linestyle="--" if dashed else "-"))


def show(ax, image: np.ndarray, boxes: np.ndarray | None = None, title: str = "", box_color: str = "#20a464", dashed: bool = False) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if boxes is not None:
        add_boxes(ax, boxes, box_color, dashed)
    ax.set_title(title, fontsize=10.5, weight="bold", pad=5)
    ax.axis("off")


def arrow(fig, left_ax, right_ax, y: float, color: str = "#64748b") -> None:
    a, b = left_ax.get_position(), right_ax.get_position()
    fig.add_artist(FancyArrowPatch((a.x1 + 0.006, y), (b.x0 - 0.006, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=15, linewidth=2.0, color=color))


def make_figure(items: list[Item], seed: int) -> plt.Figure:
    rng = random.Random(seed)
    base = choose(items, lambda x: len(x.boxes) >= 2, rng)
    single = choose(items, lambda x: len(x.boxes) >= 1, rng, 1)
    cluster = choose(items, lambda x: len(x.boxes) >= 3, rng, 2)
    large = choose(items, lambda x: max((b[2] - b[0]) * (b[3] - b[1]) for b in x.boxes) > 0, rng, 3)

    fig = plt.figure(figsize=(15.5, 15.5), dpi=180, facecolor="#f8fafc")
    gs = fig.add_gridspec(5, 4, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.72, wspace=0.28,
                          left=0.045, right=0.985, top=0.895, bottom=0.08)
    colors = ["#2563eb", "#15803d", "#c2410c", "#7c3aed", "#be123c"]
    titles = [
        ("A", "Positive object injection", "Paste one or more real objects; add labels"),
        ("B", "Structure-preserving injection", "Paste a local cluster; keep its relative layout"),
        ("C", "Geometry-conditioned injection", "Paste with controlled overlap to model crowding"),
        ("D", "Distribution-conditioned injection", "Match the target scale before pasting"),
        ("E", "Hard-negative injection", "Paste confusing patches without adding labels"),
    ]

    # A: one object, then two copies.
    target, boxes = target_with_boxes(base)
    patch, alpha, _ = object_patch(single)
    after = target.copy()
    h, w = after.shape[:2]
    pasted = []
    for x, y in [(int(w * 0.62), int(h * 0.16)), (int(w * 0.70), int(h * 0.62))]:
        after, placed = paste(after, patch, alpha, x, y, scale=0.82)
        pasted.append(placed)
    row = 0
    show(fig.add_subplot(gs[row, 0]), patch, None, "source crop", "#15803d")
    show(fig.add_subplot(gs[row, 1]), target, boxes, "target before", "#15803d")
    show(fig.add_subplot(gs[row, 2]), after, np.vstack([boxes, np.asarray(pasted, dtype=np.float32)]), "after: +2 objects", "#ef4444")
    ax_note = fig.add_subplot(gs[row, 3]); ax_note.axis("off")
    ax_note.text(0, .72, "new objects are\nvalid positives", fontsize=12, weight="bold", color=colors[row], va="top")
    ax_note.text(0, .33, "single-object unit\nrandom placement", fontsize=10, color="#475569", va="top")

    # B: cluster patch.
    target, boxes = target_with_boxes(base)
    patch, crop_box = cluster_patch(cluster)
    after = target.copy(); h, w = after.shape[:2]
    placed = (int(w * .53), int(h * .46), int(w * .53) + patch.shape[1], int(h * .46) + patch.shape[0])
    after, placed = paste(after, patch, np.full(patch.shape[:2], 210, np.uint8), placed[0], placed[1], scale=.48)
    row = 1
    show(fig.add_subplot(gs[row, 0]), read_image(cluster), np.asarray([crop_box], np.float32), "source cluster", "#15803d", True)
    show(fig.add_subplot(gs[row, 1]), target, boxes, "target before", "#15803d")
    show(fig.add_subplot(gs[row, 2]), after, np.vstack([boxes, np.asarray([placed], dtype=np.float32)]), "after: cluster paste", "#ef4444")
    ax_note = fig.add_subplot(gs[row, 3]); ax_note.axis("off")
    ax_note.text(0, .72, "local structure\nis preserved", fontsize=12, weight="bold", color=colors[row], va="top")
    ax_note.text(0, .33, "relative spacing and\norientation stay intact", fontsize=10, color="#475569", va="top")

    # C: mild and moderate overlap in one clean row.
    target, boxes = target_with_boxes(base)
    patch, alpha, _ = object_patch(single)
    after = target.copy(); h, w = after.shape[:2]
    p1 = (int(w * .50), int(h * .18)); p2 = (int(w * .58), int(h * .24))
    after, mild = paste(after, patch, alpha, *p1, scale=.9)
    after, moderate = paste(after, patch, alpha, *p2, scale=.9)
    row = 2
    show(fig.add_subplot(gs[row, 0]), target, boxes, "target before", "#15803d")
    show(fig.add_subplot(gs[row, 1]), after, np.vstack([boxes, np.asarray([mild, moderate], np.float32)]), "after: controlled overlap", "#ef4444")
    ax = fig.add_subplot(gs[row, 2]); ax.axis("off")
    ax.text(.02, .76, "mild", fontsize=12, weight="bold", color="#c2410c")
    ax.text(.02, .58, "10–30% overlap", fontsize=11, color="#475569")
    ax.text(.02, .32, "moderate", fontsize=12, weight="bold", color="#c2410c")
    ax.text(.02, .14, "20–40% overlap", fontsize=11, color="#475569")
    ax_note = fig.add_subplot(gs[row, 3]); ax_note.axis("off")
    ax_note.text(0, .72, "crowding is\nintentional", fontsize=12, weight="bold", color=colors[row], va="top")
    ax_note.text(0, .33, "visibility constraint\nprevents unrealistic paste", fontsize=10, color="#475569", va="top")

    # D: scale matched. Use a deliberately smaller pasted object.
    target, boxes = target_with_boxes(base)
    patch, alpha, _ = object_patch(large)
    after = target.copy(); h, w = after.shape[:2]
    scale = .42
    after, placed = paste(after, patch, alpha, int(w * .66), int(h * .57), scale=scale)
    row = 3
    show(fig.add_subplot(gs[row, 0]), patch, None, "source: larger object", "#15803d")
    show(fig.add_subplot(gs[row, 1]), cv2.resize(patch, (max(2, int(patch.shape[1]*scale)), max(2, int(patch.shape[0]*scale)))), None, "resized to target scale", "#15803d")
    show(fig.add_subplot(gs[row, 2]), after, np.vstack([boxes, np.asarray([placed], np.float32)]), "after: scale-matched", "#ef4444")
    ax_note = fig.add_subplot(gs[row, 3]); ax_note.axis("off")
    ax_note.text(0, .72, "size follows\nthe target scene", fontsize=12, weight="bold", color=colors[row], va="top")
    ax_note.text(0, .33, "avoids oversized\nsynthetic ships", fontsize=10, color="#475569", va="top")

    # E: blurred hard-negative patch, no new box.
    target, boxes = target_with_boxes(base)
    patch, _, _ = object_patch(single)
    negative = cv2.GaussianBlur(patch, (0, 0), sigmaX=max(2, min(patch.shape[:2]) / 8))
    after = target.copy(); h, w = after.shape[:2]
    after, placed = paste(after, negative, np.full(negative.shape[:2], 210, np.uint8), int(w * .63), int(h * .18), scale=.72)
    row = 4
    show(fig.add_subplot(gs[row, 0]), negative, None, "hard-negative patch", "#be123c")
    show(fig.add_subplot(gs[row, 1]), target, boxes, "target before", "#15803d")
    show(fig.add_subplot(gs[row, 2]), after, boxes, "after: no new label", "#be123c")
    ax_note = fig.add_subplot(gs[row, 3]); ax_note.axis("off")
    ax_note.text(0, .72, "background\nconfuser", fontsize=12, weight="bold", color=colors[row], va="top")
    ax_note.text(0, .33, "red patch is visible,\nbut no GT box is added", fontsize=10, color="#475569", va="top")

    # Row labels and arrows are added after axes exist.
    for row, (letter, title, subtitle) in enumerate(titles):
        y = 0.905 - row * 0.164
        fig.text(.048, y, f"{letter}. {title}", fontsize=15, weight="bold", color=colors[row], va="top")
        fig.text(.27, y - .002, subtitle, fontsize=10.5, color="#475569", va="top")
    fig.suptitle("Copy–Paste augmentation: five controlled hypotheses", fontsize=23, weight="bold", color="#0f172a", y=.965)
    fig.text(.5, .94, "Illustrations use real LEVIR-Ship imagery; green = original ground truth, red = pasted region.",
             ha="center", fontsize=11.5, color="#475569")
    fig.text(.5, .032, "The augmentation changes the training distribution, not the detector architecture.",
             ha="center", fontsize=11, weight="bold", color="#334155")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/augmentation_examples/copy_paste_methods.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    items = load_items(args.dataset)
    fig = make_figure(items, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {args.output} using {len(items)} annotated images")


if __name__ == "__main__":
    main()
