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
from matplotlib.patches import ConnectionPatch, Rectangle
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


def best_object_item(items: list[Item], minimum_objects: int = 1) -> Item:
    """Choose the image with the largest annotated object for a readable figure."""
    candidates = [item for item in items if len(item.boxes) >= minimum_objects]
    if not candidates:
        candidates = items
    return max(candidates, key=lambda item: max(
        float((box[2] - box[0]) * (box[3] - box[1])) for box in item.boxes
    ))


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


def zoom_group(boxes: np.ndarray) -> tuple[int, ...]:
    """Select one object, or a close pair, following the previous report logic."""
    if len(boxes) == 0:
        return ()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    candidates: list[tuple[tuple[int, float, float], tuple[int, ...]]] = []
    for index, box in enumerate(boxes):
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        candidates.append(((1, 0.0, area), (index,)))
        for other in range(index + 1, len(boxes)):
            other_box = boxes[other]
            distance = float(np.linalg.norm(centers[index] - centers[other]))
            scale = max(box[2] - box[0], box[3] - box[1], other_box[2] - other_box[0], other_box[3] - other_box[1])
            if distance / max(float(scale), 1.0) <= 2.0:
                pair_area = float((box[2] - box[0]) * (box[3] - box[1]) + (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]))
                candidates.append(((2, -distance / max(float(scale), 1.0), pair_area), (index, other)))
    return max(candidates, key=lambda candidate: candidate[0])[1]


def zoom_bounds(boxes: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Return a tight padded crop around one object or a close pair."""
    selected = boxes[list(zoom_group(boxes))]
    x1, y1 = selected[:, :2].min(axis=0)
    x2, y2 = selected[:, 2:].max(axis=0)
    padding = max(10.0, 1.25 * max(x2 - x1, y2 - y1))
    return clip_box(np.array([x1 - padding, y1 - padding, x2 + padding, y2 + padding]), width, height)


def zoom_view(image: np.ndarray, boxes: np.ndarray, margin: float = 0.34) -> tuple[np.ndarray, np.ndarray]:
    """Crop and enlarge the sampled object zone used in the earlier plots."""
    if boxes is None or len(boxes) == 0:
        return image, boxes
    height, width = image.shape[:2]
    selected = boxes[list(zoom_group(boxes))]
    left, top, right, bottom = zoom_bounds(selected, width, height)
    view = image[top:bottom, left:right]
    transformed = selected.copy()
    transformed[:, [0, 2]] -= left
    transformed[:, [1, 3]] -= top
    return view, transformed


def show(ax, image: np.ndarray, boxes: np.ndarray | None = None, title: str = "", box_color: str = "#20a464", dashed: bool = False, zoom: bool = True) -> None:
    if zoom and boxes is not None and len(boxes):
        image, boxes = zoom_view(image, boxes)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if boxes is not None:
        add_boxes(ax, boxes, box_color, dashed)
    ax.set_title(title, fontsize=10.5, weight="bold", pad=5)
    ax.axis("off")


def arrow(fig, left_ax, right_ax, y: float, color: str = "#64748b") -> None:
    a, b = left_ax.get_position(), right_ax.get_position()
    fig.add_artist(FancyArrowPatch((a.x1 + 0.006, y), (b.x0 - 0.006, y), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=15, linewidth=2.0, color=color))


def zoom_pair(fig, ax_full, ax_zoom, image: np.ndarray, boxes: np.ndarray, title: str) -> None:
    """Render a contextual object view, sampled crop zone, and linked zoom view."""
    selected_indices = zoom_group(boxes)
    left, top, right, bottom = zoom_bounds(boxes, image.shape[1], image.shape[0])
    crop = image[top:bottom, left:right]
    selected = boxes[list(selected_indices)]

    # The source tiles are 512x512 while ships can be only a few dozen pixels.
    # Show a contextual crop on the left, then the tighter sampled crop on the
    # right. This is the same visual relationship as the earlier report figure.
    x1, y1 = selected[:, :2].min(axis=0)
    x2, y2 = selected[:, 2:].max(axis=0)
    context_pad = max(2.0 * float(max(x2 - x1, y2 - y1)), 24.0)
    context_left, context_top, context_right, context_bottom = clip_box(
        np.array([x1 - context_pad, y1 - context_pad, x2 + context_pad, y2 + context_pad]),
        image.shape[1], image.shape[0]
    )
    context = image[context_top:context_bottom, context_left:context_right]
    crop_left, crop_top = left - context_left, top - context_top

    ax_full.imshow(cv2.cvtColor(context, cv2.COLOR_BGR2RGB), interpolation="nearest")
    for x1, y1, x2, y2 in boxes:
        if x2 < context_left or x1 > context_right or y2 < context_top or y1 > context_bottom:
            continue
        ax_full.add_patch(Rectangle((x1 - context_left, y1 - context_top), x2 - x1, y2 - y1,
                                    fill=False, edgecolor="#ff3b30", linewidth=1.5))
    ax_full.add_patch(Rectangle((crop_left, crop_top), right - left, bottom - top, fill=False,
                                edgecolor="#ffd60a", linewidth=2.2, linestyle="--"))
    ax_full.set_title(title, fontsize=10.5, weight="bold", pad=5)
    ax_full.axis("off")

    ax_zoom.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), interpolation="nearest")
    for x1, y1, x2, y2 in selected:
        ax_zoom.add_patch(Rectangle((x1 - left, y1 - top), x2 - x1, y2 - y1,
                                    fill=False, edgecolor="#ff3b30", linewidth=1.8))
    ax_zoom.text(0.03, 0.97, "ZOOM", transform=ax_zoom.transAxes, va="top", ha="left",
                 fontsize=10, weight="bold", color="#111111",
                 bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2})
    ax_zoom.axis("off")
    fig.add_artist(ConnectionPatch((crop_left + right - left, crop_top), (0, 1), coordsA=ax_full.transData,
                                   coordsB=ax_zoom.transAxes, color="#ffd60a", linewidth=1.2))
    fig.add_artist(ConnectionPatch((crop_left + right - left, crop_top + bottom - top), (0, 0), coordsA=ax_full.transData,
                                   coordsB=ax_zoom.transAxes, color="#ffd60a", linewidth=1.2))


def make_figure(items: list[Item], seed: int) -> plt.Figure:
    rng = random.Random(seed)
    # Use the same deterministic crop/zoom idea as the old report, but prefer
    # the largest annotated ships so the resulting panels are actually legible.
    base = best_object_item(items, 2)
    single = best_object_item(items, 1)
    cluster = best_object_item(items, 3)
    large = best_object_item(items, 1)

    fig = plt.figure(figsize=(17, 16), dpi=180, facecolor="white")
    gs = fig.add_gridspec(5, 5, width_ratios=[1.10, 0.88, 1.22, 1.22, 0.88],
                          hspace=0.82, wspace=0.24, left=0.035, right=0.985, top=0.91, bottom=0.06)
    colors = ["#2563eb", "#15803d", "#c2410c", "#7c3aed", "#be123c"]
    titles = [
        ("A", "Positive injection", "real object → valid positive"),
        ("B", "Structure-preserving injection", "local cluster stays together"),
        ("C", "Geometry-conditioned injection", "controlled overlap / crowding"),
        ("D", "Distribution-conditioned injection", "match the target scale"),
        ("E", "Hard-negative injection", "visible patch, no new label"),
    ]

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []

    # A: one object, then two copies.
    target, target_boxes = target_with_boxes(base)
    patch, alpha, _ = object_patch(single)
    after = target.copy(); h, w = after.shape[:2]; pasted = []
    for x, y in [(int(w * .62), int(h * .16)), (int(w * .70), int(h * .62))]:
        after, placed = paste(after, patch, alpha, x, y, scale=.82); pasted.append(placed)
    rows.append((patch, target, target_boxes, after, np.vstack([target_boxes, np.asarray(pasted, np.float32)]), "source crop"))

    # B: cluster patch.
    target, target_boxes = target_with_boxes(base)
    patch, _ = cluster_patch(cluster)
    after = target.copy(); h, w = after.shape[:2]
    after, placed = paste(after, patch, np.full(patch.shape[:2], 210, np.uint8), int(w * .53), int(h * .46), scale=.48)
    rows.append((patch, target, target_boxes, after, np.vstack([target_boxes, np.asarray([placed], np.float32)]), "source cluster"))

    # C: two controlled crowding placements.
    target, target_boxes = target_with_boxes(base)
    patch, alpha, _ = object_patch(single)
    after = target.copy(); h, w = after.shape[:2]
    after, mild = paste(after, patch, alpha, int(w * .50), int(h * .18), scale=.9)
    after, moderate = paste(after, patch, alpha, int(w * .58), int(h * .24), scale=.9)
    rows.append((patch, target, target_boxes, after, np.vstack([target_boxes, np.asarray([mild, moderate], np.float32)]), "source object"))

    # D: scale-matched placement.
    target, target_boxes = target_with_boxes(base)
    patch, alpha, _ = object_patch(large); scale = .42
    after = target.copy(); h, w = after.shape[:2]
    after, placed = paste(after, patch, alpha, int(w * .66), int(h * .57), scale=scale)
    resized = cv2.resize(patch, (max(2, int(patch.shape[1] * scale)), max(2, int(patch.shape[0] * scale))))
    rows.append((resized, target, target_boxes, after, np.vstack([target_boxes, np.asarray([placed], np.float32)]), "resized source"))

    # E: blurred hard-negative, intentionally omitted from the label boxes.
    target, target_boxes = target_with_boxes(base)
    patch, _, _ = object_patch(single)
    negative = cv2.GaussianBlur(patch, (0, 0), sigmaX=max(2, min(patch.shape[:2]) / 8))
    after = target.copy(); h, w = after.shape[:2]
    after, _ = paste(after, negative, np.full(negative.shape[:2], 210, np.uint8), int(w * .63), int(h * .18), scale=.72)
    rows.append((negative, target, target_boxes, after, target_boxes, "negative patch"))

    for row, (letter, title, subtitle) in enumerate(titles):
        ax_label = fig.add_subplot(gs[row, 0]); ax_label.axis("off")
        ax_label.text(0, .82, f"{letter}. {title}", fontsize=10.8, weight="bold", color=colors[row], va="top", wrap=True)
        ax_label.text(0, .54, subtitle, fontsize=8.8, color="#475569", va="top", wrap=True)
        patch, before, before_boxes, after, after_boxes, patch_title = rows[row]
        show(ax_label.figure.add_subplot(gs[row, 1]), patch, None, patch_title, colors[row], zoom=False)
        ax_full = fig.add_subplot(gs[row, 2])
        show(ax_full, before, before_boxes, "target before", "#20a464", zoom=False)
        ax_after = fig.add_subplot(gs[row, 3])
        ax_zoom = fig.add_subplot(gs[row, 4])
        zoom_pair(fig, ax_after, ax_zoom, after, after_boxes, "after / sampled zone")

    fig.suptitle("Copy–Paste augmentation: sample the object zone, then zoom it", fontsize=22, weight="bold", color="#0f172a", y=.965)
    fig.text(.5, .938, "Yellow dashed box = the sampled crop region; red = object annotations / pasted region; green = target ground truth.",
             ha="center", fontsize=10.8, color="#475569")
    fig.text(.5, .025, "The crop-and-zoom view follows the existing zoom_group / zoom_bounds visualization code.",
             ha="center", fontsize=10.5, weight="bold", color="#334155")
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
