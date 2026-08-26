#!/usr/bin/env python3
"""Create report-ready ground-truth examples for the local YOLO datasets.

Example:
    python misc/plot_dataset_examples.py

The script uses the validation split and deterministically selects images with
roughly low, medium, and high annotation counts. It writes individual figures
and a three-panel contact sheet under docs/reports/dataset_examples/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "varroa": {
        "root": ROOT / "datasets/varroa_yolo",
        "label": "Varroa",
        "class_name": "varroa",
        "targets": (1, 2, 3),
    },
    "levir_ship": {
        "root": ROOT / "datasets/levir_ship_yolo_seed42",
        "label": "LEVIR-Ship",
        "class_name": "ship",
        "targets": (1, 2, 4),
    },
    "tiny_person": {
        "root": ROOT / "datasets/local_test_set/tinyperson_seed_42_corner_sw640_sh512",
        "label": "TinyPerson",
        "class_name": "person",
        "targets": (1, 3, 8),
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        _, xc, yc, bw, bh = map(float, values[:5])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def candidates(root: Path, split: str = "val") -> list[tuple[int, Path, Path]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    rows = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        count = sum(bool(line.strip()) for line in label_path.read_text().splitlines())
        if count:
            rows.append((count, image_path, label_path))
    return rows


def choose_examples(rows: list[tuple[int, Path, Path]], targets: tuple[int, ...]) -> list[tuple[int, Path, Path]]:
    chosen = []
    remaining = rows[:]
    for target in targets:
        item = min(remaining, key=lambda row: (abs(row[0] - target), row[1].name))
        chosen.append(item)
        remaining.remove(item)
    return chosen


def plot_one(item, dataset_label: str, class_name: str, output_path: Path, number: int) -> None:
    count, image_path, label_path = item
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxes = read_boxes(label_path, width, height)

    # Scale the canvas to preserve readability for tiny 160x280 Varroa crops.
    aspect = width / height
    fig, ax = plt.subplots(figsize=(max(5.5, 6.2 * aspect), 6.2), dpi=180)
    ax.imshow(image)
    line_width = max(1.8, min(width, height) / 120)
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#ff3b30", linewidth=line_width))
    ax.set_title(f"{dataset_label} | example {number} | {len(boxes)} {class_name} annotation(s)", fontsize=13, pad=10)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sheet(items, dataset_label: str, class_name: str, output_path: Path) -> None:
    # Put every panel on the same square canvas. This keeps the three columns
    # aligned even though the source datasets have different aspect ratios.
    fig, axes = plt.subplots(1, len(items), figsize=(15, 5), dpi=180, squeeze=False)
    for index, (item, ax) in enumerate(zip(items, axes[0]), start=1):
        count, image_path, label_path = item
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        canvas_size = max(width, height)
        canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
        fitted = ImageOps.contain(image, (canvas_size, canvas_size))
        canvas.paste(fitted, ((canvas_size - fitted.width) // 2, (canvas_size - fitted.height) // 2))
        ax.imshow(canvas)
        offset_x = (canvas_size - fitted.width) / 2
        offset_y = (canvas_size - fitted.height) / 2
        for x1, y1, x2, y2 in read_boxes(label_path, width, height):
            ax.add_patch(Rectangle((x1 + offset_x, y1 + offset_y), x2 - x1, y2 - y1, fill=False, edgecolor="#ff3b30", linewidth=max(1.5, min(width, height) / 140)))
        ax.axis("off")
        ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def zoom_bounds(
    boxes: list[tuple[float, float, float, float]],
    width: int,
    height: int,
    crop_fraction: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Find a fixed-size local window containing the densest GT cluster.

    A fixed window prevents sparse objects spread over an image from making the
    crop expand to the full frame. Candidate centers come from GT centers and a
    small image grid; ties prefer the window containing larger objects.
    """
    crop_width = max(32, min(width - 1, int(width * crop_fraction[0])))
    crop_height = max(32, min(height - 1, int(height * crop_fraction[1])))
    max_left = width - crop_width
    max_top = height - crop_height
    centers = [(0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])) for box in boxes]
    candidates = centers + [
        (width * x, height * y)
        for x in (0.2, 0.4, 0.6, 0.8)
        for y in (0.2, 0.4, 0.6, 0.8)
    ]

    best = None
    for center_x, center_y in candidates:
        left = int(max(0, min(max_left, center_x - crop_width / 2)))
        top = int(max(0, min(max_top, center_y - crop_height / 2)))
        right, bottom = left + crop_width, top + crop_height
        inside = [box for box in boxes if left <= 0.5 * (box[0] + box[2]) <= right and top <= 0.5 * (box[1] + box[3]) <= bottom]
        score = (len(inside), sum((box[2] - box[0]) * (box[3] - box[1]) for box in inside))
        if best is None or score > best[0]:
            best = (score, (left, top, right, bottom))
    return best[1]


def plot_zoom_one(item, output_path: Path, crop_fraction: tuple[float, float]) -> None:
    """Plot the full image with a crop indication next to its enlarged view."""
    _, image_path, label_path = item
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxes = read_boxes(label_path, width, height)
    left, top, right, bottom = zoom_bounds(boxes, width, height, crop_fraction)
    crop = image.crop((left, top, right, bottom))

    fig, (ax_original, ax_zoom) = plt.subplots(
        1, 2, figsize=(11, 5.8), dpi=180, gridspec_kw={"width_ratios": (1.15, 1)}
    )
    box_color = "#ff3b30"
    crop_color = "#ffd60a"
    line_width = max(1.8, min(width, height) / 120)

    ax_original.imshow(image)
    for x1, y1, x2, y2 in boxes:
        ax_original.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=box_color, linewidth=line_width))
    ax_original.add_patch(
        Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor=crop_color, linewidth=line_width * 1.4, linestyle="--")
    )
    ax_original.axis("off")

    ax_zoom.imshow(crop)
    for x1, y1, x2, y2 in boxes:
        center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if left <= center_x <= right and top <= center_y <= bottom:
            ax_zoom.add_patch(Rectangle((x1 - left, y1 - top), x2 - x1, y2 - y1, fill=False, edgecolor=box_color, linewidth=line_width * 1.2))
    ax_zoom.text(0.03, 0.97, "ZOOM", transform=ax_zoom.transAxes, va="top", ha="left", color="#111111", fontsize=11, weight="bold", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2})
    ax_zoom.axis("off")

    # Explicit connectors make the crop-to-zoom relationship clear in a report.
    for source_xy, target_xy in [((right, top), (0, 1)), ((right, bottom), (0, 0))]:
        fig.add_artist(ConnectionPatch(source_xy, target_xy, coordsA=ax_original.transData, coordsB=ax_zoom.transAxes, color=crop_color, linewidth=1.4, alpha=0.9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/reports/dataset_examples")
    args = parser.parse_args()

    names = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, config in names.items():
        rows = candidates(config["root"], args.split)
        if len(rows) < 3:
            raise RuntimeError(f"Need at least three annotated images for {name}, found {len(rows)}")
        items = choose_examples(rows, config["targets"])
        dataset_dir = args.output_dir / name
        for index, item in enumerate(items, start=1):
            plot_one(item, config["label"], config["class_name"], dataset_dir / f"example_{index}.png", index)
        plot_sheet(items, config["label"], config["class_name"], dataset_dir / "contact_sheet.png")
        if name in {"levir_ship", "tiny_person"}:
            zoom_dir = args.output_dir / "zoomed_examples" / name
            crop_fraction = (0.60, 0.60) if name == "levir_ship" else (0.38, 0.48)
            for index, item in enumerate(items, start=1):
                plot_zoom_one(item, zoom_dir / f"example_{index}.png", crop_fraction)
        print(f"{name}: " + ", ".join(f"{count} boxes ({path.name})" for count, path, _ in items))
        print(f"  output: {dataset_dir}")


if __name__ == "__main__":
    main()
