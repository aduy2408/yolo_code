#!/usr/bin/env python3
"""Render two-section donor provenance figures for Copy-Paste variants.

Pic 1: donor image with red bbox and a linked zoom crop beside it.
Pic 2: target original beside the augmented target after insertion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

from misc.plot_cp_donor_provenance import image_for, transform_for
from misc.plot_cp_variants_4x2 import load_dataset, make_labels, boxes_from_labels

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    ("CP1", "cp1"),
    ("CP2", "cp2"),
    ("Negative Canvas R2", "negative_canvas_r2"),
    ("Negative Canvas R4", "negative_canvas_r4"),
]


def crop_with_margin(image: np.ndarray, box: tuple[float, float, float, float], margin: float = 1.2):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max((x2 - x1) * margin, 5.0)
    pad_y = max((y2 - y1) * margin, 5.0)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(w, int(np.ceil(x2 + pad_x)))
    bottom = min(h, int(np.ceil(y2 + pad_y)))
    return image[top:bottom, left:right].copy(), (left, top, right, bottom)


def draw_image(ax, image: np.ndarray, title: str = "", boxes=None, linewidth: float = 2.0):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#ff3b30", linewidth=linewidth))
    if title:
        ax.set_title(title, fontsize=12, weight="bold", pad=6)
    ax.axis("off")


def section_one(dataset, target_index: int, seed: int, output: Path):
    fig, axes = plt.subplots(4, 2, figsize=(12, 15), dpi=180, squeeze=False)
    axes[0, 0].set_title("Donor image with selected object", fontsize=13, weight="bold", pad=8)
    axes[0, 1].set_title("Zoom of red-boxed donor object", fontsize=13, weight="bold", pad=8)

    for row, (name, variant) in enumerate(VARIANTS):
        transform = transform_for(dataset, variant, seed + row)
        result = transform(make_labels(dataset, target_index))
        if not transform.donor_records:
            raise RuntimeError(f"{variant} did not select a donor")
        record = transform.donor_records[-1]
        donor_path = Path(dataset.im_files[record.image_index])
        donor_image = image_for(donor_path)
        donor_box = tuple(float(value) for value in record.bbox_xyxy)
        crop, crop_box = crop_with_margin(donor_image, donor_box)
        local_box = np.asarray([[donor_box[0] - crop_box[0], donor_box[1] - crop_box[1],
                                donor_box[2] - crop_box[0], donor_box[3] - crop_box[1]]], dtype=np.float32)

        axes[row, 0].set_ylabel(name, fontsize=11, weight="bold", rotation=90, labelpad=14)
        draw_image(axes[row, 0], donor_image, boxes=[donor_box], title="" if row else "")
        draw_image(axes[row, 1], crop, boxes=local_box, title="" if row else "", linewidth=2.3)
        axes[row, 0].text(0.02, 0.02, donor_path.name[:43], transform=axes[row, 0].transAxes,
                           fontsize=7.4, color="white", va="bottom", ha="left",
                           bbox={"facecolor": "black", "alpha": 0.7, "pad": 2})
        axes[row, 1].text(0.02, 0.02, "red box = donor object", transform=axes[row, 1].transAxes,
                           fontsize=8, color="white", va="bottom", ha="left",
                           bbox={"facecolor": "black", "alpha": 0.7, "pad": 2})
        left, top, right, bottom = axes[row, 0].get_position().x0, axes[row, 0].get_position().y0, axes[row, 0].get_position().x1, axes[row, 0].get_position().y1
        _ = (left, top, right, bottom)
        fig.add_artist(ConnectionPatch(
            xyA=(donor_box[2], donor_box[1]), coordsA=axes[row, 0].transData,
            xyB=(0, 1), coordsB=axes[row, 1].transAxes,
            arrowstyle="-", linewidth=1.2, linestyle="--", color="#ff3b30", alpha=0.85,
        ))
        fig.add_artist(ConnectionPatch(
            xyA=(donor_box[2], donor_box[3]), coordsA=axes[row, 0].transData,
            xyB=(0, 0), coordsB=axes[row, 1].transAxes,
            arrowstyle="-", linewidth=1.2, linestyle="--", color="#ff3b30", alpha=0.85,
        ))

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.02, hspace=0.10, wspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def section_two(dataset, target_index: int, seed: int, output: Path):
    target_path = Path(dataset.im_files[target_index])
    target_image = image_for(target_path)
    fig, axes = plt.subplots(4, 2, figsize=(12, 15), dpi=180, squeeze=False)
    axes[0, 0].set_title("Original target", fontsize=13, weight="bold", pad=8)
    axes[0, 1].set_title("After donor insertion", fontsize=13, weight="bold", pad=8)

    for row, (name, variant) in enumerate(VARIANTS):
        transform = transform_for(dataset, variant, seed + row)
        result = transform(make_labels(dataset, target_index))
        output_boxes = boxes_from_labels(result)
        axes[row, 0].set_ylabel(name, fontsize=11, weight="bold", rotation=90, labelpad=14)
        draw_image(axes[row, 0], target_image)
        draw_image(axes[row, 1], result["img"], boxes=output_boxes)
        axes[row, 0].text(0.02, 0.02, target_path.name[:43], transform=axes[row, 0].transAxes,
                           fontsize=7.4, color="white", va="bottom", ha="left",
                           bbox={"facecolor": "black", "alpha": 0.7, "pad": 2})
        axes[row, 1].text(0.02, 0.02, f"{len(output_boxes)} output label(s)", transform=axes[row, 1].transAxes,
                           fontsize=8, color="white", va="bottom", ha="left",
                           bbox={"facecolor": "black", "alpha": 0.7, "pad": 2})

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.02, hspace=0.10, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def combined(pic1: Path, pic2: Path, output: Path):
    first = cv2.imread(str(pic1), cv2.IMREAD_COLOR)
    second = cv2.imread(str(pic2), cv2.IMREAD_COLOR)
    if first is None or second is None:
        raise RuntimeError("Could not reload section figures")
    width = max(first.shape[1], second.shape[1])
    canvas = np.full((first.shape[0] + second.shape[0] + 32, width, 3), 255, dtype=np.uint8)
    canvas[: first.shape[0], : first.shape[1]] = first
    canvas[first.shape[0] + 32:, : second.shape[1]] = second
    cv2.putText(canvas, "SECTION 1 / PIC 1: DONOR SOURCE AND ZOOM", (24, first.shape[0] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/LevirShipData")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/reports/augmentation_examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    negative_indices = [i for i, label in enumerate(dataset.labels) if not len(label["bboxes"])]
    if not negative_indices:
        raise RuntimeError("Need a negative target canvas")
    target_index = negative_indices[0]
    pic1 = args.output_dir / "levir_cp_section1_donor_zoom.png"
    pic2 = args.output_dir / "levir_cp_section2_original_after.png"
    combo = args.output_dir / "levir_cp_two_sections_combined.png"
    section_one(dataset, target_index, args.seed, pic1)
    section_two(dataset, target_index, args.seed, pic2)
    combined(pic1, pic2, combo)
    print(f"Saved {pic1}")
    print(f"Saved {pic2}")
    print(f"Saved {combo}")


if __name__ == "__main__":
    main()
