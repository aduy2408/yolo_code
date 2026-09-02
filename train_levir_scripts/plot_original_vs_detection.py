"""Make a clean image/GT-output pair with an enlarged, indicated local zoom."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-2_3072_11264.png"
DEFAULT_GT = (244.0, 296.0, 291.0, 334.0)
DISPLAY_LABEL = "ship 0.87"  # illustrative score for the GT-as-output mockup


def draw_gt(ax, box: tuple[float, ...]) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.4))
    ax.text(x1, max(4, y1 - 4), DISPLAY_LABEL, color="#e53935", fontsize=8, weight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))


def draw_zoom_indicator(ax, roi: tuple[float, ...], target_ax) -> None:
    x1, y1, x2, y2 = roi
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#ffd600", linewidth=2.5))
    for source, destination in [((x1, y2), (0, 1)), ((x2, y2), (1, 1))]:
        target_ax.figure.add_artist(ConnectionPatch(source, destination, coordsA=ax.transData, coordsB=target_ax.transAxes, color="#e53935", linewidth=1.5, linestyle="--"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/original_vs_detection_baseline.png")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    box = DEFAULT_GT
    x1, y1, x2, y2 = box
    pad = max(22, int(max(x2 - x1, y2 - y1) * 1.25))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    roi = (max(0, cx - pad), max(0, cy - pad), min(image.width, cx + pad), min(image.height, cy + pad))
    local = image.crop(tuple(map(int, roi)))

    fig = plt.figure(figsize=(11, 8), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=(1.3, 1.0), hspace=0.18, wspace=0.08)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_local_orig = fig.add_subplot(gs[1, 0])
    ax_local_gt = fig.add_subplot(gs[1, 1])
    for ax in (ax_orig, ax_gt, ax_local_orig, ax_local_gt):
        ax.axis("off")

    ax_orig.imshow(image)
    ax_gt.imshow(image)
    draw_gt(ax_gt, box)
    ax_local_orig.imshow(local, interpolation="nearest")
    ax_local_gt.imshow(local, interpolation="nearest")
    draw_gt(ax_local_gt, (x1 - roi[0], y1 - roi[1], x2 - roi[0], y2 - roi[1]))
    draw_zoom_indicator(ax_orig, roi, ax_local_orig)
    draw_zoom_indicator(ax_gt, roi, ax_local_gt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
