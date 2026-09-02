"""Make a clean image/GT-output pair with an enlarged, indicated local zoom."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_9728_5120.png"
DEFAULT_GTS = ((310.0, 42.0, 338.0, 67.0), (390.0, 28.0, 415.0, 56.0))


def draw_gt(ax, boxes: tuple[tuple[float, ...], ...]) -> None:
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.4))


def draw_zoom_indicator(ax, roi: tuple[float, ...], target_ax) -> None:
    x1, y1, x2, y2 = roi
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#ffd600", linewidth=2.5))
    # Connect the lower-right ROI corner toward the local crop, matching the
    # visual convention of the supplied paper figure.
    ax.annotate("", xy=(1.0, 1.0), xycoords=target_ax.transAxes, xytext=(x2, y2), textcoords=ax.transData, arrowprops=dict(arrowstyle="-", color="#e53935", linewidth=1.6, linestyle="--"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/original_vs_detection_baseline.png")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    boxes = DEFAULT_GTS
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad = max(35, int(max(x2 - x1, y2 - y1) * 2.8))
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
    ax_orig.set_title("(a) Image input", fontsize=13, weight="bold", pad=8)
    ax_gt.imshow(image)
    draw_gt(ax_gt, boxes)
    ax_gt.set_title("(b) Ground-truth output", fontsize=13, weight="bold", pad=8)
    ax_local_orig.imshow(local, interpolation="nearest")
    ax_local_orig.set_title("(c) Local region", fontsize=12, weight="bold", pad=8)
    ax_local_gt.imshow(local, interpolation="nearest")
    draw_gt(ax_local_gt, tuple((b[0] - roi[0], b[1] - roi[1], b[2] - roi[0], b[3] - roi[1]) for b in boxes))
    ax_local_gt.set_title("(d) Local region with GT", fontsize=12, weight="bold", pad=8)
    draw_zoom_indicator(ax_orig, roi, ax_local_orig)
    draw_zoom_indicator(ax_gt, roi, ax_local_gt)

    fig.suptitle("Ground-truth localization and enlarged local region", fontsize=16, weight="bold", y=0.98)
    fig.text(0.5, 0.015, "The yellow ROI on each full image indicates the enlarged region below; no ROI box is drawn inside the zoom panels.", ha="center", fontsize=9, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
