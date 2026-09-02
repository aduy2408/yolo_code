"""Make a clean image/GT-output pair with an enlarged, indicated local zoom."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.patches import ConnectionPatch
from PIL import Image
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO
HF_REPO = "duyle2408/levir-ship-yolo-p2"
HF_FILE = "train/yolov8n_p2_baseline_seed42/weights/best.pt"
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_9728_5120.png"
DEFAULT_GT = (310.0, 42.0, 338.0, 67.0)


def draw_gt(ax, box: tuple[float, ...]) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.4))


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
    model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    result = YOLO(model_path).predict(str(args.image), imgsz=512, conf=0.25, verbose=False)[0]
    predictions = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    class_ids = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
    names = result.names
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
    for pred, confidence, class_id in zip(predictions, confidences, class_ids):
        px1, py1, px2, py2 = pred
        ax_gt.add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1, fill=False, color="#00a6ff", linewidth=2.2))
        ax_gt.text(px1, max(4, py1 - 3), f"{names[int(class_id)]} {confidence:.2f}", color="#0069a6", fontsize=8, weight="bold", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
    ax_local_orig.imshow(local, interpolation="nearest")
    ax_local_gt.imshow(local, interpolation="nearest")
    for pred, confidence, class_id in zip(predictions, confidences, class_ids):
        px1, py1, px2, py2 = pred
        ax_local_gt.add_patch(plt.Rectangle((px1 - roi[0], py1 - roi[1]), px2 - px1, py2 - py1, fill=False, color="#00a6ff", linewidth=2.0))
        ax_local_gt.text(px1 - roi[0], max(2, py1 - roi[1] - 2), f"{names[int(class_id)]} {confidence:.2f}", color="#0069a6", fontsize=7, weight="bold", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
    draw_zoom_indicator(ax_orig, roi, ax_local_orig)
    draw_zoom_indicator(ax_gt, roi, ax_local_gt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
