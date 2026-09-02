"""Plot one LEVIR image beside YOLOv8-P2 detections and the missed GT target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO  # noqa: E402

HF_REPO = "duyle2408/levir-ship-yolo-p2"
HF_FILE = "train/yolov8n_p2_baseline_seed42/weights/best.pt"
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_6144_17285.png"
DEFAULT_GT = (444.0, 254.0, 457.0, 269.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--gt-box", type=float, nargs=4, default=DEFAULT_GT, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/original_vs_detection_baseline.png")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    print(f"Downloading baseline from hf://datasets/{HF_REPO}/{HF_FILE}", flush=True)
    model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    model = YOLO(model_path)
    result = model.predict(str(args.image), imgsz=512, conf=args.conf, verbose=False)[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), facecolor="white")
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(image)
    x1, y1, x2, y2 = args.gt_box
    axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.5))
    axes[0].text(x1, max(4, y1 - 5), "GT small target", color="#e53935", fontsize=10, weight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    axes[0].set_title("(a) Original image", fontsize=13, weight="bold", pad=8)

    axes[1].imshow(image)
    detections = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    for box, confidence in zip(detections, confidences):
        dx1, dy1, dx2, dy2 = box
        axes[1].add_patch(plt.Rectangle((dx1, dy1), dx2 - dx1, dy2 - dy1, fill=False, color="#00a6ff", linewidth=2))
        axes[1].text(dx1, max(4, dy1 - 4), f"{confidence:.2f}", color="#0069a6", fontsize=9, weight="bold", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
    axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.5, linestyle="--"))
    axes[1].set_title(f"(b) Baseline detections ({len(detections)} boxes)", fontsize=13, weight="bold", pad=8)

    fig.suptitle("Original image versus baseline detection", fontsize=16, weight="bold", y=0.98)
    fig.text(0.5, 0.015, f"YOLOv8n-P2 baseline downloaded from Hugging Face · blue = prediction · red dashed = missed GT target · confidence threshold = {args.conf:.2f}", ha="center", fontsize=9, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"detections: {len(detections)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
