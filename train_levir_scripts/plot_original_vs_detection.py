"""Plot one LEVIR image beside YOLOv8-P2 detections and the missed GT target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO  # noqa: E402

HF_REPO = "duyle2408/levir-ship-yolo-p2"
HF_FILE = "train/yolov8n_p2_baseline_seed42/weights/best.pt"
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_9728_5120.png"
DEFAULT_GTS = ((310.0, 42.0, 338.0, 67.0), (390.0, 28.0, 415.0, 56.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--gt-box", type=float, nargs=4, action="append", default=None, metavar=("X1", "Y1", "X2", "Y2"))
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
    gt_boxes = args.gt_box or DEFAULT_GTS
    for x1, y1, x2, y2 in gt_boxes:
        axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.5))
    axes[0].set_title("(a) Original image", fontsize=13, weight="bold", pad=8)

    axes[1].imshow(image)
    detections = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    for box in detections:
        dx1, dy1, dx2, dy2 = box
        axes[1].add_patch(plt.Rectangle((dx1, dy1), dx2 - dx1, dy2 - dy1, fill=False, color="#00a6ff", linewidth=2))
    for x1, y1, x2, y2 in gt_boxes:
        axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.5, linestyle="--"))
    axes[1].set_title(f"(b) Baseline detections ({len(detections)} boxes)", fontsize=13, weight="bold", pad=8)

    fig.suptitle("Original image versus baseline detection", fontsize=16, weight="bold", y=0.98)
    fig.text(0.5, 0.015, f"YOLOv8n-P2 baseline downloaded from Hugging Face · blue = prediction · red dashed = ground truth · threshold = {args.conf:.2f}", ha="center", fontsize=9, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"detections: {len(detections)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
