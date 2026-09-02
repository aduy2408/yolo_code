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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), facecolor="white", gridspec_kw={"height_ratios": (1.35, 1)})
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    detections = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    axes[0].imshow(image)
    gt_boxes = args.gt_box or DEFAULT_GTS
    axes[0].set_title("(a) Original image", fontsize=13, weight="bold", pad=8)
    axes[1].imshow(image)
    axes[1].set_title(f"(b) Baseline output ({len(detections)} predictions)", fontsize=13, weight="bold", pad=8)

    # Keep both full-image panels clean. The local crops are shown separately,
    # as in the reference style, without drawing any bounding boxes.
    x1 = min(b[0] for b in gt_boxes)
    y1 = min(b[1] for b in gt_boxes)
    x2 = max(b[2] for b in gt_boxes)
    y2 = max(b[3] for b in gt_boxes)
    pad = max(28, int(max(x2 - x1, y2 - y1) * 3.0))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1, zy1 = max(0, int(cx - pad)), max(0, int(cy - pad))
    zx2, zy2 = min(image.width, int(cx + pad)), min(image.height, int(cy + pad))
    local = image.crop((zx1, zy1, zx2, zy2))
    axes[2].imshow(local, interpolation="nearest")
    axes[2].set_title("(c) Local region: original", fontsize=12, weight="bold", pad=8)
    axes[3].imshow(local, interpolation="nearest")
    axes[3].set_title("(d) Local region: baseline output", fontsize=12, weight="bold", pad=8)

    fig.suptitle("Original image versus baseline output", fontsize=16, weight="bold", y=0.98)
    fig.text(0.5, 0.015, f"YOLOv8n-P2 baseline downloaded from Hugging Face · full images and local crops shown without bbox overlays · threshold = {args.conf:.2f}", ha="center", fontsize=9, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"detections: {len(detections)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
