"""Create a clean P2 activation heatmap from the Hugging Face baseline model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO  # noqa: E402


HF_REPO = "duyle2408/levir-ship-yolo-p2"
HF_FILE = "train/yolov8n_p2_baseline_seed42/weights/best.pt"
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png"


def read_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, ...]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 5:
            _, cx, cy, bw, bh = map(float, fields[:5])
            boxes.append((cx * width - bw * width / 2, cy * height - bh * height / 2, cx * width + bw * width / 2, cy * height + bh * height / 2))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--label", type=Path, default=None)
    parser.add_argument("--box-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/baseline_p2_activation_heatmap.png")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = image.shape[:2]
    label_path = args.label or ROOT / "datasets/levir_ship_yolo_seed42/labels/test" / f"{args.image.stem}.txt"
    boxes = read_boxes(label_path, width, height)
    if not boxes or args.box_index >= len(boxes):
        raise ValueError(f"Expected box index {args.box_index} in {label_path}, found {len(boxes)} boxes")
    target = boxes[args.box_index]

    print(f"Downloading baseline from hf://datasets/{HF_REPO}/{HF_FILE}", flush=True)
    model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path).model.to(device).eval()
    capture = {}
    handle = model.model[18].register_forward_hook(lambda _, __, output: capture.__setitem__("p2", output))
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        model(tensor)
    handle.remove()

    features = capture["p2"].detach().abs().mean(dim=1)[0].cpu().numpy()
    lo, hi = np.percentile(features, [2, 99.5])
    heatmap = np.clip((features - lo) / (hi - lo + 1e-8), 0, 1)
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    x1, y1, x2, y2 = target
    pad = max(20, int(max(x2 - x1, y2 - y1) * 4))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1, zy1 = max(0, int(cx - pad)), max(0, int(cy - pad))
    zx2, zy2 = min(width, int(cx + pad)), min(height, int(cy + pad))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(13, 8), facecolor="#101419")
    gs = fig.add_gridspec(2, 2, height_ratios=(1.1, 1), hspace=0.16, wspace=0.08)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax in axes:
        ax.set_facecolor("#101419")
        ax.axis("off")

    axes[0].imshow(image)
    axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#00f5a0", linewidth=2.5))
    axes[0].set_title("Input image", color="white", pad=8)

    axes[1].imshow(image)
    im = axes[1].imshow(heatmap, cmap="inferno", alpha=0.78, norm=Normalize(0, 1))
    axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="white", linewidth=2, linestyle="--"))
    axes[1].set_title("YOLOv8n-P2 activation heatmap", color="white", pad=8)

    crop = image[zy1:zy2, zx1:zx2]
    crop_heat = heatmap[zy1:zy2, zx1:zx2]
    axes[2].imshow(crop, interpolation="nearest")
    axes[2].add_patch(plt.Rectangle((x1 - zx1, y1 - zy1), x2 - x1, y2 - y1, fill=False, color="#00f5a0", linewidth=2.5))
    axes[2].set_title("Target crop", color="white", pad=8)

    axes[3].imshow(crop, interpolation="nearest")
    axes[3].imshow(crop_heat, cmap="inferno", alpha=0.84, norm=Normalize(0, 1), interpolation="bilinear")
    axes[3].add_patch(plt.Rectangle((x1 - zx1, y1 - zy1), x2 - x1, y2 - y1, fill=False, color="white", linewidth=2, linestyle="--"))
    axes[3].set_title("Target crop + P2 activation", color="white", pad=8)

    cbar = fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.015)
    cbar.set_label("Normalized channel-mean |activation|", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    fig.suptitle("Small-object representation in a YOLOv8n-P2 baseline", color="white", fontsize=17, weight="bold", y=0.98)
    fig.text(0.5, 0.02, "Baseline checkpoint downloaded from Hugging Face · P2 feature map (stride 4) · green/white box marks the same ship", ha="center", color="#c7d0d9", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
