"""Create a paper-style figure showing progressive P2/P3/P4 downsampling."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO  # noqa: E402

HF_REPO = "duyle2408/levir-ship-yolo-p2"
HF_FILE = "train/yolov8n_p2_baseline_seed42/weights/best.pt"
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_6144_17285.png"
# Verified baseline miss with strong early activation: 13 x 15 px in the tile.
DEFAULT_BOX = (444.0, 254.0, 457.0, 269.0)
STAGES = ((18, 4, "P2", "stride 4"), (21, 8, "P3", "stride 8"), (24, 16, "P4", "stride 16"))


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [2, 99])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def target_background_ratio(feature: np.ndarray, box: tuple[float, ...], stride: int) -> float:
    height, width = feature.shape
    x1, y1, x2, y2 = box
    fx1, fy1 = max(0, int(x1 / stride)), max(0, int(y1 / stride))
    fx2, fy2 = min(width, max(fx1 + 1, int(np.ceil(x2 / stride)))), min(height, max(fy1 + 1, int(np.ceil(y2 / stride))))
    target = feature[fy1:fy2, fx1:fx2].mean()
    span_x, span_y = max(1, fx2 - fx1), max(1, fy2 - fy1)
    if fx2 + span_x <= width:
        background = feature[fy1:fy2, fx2:fx2 + span_x].mean()
    else:
        background = feature[fy1:fy2, max(0, fx1 - span_x):fx1].mean()
    return float(target / (background + 1e-8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--box", type=float, nargs=4, default=DEFAULT_BOX, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/paper_style_progressive_downsampling.png")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = args.box
    print(f"Downloading baseline from hf://datasets/{HF_REPO}/{HF_FILE}", flush=True)
    model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path).model.to(device).eval()
    captures = {}
    handles = [model.model[layer].register_forward_hook(lambda _, __, out, n=name: captures.__setitem__(n, out)) for layer, _, name, _ in STAGES]
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        model(tensor)
    for handle in handles:
        handle.remove()

    feature_maps = []
    for _, stride, name, stride_label in STAGES:
        raw = captures[name].detach().abs().mean(dim=1)[0].cpu().numpy()
        feature_maps.append((name, stride, stride_label, norm01(raw), raw.shape, target_background_ratio(raw, args.box, stride)))

    pad = max(30, int(max(x2 - x1, y2 - y1) * 3.5))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1, zy1 = max(0, int(cx - pad)), max(0, int(cy - pad))
    zx2, zy2 = min(width, int(cx + pad)), min(height, int(cy + pad))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 5.5), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.1, 1.1, 2.6), wspace=0.08)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_local = fig.add_subplot(gs[0, 1])
    ax_container = fig.add_subplot(gs[0, 2])
    ax_container.axis("off")
    ax_input.imshow(image)
    ax_input.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.4))
    ax_input.set_title("(a) Image input", pad=8)
    ax_input.axis("off")

    ax_local.imshow(image[zy1:zy2, zx1:zx2], interpolation="nearest")
    ax_local.add_patch(plt.Rectangle((x1 - zx1, y1 - zy1), x2 - x1, y2 - y1, fill=False, color="#ffd600", linewidth=2.5, linestyle="--"))
    ax_local.set_title("(b) Local region", pad=8)
    ax_local.axis("off")

    inner = gs[0, 2].subgridspec(1, 3, wspace=0.06)
    feature_axes = [fig.add_subplot(inner[0, i]) for i in range(3)]
    for ax, (name, stride, stride_label, heatmap, shape, ratio) in zip(feature_axes, feature_maps):
        ax.imshow(heatmap, cmap="plasma", interpolation="nearest")
        ax.add_patch(plt.Rectangle((x1 / stride, y1 / stride), (x2 - x1) / stride, (y2 - y1) / stride, fill=False, color="#ffd600", linewidth=2, linestyle="--"))
        ax.set_title(f"{name}\n{stride_label}\n{shape[1]}×{shape[0]}\nT/B = {ratio:.2f}", fontsize=10, pad=7)
        ax.axis("off")
    ax_container.set_title("(c) Progressive feature maps", pad=8, y=1.02)

    fig.suptitle("Small-object representation under progressive downsampling", fontsize=16, weight="bold", y=0.99)
    fig.text(0.5, 0.015, "YOLOv8n-P2 baseline downloaded from Hugging Face · yellow box marks the same 13×15 px target · T/B = target/background activation", ha="center", fontsize=9, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
