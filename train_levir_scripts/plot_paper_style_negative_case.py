"""Make a paper-style five-panel visualization for a difficult YOLOv8-P2 case."""

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
DEFAULT_IMAGE = ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF1_WFV2_E123.6_N29.3_20190910_L2A0004239231_2048_2560.png"
# A known tiny difficult case from the baseline diagnostics.
DEFAULT_BOX = (168.0, 270.0, 183.0, 283.0)


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--box", type=float, nargs=4, default=DEFAULT_BOX, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/paper_style_negative_case.png")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = args.box
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

    p2 = capture["p2"].detach().abs().mean(dim=1)[0].cpu().numpy()
    feature_map = norm01(p2)
    reconstructed = cv2.resize(feature_map, (width, height), interpolation=cv2.INTER_CUBIC)
    gray = image.mean(axis=2) / 255.0
    difference = np.abs(norm01(gray) - reconstructed)

    pad = max(28, int(max(x2 - x1, y2 - y1) * 4.5))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1, zy1 = max(0, int(cx - pad)), max(0, int(cy - pad))
    zx2, zy2 = min(width, int(cx + pad)), min(height, int(cy + pad))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=(1.15, 1, 1), hspace=0.18, wspace=0.08)
    ax_input = fig.add_subplot(gs[:, 0])
    ax_local = fig.add_subplot(gs[0, 1])
    ax_feat = fig.add_subplot(gs[0, 2])
    ax_recon = fig.add_subplot(gs[1, 1])
    ax_diff = fig.add_subplot(gs[1, 2])
    axes = [ax_input, ax_local, ax_feat, ax_recon, ax_diff]
    for ax in axes:
        ax.axis("off")

    ax_input.imshow(image)
    ax_input.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#e53935", linewidth=2.2))
    ax_input.set_title("(a) Image input", pad=8)

    local = image[zy1:zy2, zx1:zx2]
    ax_local.imshow(local)
    ax_local.add_patch(plt.Rectangle((x1 - zx1, y1 - zy1), x2 - x1, y2 - y1, fill=False, color="#ffd600", linewidth=2.2, linestyle="--"))
    ax_local.set_title("(b) Local region", pad=8)

    ax_feat.imshow(feature_map, cmap="plasma", interpolation="nearest")
    ax_feat.add_patch(plt.Rectangle((x1 / 4, y1 / 4), (x2 - x1) / 4, (y2 - y1) / 4, fill=False, color="#ffd600", linewidth=2.2, linestyle="--"))
    ax_feat.set_title("(c) P2 feature map", pad=8)

    ax_recon.imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
    ax_recon.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#ffd600", linewidth=2.2, linestyle="--"))
    ax_recon.set_title("(d) Reconstructed activation", pad=8)

    im = ax_diff.imshow(difference, cmap="Blues", vmin=0, vmax=1)
    ax_diff.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="#ffd600", linewidth=2.2, linestyle="--"))
    ax_diff.set_title("(e) Difference map", pad=8)

    fig.suptitle("Baseline failure case: tiny object after P2 encoding", fontsize=16, weight="bold", y=0.98)
    fig.text(0.5, 0.015, "YOLOv8n-P2 baseline downloaded from Hugging Face · feature map = channel-mean |activation| · yellow box marks the 15×13 px target", ha="center", fontsize=9, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
