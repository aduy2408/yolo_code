"""Visualize how a tiny LEVIR-Ship target changes across YOLOv8-P2 stages.

The figure deliberately probes backbone outputs before the detection head. Each
column shows the same target at a different spatial stride, with the target box
projected onto the feature grid. The lower plot compares target activation with
an adjacent background patch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO  # noqa: E402


STAGE_LAYERS = {
    "P1\nstride 2": (0, 2),
    "P2\nstride 4": (1, 4),
    "P3\nstride 8": (3, 8),
    "P4\nstride 16": (5, 16),
    "P5\nstride 32": (7, 32),
}


def read_box(label_path: Path, width: int, height: int, index: int = 0) -> tuple[float, ...]:
    boxes = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 5:
            _, cx, cy, bw, bh = map(float, fields[:5])
            boxes.append((cx * width, cy * height, bw * width, bh * height))
    if not boxes:
        raise ValueError(f"No boxes found in {label_path}")
    cx, cy, bw, bh = boxes[index]
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def target_and_background(feature: np.ndarray, box: tuple[float, ...], stride: int) -> tuple[float, float]:
    _, _, h, w = feature.shape
    x1, y1, x2, y2 = box
    fx1, fy1 = max(0, int(np.floor(x1 / stride))), max(0, int(np.floor(y1 / stride)))
    fx2, fy2 = min(w, max(fx1 + 1, int(np.ceil(x2 / stride)))), min(h, max(fy1 + 1, int(np.ceil(y2 / stride))))
    target = feature[0, 0, fy1:fy2, fx1:fx2]

    # Use a one-box-sized patch immediately to the right, falling back to left.
    bw, bh = max(1, fx2 - fx1), max(1, fy2 - fy1)
    candidates = [(fx2, fy1, fx2 + bw, fy1 + bh), (fx1 - bw, fy1, fx1, fy1 + bh)]
    background = feature[0, 0, fy1:fy2, fx1:fx2]
    for bx1, by1, bx2, by2 in candidates:
        if bx1 >= 0 and by1 >= 0 and bx2 <= w and by2 <= h:
            background = feature[0, 0, by1:by2, bx1:bx2]
            break
    return float(target.mean()), float(background.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "diagnostics/hf_yolov8n_p2/train/yolov8n_p2_baseline_seed42/weights/best.pt")
    parser.add_argument("--image", type=Path, default=ROOT / "datasets/levir_ship_yolo_seed42/images/test/GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png")
    parser.add_argument("--label", type=Path, default=None)
    parser.add_argument("--box-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/progressive_downsampling_small_object.png")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = image.shape[:2]
    label_path = args.label or ROOT / "datasets/levir_ship_yolo_seed42/labels/test" / f"{args.image.stem}.txt"
    box = read_box(label_path, width, height, args.box_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(args.model)).model.to(device).eval()
    captures: dict[str, torch.Tensor] = {}
    handles = []
    for name, (layer_index, _) in STAGE_LAYERS.items():
        handles.append(model.model[layer_index].register_forward_hook(lambda _, __, out, n=name: captures.__setitem__(n, out)))

    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        model(tensor)
    for handle in handles:
        handle.remove()

    stages = []
    ratios = []
    for name, (_, stride) in STAGE_LAYERS.items():
        raw = captures[name]
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"Layer output for {name} is not a tensor")
        activation = raw.detach().abs().mean(dim=1, keepdim=True).cpu().numpy()
        activation /= np.percentile(activation, 99) + 1e-8
        activation = np.clip(activation, 0, 1)[0, 0]
        stages.append((name, stride, activation))
        feature_for_stats = raw.detach().abs().mean(dim=1, keepdim=True).cpu().numpy()
        target, background = target_and_background(feature_for_stats, box, stride)
        ratios.append(target / (background + 1e-8))

    fig = plt.figure(figsize=(18, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=(2.4, 1.0))
    x1, y1, x2, y2 = box
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    ax.set_title("Input\n512×512", fontsize=11, weight="bold")
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#39ff88", linewidth=2.2))
    ax.axis("off")
    for col, (name, stride, activation) in enumerate(stages, start=1):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image)
        ax.imshow(activation, cmap="magma", alpha=0.68, extent=(0, width, height, 0), interpolation="nearest")
        ax.set_title(f"{name}\n{activation.shape[1]}×{activation.shape[0]} grid", fontsize=11, weight="bold")
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#39ff88", linewidth=2.2))
        cells = (max(1, int(np.ceil((x2 - x1) / stride))), max(1, int(np.ceil((y2 - y1) / stride))))
        ax.text(0.03, 0.97, f"target footprint ≈ {cells[0]}×{cells[1]} cells", transform=ax.transAxes, va="top", color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.65, pad=3))
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    x = np.arange(len(stages))
    ax.plot(x, ratios, marker="o", linewidth=2.5, color="#d62728", label="mean |activation|: target / adjacent background")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="equal target and background")
    ax.set_xticks(x, [f"stride {stride}" for _, stride, _ in stages])
    ax.set_ylabel("Activation contrast ratio")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Effect of progressive downsampling on small-object representation", fontsize=16, weight="bold")
    fig.text(0.5, 0.01, "Same tiny ship (green box) through YOLOv8-P2 backbone. Heatmaps show channel-mean absolute activation; each panel is independently normalized for visibility.", ha="center", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"saved: {args.output}")
    print("target/background ratios:", ", ".join(f"{r:.3f}" for r in ratios))


if __name__ == "__main__":
    main()
