#!/usr/bin/env python3
"""Compare gradient statistics (spatial & channel) w.r.t P2 feature map across 4 model checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.loss import make_anchors, v8DetectionLoss  # noqa: E402


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def read_labels(image: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [line.split() for line in labels_for(image).read_text().splitlines() if line.strip()]
    if not rows:
        return torch.zeros((0, 1)), torch.zeros((0, 4))
    cls = torch.tensor([int(row[0]) for row in rows], dtype=torch.float32).reshape(-1, 1)
    boxes = torch.tensor([[float(value) for value in row[1:5]] for row in rows], dtype=torch.float32).reshape(-1, 4)
    return cls, boxes


def analyze_checkpoint(
    checkpoint_path: Path,
    images: list[Path],
    device: str,
    imgsz: int,
    limit: int
) -> dict[str, float]:
    wrapper = YOLO(checkpoint_path)
    net = wrapper.model.to(device)
    loss_fn = v8DetectionLoss(net)
    
    # Mock net.args if SimpleNamespace details are missing
    from types import SimpleNamespace
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if not hasattr(net, "args") or isinstance(net.args, dict):
        from ultralytics.utils import DEFAULT_CFG_DICT
        cfg_dict = DEFAULT_CFG_DICT.copy()
        if train_args:
            cfg_dict.update(train_args)
        net.args = SimpleNamespace(**cfg_dict)
    elif not hasattr(net.args, "box"):
        for k, v in train_args.items():
            setattr(net.args, k, v)
        if not hasattr(net.args, "box"):
            setattr(net.args, "box", 7.5)

    # Determine P2 feature map layer (often 18 or 19 depending on config/neck structure)
    # Let's inspect model modules and grab layer 19 or 18
    # We will hook the layer just before Detect module
    # Detect module is the last module in net.model
    detect_idx = len(net.model) - 1
    # The input to Detect is a list of features from P2, P3. Let's inspect net.model[detect_idx].f
    detect_input_indices = net.model[detect_idx].f
    # P2 layer is usually the first input to Detect
    p2_layer_idx = detect_input_indices[0] if isinstance(detect_input_indices, list) else 19

    target_layer = net.model[p2_layer_idx]
    
    feature_gradients = {}
    def save_grad(grad):
        feature_gradients["p2_grad"] = grad

    def forward_hook(module, input, output):
        output.register_hook(save_grad)

    h_handle = target_layer.register_forward_hook(forward_hook)
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)

    spatial_neffs = []
    spatial_top1s = []
    spatial_entropies = []

    channel_neffs = []
    channel_cvs = []
    channel_stds = []

    net.train() # Set to train mode to get gradients

    for idx, img_path in enumerate(images):
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
        tensor.requires_grad = True

        cls, boxes = read_labels(img_path)
        if len(boxes) == 0:
            continue

        batch = {
            "batch_idx": torch.zeros((len(boxes),), dtype=torch.long, device=device),
            "cls": cls.to(device),
            "bboxes": boxes.to(device),
            "img": tensor
        }

        net.zero_grad()
        feature_gradients.clear()

        preds = net(tensor)
        parsed_preds = loss_fn.parse_output(preds)
        loss_detect = loss_fn.get_assigned_targets_and_loss(parsed_preds, batch)[1]
        l_cls = loss_detect[1] # Classification loss

        l_cls.backward()

        g_cls = feature_gradients.get("p2_grad")
        if g_cls is None:
            continue
        
        # Absolute gradient magnitude: shape (B, C, H, W)
        G = g_cls.abs().squeeze(0) # (C, H, W)
        C, H_p2, W_p2 = G.shape

        # Create P2 grid mask matching GT boxes
        mask_p2 = torch.zeros((H_p2, W_p2), device=device, dtype=torch.bool)
        for box in boxes:
            cx, cy, w, h = box[0]*W_p2, box[1]*H_p2, box[2]*W_p2, box[3]*H_p2
            x1, y1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
            x2, y2 = int(min(W_p2, cx + w/2)), int(min(H_p2, cy + h/2))
            mask_p2[y1:y2, x1:x2] = True

        if not mask_p2.any():
            continue

        # Spatial axis stats
        # Sum along channel dimension for cells inside GT region
        g_spatial = G[:, mask_p2].sum(dim=0) # (N_cells,)
        if len(g_spatial) > 0:
            sum_g = g_spatial.sum()
            if sum_g > 1e-12:
                # N_eff = (sum(g))^2 / sum(g^2)
                neff = (sum_g.pow(2) / g_spatial.pow(2).sum()).item()
                top1 = (g_spatial.max() / sum_g).item()
                p = g_spatial / sum_g
                entropy = -(p * p.clamp_min(1e-12).log()).sum().item()

                spatial_neffs.append(neff)
                spatial_top1s.append(top1)
                spatial_entropies.append(entropy)

        # Channel axis stats
        # Sum along spatial dimension for coordinates inside GT region
        G_gt = G[:, mask_p2] # (C, N_cells)
        g_channel = G_gt.sum(dim=1) # (C,)
        sum_ch = g_channel.sum()
        if sum_ch > 1e-12:
            ch_neff = (g_channel.sum().pow(2) / g_channel.pow(2).sum()).item()
            std_ch = g_channel.std().item()
            mean_ch = g_channel.mean().item()
            cv_ch = (std_ch / mean_ch) if mean_ch > 1e-12 else 0.0

            channel_neffs.append(ch_neff)
            channel_stds.append(std_ch)
            channel_cvs.append(cv_ch)

    h_handle.remove()

    return {
        "spatial_neff_mean": float(np.mean(spatial_neffs)) if spatial_neffs else 0.0,
        "spatial_top1_mean": float(np.mean(spatial_top1s)) if spatial_top1s else 0.0,
        "spatial_entropy_mean": float(np.mean(spatial_entropies)) if spatial_entropies else 0.0,
        "channel_neff_mean": float(np.mean(channel_neffs)) if channel_neffs else 0.0,
        "channel_std_mean": float(np.mean(channel_stds)) if channel_stds else 0.0,
        "channel_cv_mean": float(np.mean(channel_cvs)) if channel_cvs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device

    checkpoints = {
        "Plain": Path("/marimo/yolo_code/runs/checkpoints_4way/runs/plain_p2p3/seed_42/weights/best.pt"),
        "Plain + FTAL": Path("/marimo/yolo_code/runs/checkpoints_4way/runs/plain_p2_factorized_k15/seed_42/weights/best.pt"),
        "GAP": Path("/marimo/yolo_code/hf_cache/levir-yolov8n-p2-channel-descriptor-seed42/runs/gap/seed_42/weights/best.pt"),
        "GAP + FTAL": Path("/marimo/yolo_code/runs/checkpoints_4way/runs/gap_factorized_k15/seed_42/weights/best.pt"),
    }

    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[:args.limit]

    results = {}
    for name, path in checkpoints.items():
        print(f"Analyzing {name} checkpoint...")
        results[name] = analyze_checkpoint(path, images, device, args.imgsz, args.limit)

    print("\n=== 4-Way Gradient Distribution Results ===")
    print(json.dumps(results, indent=2))

    output_dir = ROOT / "runs/gradient_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gradient_distribution_4way.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
