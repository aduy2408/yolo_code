#!/usr/bin/env python3
"""Measure cosine similarity between Detect classification and Support auxiliary gradients at P2 representation."""

from __future__ import annotations

import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    wrapper = YOLO(args.checkpoint)
    net = wrapper.model.to(device)
    
    # Initialize basic model args if missing or if it doesn't support dot notation
    from types import SimpleNamespace
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if not hasattr(net, "args") or isinstance(net.args, dict):
        # Merge default configs and train_args into a SimpleNamespace
        from ultralytics.utils import DEFAULT_CFG_DICT
        cfg_dict = DEFAULT_CFG_DICT.copy()
        if train_args:
            cfg_dict.update(train_args)
        net.args = SimpleNamespace(**cfg_dict)
    elif not hasattr(net.args, "box"):
        for k, v in (getattr(wrapper, "ckpt", None) or {}).get("train_args", {}).items():
            setattr(net.args, k, v)
        if not hasattr(net.args, "box"):
            setattr(net.args, "box", 7.5) # Fallback gain

    loss_fn = v8DetectionLoss(net)

    # Find the GAP module output on layer 19 (P2 ChannelAttention)
    # We want to attach hook or register tracking to get gradients w.r.t F_P2
    # In YOLOv8 execution, the layers are sequential. Let's trace the modules:
    # layer 19: ChannelAttention
    # layer 20: ChannelAttention (P3)
    # layer 21: FactorizedSupportAux
    # layer 22: Detect
    
    # We can capture the activations on layer 19 output
    feature_gradients = {}

    def save_grad(name):
        def hook(grad):
            feature_gradients[name] = grad
        return hook

    # Hook the output of model.model[19] (P2 GAP layer)
    gap_module = net.model[19]
    activations = {}

    def forward_hook(module, input, output):
        activations["gap_out"] = output
        output.register_hook(save_grad("gap_out"))

    h_handle = gap_module.register_forward_hook(forward_hook)

    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[:args.limit]

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    results = []

    for idx, img_path in enumerate(images, 1):
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        h_orig, w_orig = original.shape[:2]
        
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

        # Enable grads
        net.train() # needs to be in training mode to compute loss & support aux
        
        # Reset gradients
        net.zero_grad()
        feature_gradients.clear()
        activations.clear()

        # Forward
        preds = net(tensor)
        
        # Decode targets for losses
        parsed_preds = loss_fn.parse_output(preds)
        
        # Calculate standard losses
        loss_detect, loss_detach = loss_fn.get_assigned_targets_and_loss(parsed_preds, batch)[1:]
        l_cls = loss_detect[1] # Classification loss

        # Calculate Support Aux loss separately if active
        l_sup = torch.tensor(0.0, device=device)
        if loss_fn.support_module is not None and loss_fn.support_module.support_logits is not None:
            h2, w2 = parsed_preds["feats"][0].shape[-2:]
            n_p2 = h2 * w2
            pred_distri = parsed_preds["boxes"].permute(0, 2, 1).contiguous()
            anchor_points_all, stride_tensor_all = make_anchors(parsed_preds["feats"], loss_fn.stride, 0.5)
            pred_bboxes_all = loss_fn.bbox_decode(anchor_points_all, pred_distri, None, stride_tensor_all)

            imgsz = torch.tensor(parsed_preds["feats"][0].shape[2:], device=device, dtype=pred_bboxes_all.dtype) * loss_fn.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = loss_fn.preprocess(targets, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
            _, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            anchor_points_p2 = anchor_points_all[:n_p2] * stride_tensor_all[:n_p2]
            pred_bboxes_p2 = pred_bboxes_all[:, :n_p2].detach() * stride_tensor_all[:n_p2]

            l_sup = loss_fn.compute_factorized_support_loss(
                logits=loss_fn.support_module.support_logits,
                pred_bboxes=pred_bboxes_p2,
                anchor_points=anchor_points_p2,
                gt_bboxes=gt_bboxes,
                mask_gt=mask_gt,
                tau=loss_fn.support_tau,
                kappa=loss_fn.support_kappa,
                blend=loss_fn.support_blend,
                topk=loss_fn.support_topk,
                s_max=loss_fn.support_s_max,
            )

        if l_sup.item() == 0.0:
            continue

        # 1. Backprop classification loss
        l_cls.backward(retain_graph=True)
        g_cls = feature_gradients.get("gap_out")
        if g_cls is None:
            continue
        g_cls = g_cls.clone()

        # 2. Reset and backprop support loss
        net.zero_grad()
        feature_gradients.clear()
        l_sup.backward()
        g_sup = feature_gradients.get("gap_out")
        if g_sup is None:
            continue
        g_sup = g_sup.clone()

        # Calculate Cosine Similarity
        # Global
        g_cls_flat = g_cls.view(-1)
        g_sup_flat = g_sup.view(-1)
        cos_global = F.cosine_similarity(g_cls_flat, g_sup_flat, dim=0).item()

        # Inside GT mask
        # Create P2 grid mask matching GT boxes
        _, h_p2, w_p2 = g_cls.shape[1:]
        mask_p2 = torch.zeros((h_p2, w_p2), device=device, dtype=torch.bool)
        
        # Scale GTs to P2 scale
        for box in boxes:
            # box is cxcywh normalized
            cx, cy, w, h = box[0]*w_p2, box[1]*h_p2, box[2]*w_p2, box[3]*h_p2
            x1, y1 = int(max(0, cx - w/2)), int(max(0, cy - h/2))
            x2, y2 = int(min(w_p2, cx + w/2)), int(min(h_p2, cy + h/2))
            mask_p2[y1:y2, x1:x2] = True

        mask_p2_expanded = mask_p2.unsqueeze(0).unsqueeze(0).expand(g_cls.shape[0], g_cls.shape[1], -1, -1) # B, C, H, W
        
        g_cls_local = g_cls[mask_p2_expanded].view(-1)
        g_sup_local = g_sup[mask_p2_expanded].view(-1)
        
        if len(g_cls_local) > 0:
            cos_local = F.cosine_similarity(g_cls_local, g_sup_local, dim=0).item()
        else:
            cos_local = float("nan")

        results.append({
            "image": img_path.name,
            "cos_global": cos_global,
            "cos_local": cos_local,
        })

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(images)} images")

    h_handle.remove()

    # Summarize results
    cos_globals = [r["cos_global"] for r in results if not math.isnan(r["cos_global"])]
    cos_locals = [r["cos_local"] for r in results if not math.isnan(r["cos_local"])]

    summary = {
        "mean_cos_global": float(np.mean(cos_globals)) if cos_globals else None,
        "mean_cos_local": float(np.mean(cos_locals)) if cos_locals else None,
        "median_cos_global": float(np.median(cos_globals)) if cos_globals else None,
        "median_cos_local": float(np.median(cos_locals)) if cos_locals else None,
    }

    print("\n=== Gradient Shaping Results ===")
    print(json.dumps(summary, indent=2))

    output_dir = ROOT / "runs/gradient_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gradient_shaping_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
