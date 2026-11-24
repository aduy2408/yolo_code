#!/usr/bin/env python3
"""Measure the local contrast vs raw RGB disagreement map D across WP, TP, and BG anchors."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator

def main():
    # 1. Download pretrained contrast_basis best.pt from HF
    repo_id = "duyle2408/levir-yolov8n-p2-contrast-basis-seed42"
    print(f"Downloading best.pt from HF dataset {repo_id}...")
    token = os.environ.get("HF_TOKEN") or (open("/root/.cache/huggingface/token").read().strip() if os.path.exists("/root/.cache/huggingface/token") else None)
    
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="runs/contrast_basis/seed_42/weights/best.pt",
        repo_type="dataset",
        token=token
    )
    print(f"Downloaded weights to: {weights_path}")

    # 2. Load model
    model = YOLO(weights_path)
    model.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    # 3. Setup dataset validator
    dataset_yaml = ROOT.parent / "datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    if not dataset_yaml.exists():
        print(f"Dataset yaml not found at {dataset_yaml}")
        return

    # Use Ultralytics DetectionValidator to get the val loader
    args = dict(model=weights_path, data=str(dataset_yaml), imgsz=512, batch=8, device=str(device))
    validator = DetectionValidator(args=args)
    validator.data = check_det_dataset(dataset_yaml)
    val_loader = validator.get_dataloader(validator.data["val"], batch_size=8)

    # Accumulators for disagreement D
    wp_vals = []
    tp_vals = []
    bg_vals = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["img"].to(device).float() / 255.0  # [B, 3, 512, 512]
            B, _, H_img, W_img = imgs.shape

            # Capture the input to the Detect head using a forward hook
            detect_head = model.model.model[-1]
            p2_feature = None
            def hook_fn(module, input_args, output):
                nonlocal p2_feature
                # input_args[0] is the list of feature maps passed to the Detect head.
                # For P2-only, it has 1 element which is the P2 feature map.
                p2_feature = input_args[0][0].detach()

            hook = detect_head.register_forward_hook(hook_fn)
            try:
                preds = model.model(imgs)
            finally:
                hook.remove()

            # Get the P2 local contrast stem module (layer 0)
            stem = model.model.model[0]
            if not hasattr(stem, "last_D"):
                print("Error: stem does not have last_D attribute. Check if local_contrast.py was modified correctly.")
                return
            
            last_D = stem.last_D.detach()  # [B, H_p2, W_p2] (e.g. [B, 128, 128])
            B, H_p2, W_p2 = last_D.shape

            # x is the P2 feature map of shape [B, 128, H_p2, W_p2]
            p2_scores = detect_head.cv3[0](p2_feature).sigmoid().squeeze(1) # [B, H_p2, W_p2]

            # Parse ground truth boxes
            # batch["bboxes"] has shape [num_total_gts, 4] (normalized xywh or xyxy?)
            # batch["cls"] has shape [num_total_gts, 1]
            # batch["batch_idx"] has shape [num_total_gts]
            gt_bboxes = batch["bboxes"]
            gt_batch_idx = batch["batch_idx"]

            for b in range(B):
                # Mask of P2 pixels inside any GT box for this image
                pos_mask = torch.zeros((H_p2, W_p2), dtype=torch.bool, device=device)
                
                # Filter GTs for this image
                img_gts = gt_bboxes[gt_batch_idx == b] # [num_gts, 4] in normalized xywh
                for gt in img_gts:
                    # Convert normalized xywh to pixel coordinates in P2
                    x_center, y_center, w, h = gt
                    x1 = int((x_center - w / 2) * W_p2)
                    y1 = int((y_center - h / 2) * H_p2)
                    x2 = int((x_center + w / 2) * W_p2) + 1
                    y2 = int((y_center + h / 2) * H_p2) + 1
                    
                    x1 = max(0, min(x1, W_p2 - 1))
                    y1 = max(0, min(y1, H_p2 - 1))
                    x2 = max(0, min(x2, W_p2))
                    y2 = max(0, min(y2, H_p2))
                    
                    pos_mask[y1:y2, x1:x2] = True

                neg_mask = ~pos_mask
                
                # Fetch scores and disagreement map D for this image
                img_scores = p2_scores[b]  # [H_p2, W_p2]
                img_D = last_D[b]          # [H_p2, W_p2]

                # Categorize
                tp_indices = pos_mask & (img_scores >= 0.5)
                wp_indices = pos_mask & (img_scores < 0.3)
                bg_indices = neg_mask & (img_scores < 0.1)

                if tp_indices.any():
                    tp_vals.extend(img_D[tp_indices].cpu().numpy().tolist())
                if wp_indices.any():
                    wp_vals.extend(img_D[wp_indices].cpu().numpy().tolist())
                if bg_indices.any():
                    bg_vals.extend(img_D[bg_indices].cpu().numpy().tolist())

    # 4. Report statistics
    print("\n--- Disagreement Map D Statistics ---")
    for name, vals in [("True Positives (TP, score >= 0.5)", tp_vals), 
                       ("Weak Positives (WP, score < 0.3)", wp_vals), 
                       ("Background (BG, score < 0.1)", bg_vals)]:
        if len(vals) > 0:
            arr = np.array(vals)
            print(f"{name}:")
            print(f"  Count:  {len(arr)}")
            print(f"  Mean:   {arr.mean():.4f}")
            print(f"  Std:    {arr.std():.4f}")
            print(f"  Min:    {arr.min():.4f}")
            print(f"  Max:    {arr.max():.4f}")
            print(f"  Median: {np.median(arr):.4f}")
        else:
            print(f"{name}: No anchors matched")
    print("--------------------------------------")

if __name__ == "__main__":
    main()
