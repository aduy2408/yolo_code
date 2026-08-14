#!/usr/bin/env python3
"""Run 7 diagnostic representation probes on hard-missed ships using GAP + FTAL checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import scale_boxes, xywh2xyxy  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def read_labels(image: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [line.split() for line in labels_for(image).read_text().splitlines() if line.strip()]
    if not rows:
        return torch.zeros((0, 1)), torch.zeros((0, 4))
    cls = torch.tensor([int(row[0]) for row in rows], dtype=torch.float32).reshape(-1, 1)
    boxes = torch.tensor([[float(value) for value in row[1:5]] for row in rows], dtype=torch.float32).reshape(-1, 4)
    return cls, boxes


def extract_ring_mean(F_in: torch.Tensor, r_in: int = 2, r_out: int = 5) -> torch.Tensor:
    # Average of ring context not containing center (annulus mask)
    device = F_in.device
    C, H, W = F_in.shape
    
    # Create annulus kernel
    size = 2 * r_out + 1
    center = r_out
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist = ((x - center)**2 + (y - center)**2).float().sqrt()
    mask = (dist >= r_in) & (dist <= r_out)
    kernel = mask.float()
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size).repeat(C, 1, 1, 1).to(device)

    # Depthwise convolution
    F_pad = F.pad(F_in.unsqueeze(0), (r_out, r_out, r_out, r_out), mode="replicate")
    out = F.conv2d(F_pad, kernel, groups=C).squeeze(0)
    return out


def extract_local_rarity(F_in: torch.Tensor, k_size: int = 5) -> torch.Tensor:
    # 1 - local mean cosine similarity
    device = F_in.device
    C, H, W = F_in.shape
    pad = k_size // 2
    
    # Normalize features
    F_norm = F_in / (F_in.norm(dim=0, keepdim=True) + 1e-8) # (C, H, W)
    F_unfolded = F.unfold(F_norm.unsqueeze(0), kernel_size=k_size, padding=pad).squeeze(0) # (C * k*k, H*W)
    F_unfolded = F_unfolded.view(C, k_size*k_size, H*W) # (C, k*k, H*W)
    
    # Dot product of normalized features
    F_center = F_norm.view(C, 1, H*W)
    cos_sim = (F_unfolded * F_center).sum(dim=0) # (k*k, H*W)
    
    mean_cos = cos_sim.mean(dim=0).view(H, W)
    return (1.0 - mean_cos).unsqueeze(0) # (1, H, W)


def extract_local_coherence(F_in: torch.Tensor, k_size: int = 5) -> torch.Tensor:
    device = F_in.device
    C, H, W = F_in.shape
    pad = k_size // 2
    
    # Normalize features
    F_norm = F_in / (F_in.norm(dim=0, keepdim=True) + 1e-8)
    F_unfolded = F.unfold(F_norm.unsqueeze(0), kernel_size=k_size, padding=pad).squeeze(0)
    F_unfolded = F_unfolded.view(C, k_size*k_size, H*W)
    
    # Average cosine similarity in neighborhood
    # similarity between all pairs in the window? Let's do simple mean local similarity
    F_center = F_norm.view(C, 1, H*W)
    cos_sim = (F_unfolded * F_center).sum(dim=0)
    return cos_sim.mean(dim=0).view(H, W).unsqueeze(0)


def find_hard_misses(
    wrapper: YOLO,
    images: list[Path],
    device: str,
    imgsz: int
) -> list[dict]:
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    hard_misses = []
    
    for img_path in images:
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        h_orig, w_orig = original.shape[:2]
        
        # Run inference
        results = wrapper.predict(img_path, conf=0.001, iou=0.5, imgsz=imgsz, device=device, verbose=False)
        pred_boxes = results[0].boxes.xyxy # (N, 4) in original px
        pred_scores = results[0].boxes.conf
        
        cls, boxes = read_labels(img_path)
        if len(boxes) == 0:
            continue
        
        # Convert GT normalized cxcywh to xyxy original px
        gt_xyxy = xywh2xyxy(boxes)
        gt_xyxy[:, [0, 2]] *= w_orig
        gt_xyxy[:, [1, 3]] *= h_orig
        
        for g_idx, gt in enumerate(gt_xyxy):
            area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            # Categorize size groups
            size_side = math.sqrt(area.item())
            if area.item() < 100.0:
                size_group = "tiny"
            elif area.item() <= 400.0:
                size_group = "small"
            elif area.item() <= 1024.0:
                size_group = "medium"
            else:
                size_group = "large"
                
            # Compute IoU with all predictions
            if len(pred_boxes) == 0:
                max_iou = 0.0
            else:
                # Simple IoU
                x1 = torch.clamp(pred_boxes[:, 0], min=gt[0].item())
                y1 = torch.clamp(pred_boxes[:, 1], min=gt[1].item())
                x2 = torch.clamp(pred_boxes[:, 2], max=gt[2].item())
                y2 = torch.clamp(pred_boxes[:, 3], max=gt[3].item())
                inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
                union = area + (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]) - inter
                iou = inter / (union + 1e-8)
                max_iou = iou.max().item()
                
            if max_iou < 0.1:
                # Identify false positives as candidate hard background
                false_positives = []
                for p_idx, pred in enumerate(pred_boxes):
                    if pred_scores[p_idx] > 0.1:
                         # Check IoU with all GTs on CPU
                         pred_cpu = pred.cpu()
                         gt_xyxy_cpu = gt_xyxy.cpu()
                         x1_fp = torch.clamp(gt_xyxy_cpu[:, 0], min=pred_cpu[0].item())
                         y1_fp = torch.clamp(gt_xyxy_cpu[:, 1], min=pred_cpu[1].item())
                         x2_fp = torch.clamp(gt_xyxy_cpu[:, 2], max=pred_cpu[2].item())
                         y2_fp = torch.clamp(gt_xyxy_cpu[:, 3], max=pred_cpu[3].item())
                         inter_fp = torch.clamp(x2_fp - x1_fp, min=0) * torch.clamp(y2_fp - y1_fp, min=0)
                         union_fp = area + (pred_cpu[2] - pred_cpu[0]) * (pred_cpu[3] - pred_cpu[1]) - inter_fp
                         iou_fp = inter_fp / (union_fp + 1e-8)
                         if iou_fp.max() < 0.1:
                             false_positives.append(pred)

                hard_misses.append({
                    "image_path": img_path,
                    "gt_box": gt,
                    "gt_idx": g_idx,
                    "size_group": size_group,
                    "false_positives": false_positives
                })
                if len(hard_misses) >= 200:
                    return hard_misses
                    
    return hard_misses


def evaluate_probe(
    x_train: torch.Tensor, # (N, D)
    y_train: torch.Tensor, # (N,) binary
    epochs: int = 150
) -> tuple[float, float, float]:
    device = x_train.device
    D = x_train.shape[1]
    
    model = nn.Linear(D, 1).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    
    # Class weights for positive sample balancing
    pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float().clamp_min(1.0)
    criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train).squeeze(-1)
        loss = criterion_weighted(logits, y_train.float())
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        logits = model(x_train).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        # Calculate statistics
        pos_probs = probs[y_train == 1]
        neg_probs = probs[y_train == 0]
        
        gt_peak = pos_probs.max().item() if len(pos_probs) else 0.0
        bg_peak = neg_probs.max().item() if len(neg_probs) else 0.0
        margin = gt_peak - bg_peak
        
        # Cell AP
        # Sort targets
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_y = y_train[sorted_indices]
        
        tp_cumsum = sorted_y.cumsum(0)
        fp_cumsum = (1 - sorted_y).cumsum(0)
        recalls = tp_cumsum / sorted_y.sum().clamp_min(1.0)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum).clamp_min(1.0)
        
        # Compute AP (trapezoidal rule or standard step)
        ap = 0.0
        for i in range(1, len(precisions)):
            ap += (recalls[i] - recalls[i-1]).item() * precisions[i].item()
            
    return gt_peak, bg_peak, ap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    wrapper = YOLO(args.checkpoint)
    net = wrapper.model.to(device).eval()

    # Locate layer 19 (P2 ChannelAttention output)
    target_layer = net.model[19]
    activations = {}
    
    def forward_hook(module, input, output):
        activations["p2"] = output.squeeze(0)

    h_handle = target_layer.register_forward_hook(forward_hook)

    # Scan test split for hard misses
    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    
    print("Scanning test set for hard misses...")
    miss_list = find_hard_misses(wrapper, images, device, args.imgsz)
    print(f"Found {len(miss_list)} hard missed objects.")

    if not miss_list:
        print("No hard missed objects found!")
        h_handle.remove()
        return

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    
    probe_names = [
        "Raw F", "F Normalized", "Center-Ring Contrast", 
        "Local Rarity", "Local Coherence", 
        "BG Prototype Residual", "Hard-BG Prototype Residual"
    ]
    
    # Accumulate results across all misses
    probe_results = {name: {"gt_peak": [], "bg_peak": [], "ap": [], "size_group": [], "rescued": 0} for name in probe_names}

    for idx, miss in enumerate(miss_list, 1):
        img_path = miss["image_path"]
        gt = miss["gt_box"]
        fps = miss["false_positives"]
        
        # Forward pass to get activations
        original = cv2.imread(str(img_path))
        h_orig, w_orig = original.shape[:2]
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
        
        with torch.no_grad():
            net(tensor)
            
        F_p2 = activations["p2"] # (C, H_p2, W_p2)
        C, H_p2, W_p2 = F_p2.shape
        
        # Create mask for current GT box on P2 scale
        # Scale GT coords to P2 grid size
        scale_x = W_p2 / w_orig
        scale_y = H_p2 / h_orig
        x1_p2, y1_p2 = int(gt[0].item() * scale_x), int(gt[1].item() * scale_y)
        x2_p2, y2_p2 = int(gt[2].item() * scale_x), int(gt[3].item() * scale_y)
        
        # Ensure it occupies at least 1 cell
        x2_p2 = max(x2_p2, x1_p2 + 1)
        y2_p2 = max(y2_p2, y1_p2 + 1)
        
        gt_mask = torch.zeros((H_p2, W_p2), dtype=torch.bool, device=device)
        gt_mask[y1_p2:y2_p2, x1_p2:x2_p2] = True
        
        # Create hard-background mask if false positives exist
        hard_bg_mask = torch.zeros((H_p2, W_p2), dtype=torch.bool, device=device)
        for fp in fps:
            xf1, yf1 = int(fp[0].item() * scale_x), int(fp[1].item() * scale_y)
            xf2, yf2 = int(fp[2].item() * scale_x), int(fp[3].item() * scale_y)
            hard_bg_mask[yf1:max(yf2, yf1+1), xf1:max(xf2, xf1+1)] = True
            
        # Target labels
        y = gt_mask.view(-1).long() # 1 inside GT, 0 background
        
        # Define representation inputs for each probe
        probes = {}
        
        # 1. Raw F
        probes["Raw F"] = F_p2.permute(1, 2, 0).reshape(-1, C)
        
        # 2. F Normalized
        F_norm = F_p2 / (F_p2.norm(dim=0, keepdim=True) + 1e-8)
        probes["F Normalized"] = F_norm.permute(1, 2, 0).reshape(-1, C)
        
        # 3. Center-Ring Contrast
        F_ring = extract_ring_mean(F_p2)
        F_contrast = F_p2 - F_ring
        probes["Center-Ring Contrast"] = torch.cat([F_p2, F_contrast], dim=0).permute(1, 2, 0).reshape(-1, 2*C)
        
        # 4. Local Rarity
        F_rarity = extract_local_rarity(F_p2)
        probes["Local Rarity"] = torch.cat([F_p2, F_rarity], dim=0).permute(1, 2, 0).reshape(-1, C+1)
        
        # 5. Local Coherence
        F_coherence = extract_local_coherence(F_p2)
        probes["Local Coherence"] = torch.cat([F_p2, F_coherence], dim=0).permute(1, 2, 0).reshape(-1, C+1)
        
        # 6. BG Prototype Residual
        # Average feature of background regions
        bg_features = F_p2[:, ~gt_mask]
        p_bg = bg_features.mean(dim=1).view(C, 1, 1) # (C, 1, 1)
        F_bg_res = F_p2 - p_bg
        probes["BG Prototype Residual"] = F_bg_res.permute(1, 2, 0).reshape(-1, C)
        
        # 7. Hard-BG Prototype Residual
        if hard_bg_mask.any():
            p_hard = F_p2[:, hard_bg_mask].mean(dim=1).view(C, 1, 1)
        else:
            # Fallback to general background if no false positives
            p_hard = p_bg
        F_hard_res = F_p2 - p_hard
        probes["Hard-BG Prototype Residual"] = F_hard_res.permute(1, 2, 0).reshape(-1, C)

        # Train and evaluate each probe
        for name, x in probes.items():
            gt_peak, bg_peak, ap = evaluate_probe(x, y)
            
            probe_results[name]["gt_peak"].append(gt_peak)
            probe_results[name]["bg_peak"].append(bg_peak)
            probe_results[name]["ap"].append(ap)
            probe_results[name]["size_group"].append(miss["size_group"])
            if ap > 0.5: # Criteria for successful rescue
                probe_results[name]["rescued"] += 1

    h_handle.remove()

    # Summarize all results by size groups
    summary = {}
    size_groups_list = ["all", "tiny", "small", "medium", "large"]
    
    for name in probe_names:
        summary[name] = {}
        for group in size_groups_list:
            if group == "all":
                selected_indices = list(range(len(miss_list)))
            else:
                selected_indices = [i for i, sg in enumerate(probe_results[name]["size_group"]) if sg == group]
                
            if not selected_indices:
                continue
                
            gt_peaks = [probe_results[name]["gt_peak"][i] for i in selected_indices]
            bg_peaks = [probe_results[name]["bg_peak"][i] for i in selected_indices]
            aps = [probe_results[name]["ap"][i] for i in selected_indices]
            
            rescued_count = sum(1 for i in selected_indices if probe_results[name]["ap"][i] > 0.5)
            total_count = len(selected_indices)
            rescue_rate = rescued_count / total_count if total_count else 0.0
            
            summary[name][group] = {
                "mean_gt_peak": float(np.mean(gt_peaks)),
                "mean_bg_peak": float(np.mean(bg_peaks)),
                "mean_ap": float(np.mean(aps)),
                "rescue_rate": float(rescue_rate),
                "rescued_count": rescued_count,
                "total_count": total_count
            }

    print("\n=== Hard Misses Oracle Probes Results ===")
    print(json.dumps(summary, indent=2))

    output_dir = ROOT / "runs/gradient_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "probe_hard_misses_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
