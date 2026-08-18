#!/usr/bin/env python3
"""Unified color-space and geometry-conditioned linear probing script.

Evaluates baseline M (c2f_fused) and auxiliary representations (RGB, Y, CbCr, Opponent)
using patches (3x3 and 5x5 cells) and box-aligned geometry regions.
Computes PairAcc, Rank, Spearman Rho, Recall@1/5, Regret, Rescue and Damage rates.
Averages results across seeds 42, 43, 44 using checkpoints from HuggingFace.
Highly optimized: runs YOLO forward pass once per image and performs all cropping/pooling on GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402

from probe_center_ring_cohorts import iou_matrix, read_labels  # noqa: E402


# ---------------------------------------------------------------------------
# Core Probe Training & Evaluation Logic
# ---------------------------------------------------------------------------

def ap_auc(scores: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    y = y.to(scores.device)
    order = torch.argsort(scores, descending=True)
    sorted_y = y[order].float()
    tp = sorted_y.cumsum(0)
    fp = (1 - sorted_y).cumsum(0)
    recall = tp / sorted_y.sum().clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    ap = float(((recall[1:] - recall[:-1]) * precision[1:]).sum().item())
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return ap, float("nan")
    auc = float(
        ((pos[:, None] > neg[None]).float().mean()
         + 0.5 * (pos[:, None] == neg[None]).float().mean()).item()
    )
    return ap, auc


def train_probe(x: torch.Tensor, y: torch.Tensor, epochs: int) -> nn.Linear:
    model = nn.Linear(x.shape[1], 1).to(x.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=(y == 0).sum().float() / (y == 1).sum().float().clamp_min(1)
    )
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x).squeeze(1), y.float()).backward()
        opt.step()
    return model


def evaluate_case(probe: nn.Linear, x_case: torch.Tensor, local_ious: torch.Tensor,
                  y: torch.Tensor, base_scores_case: torch.Tensor | None = None) -> dict:
    device = x_case.device
    y = y.to(device)
    local_ious = local_ious.to(device)
    if base_scores_case is not None:
        base_scores_case = base_scores_case.to(device)

    with torch.no_grad():
        logits = probe(x_case).squeeze(1)
        scores = torch.sigmoid(logits)

    ap, auc = ap_auc(scores, y)
    order = torch.argsort(scores, descending=True)
    best_iou_idx = int(torch.argmax(local_ious))
    rank = int((order == best_iou_idx).nonzero(as_tuple=False)[0].item()) + 1

    # Recall
    recall_at_1 = float((local_ious[order[:1]] >= 0.5).any().item())
    recall_at_5 = float((local_ious[order[:5]] >= 0.5).any().item())

    # Regret
    max_iou = float(local_ious.max().item())
    top1_iou = float(local_ious[order[0]].item())
    regret = max_iou - top1_iou

    # Spearman correlation
    scores_cpu = scores.cpu().numpy()
    ious_cpu = local_ious.cpu().numpy()
    if len(scores_cpu) > 1 and len(np.unique(scores_cpu)) > 1 and len(np.unique(ious_cpu)) > 1:
        spearman_rho, _ = scipy.stats.spearmanr(scores_cpu, ious_cpu)
    else:
        spearman_rho = float("nan")

    # Pairwise statistics (positive-positive pairs with delta IoU > 0.05)
    pos_indices = (y == 1).nonzero(as_tuple=False).flatten()
    correct_pairs = 0
    total_pairs = 0

    rescue_numerator = 0
    rescue_denominator = 0
    damage_numerator = 0
    damage_denominator = 0

    if len(pos_indices) > 1:
        ious_pos = local_ious[pos_indices]
        scores_pos = scores[pos_indices]

        if base_scores_case is not None:
            base_scores_pos = base_scores_case[pos_indices]
        else:
            base_scores_pos = None

        for i in range(len(pos_indices)):
            for j in range(len(pos_indices)):
                if ious_pos[i] > ious_pos[j] + 0.05:
                    total_pairs += 1
                    is_correct = bool(scores_pos[i] > scores_pos[j])
                    if is_correct:
                        correct_pairs += 1

                    if base_scores_pos is not None:
                        is_base_correct = bool(base_scores_pos[i] > base_scores_pos[j])
                        if not is_base_correct:
                            rescue_denominator += 1
                            if is_correct:
                                rescue_numerator += 1
                        else:
                            damage_denominator += 1
                            if not is_correct:
                                damage_numerator += 1

    return {
        "ap": ap,
        "auc": auc,
        "rank": rank,
        "spearman": spearman_rho,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "regret": regret,
        "correct_pairs": correct_pairs,
        "total_pairs": total_pairs,
        "rescue_num": rescue_numerator,
        "rescue_den": rescue_denominator,
        "damage_num": damage_numerator,
        "damage_den": damage_denominator,
        "scores": scores,
    }


def load_split_samples(dataset_root: Path, split: str) -> list[dict]:
    yaml_path = dataset_root / "levir_ship_yolo_seed42/levir_ship.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    split_dir = dataset_root / "levir_ship_yolo_seed42" / config[split]
    samples = []
    for p in sorted(split_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            samples.append({"image": p, "boxes": read_labels(p)})
    return samples


def get_candidates_precomputed(decoded: torch.Tensor, preds: dict, original_shape: tuple[int, int],
                               gt: torch.Tensor, device: str, near_cells: int = 8,
                               top_neg: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = original_shape[:2]
    p2 = preds["feats"][0].squeeze(0)
    c, hp, wp = p2.shape
    n_p2 = hp * wp
    boxes = xywh2xyxy(decoded[0, :4, :n_p2].T)
    logits = preds["scores"][0, 0, :n_p2]

    ious = iou_matrix(gt.view(1, 4), boxes).squeeze(0)
    yy, xx = torch.meshgrid(
        torch.arange(hp, device=device),
        torch.arange(wp, device=device),
        indexing="ij",
    )
    gx = (gt[0] + gt[2]) * 0.5 * wp / w
    gy = (gt[1] + gt[3]) * 0.5 * hp / h
    dist = ((xx.flatten() - gx) ** 2 + (yy.flatten() - gy) ** 2).sqrt()

    pos = ious >= 0.1
    neg_pool = (ious < 0.05) & (dist <= near_cells)
    if neg_pool.sum() < 10:
        neg_pool = ious < 0.05
    neg_idx = torch.nonzero(neg_pool, as_tuple=False).flatten()
    if len(neg_idx) > top_neg:
        top = torch.argsort(logits[neg_idx], descending=True)[:top_neg]
        neg_idx = neg_idx[top]
    idx = torch.cat([torch.nonzero(pos, as_tuple=False).flatten(), neg_idx]).unique()
    return idx, ious[idx], (ious[idx] >= 0.1).long(), boxes[idx]


# ---------------------------------------------------------------------------
# GPU Feature Extraction Functions (RGB, Y, CbCr, Opponent)
# ---------------------------------------------------------------------------

def compute_color_maps_gpu(image_bgr: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    """Convert BGR image to normalized RGB, Y, CbCr, and Opponent color spaces on GPU."""
    img_t = torch.from_numpy(image_bgr[..., ::-1].copy()).to(device).float() / 255.0
    r = img_t[..., 0]
    g = img_t[..., 1]
    b = img_t[..., 2]

    # Y (Luminance)
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # CbCr (Pure chroma)
    cb = 0.5 - 0.1687 * r - 0.3313 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.4187 * g - 0.0813 * b

    # Opponent color
    o1 = r - g
    o2 = b - 0.5 * (r + g)

    return {
        "rgb": img_t,
        "y": y.unsqueeze(-1),
        "cbcr": torch.stack([cb, cr], dim=-1),
        "opp": torch.stack([o1, o2], dim=-1),
    }


def get_patch_feature_gpu(color_map_t: torch.Tensor, cx: float, cy: float, patch_cells: int) -> torch.Tensor:
    """Crop patch_cells around candidate center and adaptive average pool to 3x3 on GPU."""
    H, W, C = color_map_t.shape
    half = 2 * patch_cells
    ymin, ymax = int(max(0, cy - half)), int(min(H, cy + half))
    xmin, xmax = int(max(0, cx - half)), int(min(W, cx + half))

    patch = color_map_t[ymin:ymax, xmin:xmax]
    patch_t = patch.permute(2, 0, 1).unsqueeze(0) # (1, C, h, w)
    pooled = F.adaptive_avg_pool2d(patch_t, (3, 3)).squeeze(0) # (C, 3, 3)
    return pooled.flatten()


def get_region_stats_gpu(crop_cbcr: torch.Tensor, mask: torch.Tensor) -> list[float]:
    if mask.sum() > 0:
        pixels = crop_cbcr[mask] # (N, 2)
        mean = pixels.mean(dim=0)
        std = pixels.std(dim=0)
        return [float(mean[0].item()), float(mean[1].item()), float(std[0].item()), float(std[1].item())]
    return [0.0, 0.0, 0.0, 0.0]


def get_strip_mean_gpu(color_map_t: torch.Tensor, x_range: list[float], y_range: list[float]) -> torch.Tensor:
    H, W, C = color_map_t.shape
    x1, x2 = int(max(0, x_range[0])), int(min(W, x_range[1]))
    y1, y2 = int(max(0, y_range[0])), int(min(H, y_range[1]))
    if (x2 > x1) and (y2 > y1):
        return color_map_t[y1:y2, x1:x2].mean(dim=(0, 1))
    return torch.zeros(C, device=color_map_t.device)


def extract_candidate_features_gpu(gpu_maps: dict[str, torch.Tensor], cx: float, cy: float,
                                   box: torch.Tensor, H: int, W: int) -> dict[str, torch.Tensor]:
    feats = {}
    device = box.device

    # Stage A: Patches (3x3 and 5x5 cells)
    for space in ["rgb", "y", "cbcr", "opp"]:
        for size in [3, 5]:
            feats[f"{space}_{size}"] = get_patch_feature_gpu(gpu_maps[space], cx, cy, size)

    # Stage B: Geometry features
    x1, y1, x2, y2 = float(box[0].item()), float(box[1].item()), float(box[2].item()), float(box[3].item())
    w_box, h_box = x2 - x1, y2 - y1

    # Expanded box bounds
    w_exp, h_exp = w_box * 1.25, h_box * 1.25
    x1_exp, y1_exp = cx - w_exp * 0.5, cy - h_exp * 0.5
    x2_exp, y2_exp = cx + w_exp * 0.5, cy + h_exp * 0.5

    y1_e, y2_e = int(max(0, y1_exp)), int(min(H, y2_exp))
    x1_e, x2_e = int(max(0, x1_exp)), int(min(W, x2_exp))

    crop_cbcr = gpu_maps["cbcr"][y1_e:y2_e, x1_e:x2_e]

    # Region masks relative to the crop coordinates
    grid_y = torch.arange(y1_e, y2_e, device=device).view(-1, 1)
    grid_x = torch.arange(x1_e, x2_e, device=device).view(1, -1)
    w_in, h_in = w_box * 0.7, h_box * 0.7
    x1_in, y1_in = cx - w_in * 0.5, cy - h_in * 0.5
    x2_in, y2_in = cx + w_in * 0.5, cy + h_in * 0.5

    mask_inner = (grid_x >= x1_in) & (grid_x <= x2_in) & (grid_y >= y1_in) & (grid_y <= y2_in)
    mask_box = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
    mask_border = mask_box & (~mask_inner)
    mask_outer = ~mask_box

    # Probe 5: Inner / Border / Outer region stats
    p5_feats = []
    p5_feats.extend(get_region_stats_gpu(crop_cbcr, mask_inner))
    p5_feats.extend(get_region_stats_gpu(crop_cbcr, mask_border))
    p5_feats.extend(get_region_stats_gpu(crop_cbcr, mask_outer))
    feats["inner_outer"] = torch.tensor(p5_feats, dtype=torch.float32, device=device)

    # Probe 6: Four-side chroma geometry
    d = max(2.0, 0.1 * min(w_box, h_box))
    p6_feats = []
    # Left
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x1, x1 + d], [y1, y2])) # inside
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x1 - d, x1], [y1, y2])) # outside
    # Right
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x2 - d, x2], [y1, y2])) # inside
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x2, x2 + d], [y1, y2])) # outside
    # Top
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x1, x2], [y1, y1 + d])) # inside
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x1, x2], [y1 - d, y1])) # outside
    # Bottom
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x1, x2], [y2 - d, y2])) # inside
    p6_feats.append(get_strip_mean_gpu(gpu_maps["cbcr"], [x1, x2], [y2, y2 + d])) # outside
    feats["four_side"] = torch.cat(p6_feats, dim=0)

    # Probe 7: Spatial chroma map 5x5
    crop_t = crop_cbcr.permute(2, 0, 1).unsqueeze(0)
    feats["spatial_map"] = F.adaptive_avg_pool2d(crop_t, (5, 5)).squeeze(0).flatten()

    return feats


# ---------------------------------------------------------------------------
# Comprehensive Probing Process
# ---------------------------------------------------------------------------

def collect_features(net, hooked, samples, device, letterbox, args, split_name) -> tuple[dict, torch.Tensor, list]:
    data = {}
    all_y = []
    case_feats_list = []

    for sample in samples:
        original = cv2.imread(str(sample["image"]))
        if original is None:
            continue
        gt_boxes = sample["boxes"].to(device)
        if len(gt_boxes) == 0:
            continue
        gt_xyxy = xywh2xyxy(gt_boxes)
        gt_xyxy[:, [0, 2]] *= original.shape[1]
        gt_xyxy[:, [1, 3]] *= original.shape[0]

        # Convert colors and run forward pass ONCE per image
        original_letter = letterbox(image=original)
        gpu_maps = compute_color_maps_gpu(original_letter, device)
        H_img, W_img, _ = original_letter.shape

        tensor = (
            torch.from_numpy(original_letter[..., ::-1].copy())
            .to(device).permute(2, 0, 1).float()[None] / 255.0
        )
        with torch.no_grad():
            decoded, preds = net(tensor)

        if "c2f_fused" not in hooked:
            continue
        t_fused = hooked["c2f_fused"]
        c, hf, wf = t_fused.shape
        n = hf * wf

        for gt in gt_xyxy:
            idx, local_ious, y, boxes = get_candidates_precomputed(
                decoded, preds, original.shape, gt, device
            )
            safe_idx = idx[idx < n]
            if len(safe_idx) < 2 or y.sum() == 0:
                continue

            m_feats = t_fused.permute(1, 2, 0).reshape(n, c)[safe_idx]

            # Collect color and geometry features for each candidate
            case_feats = {
                "m": m_feats.cpu()
            }
            temp_feats = {}
            for k_idx, k in enumerate(safe_idx.tolist()):
                grid_y = k // wf
                grid_x = k % wf
                cy = (grid_y + 0.5) * 4
                cx = (grid_x + 0.5) * 4
                cand_feats = extract_candidate_features_gpu(gpu_maps, cx, cy, boxes[k_idx], H_img, W_img)
                for key, val in cand_feats.items():
                    if key not in temp_feats:
                        temp_feats[key] = []
                    temp_feats[key].append(val)

            for key, val in temp_feats.items():
                case_feats[key] = torch.stack(val, dim=0).cpu()

            case_feats_list.append((case_feats, local_ious.cpu(), y.cpu()))

            # Append to flat lists for VAL training
            if "m" not in data:
                data["m"] = []
            data["m"].append(case_feats["m"])
            for key in temp_feats:
                if key not in data:
                    data[key] = []
                data[key].append(case_feats[key])
            all_y.append(y.cpu())

    flat_data = {}
    flat_y = torch.tensor([])
    if all_y:
        flat_data = {k: torch.cat(data[k], dim=0) for k in data}
        flat_y = torch.cat(all_y, dim=0)

    print(f"  [{split_name}] collected {len(case_feats_list)} GT groups")
    return flat_data, flat_y, case_feats_list


def probe_one_seed(ckpt: Path, val_samples, test_samples, device, letterbox, args) -> dict:
    wrapper = YOLO(str(ckpt))
    net = wrapper.model.to(device).eval()

    hooked: dict = {}
    handles = []
    for key, layer_idx in [("c2f_fused", 18)]:
        def _hook(mod, inp, out, k=key):
            hooked[k] = out if isinstance(out, torch.Tensor) else out[0]
            hooked[k] = hooked[k].squeeze(0)
        handles.append(net.model[layer_idx].register_forward_hook(_hook))

    # ---- VAL ----
    val_data, val_y_cat, _ = collect_features(
        net, hooked, val_samples, device, letterbox, args, "VAL"
    )
    if not val_data:
        for h in handles:
            h.remove()
        return {}

    val_y_cat = val_y_cat.to(device)
    val_X = {k: val_data[k].to(device) for k in val_data}

    # Define representations
    rep_keys = [k for k in val_data.keys() if k != "m"]

    # Fit standalone probes
    probes_standalone = {}
    probes_combined = {}

    # 1. Fit baseline M
    probes_standalone["m"] = train_probe(val_X["m"], val_y_cat, args.epochs)

    # 2. Fit standalone and combined for each auxiliary key
    for k in rep_keys:
        probes_standalone[k] = train_probe(val_X[k], val_y_cat, args.epochs)
        # Combined [M, R]
        combined_val = torch.cat([val_X["m"], val_X[k]], dim=1)
        probes_combined[k] = train_probe(combined_val, val_y_cat, args.epochs)

    # ---- TEST ----
    _, _, test_cases = collect_features(
        net, hooked, test_samples, device, letterbox, args, "TEST"
    )

    for h in handles:
        h.remove()

    # Pre-evaluate baseline M to collect base_scores_case for rescue/damage computations
    base_scores_list = []
    for case_feats, local_ious, y in test_cases:
        with torch.no_grad():
            x_m = case_feats["m"].to(device)
            m_scores = torch.sigmoid(probes_standalone["m"](x_m).squeeze(1))
            base_scores_list.append(m_scores.cpu())

    # Evaluate all variants
    variant_results = {}
    
    # Baseline M evaluation
    m_evals = []
    for idx, (case_feats, local_ious, y) in enumerate(test_cases):
        res = evaluate_case(probes_standalone["m"], case_feats["m"].to(device), local_ious, y)
        m_evals.append(res)
    variant_results["m"] = m_evals

    # Standalone and Combined evaluations
    for k in rep_keys:
        # Standalone
        st_evals = []
        for idx, (case_feats, local_ious, y) in enumerate(test_cases):
            res = evaluate_case(probes_standalone[k], case_feats[k].to(device), local_ious, y)
            st_evals.append(res)
        variant_results[k] = st_evals

        # Combined
        cb_evals = []
        for idx, (case_feats, local_ious, y) in enumerate(test_cases):
            comb_x = torch.cat([case_feats["m"], case_feats[k]], dim=1).to(device)
            res = evaluate_case(probes_combined[k], comb_x, local_ious, y, base_scores_case=base_scores_list[idx])
            cb_evals.append(res)
        variant_results[f"m_{k}"] = cb_evals

    # Average metrics over all test cases
    summary = {}
    for key, rows in variant_results.items():
        if not rows:
            continue
        
        # Calculate positive-positive pair statistics globally (weighted sum)
        total_p = sum(r["total_pairs"] for r in rows)
        correct_p = sum(r["correct_pairs"] for r in rows)
        pair_acc = float(correct_p / total_p) if total_p > 0 else float("nan")

        rescue_den = sum(r["rescue_den"] for r in rows)
        rescue_num = sum(r["rescue_num"] for r in rows)
        rescue_rate = float(rescue_num / rescue_den) if rescue_den > 0 else float("nan")

        damage_den = sum(r["damage_den"] for r in rows)
        damage_num = sum(r["damage_num"] for r in rows)
        damage_rate = float(damage_num / damage_den) if damage_den > 0 else float("nan")

        # Average other scalar metrics
        keys_to_avg = ["ap", "auc", "rank", "spearman", "recall_at_1", "recall_at_5", "regret"]
        summary[key] = {
            mk: float(np.mean([r[mk] for r in rows if not np.isnan(r[mk])]))
            if [r[mk] for r in rows if not np.isnan(r[mk])] else float("nan")
            for mk in keys_to_avg
        }
        summary[key].update({
            "pair_acc": pair_acc,
            "rescue_rate": rescue_rate,
            "damage_rate": damage_rate,
            "count": len(rows),
        })

    return summary


def average_seed_results(seed_results: list[dict]) -> dict:
    valid = [r for r in seed_results if r]
    if not valid:
        return {}
    all_keys = set().union(*(r.keys() for r in valid))
    out = {}
    for k in all_keys:
        rows = [r[k] for r in valid if k in r]
        metric_keys = rows[0].keys()
        out[k] = {
            mk: float(np.mean([r[mk] for r in rows if not np.isnan(r[mk])])
                      if [r[mk] for r in rows if not np.isnan(r[mk])] else float("nan"))
            for mk in metric_keys
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--ckpt-root", type=Path,
                        default=ROOT / "runs/levir_yolov8n_p2_plain")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    random.seed(42)
    torch.manual_seed(42)

    val_samples = load_split_samples(args.dataset_root, "val")
    test_samples = load_split_samples(args.dataset_root, "test")
    print(f"VAL: {len(val_samples)} images, TEST: {len(test_samples)} images")

    if args.smoke_only:
        val_samples = [s for s in val_samples if len(s["boxes"]) > 0][:20]
        test_samples = [s for s in test_samples if len(s["boxes"]) > 0][:20]
        print(f"[SMOKE] VAL: {len(val_samples)}, TEST: {len(test_samples)}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    seed_results = []
    used_ckpts = {}
    for seed in args.seeds:
        ckpt = args.ckpt_root / f"seed_{seed}" / "weights" / "best.pt"
        if not ckpt.is_file():
            print(f"[SKIP] seed {seed}: {ckpt}")
            continue
        print(f"\n=== seed {seed} ===")
        used_ckpts[seed] = str(ckpt)
        result = probe_one_seed(ckpt, val_samples, test_samples, device, letterbox, args)
        seed_results.append(result)

    if not seed_results:
        print("No valid seed results.")
        sys.exit(1)

    averaged = average_seed_results(seed_results)
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_plain_p2_chroma_results.json"

    output = {
        "protocol": {
            "model": "yolov8n_p2_baseline",
            "hf_repo": "duyle2408/levir-ship-yolo-p2",
            "nms_iou": 0.5,
            "split_eval": "test",
            "probe_trained_on": "val",
            "positive_def": "IoU >= 0.1",
            "seeds": list(used_ckpts.keys()),
        },
        "per_seed": seed_results,
        "averaged": averaged,
    }

    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nWrote results to {out_file}")

    # Output clean tables
    print("\n" + "="*80)
    print("=== SUMMARY AVERAGED PROBING RESULTS ===")
    print("="*80)

    # Filter to print Stage A Table
    print("\nStage A - Patch Representations (3x3 and 5x5 cells)")
    print("| Representation | PairAcc | Best Rank | Spearman | Recall@1 | Regret | Rescue | Damage |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    # Baseline M
    m = averaged["m"]
    print(f"| **M (c2f_fused)** | {m['pair_acc']:.4f} | {m['rank']:.2f} | {m['spearman']:.4f} | {m['recall_at_1']:.4f} | {m['regret']:.4f} | — | — |")
    
    stage_a_keys = [
        ("RGB 3x3", "rgb_3"), ("RGB 5x5", "rgb_5"),
        ("Y 3x3", "y_3"), ("Y 5x5", "y_5"),
        ("CbCr 3x3", "cbcr_3"), ("CbCr 5x5", "cbcr_5"),
        ("Opponent 3x3", "opp_3"), ("Opponent 5x5", "opp_5")
    ]
    for label, key in stage_a_keys:
        st = averaged[key]
        cb = averaged[f"m_{key}"]
        print(f"| {label} (Standalone) | {st['pair_acc']:.4f} | {st['rank']:.2f} | {st['spearman']:.4f} | {st['recall_at_1']:.4f} | {st['regret']:.4f} | — | — |")
        print(f"| **[M, {label}]** | **{cb['pair_acc']:.4f}** | **{cb['rank']:.2f}** | **{cb['spearman']:.4f}** | **{cb['recall_at_1']:.4f}** | **{cb['regret']:.4f}** | {cb['rescue_rate']:.4f} | {cb['damage_rate']:.4f} |")

    print("\nStage B - Geometry & Box-aligned Representations")
    print("| Representation | PairAcc | Best Rank | Spearman | Recall@1 | Regret | Rescue | Damage |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    print(f"| **M (c2f_fused)** | {m['pair_acc']:.4f} | {m['rank']:.2f} | {m['spearman']:.4f} | {m['recall_at_1']:.4f} | {m['regret']:.4f} | — | — |")
    
    stage_b_keys = [
        ("Inner/Outer (12d)", "inner_outer"),
        ("Four-side (16d)", "four_side"),
        ("Spatial Map 5x5 (50d)", "spatial_map")
    ]
    for label, key in stage_b_keys:
        st = averaged[key]
        cb = averaged[f"m_{key}"]
        print(f"| {label} (Standalone) | {st['pair_acc']:.4f} | {st['rank']:.2f} | {st['spearman']:.4f} | {st['recall_at_1']:.4f} | {st['regret']:.4f} | — | — |")
        print(f"| **[M, {label}]** | **{cb['pair_acc']:.4f}** | **{cb['rank']:.2f}** | **{cb['spearman']:.4f}** | **{cb['recall_at_1']:.4f}** | **{cb['regret']:.4f}** | {cb['rescue_rate']:.4f} | {cb['damage_rate']:.4f} |")


if __name__ == "__main__":
    main()
