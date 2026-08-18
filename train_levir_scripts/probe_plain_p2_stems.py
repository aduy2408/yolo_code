#!/usr/bin/env python3
"""Unified color-space and backbone/FPN linear probing script.

Evaluates baseline B (backbone c2f), baseline F (fused FPN c2f_fused),
and auxiliary color spaces (RGB, Y, CbCr, Opponent) mapped directly via GPU grid pooling.
Computes PairAcc, Rank, Spearman Rho, Recall@1/5, Regret, Rescue and Damage rates.
Averages results across seeds 42, 43, 44 using checkpoints from HuggingFace.

Optimizations:
1. Runs YOLO forward pass once per image.
2. Performs all color space mapping and pooling once per image (no loop or crop bottlenecks).
3. Leverages GPU-native AvgPool2d to construct multi-scale context features.
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
# GPU Multiscale Grid Feature Extraction
# ---------------------------------------------------------------------------

def compute_chroma_features_gpu(image_bgr_letter: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    """Convert BGR image to RGB/Y/CbCr/Opponent spaces and perform multi-scale grid pooling on GPU."""
    img_t = torch.from_numpy(image_bgr_letter[..., ::-1].copy()).to(device).float() / 255.0 # (H, W, 3)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0) # (1, 3, 512, 512)
    
    r = img_t[:, 0:1]
    g = img_t[:, 1:2]
    b = img_t[:, 2:3]
    
    # Y
    y = 0.299 * r + 0.587 * g + 0.114 * b
    
    # CbCr
    cb = 0.5 - 0.1687 * r - 0.3313 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.4187 * g - 0.0813 * b
    cbcr = torch.cat([cb, cr], dim=1) # (1, 2, 512, 512)
    
    # Opponent
    o1 = r - g
    o2 = b - 0.5 * (r + g)
    opp = torch.cat([o1, o2], dim=1) # (1, 2, 512, 512)
    
    # Pool to stride 4 (128x128 P2 resolution)
    rgb0 = F.avg_pool2d(img_t, kernel_size=4, stride=4) # (1, 3, 128, 128)
    y0 = F.avg_pool2d(y, kernel_size=4, stride=4) # (1, 1, 128, 128)
    c0 = F.avg_pool2d(cbcr, kernel_size=4, stride=4) # (1, 2, 128, 128)
    opp0 = F.avg_pool2d(opp, kernel_size=4, stride=4) # (1, 2, 128, 128)
    
    # Compute multi-scale context maps
    features = {}
    for name, t0 in [("rgb", rgb0), ("y", y0), ("cbcr", c0), ("opp", opp0)]:
        t3 = F.avg_pool2d(t0, kernel_size=3, stride=1, padding=1)
        t5 = F.avg_pool2d(t0, kernel_size=5, stride=1, padding=2)
        # Concatenate along channel dimension
        features[name] = torch.cat([t0, t3, t5], dim=1).squeeze(0) # (C_total, 128, 128)
        
    return features


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

        # 1. Run YOLO forward pass ONCE per image (requires 512x512 letterbox input)
        original_letter = letterbox(image=original)
        tensor = (
            torch.from_numpy(original_letter[..., ::-1].copy())
            .to(device).permute(2, 0, 1).float()[None] / 255.0
        )
        with torch.no_grad():
            decoded, preds = net(tensor)

        # 2. Extract backbone c2f (layer 2) and FPN c2f_fused (layer 18)
        if "c2f" not in hooked or "c2f_fused" not in hooked:
            continue
        t_backbone = hooked["c2f"]  # (C_b, 128, 128)
        t_fused = hooked["c2f_fused"]  # (C_f, 128, 128)
        
        c_b, h_b, w_b = t_backbone.shape
        c_f, h_f, w_f = t_fused.shape
        n = h_f * w_f

        # 3. Build chroma and other color maps at stride 4 (128x128 grid)
        gpu_color_features = compute_chroma_features_gpu(original_letter, device)

        for gt in gt_xyxy:
            idx, local_ious, y, boxes = get_candidates_precomputed(
                decoded, preds, original.shape, gt, device
            )
            safe_idx = idx[idx < n]
            if len(safe_idx) < 2 or y.sum() == 0:
                continue

            grid_y = safe_idx // w_f
            grid_x = safe_idx % w_f

            # Extract features in one vectorized GPU slice
            b_feats = t_backbone[:, grid_y, grid_x].permute(1, 0).cpu() # (num_candidates, C_b)
            f_feats = t_fused[:, grid_y, grid_x].permute(1, 0).cpu() # (num_candidates, C_f)

            case_feats = {
                "b": b_feats,
                "f": f_feats,
            }
            for name, feat_map in gpu_color_features.items():
                case_feats[name] = feat_map[:, grid_y, grid_x].permute(1, 0).cpu()

            case_feats_list.append((case_feats, local_ious.cpu(), y.cpu()))

            # Append to flat lists for VAL training
            for key in case_feats:
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
    # Hook Layer 2 (backbone P2 output c2f) and Layer 18 (FPN output c2f_fused)
    for key, layer_idx in [("c2f", 2), ("c2f_fused", 18)]:
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
    rep_keys = ["rgb", "y", "cbcr", "opp"]

    # Fit standalone and combined probes
    probes = {}
    
    # Standalones B and F
    probes["b"] = train_probe(val_X["b"], val_y_cat, args.epochs)
    probes["f"] = train_probe(val_X["f"], val_y_cat, args.epochs)
    
    # Combined [F, B]
    probes["f_b"] = train_probe(torch.cat([val_X["f"], val_X["b"]], dim=1), val_y_cat, args.epochs)

    # Auxiliary color space probes
    for c in rep_keys:
        # Standalone color probe
        probes[c] = train_probe(val_X[c], val_y_cat, args.epochs)
        # Combined [B, C]
        probes[f"b_{c}"] = train_probe(torch.cat([val_X["b"], val_X[c]], dim=1), val_y_cat, args.epochs)
        # Combined [F, C]
        probes[f"f_{c}"] = train_probe(torch.cat([val_X["f"], val_X[c]], dim=1), val_y_cat, args.epochs)
        # Combined [F, B, C]
        probes[f"f_b_{c}"] = train_probe(torch.cat([val_X["f"], val_X["b"], val_X[c]], dim=1), val_y_cat, args.epochs)

    # ---- TEST ----
    _, _, test_cases = collect_features(
        net, hooked, test_samples, device, letterbox, args, "TEST"
    )

    for h in handles:
        h.remove()

    # Pre-evaluate baseline F (fused) to collect base_scores_case for rescue/damage computations
    base_scores_list = []
    for case_feats, local_ious, y in test_cases:
        with torch.no_grad():
            x_f = case_feats["f"].to(device)
            f_scores = torch.sigmoid(probes["f"](x_f).squeeze(1))
            base_scores_list.append(f_scores.cpu())

    # Evaluate all variants
    variant_results = {}
    
    # Define keys to evaluate
    eval_keys = ["b", "f", "f_b"]
    for c in rep_keys:
        eval_keys.extend([c, f"b_{c}", f"f_{c}", f"f_b_{c}"])

    for key in eval_keys:
        evals = []
        for idx, (case_feats, local_ious, y) in enumerate(test_cases):
            # Construct input feature tensor
            if key == "b":
                x_in = case_feats["b"]
            elif key == "f":
                x_in = case_feats["f"]
            elif key == "f_b":
                x_in = torch.cat([case_feats["f"], case_feats["b"]], dim=1)
            elif key in rep_keys:
                x_in = case_feats[key]
            elif key.startswith("b_"):
                c_name = key.split("_")[1]
                x_in = torch.cat([case_feats["b"], case_feats[c_name]], dim=1)
            elif key.startswith("f_b_"):
                c_name = key.split("_")[2]
                x_in = torch.cat([case_feats["f"], case_feats["b"], case_feats[c_name]], dim=1)
            elif key.startswith("f_"):
                c_name = key.split("_")[1]
                x_in = torch.cat([case_feats["f"], case_feats[c_name]], dim=1)
                
            res = evaluate_case(probes[key], x_in.to(device), local_ious, y, base_scores_case=base_scores_list[idx])
            evals.append(res)
        variant_results[key] = evals

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

    # We print a structured table of baselines and auxiliary representations
    print("\nProbing Results Table (Averaged 3 seeds)")
    print("| Representation | PairAcc | Best Rank | Spearman | Recall@1 | Regret | Rescue (on F) | Damage (on F) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    # Baselines
    b = averaged["b"]
    f = averaged["f"]
    fb = averaged["f_b"]
    print(f"| **B (backbone P2)** | {b['pair_acc']:.4f} | {b['rank']:.2f} | {b['spearman']:.4f} | {b['recall_at_1']:.4f} | {b['regret']:.4f} | — | — |")
    print(f"| **F (fused FPN P2)** | {f['pair_acc']:.4f} | {f['rank']:.2f} | {f['spearman']:.4f} | {f['recall_at_1']:.4f} | {f['regret']:.4f} | — | — |")
    print(f"| **[F, B]** | {fb['pair_acc']:.4f} | {fb['rank']:.2f} | {fb['spearman']:.4f} | {fb['recall_at_1']:.4f} | {fb['regret']:.4f} | {fb['rescue_rate']:.4f} | {fb['damage_rate']:.4f} |")

    # Color space variants
    for label, c_key in [("RGB", "rgb"), ("Y (Luminance)", "y"), ("CbCr (Chroma)", "cbcr"), ("Opponent", "opp")]:
        c = averaged[c_key]
        bc = averaged[f"b_{c_key}"]
        fc = averaged[f"f_{c_key}"]
        fbc = averaged[f"f_b_{c_key}"]
        
        print(f"|--- | --- | --- | --- | --- | --- | --- | --- |")
        print(f"| {label} (Standalone) | {c['pair_acc']:.4f} | {c['rank']:.2f} | {c['spearman']:.4f} | {c['recall_at_1']:.4f} | {c['regret']:.4f} | — | — |")
        print(f"| **[B, {label}]** | {bc['pair_acc']:.4f} | {bc['rank']:.2f} | {bc['spearman']:.4f} | {bc['recall_at_1']:.4f} | {bc['regret']:.4f} | — | — |")
        print(f"| **[F, {label}]** | **{fc['pair_acc']:.4f}** | **{fc['rank']:.2f}** | **{fc['spearman']:.4f}** | **{fc['recall_at_1']:.4f}** | **{fc['regret']:.4f}** | {fc['rescue_rate']:.4f} | {fc['damage_rate']:.4f} |")
        print(f"| **[F, B, {label}]** | **{fbc['pair_acc']:.4f}** | **{fbc['rank']:.2f}** | **{fbc['spearman']:.4f}** | **{fbc['recall_at_1']:.4f}** | **{fbc['regret']:.4f}** | {fbc['rescue_rate']:.4f} | {fbc['damage_rate']:.4f} |")


if __name__ == "__main__":
    main()
