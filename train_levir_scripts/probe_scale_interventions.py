#!/usr/bin/env python3
"""Unified localized scale interaction probing via inference interventions on contrast_no_cross."""

from __future__ import annotations

import argparse
import json
import random
import sys
import yaml
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402

from probe_center_ring_cohorts import iou_matrix, read_labels  # noqa: E402
from probe_contrast_basis_stems import ap_auc, train_probe, evaluate_probe, load_split_samples, get_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--final-conf", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    random.seed(42)
    torch.manual_seed(42)

    ckpt_path = ROOT / "runs/checkpoints_full/contrast_no_cross_best.pt"
    if not ckpt_path.is_file():
        print(f"contrast_no_cross checkpoint not found at {ckpt_path}. Exiting.")
        sys.exit(1)

    print("Loading contrast_no_cross checkpoint...")
    wrapper = YOLO(ckpt_path)
    net = wrapper.model.to(device).eval()

    val_samples = load_split_samples(args.dataset_root, "val")
    test_samples = load_split_samples(args.dataset_root, "test")

    if args.smoke_only:
        val_samples = [s for s in val_samples if len(s["boxes"]) > 0][:20]
        test_samples = [s for s in test_samples if len(s["boxes"]) > 0][:20]
        print(f"[SMOKE] Filtered VAL samples: {len(val_samples)}, TEST samples: {len(test_samples)}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    # Hooks setup to intercept rel_encoder outputs
    rel_encoder_calls = []

    def hook_fn(_mod, _inp, out):
        if isinstance(out, tuple):
            rel_encoder_calls.append(out[0].squeeze(0))
        else:
            rel_encoder_calls.append(out.squeeze(0))

    stem = net.model[0]
    handle = stem.rel_encoder.register_forward_hook(hook_fn)

    stages = ["R4_dual", "R4_large_only", "R4_small_only", "R4_shuffle_small"]
    val_data = {k: [] for k in stages}
    val_ious, val_y = [], []

    # VAL collection
    for sample in val_samples:
        original = cv2.imread(str(sample["image"]))
        if original is None:
            continue
        gt_boxes = sample["boxes"].to(device)
        if len(gt_boxes) == 0:
            continue
        gt_xyxy = xywh2xyxy(gt_boxes)
        gt_xyxy[:, [0, 2]] *= original.shape[1]
        gt_xyxy[:, [1, 3]] *= original.shape[0]

        for gt in gt_xyxy:
            rel_encoder_calls.clear()
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0 or len(rel_encoder_calls) < 2:
                continue

            rs = rel_encoder_calls[0] # [C, H, W]
            rl = rel_encoder_calls[1] # [C, H, W]

            # We perform inference intervention directly on the scale_state tensor at full map resolution
            # before scale_formation gets called, and slice the resulting hooked R4
            with torch.no_grad():
                # Normal forward R4
                # S_dual = [rs, rl, rs, rl]
                s_dual = torch.cat((rs, rl, rs, rl), dim=0).unsqueeze(0)
                r4_dual = stem.scale_formation(s_dual).squeeze(0)

                # Large-only: S_large = [rl, rl, rl, rl]
                s_large = torch.cat((rl, rl, rl, rl), dim=0).unsqueeze(0)
                r4_large = stem.scale_formation(s_large).squeeze(0)

                # Small-only: S_small = [rs, rs, rs, rs]
                s_small = torch.cat((rs, rs, rs, rs), dim=0).unsqueeze(0)
                r4_small = stem.scale_formation(s_small).squeeze(0)

                # Shuffle-small: S_shuffle_small = [rs_shuffled, rl, rs_shuffled, rl]
                c_rs, h_rs, w_rs = rs.shape
                rs_flat = rs.permute(1, 2, 0).reshape(h_rs * w_rs, c_rs)
                rs_shuffled_flat = rs_flat[torch.randperm(len(rs_flat), device=device)]
                rs_shuffled = rs_shuffled_flat.reshape(h_rs, w_rs, c_rs).permute(2, 0, 1)
                
                s_shuf = torch.cat((rs_shuffled, rl, rs_shuffled, rl), dim=0).unsqueeze(0)
                r4_shuf = stem.scale_formation(s_shuf).squeeze(0)

            c_r4, h_r4, w_r4 = r4_dual.shape
            val_data["R4_dual"].append(r4_dual.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].cpu().detach().clone())
            val_data["R4_large_only"].append(r4_large.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].cpu().detach().clone())
            val_data["R4_small_only"].append(r4_small.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].cpu().detach().clone())
            val_data["R4_shuffle_small"].append(r4_shuf.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].cpu().detach().clone())
            
            val_ious.append(local_ious.cpu().detach().clone())
            val_y.append(y.cpu().detach().clone())

    if not val_y:
        print("No valid candidates collected in VAL split.")
        handle.remove()
        sys.exit(1)

    val_X = {k: torch.cat(val_data[k], dim=0).to(device) for k in stages}
    val_y_cat = torch.cat(val_y, dim=0).to(device)

    # Train Unified Linear Probes on VAL
    probes = {}
    for k in stages:
        probes[k] = train_probe(val_X[k].detach(), val_y_cat, args.epochs)

    # TEST collection
    test_case_groups = []
    for sample in test_samples:
        original = cv2.imread(str(sample["image"]))
        if original is None:
            continue
        gt_boxes = sample["boxes"].to(device)
        if len(gt_boxes) == 0:
            continue
        gt_xyxy = xywh2xyxy(gt_boxes)
        gt_xyxy[:, [0, 2]] *= original.shape[1]
        gt_xyxy[:, [1, 3]] *= original.shape[0]

        for gt in gt_xyxy:
            rel_encoder_calls.clear()
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0 or len(rel_encoder_calls) < 2:
                continue

            rs = rel_encoder_calls[0]
            rl = rel_encoder_calls[1]

            with torch.no_grad():
                s_dual = torch.cat((rs, rl, rs, rl), dim=0).unsqueeze(0)
                r4_dual = stem.scale_formation(s_dual).squeeze(0)

                s_large = torch.cat((rl, rl, rl, rl), dim=0).unsqueeze(0)
                r4_large = stem.scale_formation(s_large).squeeze(0)

                s_small = torch.cat((rs, rs, rs, rs), dim=0).unsqueeze(0)
                r4_small = stem.scale_formation(s_small).squeeze(0)

                c_rs, h_rs, w_rs = rs.shape
                rs_flat = rs.permute(1, 2, 0).reshape(h_rs * w_rs, c_rs)
                rs_shuffled_flat = rs_flat[torch.randperm(len(rs_flat), device=device)]
                rs_shuffled = rs_shuffled_flat.reshape(h_rs, w_rs, c_rs).permute(2, 0, 1)
                
                s_shuf = torch.cat((rs_shuffled, rl, rs_shuffled, rl), dim=0).unsqueeze(0)
                r4_shuf = stem.scale_formation(s_shuf).squeeze(0)

            c_r4, h_r4, w_r4 = r4_dual.shape
            case_rep = {
                "R4_dual": r4_dual.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].to(device),
                "R4_large_only": r4_large.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].to(device),
                "R4_small_only": r4_small.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].to(device),
                "R4_shuffle_small": r4_shuf.permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].to(device),
            }
            test_case_groups.append((case_rep, local_ious, y))

    handle.remove()

    # Evaluate on TEST split
    final_results = {}
    for k in stages:
        case_results = []
        for case_rep, local_ious, y in test_case_groups:
            eval_metrics = evaluate_probe(probes[k], case_rep[k].detach(), local_ious, y)
            case_results.append(eval_metrics)

        # Average metrics
        summary_metrics = {}
        keys = case_results[0].keys()
        for key in keys:
            vals = [r[key] for r in case_results if not np.isnan(r[key])]
            summary_metrics[key] = float(np.mean(vals)) if vals else float("nan")
        summary_metrics["count"] = len(case_results)
        final_results[k] = summary_metrics

    # Save and output
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_scale_interventions_results.json"
    
    output = {
        "protocol": {
            "split": "test",
            "nms_iou_for_cohort_selection": 0.5,
            "raw_candidate_source": "P2 decoded pre-NMS head output",
            "candidate_conf": None,
            "positive": "IoU >= 0.1",
            "checkpoint": str(ckpt_path),
        },
        "summary": final_results,
    }
    
    out_file.write_text(json.dumps(output, indent=2))
    print("\nSCALE INTERVENTION PROBING RESULTS:")
    print(json.dumps(output, indent=2))
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
