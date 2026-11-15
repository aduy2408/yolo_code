#!/usr/bin/env python3
"""Unified localized probing stage-wise topological analysis for raw_control."""

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

    ckpt_path = ROOT / "runs/levir_yolov8n_p2_contrast_basis/_smoke/raw_control/weights/best.pt"
    if not ckpt_path.is_file():
        print(f"raw_control checkpoint not found at {ckpt_path}. Exiting.")
        sys.exit(1)

    print("Loading raw_control model checkpoint...")
    wrapper = YOLO(ckpt_path)
    net = wrapper.model.to(device).eval()

    val_samples = load_split_samples(args.dataset_root, "val")
    test_samples = load_split_samples(args.dataset_root, "test")

    if args.smoke_only:
        val_samples = [s for s in val_samples if len(s["boxes"]) > 0][:20]
        test_samples = [s for s in test_samples if len(s["boxes"]) > 0][:20]
        print(f"[SMOKE] Filtered VAL samples: {len(val_samples)}, TEST samples: {len(test_samples)}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    # Hooks setup
    hooked_tensors = {}
    handles = []

    def register_hook(name, module):
        def hook_fn(_mod, _inp, out):
            if isinstance(out, tuple):
                hooked_tensors[name] = out[0].squeeze(0)
            else:
                hooked_tensors[name] = out.squeeze(0)
        handles.append(module.register_forward_hook(hook_fn))

    # Stride-4 layer hooks mapping
    stem = net.model[0]
    register_hook("M2", stem.main_cv2)
    register_hook("M3", stem.main_c2f)
    register_hook("R2", stem.rel_encoder[1])
    register_hook("R3", stem.rel_encoder[2])
    register_hook("R4", stem.scale_formation)

    stages = ["M2", "M3", "R2", "R3", "[R3,R3_squared]", "R4", "R3_no_mag", "R4_no_mag"]
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
            # 1. Standard Forward Pass
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0:
                continue

            # Slice standard features
            feats = {}
            for name in ["M2", "M3", "R2", "R3", "R4"]:
                t = hooked_tensors[name]
                c, h_f, w_f = t.shape
                feats[name] = t.permute(1, 2, 0).reshape(h_f * w_f, c)[idx].detach().cpu().clone()

            # Construct [R3, R3^2]
            feats["[R3,R3_squared]"] = torch.cat([feats["R3"], feats["R3"] ** 2], dim=1)

            # 2. Intervention Forward Pass: Mask magnitude channel to zero
            original_raw_basis = stem._raw_basis
            
            def no_mag_raw_basis(x: torch.Tensor) -> torch.Tensor:
                base = original_raw_basis(x)
                base[:, -1:, :, :] = 0.0
                return base
            
            stem._raw_basis = no_mag_raw_basis
            
            with torch.no_grad():
                image = letterbox(image=original)
                tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255.0
                net(tensor)
                
            for name in ["R3", "R4"]:
                t = hooked_tensors[name]
                c, h_f, w_f = t.shape
                feats[f"{name}_no_mag"] = t.permute(1, 2, 0).reshape(h_f * w_f, c)[idx].detach().cpu().clone()
            
            # Restore stem function
            stem._raw_basis = original_raw_basis

            for k in stages:
                val_data[k].append(feats[k])
            val_ious.append(local_ious.cpu().detach().clone())
            val_y.append(y.cpu().detach().clone())

    if not val_y:
        print("No valid candidates were collected in VAL split.")
        for h in handles:
            h.remove()
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
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0:
                continue

            feats = {}
            for name in ["M2", "M3", "R2", "R3", "R4"]:
                t = hooked_tensors[name]
                c, h_f, w_f = t.shape
                feats[name] = t.permute(1, 2, 0).reshape(h_f * w_f, c)[idx].detach().cpu().clone()

            feats["[R3,R3_squared]"] = torch.cat([feats["R3"], feats["R3"] ** 2], dim=1)

            # Intervention Pass
            original_raw_basis = stem._raw_basis
            def no_mag_raw_basis(x: torch.Tensor) -> torch.Tensor:
                base = original_raw_basis(x)
                base[:, -1:, :, :] = 0.0
                return base
            stem._raw_basis = no_mag_raw_basis
            
            with torch.no_grad():
                image = letterbox(image=original)
                tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255.0
                net(tensor)
                
            for name in ["R3", "R4"]:
                t = hooked_tensors[name]
                c, h_f, w_f = t.shape
                feats[f"{name}_no_mag"] = t.permute(1, 2, 0).reshape(h_f * w_f, c)[idx].detach().cpu().clone()
            
            stem._raw_basis = original_raw_basis

            case_rep = {k: feats[k].to(device) for k in stages}
            test_case_groups.append((case_rep, local_ious, y))

    for h in handles:
        h.remove()

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
    out_file = out_dir / "probe_stage_wise_results.json"
    
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
    print("\nSTAGE-WISE PROBING RESULTS:")
    print(json.dumps(output, indent=2))
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
