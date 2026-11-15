#!/usr/bin/env python3
"""Unified localized probing stem feature-formation ablation analysis."""

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
from ultralytics.nn.modules import C2f  # noqa: E402

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

    stem = net.model[0]
    register_hook("M3", stem.main_c2f)
    register_hook("R3", stem.rel_encoder[2])
    register_hook("R4", stem.scale_formation)

    # VAL collection
    val_M3_list, val_R3_list, val_R4_list = [], [], []
    val_ious, val_y = [], []

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
            idx, local_ious, y = get_candidates(
                net, original.shape, letterbox, sample["image"], gt, device
            )
            if len(idx) < 2 or y.sum() == 0:
                continue

            c_m, h_m, w_m = hooked_tensors["M3"].shape
            sliced_M3 = hooked_tensors["M3"].permute(1, 2, 0).reshape(h_m * w_m, c_m)[idx].detach().cpu().clone()
            
            c_r3, h_r3, w_r3 = hooked_tensors["R3"].shape
            sliced_R3 = hooked_tensors["R3"].permute(1, 2, 0).reshape(h_r3 * w_r3, c_r3)[idx].detach().cpu().clone()
            
            c_r4, h_r4, w_r4 = hooked_tensors["R4"].shape
            sliced_R4 = hooked_tensors["R4"].permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].detach().cpu().clone()

            val_M3_list.append(sliced_M3)
            val_R3_list.append(sliced_R3)
            val_R4_list.append(sliced_R4)
            val_ious.append(local_ious.cpu().detach().clone())
            val_y.append(y.cpu().detach().clone())

    if not val_y:
        print("No valid candidates collected in VAL split.")
        sys.exit(1)

    val_M3 = torch.cat(val_M3_list, dim=0).to(device)
    val_R3 = torch.cat(val_R3_list, dim=0).to(device)
    val_R4 = torch.cat(val_R4_list, dim=0).to(device)
    val_y_cat = torch.cat(val_y, dim=0).to(device)

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

            c_m, h_m, w_m = hooked_tensors["M3"].shape
            sliced_M3 = hooked_tensors["M3"].permute(1, 2, 0).reshape(h_m * w_m, c_m)[idx].detach().cpu().clone()
            
            c_r3, h_r3, w_r3 = hooked_tensors["R3"].shape
            sliced_R3 = hooked_tensors["R3"].permute(1, 2, 0).reshape(h_r3 * w_r3, c_r3)[idx].detach().cpu().clone()
            
            c_r4, h_r4, w_r4 = hooked_tensors["R4"].shape
            sliced_R4 = hooked_tensors["R4"].permute(1, 2, 0).reshape(h_r4 * w_r4, c_r4)[idx].detach().cpu().clone()

            test_case_groups.append((sliced_M3.to(device), sliced_R3.to(device), sliced_R4.to(device), local_ious.to(device), y.to(device)))

    for h in handles:
        h.remove()

    # Define the 6 ablation formations
    c_m3 = val_M3.shape[1]
    c_r3 = val_R3.shape[1]
    
    # Modules to fit parameters
    r3_to_conv1x1 = nn.Conv2d(c_r3, 32, 1).to(device)
    r3_to_c2f = C2f(c_r3, 32, n=1, shortcut=False).to(device)
    m3_to_c2f = C2f(c_m3, 32, n=1, shortcut=False).to(device)

    # Train modules on VAL first to learn expansion/consolidation mapping
    # Since our inputs are flattened candidate points [N, C], we reshape to [N, C, 1, 1] for Conv2d/C2f
    val_R3_4d = val_R3.unsqueeze(2).unsqueeze(3)
    val_M3_4d = val_M3.unsqueeze(2).unsqueeze(3)
    
    optimizer = optim.Adam(list(r3_to_conv1x1.parameters()) + list(r3_to_c2f.parameters()) + list(m3_to_c2f.parameters()), lr=0.01)
    
    # We train the modules jointly with a dummy task or reconstruction?
    # No, we fit them jointly with the linear probe end-to-end to maximize cross-entropy classification on VAL!
    probe_conv = nn.Linear(32, 1).to(device)
    probe_c2f = nn.Linear(32, 1).to(device)
    probe_m4 = nn.Linear(32, 1).to(device)
    
    optimizer_all = optim.Adam(
        list(r3_to_conv1x1.parameters()) + list(r3_to_c2f.parameters()) + list(m3_to_c2f.parameters()) +
        list(probe_conv.parameters()) + list(probe_c2f.parameters()) + list(probe_m4.parameters()),
        lr=0.01
    )
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(val_y_cat == 0).sum().float() / (val_y_cat == 1).sum().float().clamp_min(1))
    
    print("Fitting expansion modules on VAL candidates...")
    for _ in range(args.epochs):
        optimizer_all.zero_grad()
        
        # Conv1x1 path
        out_conv = r3_to_conv1x1(val_R3_4d).squeeze(3).squeeze(2)
        loss_conv = loss_fn(probe_conv(out_conv).squeeze(1), val_y_cat.float())
        
        # C2f path
        out_c2f = r3_to_c2f(val_R3_4d).squeeze(3).squeeze(2)
        loss_c2f = loss_fn(probe_c2f(out_c2f).squeeze(1), val_y_cat.float())
        
        # M3 -> C2f path (M4)
        out_m4 = m3_to_c2f(val_M3_4d).squeeze(3).squeeze(2)
        loss_m4 = loss_fn(probe_m4(out_m4).squeeze(1), val_y_cat.float())
        
        total_loss = loss_conv + loss_c2f + loss_m4
        total_loss.backward()
        optimizer_all.step()

    # Fit unified probe baseline for raw M3, R3, R4
    probe_m3 = train_probe(val_M3, val_y_cat, args.epochs)
    probe_r3 = train_probe(val_R3, val_y_cat, args.epochs)
    probe_r4 = train_probe(val_R4, val_y_cat, args.epochs)

    # Evaluate on TEST split
    case_results = {
        "R3_baseline": [],
        "R3_Conv1x1": [],
        "R3_C2f": [],
        "R4_standard": [],
        "M3_baseline": [],
        "M3_C2f_M4": [],
    }

    r3_to_conv1x1.eval()
    r3_to_c2f.eval()
    m3_to_c2f.eval()

    for test_M3, test_R3, test_R4, local_ious, y in test_case_groups:
        test_R3_4d = test_R3.unsqueeze(2).unsqueeze(3)
        test_M3_4d = test_M3.unsqueeze(2).unsqueeze(3)
        
        with torch.no_grad():
            out_conv = r3_to_conv1x1(test_R3_4d).squeeze(3).squeeze(2)
            out_c2f = r3_to_c2f(test_R3_4d).squeeze(3).squeeze(2)
            out_m4 = m3_to_c2f(test_M3_4d).squeeze(3).squeeze(2)

        case_results["R3_baseline"].append(evaluate_probe(probe_r3, test_R3, local_ious, y))
        case_results["R3_Conv1x1"].append(evaluate_probe(probe_conv, out_conv, local_ious, y))
        case_results["R3_C2f"].append(evaluate_probe(probe_c2f, out_c2f, local_ious, y))
        case_results["R4_standard"].append(evaluate_probe(probe_r4, test_R4, local_ious, y))
        case_results["M3_baseline"].append(evaluate_probe(probe_m3, test_M3, local_ious, y))
        case_results["M3_C2f_M4"].append(evaluate_probe(probe_m4, out_m4, local_ious, y))

    # Average metrics
    summary_results = {}
    for name, vals in case_results.items():
        summary_results[name] = {}
        keys = vals[0].keys()
        for key in keys:
            v_list = [r[key] for r in vals if not np.isnan(r[key])]
            summary_results[name][key] = float(np.mean(v_list)) if v_list else float("nan")
        summary_results[name]["count"] = len(vals)

    # Save to file
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_formation_ablation_results.json"
    
    output = {
        "protocol": {
            "split": "test",
            "nms_iou_for_cohort_selection": 0.5,
            "raw_candidate_source": "P2 decoded pre-NMS head output",
            "candidate_conf": None,
            "positive": "IoU >= 0.1",
            "checkpoint": str(ckpt_path),
        },
        "summary": summary_results,
    }
    
    out_file.write_text(json.dumps(output, indent=2))
    print("\nFORMATION ABLATION RESULTS:")
    print(json.dumps(output, indent=2))
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
