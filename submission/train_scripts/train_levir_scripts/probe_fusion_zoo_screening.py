#!/usr/bin/env python3
"""Unified localized probing screening runner for 12 fusion operators and controls."""

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


class CustomProjection(nn.Module):
    """Helper module to project variables to d=32 dimension."""
    def __init__(self, in_features: int, out_features: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def apply_operator(op_idx: int, M: torch.Tensor, R: torch.Tensor, proj_M: nn.Module, proj_R: nn.Module, proj_joint: nn.Module, params: dict = None) -> torch.Tensor:
    """Apply one of the 12 fusion zoo operators returning a d=32 dimension tensor."""
    if op_idx == 1:
        # R_only
        return proj_R(R)
    elif op_idx == 2:
        # M_only
        return proj_M(M)
    elif op_idx == 3:
        # Normalized Concat
        mean_M = M.mean(dim=0, keepdim=True)
        std_M = M.std(dim=0, keepdim=True).clamp_min(1e-5)
        mean_R = R.mean(dim=0, keepdim=True)
        std_R = R.std(dim=0, keepdim=True).clamp_min(1e-5)
        norm_M = (M - mean_M) / std_M
        norm_R = (R - mean_R) / std_R
        return proj_joint(torch.cat([norm_M, norm_R], dim=1))
    elif op_idx == 4:
        # Projected Sum
        return proj_M(M) + proj_R(R)
    elif op_idx == 5:
        # R-anchored Residual
        alpha = params.get("alpha", 0.5) if params else 0.5
        return proj_R(R) + alpha * proj_M(M)
    elif op_idx == 6:
        # M-anchored Residual
        alpha = params.get("alpha", 0.5) if params else 0.5
        return proj_M(M) + alpha * proj_R(R)
    elif op_idx == 7:
        # Preserved Channel Split (split into 16 channels each)
        u = proj_M(M)[:, :16]
        v = proj_R(R)[:, :16]
        return torch.cat([u, v], dim=1)
    elif op_idx == 8:
        # Projected Hadamard
        u = proj_M(M)
        v = proj_R(R)
        return proj_joint(torch.cat([u, v, u * v], dim=1))
    elif op_idx == 9:
        # Difference Interaction
        u = proj_M(M)
        v = proj_R(R)
        return proj_joint(torch.cat([u, v, torch.abs(u - v)], dim=1))
    elif op_idx == 10:
        # Full Cheap Basis
        u = proj_M(M)
        v = proj_R(R)
        return proj_joint(torch.cat([u, v, u * v, torch.abs(u - v)], dim=1))
    elif op_idx == 11:
        # Low-rank Local Bilinear
        u = proj_M(M)[:, :16]
        v = proj_R(R)[:, :16]
        z = u * v
        # project combined M, R and z
        return proj_joint(torch.cat([M, R, z], dim=1))
    elif op_idx == 12:
        # Scalar Alpha Mixture
        alpha = params.get("alpha", 0.5) if params else 0.5
        return proj_joint(alpha * M + (1 - alpha) * R)
    raise ValueError(f"Unknown operator index: {op_idx}")


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

    checkpoints = {
        "contrast_basis": Path("runs/levir_yolov8n_p2_contrast_basis/_smoke/contrast_basis/weights/best.pt"),
        "contrast_basis_ffm": Path("runs/levir_yolov8n_p2_contrast_basis_ffm/_smoke/contrast_basis_ffm/weights/best.pt"),
    }

    available_ckpts = {name: Path(ROOT / path) for name, path in checkpoints.items() if (ROOT / path).is_file()}
    if not available_ckpts:
        print("No checkpoints found.")
        sys.exit(1)

    val_samples = load_split_samples(args.dataset_root, "val")
    test_samples = load_split_samples(args.dataset_root, "test")

    if args.smoke_only:
        val_samples = [s for s in val_samples if len(s["boxes"]) > 0][:20]
        test_samples = [s for s in test_samples if len(s["boxes"]) > 0][:20]
        print(f"[SMOKE] Filtered VAL samples: {len(val_samples)}, TEST samples: {len(test_samples)}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    final_results = {}

    operators = [
        {"idx": 1, "name": "R_only"},
        {"idx": 2, "name": "M_only"},
        {"idx": 3, "name": "Normalized_Concat"},
        {"idx": 4, "name": "Projected_Sum"},
        {"idx": 5, "name": "R-anchored_Residual_0.1", "params": {"alpha": 0.1}},
        {"idx": 5, "name": "R-anchored_Residual_0.5", "params": {"alpha": 0.5}},
        {"idx": 5, "name": "R-anchored_Residual_1.0", "params": {"alpha": 1.0}},
        {"idx": 6, "name": "M-anchored_Residual_0.1", "params": {"alpha": 0.1}},
        {"idx": 6, "name": "M-anchored_Residual_0.5", "params": {"alpha": 0.5}},
        {"idx": 6, "name": "M-anchored_Residual_1.0", "params": {"alpha": 1.0}},
        {"idx": 7, "name": "Preserved_Channel_Split"},
        {"idx": 8, "name": "Projected_Hadamard"},
        {"idx": 9, "name": "Difference_Interaction"},
        {"idx": 10, "name": "Full_Cheap_Basis"},
        {"idx": 11, "name": "Low-rank_Local_Bilinear"},
        {"idx": 12, "name": "Scalar_Mixture_0.25", "params": {"alpha": 0.25}},
        {"idx": 12, "name": "Scalar_Mixture_0.5", "params": {"alpha": 0.5}},
        {"idx": 12, "name": "Scalar_Mixture_0.75", "params": {"alpha": 0.75}},
    ]

    for variant, ckpt in available_ckpts.items():
        print(f"\n--- Probing variant: {variant} ---")
        wrapper = YOLO(ckpt)
        net = wrapper.model.to(device).eval()

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
        register_hook("M", stem.main_c2f)
        register_hook("R", stem.scale_formation)

        # Collect activations for VAL
        val_M_list, val_R_list = [], []
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

                # Slice hooked tensors with candidate indices
                c_m, h_m, w_m = hooked_tensors["M"].shape
                sliced_M = hooked_tensors["M"].permute(1, 2, 0).reshape(h_m * w_m, c_m)[idx]
                c_r, h_r, w_r = hooked_tensors["R"].shape
                sliced_R = hooked_tensors["R"].permute(1, 2, 0).reshape(h_r * w_r, c_r)[idx]

                # We detach and clone immediately to ensure absolutely no reference to autograd graph
                val_M_list.append(sliced_M.detach().cpu().clone())
                val_R_list.append(sliced_R.detach().cpu().clone())
                val_ious.append(local_ious.detach().cpu().clone())
                val_y.append(y.detach().cpu().clone())

        if not val_y:
            print(f"Skipping {variant} because no valid candidates were collected in VAL.")
            for h in handles:
                h.remove()
            continue

        val_M = torch.cat(val_M_list, dim=0).to(device)
        val_R = torch.cat(val_R_list, dim=0).to(device)
        val_y_cat = torch.cat(val_y, dim=0).to(device)

        # Collect activations for TEST
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

                c_m, h_m, w_m = hooked_tensors["M"].shape
                sliced_M = hooked_tensors["M"].permute(1, 2, 0).reshape(h_m * w_m, c_m)[idx]
                c_r, h_r, w_r = hooked_tensors["R"].shape
                sliced_R = hooked_tensors["R"].permute(1, 2, 0).reshape(h_r * w_r, c_r)[idx]

                test_case_groups.append((sliced_M.detach().cpu().clone().to(device), sliced_R.detach().cpu().clone().to(device), local_ious.detach().cpu().clone().to(device), y.detach().cpu().clone().to(device)))

        for h in handles:
            h.remove()

        final_results[variant] = {}

        # Screen each operator
        for op in operators:
            name = op["name"]
            print(f"  Screening operator: {name} ...")
            
            # Setup projections
            c_m, c_r = val_M.shape[1], val_R.shape[1]
            proj_M = CustomProjection(c_m, 32).to(device)
            proj_R = CustomProjection(c_r, 32).to(device)
            
            # Dimension depends on input channels
            if op["idx"] == 3:
                proj_joint = CustomProjection(c_m + c_r, 32).to(device)
            elif op["idx"] in {8, 9}:
                proj_joint = CustomProjection(96, 32).to(device)
            elif op["idx"] == 10:
                proj_joint = CustomProjection(128, 32).to(device)
            elif op["idx"] == 11:
                proj_joint = CustomProjection(c_m + c_r + 16, 32).to(device)
            elif op["idx"] == 12:
                proj_joint = CustomProjection(c_m, 32).to(device)
            else:
                proj_joint = None

            # Modes list: [Standard, R_shuffle, R_random_proj]
            modes = ["standard", "R_shuffle", "R_random_proj"]
            op_summary = {}

            for mode in modes:
                # 1. Apply controls to VAL features
                if mode == "R_shuffle":
                    val_R_in = val_R[torch.randperm(len(val_R), device=device)]
                elif mode == "R_random_proj":
                    # Random orthogonal projection control
                    with torch.no_grad():
                        W = torch.randn(c_r, c_r, device=device)
                        Q, _ = torch.linalg.qr(W)
                    val_R_in = val_R @ Q
                else:
                    val_R_in = val_R

                # Form operator representation on VAL
                val_F = apply_operator(op["idx"], val_M, val_R_in, proj_M, proj_R, proj_joint, op.get("params"))
                
                # Fit Unified Linear Probe on VAL
                # We detach the projection outputs to make sure backward gradients only flow through the linear probe parameter weights during fitting!
                probe = train_probe(val_F.detach(), val_y_cat, args.epochs)

                # Evaluate on TEST split
                case_results = []
                for test_M, test_R, local_ious, y in test_case_groups:
                    if mode == "R_shuffle":
                        test_R_in = test_R[torch.randperm(len(test_R), device=device)]
                    elif mode == "R_random_proj":
                        test_R_in = test_R @ Q
                    else:
                        test_R_in = test_R

                    test_F = apply_operator(op["idx"], test_M, test_R_in, proj_M, proj_R, proj_joint, op.get("params"))
                    eval_metrics = evaluate_probe(probe, test_F.detach(), local_ious, y)
                    case_results.append(eval_metrics)

                # Average metrics over test cases
                mode_metrics = {}
                keys = case_results[0].keys()
                for key in keys:
                    vals = [r[key] for r in case_results if not np.isnan(r[key])]
                    mode_metrics[key] = float(np.mean(vals)) if vals else float("nan")
                mode_metrics["count"] = len(case_results)
                
                op_summary[mode] = mode_metrics

            final_results[variant][name] = op_summary

    # Save to file
    out_dir = ROOT / "runs/gradient_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probe_fusion_zoo_results.json"
    
    output = {
        "protocol": {
            "split": "test",
            "nms_iou_for_cohort_selection": 0.5,
            "raw_candidate_source": "P2 decoded pre-NMS head output",
            "candidate_conf": None,
            "positive": "IoU >= 0.1",
            "checkpoints": {k: str(v) for k, v in available_ckpts.items()},
        },
        "summary": final_results,
    }
    
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
