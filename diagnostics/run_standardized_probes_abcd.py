#!/usr/bin/env python3
"""Standardized Bradley-Terry Pairwise Quality-Ranking Probes A, B, C, and D.

Evaluates raw P2 pre-NMS candidate quality ranking on true positive candidates (IoU >= 0.5).

Protocol:
1. Extract raw P2 candidates per GT with IoU >= 0.5.
2. Standardize base components (F, B, C) using VAL statistics.
3. Fit VAL transforms (PCA8 on B, Ridge regression for D = B - AF, PCA8 on D, bilinear interactions).
4. Train Bradley-Terry pairwise linear rankers on VAL difference vectors (delta IoU > 0.05).
5. Freeze all parameters and evaluate on TEST.

Metrics reported per probe:
- Overall PairAcc
- PairAcc buckets: ΔIoU 0.05-0.10, 0.10-0.20, >0.20
- Best-IoU Rank
- Spearman Rho
- Regret
- GT-level: % Improved, % Degraded, Median ΔPairAcc
- Pairwise Rescue & Damage rates relative to baseline F.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "models_related/ultralytics") not in sys.path:
    sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402

from train_levir_scripts.probe_center_ring_cohorts import iou_matrix, read_labels  # noqa: E402


def build_hooked_yolo_model(weights_path: Path, device: str):
    wrapper = YOLO(str(weights_path))
    net = wrapper.model.to(device).eval()

    def debug_predict_once(x, profile=False, visualize=False, embed=None):
        y = []
        for idx, m in enumerate(net.model):
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    prev_x = x
                    x = []
                    for j in m.f:
                        if j == -1:
                            x.append(prev_x)
                        else:
                            x.append(y[j])
            x = m(x)
            y.append(x if idx in net.save else None)
        return x

    net._predict_once = debug_predict_once

    hooked_feats = {}
    def hook_layer2(module, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        hooked_feats["c2f"] = t.squeeze(0)

    def hook_layer18(module, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        hooked_feats["c2f_fused"] = t.squeeze(0)

    net.model[2].register_forward_hook(hook_layer2)
    net.model[18].register_forward_hook(hook_layer18)

    class ForwardWrapper(nn.Module):
        def forward(self, x):
            hooked_feats.clear()
            decoded, preds = net(x)
            return decoded, preds

    return ForwardWrapper(), net, hooked_feats


def train_pairwise_ranking_probe(diff_vectors: torch.Tensor, epochs: int = 150) -> nn.Linear:
    """Train linear Bradley-Terry model w^T (x_i - x_j) with BCE loss."""
    model = nn.Linear(diff_vectors.shape[1], 1, bias=False).to(diff_vectors.device)
    nn.init.zeros_(model.weight)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    target = torch.ones(len(diff_vectors), device=diff_vectors.device)

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(diff_vectors).squeeze(1)
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
    return model


def evaluate_probe(probe: nn.Linear, baseline_probe: nn.Linear | None,
                   cases: list, transform_fn, baseline_transform_fn,
                   device: str) -> dict:
    """Evaluate probe on TEST cases with detailed 8 metrics + Rescue/Damage against baseline."""
    all_ranks = []
    all_regrets = []
    all_spearmans = []

    bucket_correct = {"005_010": 0, "010_020": 0, "gt_020": 0}
    bucket_total = {"005_010": 0, "010_020": 0, "gt_020": 0}

    # GT-level tracking
    gt_improved = 0
    gt_degraded = 0
    gt_delta_pas = []

    # Rescue & Damage tracking over all valid pairs (|ΔIoU| >= 0.05)
    rescue_count = 0
    damage_count = 0
    baseline_wrong_total = 0
    baseline_correct_total = 0

    probe.eval()
    if baseline_probe is not None:
        baseline_probe.eval()

    for case_feats, local_ious in cases:
        if len(local_ious) < 2:
            continue

        local_ious = local_ious.to(device)
        x_in = transform_fn(case_feats).to(device)

        with torch.no_grad():
            scores = probe(x_in).squeeze(1)
            base_scores = baseline_probe(baseline_transform_fn(case_feats).to(device)).squeeze(1) if baseline_probe is not None else None

        # Rank metrics
        order = torch.argsort(scores, descending=True)
        best_iou_idx = int(torch.argmax(local_ious))
        rank = int((order == best_iou_idx).nonzero(as_tuple=False)[0].item()) + 1
        all_ranks.append(rank)

        max_iou = float(local_ious.max().item())
        top1_iou = float(local_ious[order[0]].item())
        all_regrets.append(max_iou - top1_iou)

        # Spearman rank correlation
        scores_cpu = scores.cpu().numpy()
        ious_cpu = local_ious.cpu().numpy()
        if len(scores_cpu) > 1 and len(np.unique(scores_cpu)) > 1 and len(np.unique(ious_cpu)) > 1:
            rho, _ = scipy.stats.spearmanr(scores_cpu, ious_cpu)
            if not np.isnan(rho):
                all_spearmans.append(rho)

        # Pairwise evaluation in buckets & rescue/damage
        delta_iou = local_ious[:, None] - local_ious[None, :]
        score_diff = scores[:, None] > scores[None, :]

        m_valid = delta_iou > 0.05
        if not m_valid.any():
            continue

        # Bucket 1: 0.05 < delta <= 0.10
        m1 = (delta_iou > 0.05) & (delta_iou <= 0.10)
        if m1.any():
            bucket_total["005_010"] += int(m1.sum().item())
            bucket_correct["005_010"] += int((score_diff & m1).sum().item())

        # Bucket 2: 0.10 < delta <= 0.20
        m2 = (delta_iou > 0.10) & (delta_iou <= 0.20)
        if m2.any():
            bucket_total["010_020"] += int(m2.sum().item())
            bucket_correct["010_020"] += int((score_diff & m2).sum().item())

        # Bucket 3: delta > 0.20
        m3 = delta_iou > 0.20
        if m3.any():
            bucket_total["gt_020"] += int(m3.sum().item())
            bucket_correct["gt_020"] += int((score_diff & m3).sum().item())

        # Rescue & Damage
        if base_scores is not None:
            base_score_diff = base_scores[:, None] > base_scores[None, :]

            probe_correct = score_diff & m_valid
            base_correct = base_score_diff & m_valid
            base_wrong = (~base_score_diff) & m_valid

            rescue_count += int((probe_correct & base_wrong).sum().item())
            damage_count += int(((~probe_correct) & base_correct).sum().item())
            baseline_wrong_total += int(base_wrong.sum().item())
            baseline_correct_total += int(base_correct.sum().item())

            # GT-level pair accuracy
            n_gt_pairs = int(m_valid.sum().item())
            if n_gt_pairs > 0:
                pa_probe_gt = float(probe_correct.sum().item()) / n_gt_pairs
                pa_base_gt = float(base_correct.sum().item()) / n_gt_pairs
                d_pa = pa_probe_gt - pa_base_gt
                gt_delta_pas.append(d_pa)
                if d_pa > 0.001:
                    gt_improved += 1
                elif d_pa < -0.001:
                    gt_degraded += 1

    total_correct = sum(bucket_correct.values())
    total_pairs = sum(bucket_total.values())

    return {
        "pair_acc": float(total_correct / total_pairs) if total_pairs > 0 else float("nan"),
        "pair_acc_005_010": float(bucket_correct["005_010"] / bucket_total["005_010"]) if bucket_total["005_010"] > 0 else float("nan"),
        "pair_acc_010_020": float(bucket_correct["010_020"] / bucket_total["010_020"]) if bucket_total["010_020"] > 0 else float("nan"),
        "pair_acc_gt_020": float(bucket_correct["gt_020"] / bucket_total["gt_020"]) if bucket_total["gt_020"] > 0 else float("nan"),
        "rank": float(np.mean(all_ranks)) if all_ranks else float("nan"),
        "spearman": float(np.mean(all_spearmans)) if all_spearmans else float("nan"),
        "regret": float(np.mean(all_regrets)) if all_regrets else float("nan"),
        "total_pairs": total_pairs,
        "gt_improved_pct": float(gt_improved / len(gt_delta_pas) * 100) if gt_delta_pas else float("nan"),
        "gt_degraded_pct": float(gt_degraded / len(gt_delta_pas) * 100) if gt_delta_pas else float("nan"),
        "median_gt_delta_pa": float(np.median(gt_delta_pas)) if gt_delta_pas else float("nan"),
        "rescue_pct": float(rescue_count / baseline_wrong_total * 100) if baseline_wrong_total > 0 else float("nan"),
        "damage_pct": float(damage_count / baseline_correct_total * 100) if baseline_correct_total > 0 else float("nan"),
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


def get_candidates_quality_precomputed(decoded: torch.Tensor, preds: dict, original_shape: tuple[int, int],
                                         gt: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = original_shape[:2]
    p2 = preds["feats"][0].squeeze(0)
    c, hp, wp = p2.shape
    n_p2 = hp * wp
    boxes = xywh2xyxy(decoded[0, :4, :n_p2].T)

    ious = iou_matrix(gt.view(1, 4), boxes).squeeze(0)
    idx = torch.nonzero(ious >= 0.5, as_tuple=False).flatten()
    return idx, ious[idx]


def compute_chroma_features_gpu(image_bgr_letter: np.ndarray, device: str) -> torch.Tensor:
    img_t = torch.from_numpy(image_bgr_letter[..., ::-1].copy()).to(device).float() / 255.0  # (H, W, 3)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 512, 512)

    r, g, b = img_t[:, 0:1], img_t[:, 1:2], img_t[:, 2:3]
    cb = 0.5 - 0.1687 * r - 0.3313 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.4187 * g - 0.0813 * b
    cbcr = torch.cat([cb, cr], dim=1)  # (1, 2, 512, 512)

    c0 = F.avg_pool2d(cbcr, kernel_size=4, stride=4)
    c3 = F.avg_pool2d(c0, kernel_size=3, stride=1, padding=1)
    c5 = F.avg_pool2d(c0, kernel_size=5, stride=1, padding=2)

    return torch.cat([c0, c3, c5], dim=1).squeeze(0)  # (6, 128, 128)


def collect_quality_features(net, hooked, samples, device, letterbox) -> tuple[dict[str, torch.Tensor], list]:
    data = {"b": [], "f": [], "cbcr": []}
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

        original_letter = letterbox(image=original)
        tensor = torch.from_numpy(original_letter[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            decoded, preds = net(tensor)

        t_backbone = hooked["c2f"]       # (32, 128, 128)
        t_fused = hooked["c2f_fused"]    # (32, 128, 128)
        c_cbcr = compute_chroma_features_gpu(original_letter, device)  # (6, 128, 128)

        _, h_f, w_f = t_fused.shape
        n = h_f * w_f

        for gt in gt_xyxy:
            idx, local_ious = get_candidates_quality_precomputed(
                decoded, preds, original.shape, gt, device
            )
            safe_idx = idx[idx < n]
            if len(safe_idx) < 2:
                continue

            grid_y = safe_idx // w_f
            grid_x = safe_idx % w_f

            b_feats = t_backbone[:, grid_y, grid_x].permute(1, 0).cpu()
            f_feats = t_fused[:, grid_y, grid_x].permute(1, 0).cpu()
            c_feats = c_cbcr[:, grid_y, grid_x].permute(1, 0).cpu()

            case_feats = {"b": b_feats, "f": f_feats, "cbcr": c_feats}
            case_feats_list.append((case_feats, local_ious.cpu()))

            data["b"].append(b_feats)
            data["f"].append(f_feats)
            data["cbcr"].append(c_feats)

    flat_data = {k: torch.cat(data[k], dim=0) for k in data} if data["b"] else {}
    return flat_data, case_feats_list


def create_difference_vectors(val_cases: list, transform_fn) -> torch.Tensor:
    """Create pairwise difference vectors d_ij = x_i - x_j for pairs with IoU_i - IoU_j > 0.05."""
    diff_list = []
    for case_feats, local_ious in val_cases:
        if len(local_ious) < 2:
            continue
        x = transform_fn(case_feats)
        delta_iou = local_ious[:, None] - local_ious[None, :]
        pairs = torch.nonzero(delta_iou > 0.05, as_tuple=False)
        if len(pairs):
            diff = x[pairs[:, 0]] - x[pairs[:, 1]]
            diff_list.append(diff)
    return torch.cat(diff_list, dim=0) if diff_list else torch.empty((0, 1))


class StandardizedProbeSuite:
    """Fits VAL standardization, PCA, and Ridge parameters, then builds transformed representations."""

    def __init__(self, val_flat_data: dict[str, torch.Tensor]):
        # 1. Standardization stats fit on VAL
        self.mu_f = val_flat_data["f"].mean(dim=0, keepdim=True)
        self.std_f = val_flat_data["f"].std(dim=0, keepdim=True) + 1e-6

        self.mu_b = val_flat_data["b"].mean(dim=0, keepdim=True)
        self.std_b = val_flat_data["b"].std(dim=0, keepdim=True) + 1e-6

        self.mu_c = val_flat_data["cbcr"].mean(dim=0, keepdim=True)
        self.std_c = val_flat_data["cbcr"].std(dim=0, keepdim=True) + 1e-6

        # Standardized VAL matrices
        f_std = (val_flat_data["f"] - self.mu_f) / self.std_f
        b_std = (val_flat_data["b"] - self.mu_b) / self.std_b
        c_std = (val_flat_data["cbcr"] - self.mu_c) / self.std_c

        # 2. PCA8 on B
        _, _, v_b = torch.pca_lowrank(b_std, q=8)
        self.pca_b = v_b[:, :8]  # (32, 8)

        # 3. Ridge Regression on VAL: B_std = F_std @ A^T -> A = (F^T F + lambda I)^-1 F^T B
        # Predict B from F to find Discrepancy D = B_std - F_std @ A^T
        lam = 1.0
        ff = f_std.T @ f_std + lam * torch.eye(f_std.shape[1])
        fb = f_std.T @ b_std
        self.ridge_a = torch.linalg.solve(ff, fb)  # (32, 32)

        # Discrepancy on VAL
        d_val = b_std - f_std @ self.ridge_a
        _, _, v_d = torch.pca_lowrank(d_val, q=8)
        self.pca_d = v_d[:, :8]  # (32, 8)

    def standardize(self, feats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_std = (feats["f"] - self.mu_f) / self.std_f
        b_std = (feats["b"] - self.mu_b) / self.std_b
        c_std = (feats["cbcr"] - self.mu_c) / self.std_c
        return f_std, b_std, c_std

    def compute_interaction(self, z: torch.Tensor, c_std: torch.Tensor) -> torch.Tensor:
        """Compute outer-product interaction vector Z (N, 8) ⊗ C (N, 6) -> (N, 48)."""
        # z: (N, 8), c_std: (N, 6) -> (N, 8, 6) -> reshape to (N, 48)
        outer = z.unsqueeze(2) * c_std.unsqueeze(1)
        return outer.view(z.shape[0], -1)

    # Transform builders for candidate dict
    def transform_F(self, feats: dict) -> torch.Tensor:
        f_std, _, _ = self.standardize(feats)
        return f_std

    def transform_B(self, feats: dict) -> torch.Tensor:
        _, b_std, _ = self.standardize(feats)
        return b_std

    def transform_FB(self, feats: dict) -> torch.Tensor:
        f_std, b_std, _ = self.standardize(feats)
        return torch.cat([f_std, b_std], dim=-1)

    def transform_FBC(self, feats: dict) -> torch.Tensor:
        f_std, b_std, c_std = self.standardize(feats)
        return torch.cat([f_std, b_std, c_std], dim=-1)

    def transform_IBC(self, feats: dict) -> torch.Tensor:
        _, b_std, c_std = self.standardize(feats)
        zb = b_std @ self.pca_b
        return self.compute_interaction(zb, c_std)

    def transform_FBC_IBC(self, feats: dict) -> torch.Tensor:
        f_std, b_std, c_std = self.standardize(feats)
        zb = b_std @ self.pca_b
        i_bc = self.compute_interaction(zb, c_std)
        return torch.cat([f_std, b_std, c_std, i_bc], dim=-1)

    def transform_FD(self, feats: dict) -> torch.Tensor:
        f_std, b_std, _ = self.standardize(feats)
        d = b_std - f_std @ self.ridge_a
        zd = d @ self.pca_d
        return torch.cat([f_std, zd], dim=-1)

    def transform_FDC(self, feats: dict) -> torch.Tensor:
        f_std, b_std, c_std = self.standardize(feats)
        d = b_std - f_std @ self.ridge_a
        zd = d @ self.pca_d
        return torch.cat([f_std, zd, c_std], dim=-1)

    def transform_FDC_IDC(self, feats: dict) -> torch.Tensor:
        f_std, b_std, c_std = self.standardize(feats)
        d = b_std - f_std @ self.ridge_a
        zd = d @ self.pca_d
        i_dc = self.compute_interaction(zd, c_std)
        return torch.cat([f_std, zd, c_std, i_dc], dim=-1)

    def transform_FZB(self, feats: dict) -> torch.Tensor:
        f_std, b_std, _ = self.standardize(feats)
        zb = b_std @ self.pca_b
        return torch.cat([f_std, zb], dim=-1)

    def transform_FC(self, feats: dict) -> torch.Tensor:
        f_std, _, c_std = self.standardize(feats)
        return torch.cat([f_std, c_std], dim=-1)

    def transform_FZBC(self, feats: dict) -> torch.Tensor:
        f_std, b_std, c_std = self.standardize(feats)
        zb = b_std @ self.pca_b
        return torch.cat([f_std, zb, c_std], dim=-1)

    def transform_FZBC_IBC(self, feats: dict) -> torch.Tensor:
        f_std, b_std, c_std = self.standardize(feats)
        zb = b_std @ self.pca_b
        i_bc = self.compute_interaction(zb, c_std)
        return torch.cat([f_std, zb, c_std, i_bc], dim=-1)


def print_suite_table(suite_name: str, results: dict, key_decision_pair: tuple[str, str]):
    print(f"\n=== PROBE SUITE {suite_name} METRICS ===")
    headers = ["Representation", "PairAcc", "Δ.05-.10", "Δ.10-.20", "Δ>.20", "BestRank", "Spearman", "Regret"]
    row_fmt = "{:<25} | {:>7} | {:>8} | {:>8} | {:>7} | {:>8} | {:>8} | {:>7}"
    print(row_fmt.format(*headers))
    print("-" * 105)

    for p_name, res in results.items():
        print(row_fmt.format(
            p_name,
            f"{res['pair_acc']:.4f}",
            f"{res['pair_acc_005_010']:.4f}",
            f"{res['pair_acc_010_020']:.4f}",
            f"{res['pair_acc_gt_020']:.4f}",
            f"{res['rank']:.2f}",
            f"{res['spearman']:.4f}",
            f"{res['regret']:.4f}",
        ))

    target_name, baseline_name = key_decision_pair
    if target_name in results and baseline_name in results:
        res_t = results[target_name]
        d_pa = res_t["pair_acc"] - results[baseline_name]["pair_acc"]
        print(f"\n--- GT-level & Decision Analysis ({target_name} vs {baseline_name}) ---")
        print(f"ΔPairAcc: {d_pa:+.4f}")
        print(f"GT Improved: {res_t['gt_improved_pct']:.1f}% | GT Degraded: {res_t['gt_degraded_pct']:.1f}% | Median ΔPA: {res_t['median_gt_delta_pa']:+.4f}")
        print(f"Rescue: {res_t['rescue_pct']:.1f}% | Damage: {res_t['damage_pct']:.1f}%")

        if d_pa >= 0.02 and res_t["rescue_pct"] > res_t["damage_pct"]:
            gate = "STRONG PASS"
        elif d_pa >= 0.015 and res_t["rescue_pct"] > res_t["damage_pct"]:
            gate = "PASS"
        elif d_pa >= 0.005 and res_t["rescue_pct"] > res_t["damage_pct"]:
            gate = "WEAK SIGNAL"
        else:
            gate = "FAIL"
        print(f"Decision Gate Classification: [{gate}]")


def main():
    parser = argparse.ArgumentParser(description="Run standardized probe suites A, B, C, and D.")
    parser.add_argument("--data-root", type=str, default="/mnt/data/varroa/yolo_related/datasets")
    parser.add_argument("--weights", type=str, default="/mnt/data/varroa/yolo_related/yolov8n.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data_root = Path(args.data_root)
    val_samples = load_split_samples(data_root, "val")
    test_samples = load_split_samples(data_root, "test")

    print(f"Loaded {len(val_samples)} VAL samples, {len(test_samples)} TEST samples.")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        fallbacks = [
            ROOT / "runs/plain_p2_only/seed_42/weights/best.pt",
            ROOT / "runs/levir_yolov8n_p2_plain/seed_42/weights/best.pt",
            ROOT / "runs/checkpoint_cache/duyle2408_levir-ship-yolo-p2_train_yolov8n_p2_baseline_seed42_weights_best.pt",
            Path("/marimo/yolo_code/runs/levir_yolov8n_p2_plain/seed_42/weights/best.pt"),
        ]
        for fb in fallbacks:
            if fb.exists():
                weights_path = fb
                break
        else:
            raise FileNotFoundError(f"Checkpoint {args.weights} not found and no fallback checkpoint available.")

    print(f"Using checkpoint weights: {weights_path}")
    wrapper, net, hooked_feats = build_hooked_yolo_model(weights_path, args.device)
    letterbox = LetterBox(new_shape=(512, 512), auto=False, scale_fill=False, scaleup=False, stride=32)

    print("Collecting candidate-quality features on VAL...")
    val_flat, val_cases = collect_quality_features(wrapper, hooked_feats, val_samples, args.device, letterbox)
    print("Collecting candidate-quality features on TEST...")
    test_flat, test_cases = collect_quality_features(wrapper, hooked_feats, test_samples, args.device, letterbox)

    print(f"VAL total candidate vectors: {len(val_flat['f']) if val_flat else 0}")
    print(f"TEST total candidate vectors: {len(test_flat['f']) if test_flat else 0}")

    suite = StandardizedProbeSuite(val_flat)

    # Train Baseline F probe
    print("\nTraining Baseline P0 = F probe...")
    diff_f = create_difference_vectors(val_cases, suite.transform_F).to(args.device)
    p0_probe = train_pairwise_ranking_probe(diff_f, epochs=args.epochs)
    f_results = evaluate_probe(p0_probe, None, test_cases, suite.transform_F, suite.transform_F, args.device)

    # ==================== SUITE A ====================
    print("\n--- Running Probe Suite A (Channel Replacement F24 + B8) ---")
    probes_a_def = [
        ("P0: F", suite.transform_F),
        ("P1: B", suite.transform_B),
        ("P2: [F, B]", suite.transform_FB),
    ]
    results_a = {"P0: F": f_results}
    for name, transform_fn in probes_a_def[1:]:
        diff_v = create_difference_vectors(val_cases, transform_fn).to(args.device)
        probe = train_pairwise_ranking_probe(diff_v, epochs=args.epochs)
        results_a[name] = evaluate_probe(probe, p0_probe, test_cases, transform_fn, suite.transform_F, args.device)
    print_suite_table("A (F24 + B8 Channel Replacement)", results_a, ("P2: [F, B]", "P0: F"))

    # ==================== SUITE B ====================
    print("\n--- Running Probe Suite B (Cue-Formed Evidence E = φ(B, C)) ---")
    probes_b_def = [
        ("P0: F", suite.transform_F),
        ("P1: [F, B, C]", suite.transform_FBC),
        ("P2: I_BC", suite.transform_IBC),
        ("P3: [F, B, C, I_BC]", suite.transform_FBC_IBC),
    ]
    results_b = {"P0: F": f_results}
    diff_fbc = create_difference_vectors(val_cases, suite.transform_FBC).to(args.device)
    p1_fbc_probe = train_pairwise_ranking_probe(diff_fbc, epochs=args.epochs)

    for name, transform_fn in probes_b_def[1:]:
        diff_v = create_difference_vectors(val_cases, transform_fn).to(args.device)
        probe = train_pairwise_ranking_probe(diff_v, epochs=args.epochs)
        base_pr = p1_fbc_probe if name == "P3: [F, B, C, I_BC]" else p0_probe
        base_tf = suite.transform_FBC if name == "P3: [F, B, C, I_BC]" else suite.transform_F
        results_b[name] = evaluate_probe(probe, base_pr, test_cases, transform_fn, base_tf, args.device)
    print_suite_table("B (Cue-Formed Interaction E = φ(B, C))", results_b, ("P3: [F, B, C, I_BC]", "P1: [F, B, C]"))

    # ==================== SUITE C ====================
    print("\n--- Running Probe Suite C (Discrepancy-Guided Evidence φ(B-F, C)) ---")
    probes_c_def = [
        ("P0: F", suite.transform_F),
        ("P1: [F, B, C]", suite.transform_FBC),
        ("P2: [F, D]", suite.transform_FD),
        ("P3: [F, D, C]", suite.transform_FDC),
        ("P4: [F, D, C, I_DC]", suite.transform_FDC_IDC),
    ]
    results_c = {"P0: F": f_results}
    for name, transform_fn in probes_c_def[1:]:
        diff_v = create_difference_vectors(val_cases, transform_fn).to(args.device)
        probe = train_pairwise_ranking_probe(diff_v, epochs=args.epochs)
        base_pr = p1_fbc_probe if name == "P4: [F, D, C, I_DC]" else p0_probe
        base_tf = suite.transform_FBC if name == "P4: [F, D, C, I_DC]" else suite.transform_F
        results_c[name] = evaluate_probe(probe, base_pr, test_cases, transform_fn, base_tf, args.device)
    print_suite_table("C (Discrepancy-Guided φ(B-F, C))", results_c, ("P4: [F, D, C, I_DC]", "P1: [F, B, C]"))

    # ==================== SUITE D ====================
    print("\n--- Running Probe Suite D (Cue-Conditioned Basis Synthesis) ---")
    diff_fzbc = create_difference_vectors(val_cases, suite.transform_FZBC).to(args.device)
    p3_fzbc_probe = train_pairwise_ranking_probe(diff_fzbc, epochs=args.epochs)

    probes_d_def = [
        ("P0: F", suite.transform_F),
        ("P1: [F, Z_B]", suite.transform_FZB),
        ("P2: [F, C]", suite.transform_FC),
        ("P3: [F, Z_B, C]", suite.transform_FZBC),
        ("P4: [F, Z_B, C, Z_B ⊗ C]", suite.transform_FZBC_IBC),
    ]
    results_d = {"P0: F": f_results}
    for name, transform_fn in probes_d_def[1:]:
        diff_v = create_difference_vectors(val_cases, transform_fn).to(args.device)
        probe = train_pairwise_ranking_probe(diff_v, epochs=args.epochs)
        base_pr = p3_fzbc_probe if name == "P4: [F, Z_B, C, Z_B ⊗ C]" else p0_probe
        base_tf = suite.transform_FZBC if name == "P4: [F, Z_B, C, Z_B ⊗ C]" else suite.transform_F
        results_d[name] = evaluate_probe(probe, base_pr, test_cases, transform_fn, base_tf, args.device)
    print_suite_table("D (Cue-Conditioned Basis Synthesis)", results_d, ("P4: [F, Z_B, C, Z_B ⊗ C]", "P3: [F, Z_B, C]"))

    # Save full JSON report
    report = {
        "suite_a": results_a,
        "suite_b": results_b,
        "suite_c": results_c,
        "suite_d": results_d,
    }
    report_path = ROOT / "runs" / "standardized_probes_abcd_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull probe report saved to {report_path}")


if __name__ == "__main__":
    main()
