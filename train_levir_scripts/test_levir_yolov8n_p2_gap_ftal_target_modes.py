#!/usr/bin/env python3
"""Focused checks for GAP+FTAL target modes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
sys.path.insert(0, str(ROOT))

from ultralytics.utils.loss import v8DetectionLoss

from train_levir_scripts import train_all_levir_yolov8n_p2_gap_ftal_target_modes as train


def _loss(mode: str) -> v8DetectionLoss:
    loss = v8DetectionLoss.__new__(v8DetectionLoss)
    loss.factorized_tal_mode = mode
    loss.factorized_tal_tau = 0.75
    loss.factorized_tal_kappa = 1.5
    return loss


def test_mass_preserve_conserves_unclamped_mass():
    q = torch.tensor([[0.20], [0.10], [0.05]])
    u = torch.tensor([0.30, 0.20, 0.10])
    out, metrics = _loss("mass_preserve").factorize_tal_targets(q, u, lam=1.0)
    assert out.min() >= 0 and out.max() <= 1
    torch.testing.assert_close(out.sum(), q.sum(), rtol=1e-5, atol=1e-6)
    assert metrics["mass_ratio_after"] > 0.99


def test_geometry_ranks_by_iou_not_old_q():
    q = torch.tensor([[0.20], [0.90], [0.40]])
    u = torch.tensor([0.80, 0.60, 0.40])
    out, _ = _loss("geometry").factorize_tal_targets(q, u, lam=1.0)
    assert out[0, 0] > out[1, 0] > out[2, 0]


def test_agreement_gate_passes_and_bypasses():
    q = torch.tensor([[0.90], [0.60], [0.40]])
    u = torch.tensor([0.80, 0.70, 0.50])
    out, metrics = _loss("agreement_gate").factorize_tal_targets(q, u, lam=0.5)
    assert not torch.allclose(out, q)
    assert metrics["gate_on_fraction"] == 1.0

    q = torch.tensor([[0.60], [0.90], [0.40]])
    out, metrics = _loss("agreement_gate").factorize_tal_targets(q, u, lam=0.5)
    torch.testing.assert_close(out, q)
    assert metrics["gate_on_fraction"] == 0.0


def test_invalid_mode_and_runner_defaults():
    try:
        _loss("bad").factorize_tal_targets(torch.ones(1, 1), torch.ones(1), lam=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid factorized_tal_mode should fail")

    args = train.parse_args()
    assert args.variants == [
        "gap_ftal_mass_preserve_l05",
        "gap_ftal_mass_preserve_l1",
        "gap_ftal_geometry_l05",
        "gap_ftal_agreement_gate_l05",
    ]
    assert args.seeds == [42]
    assert args.split_seed == 42
    assert args.epochs == 100 and args.imgsz == 512
    assert args.hf_repo_id == "duyle2408/levir-yolov8n-p2-gap-ftal-target-modes-seed42"


if __name__ == "__main__":
    test_mass_preserve_conserves_unclamped_mass()
    test_geometry_ranks_by_iou_not_old_q()
    test_agreement_gate_passes_and_bypasses()
    test_invalid_mode_and_runner_defaults()
    print("GAP+FTAL target-mode tests passed")
