#!/usr/bin/env python3
"""Focused tests for GAP+FTAL shared Detect head variants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import ChannelAttention, Detect

from train_levir_scripts import train_all_levir_yolov8n_p2_gap_ftal_shared_head as train


def _grad_sum(module: torch.nn.Module) -> torch.Tensor:
    return sum((p.grad.abs().sum() for p in module.parameters() if p.grad is not None), torch.tensor(0.0))


def test_detect_head_share_modes_shapes_and_gradients():
    x = torch.randn(2, 32, 16, 16)
    for mode in ("none", "share1", "full"):
        head = Detect(nc=1, ch=(32,), head_share_mode=mode)
        out = head.forward_head([x], **head.one2many)
        assert out["boxes"].shape == (2, 64, 256)
        assert out["scores"].shape == (2, 1, 256)
        if mode == "none":
            assert len(head.shared_head) == 0
            continue

        head.zero_grad(set_to_none=True)
        out["boxes"].sum().backward(retain_graph=True)
        box_grad = _grad_sum(head.shared_head)
        head.zero_grad(set_to_none=True)
        out["scores"].sum().backward()
        cls_grad = _grad_sum(head.shared_head)
        assert box_grad > 0 and cls_grad > 0


def test_model_topology_and_runner_defaults():
    for variant, expected in train.EXPECTED_MODE.items():
        model = YOLO(train.workflow.VARIANTS[variant])
        layers = model.model.model
        head = layers[-1]
        assert isinstance(layers[19], ChannelAttention)
        assert isinstance(head, Detect)
        assert head.f == [19]
        assert head.stride.tolist() == [4.0]
        assert head.head_share_mode == expected
        assert not head.ring_context

    args = train.parse_args([])
    assert args.variants == ["gap_ftal_decoupled", "gap_ftal_share1", "gap_ftal_fully_shared"]
    assert args.seeds == [42]
    kwargs = train.train_kwargs(args, Path("data.yaml"), 42, True)
    assert kwargs["factorized_tal_target"] is True
    assert kwargs["factorized_tal_kappa"] == 1.5
    assert kwargs["factorized_tal_lambda"] == 0.5
    assert kwargs["factorized_tal_p2_only"] is True


if __name__ == "__main__":
    test_detect_head_share_modes_shapes_and_gradients()
    test_model_topology_and_runner_defaults()
    print("GAP+FTAL shared-head tests passed")
