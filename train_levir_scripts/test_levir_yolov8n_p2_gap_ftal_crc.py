#!/usr/bin/env python3
"""Focused tests for the GAP+FTAL CRC experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import ChannelAttention, Detect, RingContextCls, RingPoolR5

from train_levir_scripts import train_all_levir_yolov8n_p2_gap_ftal_crc as train


def _grad_sum(module: torch.nn.Module) -> torch.Tensor:
    return sum((p.grad.abs().sum() for p in module.parameters() if p.grad is not None), torch.tensor(0.0))


def test_ring_pool_matches_r5_annulus_shape_and_params():
    pool = RingPoolR5(4, radius=5)
    x = torch.randn(2, 4, 8, 9)
    y = pool(x)
    assert y.shape == x.shape
    assert sum(p.numel() for p in pool.parameters()) == 0
    assert pool.weight.shape == (4, 1, 11, 11)
    assert torch.allclose(pool.weight[0].sum(), torch.tensor(1.0))


def test_zero_init_crc_preserves_scores_and_boxes_and_is_cls_only():
    plain = Detect(nc=1, ch=(32,))
    crc = Detect(nc=1, ch=(32,), ring_context=True, ring_radius=5)
    crc.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(2, 32, 16, 16)

    with torch.no_grad():
        plain_out = plain.forward_head([x], **plain.one2many)
        crc_out = crc.forward_head([x], **crc.one2many)
    torch.testing.assert_close(crc_out["boxes"], plain_out["boxes"])
    torch.testing.assert_close(crc_out["scores"], plain_out["scores"])

    crc.train()
    crc.forward_head([x], **crc.one2many)["scores"].sum().backward()
    assert _grad_sum(crc.cls_ring_context) > 0
    assert _grad_sum(crc.cv3) > 0
    assert _grad_sum(crc.cv2) == 0

    crc.zero_grad(set_to_none=True)
    crc.forward_head([x], **crc.one2many)["boxes"].sum().backward()
    assert _grad_sum(crc.cls_ring_context) == 0
    assert _grad_sum(crc.cv2) > 0
    assert _grad_sum(crc.cv3) == 0


def test_model_topology_and_runner_defaults():
    model = YOLO(ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_gap_ring_context_cls.yaml")
    layers = model.model.model
    head = layers[-1]
    assert isinstance(layers[19], ChannelAttention)
    assert isinstance(head, Detect)
    assert head.f == [19]
    assert head.stride.tolist() == [4.0]
    assert isinstance(head.cls_ring_context[0], RingContextCls)
    assert head.ring_context and head.ring_radius == 5

    args = train.parse_args()
    assert args.variants == ["gap_ftal_crc"]
    assert args.seeds == [42]
    assert args.split_seed == 42
    assert args.epochs == 100 and args.imgsz == 512
    kwargs = train.train_kwargs(args, Path("data.yaml"), 42, True)
    assert kwargs["factorized_tal_target"] is True
    assert kwargs["factorized_tal_tau"] == 0.75
    assert kwargs["factorized_tal_kappa"] == 1.5
    assert kwargs["factorized_tal_lambda"] == 0.5
    assert kwargs["factorized_tal_p2_only"] is True


if __name__ == "__main__":
    test_ring_pool_matches_r5_annulus_shape_and_params()
    test_zero_init_crc_preserves_scores_and_boxes_and_is_cls_only()
    test_model_topology_and_runner_defaults()
    print("GAP+FTAL CRC tests passed")
