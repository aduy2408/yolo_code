#!/usr/bin/env python3
"""Focused tests for GAP+FTAL classifier-capacity Detect variants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import ChannelAttention, Detect
from ultralytics.nn.modules.conv import DWConv

from train_levir_scripts import train_all_levir_yolov8n_p2_gap_ftal_cls_capacity as train


def test_detect_cls_capacity_shapes():
    x = torch.randn(2, 32, 16, 16)
    for dense in (False, True):
        head = Detect(nc=1, ch=(32,), cls_head_width=64, cls_head_dense=dense)
        out = head.forward_head([x], **head.one2many)
        assert out["boxes"].shape == (2, 64, 256)
        assert out["scores"].shape == (2, 1, 256)
        assert head.cv2[0][0].conv.out_channels == 64
        if dense:
            assert head.cv3[0][0].conv.out_channels == 64
            assert head.cv3[0][1].conv.out_channels == 64
        else:
            assert isinstance(head.cv3[0][0][0], DWConv)
            assert head.cv3[0][0][1].conv.out_channels == 64


def test_model_topology_and_runner_defaults():
    for variant, (width, dense) in train.EXPECTED.items():
        model = YOLO(train.workflow.VARIANTS[variant])
        layers = model.model.model
        head = layers[-1]
        assert isinstance(layers[19], ChannelAttention)
        assert isinstance(head, Detect)
        assert head.f == [19]
        assert head.stride.tolist() == [4.0]
        assert head.head_share_mode == "none"
        assert head.cls_head_width == width
        assert head.cls_head_dense is dense
        assert not head.ring_context

    args = train.parse_args([])
    assert args.variants == ["gap_ftal_reg64_cls64_dw", "gap_ftal_reg64_cls64_fullconv"]
    assert args.seeds == [42]
    kwargs = train.train_kwargs(args, Path("data.yaml"), 42, True)
    assert kwargs["factorized_tal_target"] is True
    assert kwargs["factorized_tal_kappa"] == 1.5
    assert kwargs["factorized_tal_lambda"] == 0.5


if __name__ == "__main__":
    test_detect_cls_capacity_shapes()
    test_model_topology_and_runner_defaults()
    print("GAP+FTAL cls-capacity tests passed")
