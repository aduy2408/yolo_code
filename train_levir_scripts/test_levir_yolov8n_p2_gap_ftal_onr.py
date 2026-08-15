#!/usr/bin/env python3
"""Focused tests for GAP+FTAL ONR refinement."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.nn.modules import ChannelAttention, Detect
from ultralytics.utils.loss import v8DetectionLoss

from train_levir_scripts import train_all_levir_yolov8n_p2_gap_ftal_onr as train


def test_zero_init_onr_preserves_coarse_scores_and_boxes():
    plain = Detect(nc=1, ch=(32,))
    onr = Detect(nc=1, ch=(32,), onr_refine=True, onr_sampling="object")
    plain.stride = onr.stride = torch.tensor([4.0])
    onr.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(2, 32, 16, 16)

    with torch.no_grad():
        plain_out = plain.forward_head([x], **plain.one2many)
        onr_out = onr.forward_head([x], **onr.one2many)

    torch.testing.assert_close(onr_out["boxes"], plain_out["boxes"])
    torch.testing.assert_close(onr_out["scores"], plain_out["scores"])
    torch.testing.assert_close(onr_out["refined_scores"], plain_out["scores"])
    torch.testing.assert_close(onr_out["refined_bboxes"], onr_out["coarse_bboxes"])
    assert onr_out["refined_bboxes"].shape == (2, 4, 256)


def test_onr_sampling_is_detached_from_coarse_box_logits():
    onr = Detect(nc=1, ch=(8,), onr_refine=True, onr_sampling="object")
    onr.stride = torch.tensor([4.0])
    x = torch.randn(1, 8, 6, 6, requires_grad=True)
    out = onr.forward_head([x], **onr.one2many)
    out["boxes"].retain_grad()
    out["refined_bboxes"].sum().backward()
    assert out["boxes"].grad is None


def test_fixed_and_object_sampling_can_diverge():
    onr = Detect(nc=1, ch=(8,), onr_refine=True, onr_sampling="fixed")
    onr.stride = torch.tensor([4.0])
    with torch.no_grad():
        onr.onr_refiner.cls.weight.fill_(1.0)
        onr.onr_refiner.cls.bias.zero_()
    x = torch.randn(1, 8, 7, 9)
    fixed = onr.forward_head([x], **onr.one2many)["refined_scores"]
    onr.onr_sampling = "object"
    obj = onr.forward_head([x], **onr.one2many)["refined_scores"]
    assert fixed.shape == obj.shape == (1, 1, 63)
    assert not torch.allclose(fixed, obj)


def test_model_topology_and_runner_defaults():
    model = YOLO(ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_gap_onr.yaml")
    layers = model.model.model
    head = layers[-1]
    assert isinstance(layers[19], ChannelAttention)
    assert isinstance(head, Detect)
    assert head.f == [19]
    assert head.stride.tolist() == [4.0]
    assert head.onr_refine and head.onr_sampling == "object"

    args = train.parse_args([])
    assert args.variants[0] == "R1_fixed_after_tal"
    assert args.seeds == [42]
    assert args.epochs == 100 and args.imgsz == 512
    args._variant = "R3_object_before_tal"
    kwargs = train.train_kwargs(args, Path("data.yaml"), 42, True)
    assert kwargs["factorized_tal_target"] is True
    assert kwargs["factorized_tal_lambda"] == 0.5
    assert kwargs["onr_assign_refined"] is True
    assert kwargs["onr_tal_diagnostics"] is True

    args._variant = "R4_object_standard_tal"
    kwargs = train.train_kwargs(args, Path("data.yaml"), 42, True)
    assert kwargs["factorized_tal_target"] is False


def test_synthetic_loss_uses_onr_outputs():
    model = YOLO(ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_gap_onr.yaml").model
    model.args = get_cfg(DEFAULT_CFG_DICT, model.args)
    model.args.onr_assign_refined = True
    model.args.onr_tal_diagnostics = True
    model.train()
    preds = model(torch.randn(1, 3, 64, 64))
    loss, items = v8DetectionLoss(model)(preds, {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
    })
    assert loss.shape == items.shape == (3,)
    assert torch.isfinite(loss).all()


if __name__ == "__main__":
    test_zero_init_onr_preserves_coarse_scores_and_boxes()
    test_onr_sampling_is_detached_from_coarse_box_logits()
    test_fixed_and_object_sampling_can_diverge()
    test_model_topology_and_runner_defaults()
    test_synthetic_loss_uses_onr_outputs()
    print("GAP+FTAL ONR tests passed")
