#!/usr/bin/env python3
"""Smoke tests for P2 ResidualDWConv5 + GAP topology."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import ChannelAttention, ResidualDWConv, ResidualDWConv5


def test_module_behavior():
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    m = ResidualDWConv5(32, alpha=0.1)
    y = m(x)
    delta = y - x
    assert y.shape == x.shape
    assert m.dw.groups == 32
    assert abs(float(m.alpha.detach()) - 0.1) < 1e-6
    assert delta.abs().mean() < x.abs().mean()
    y.mean().backward()
    assert m.alpha.grad is not None and torch.isfinite(m.alpha.grad).all()
    assert m.dw.weight.grad is not None and torch.isfinite(m.dw.weight.grad).all()

    partial = ResidualDWConv(32, k=5, alpha=0.1, partial_ratio=0.5)
    y = partial(x.detach())
    assert y.shape == x.shape
    assert partial.active_channels == 16
    assert torch.equal(y[:, 16:], x.detach()[:, 16:])


def test_model_topology():
    cases = {
        "yolov8n_p2_fpn_only_repdw5_gap.yaml": (19, 20, 21, 5, 1.0),
        "yolov8n_p2_fpn_only_repdw3_gap.yaml": (19, 20, 21, 3, 1.0),
        "yolov8n_p2_fpn_only_partial_repdw5_gap.yaml": (19, 20, 21, 5, 0.5),
        "yolov8n_p2_fpn_only_gap_repdw5.yaml": (20, 19, 21, 5, 1.0),
    }
    for yaml_name, (dw_i, gap_i, detect_i, k, ratio) in cases.items():
        model = YOLO(ROOT / f"models_related/models_config/yolov8/levir/{yaml_name}")
        layers = model.model.model
        assert isinstance(layers[dw_i], ResidualDWConv)
        assert layers[dw_i].k == k
        assert layers[dw_i].partial_ratio == ratio
        assert isinstance(layers[gap_i], ChannelAttention)
        assert layers[gap_i].descriptor == "avg"
        assert layers[detect_i].f == [20]
        assert layers[detect_i].stride.tolist() == [4.0]
        with torch.no_grad():
            output = model.model(torch.randn(1, 3, 128, 128))
        assert output is not None


if __name__ == "__main__":
    test_module_behavior()
    test_model_topology()
    print("ResidualDWConv5 + GAP tests passed")
