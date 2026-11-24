#!/usr/bin/env python3
"""Smoke tests for P2 C2f_PConv + GAP topology."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import ChannelAttention, C2f_PConv


def test_module_behavior():
    x = torch.randn(2, 64, 16, 16, requires_grad=True)
    m = C2f_PConv(64, 64, n=2, shortcut=True, n_div=4)
    y = m(x)
    assert y.shape == x.shape
    y.mean().backward()
    # Check gradients
    for name, param in m.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), f"Gradient of {name} is invalid"


def test_model_topology():
    yaml_name = "yolov8n_p2_levir_pconv_gap.yaml"
    model = YOLO(ROOT / f"models_related/models_config/yolov8/levir/{yaml_name}")
    layers = model.model.model
    
    # Check that C2f_PConv layers are present
    pconv_layers = [i for i, l in enumerate(layers) if isinstance(l, C2f_PConv)]
    assert len(pconv_layers) > 0, "No C2f_PConv layers found in model"
    
    gap_i = 19
    detect_i = 20
    assert isinstance(layers[gap_i], ChannelAttention)
    assert layers[gap_i].descriptor == "avg"
    assert layers[detect_i].f == [19]
    assert layers[detect_i].stride.tolist() == [4.0]
    
    with torch.no_grad():
        output = model.model(torch.randn(1, 3, 128, 128))
    assert output is not None


if __name__ == "__main__":
    test_module_behavior()
    test_model_topology()
    print("P2 C2f_PConv + GAP tests passed successfully!")
