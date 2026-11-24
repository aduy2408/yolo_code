#!/usr/bin/env python3
"""Smoke tests for the P1->P2 local-basis residual path."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, LocalBasisDownsample, LocalBasisDownsampleExpanded


def test_module_behavior():
    x = torch.randn(2, 16, 15, 17, requires_grad=True)
    module = LocalBasisDownsample(16, 32, adaptive=False).eval()
    y = module(x)
    assert y.shape == (2, 32, 8, 9)
    assert module.gamma.item() == 0.0
    assert module.branches[0].dw.groups == 16
    assert torch.equal(module.basis, torch.tensor(module._haar, dtype=torch.float32) / 2)
    y.square().mean().backward()
    assert module.gamma.grad is not None and torch.isfinite(module.gamma.grad).all()

    adaptive = LocalBasisDownsample(16, 32, adaptive=True)
    assert torch.equal(adaptive.delta_basis, torch.zeros(4, 4))
    assert torch.equal(adaptive.basis, module.basis)


def test_model_topology_and_transfer_keys():
    baseline = YOLO(ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml").model
    for name in ("lbd_fixed", "lbd_adaptive"):
        model = YOLO(ROOT / f"models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_{name}.yaml").model
        stem = model.model[1]
        assert isinstance(stem, LocalBasisDownsample)
        assert model.model[20].f == [19]
        assert model.model[20].stride.tolist() == [4.0]
        source = baseline.model[1].state_dict()
        missing, unexpected = stem.load_state_dict(source, strict=False)
        assert not unexpected
        expected_missing = {
            key for key in stem.state_dict() if key not in source and not key.endswith("num_batches_tracked")
        }
        assert set(missing) == expected_missing
        stem.eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 128, 128)) is not None


def test_expanded_channel_controls():
    configs = {
        "yolov8n_p2_fpn_only_conv48.yaml": Conv,
        "yolov8n_p2_fpn_only_lbd48_fixed.yaml": LocalBasisDownsampleExpanded,
        "yolov8n_p2_fpn_only_lbd48_adaptive.yaml": LocalBasisDownsampleExpanded,
    }
    for yaml_name, layer_type in configs.items():
        model = YOLO(ROOT / f"models_related/models_config/yolov8/levir/{yaml_name}").model
        if layer_type is Conv:
            assert type(model.model[1]) is Conv
        else:
            assert isinstance(model.model[1], layer_type)
        assert model.model[2].cv1.conv.in_channels == 48
        assert model.model[20].f == [19]
        with torch.no_grad():
            assert model(torch.randn(1, 3, 128, 128)) is not None


if __name__ == "__main__":
    test_module_behavior()
    test_model_topology_and_transfer_keys()
    test_expanded_channel_controls()
    print("LocalBasisDownsample tests passed")
