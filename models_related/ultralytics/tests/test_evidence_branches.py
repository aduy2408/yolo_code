"""Focused tests for the P2 evidence branch experiment plumbing."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("cv2")

from ultralytics import YOLO
from ultralytics.nn.modules import AugmentationAwareEvidence, GradientIsolatedEvidence, ScaleDisappearanceEvidence


CONFIG_ROOT = Path(__file__).parents[2] / "models_config/yolov8/levir"


def test_gradient_isolated_evidence_separates_detection_and_auxiliary_gradients():
    module = GradientIsolatedEvidence(32, evidence_ch=8, detach_detection=True, aux_enabled=True).train()
    p2 = torch.randn(1, 32, 16, 16, requires_grad=True)
    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    output = module(p2, image)
    assert output.shape == (1, 40, 16, 16)
    output[:, 32:].sum().backward()
    assert image.grad is None
    assert all(parameter.grad is None for parameter in module.stem.parameters())
    module.zero_grad(set_to_none=True)
    module.last_aux["evidence_heatmap"].sum().backward()
    assert any(parameter.grad is not None for parameter in module.stem.parameters())


def test_scale_disappearance_evidence_detaches_sources_and_preserves_shape():
    module = ScaleDisappearanceEvidence(32, 64, out_ch=8, hidden=16, detach_sources=True).train()
    fine = torch.randn(1, 32, 16, 16, requires_grad=True)
    coarse = torch.randn(1, 64, 8, 8, requires_grad=True)
    output = module([fine, coarse])
    assert output.shape == (1, 8, 16, 16)
    output.sum().backward()
    assert fine.grad is None and coarse.grad is None
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_augmentation_aware_evidence_emits_bounded_state_and_eight_channels():
    module = AugmentationAwareEvidence(32, evidence_ch=8).train()
    output = module(torch.randn(2, 32, 16, 16), torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 40, 16, 16)
    state = module.last_aux["resolution_pred"]
    assert state.shape == (2, 1)
    assert torch.all((state >= 0) & (state <= 1))


@pytest.mark.parametrize(
    "name, expected_module",
    [
        ("yolov8n_p2_extra8_joint.yaml", GradientIsolatedEvidence),
        ("yolov8n_p2_gradient_isolated_evidence.yaml", GradientIsolatedEvidence),
        ("yolov8n_p2_resolution_conditioned_evidence.yaml", AugmentationAwareEvidence),
        ("yolov8n_p2_scale_disappearance_evidence.yaml", ScaleDisappearanceEvidence),
    ],
)
def test_evidence_yaml_builds_p2_only_detect(name, expected_module):
    model = YOLO(CONFIG_ROOT / name).model
    assert isinstance(model.model[-1], torch.nn.Module)
    assert model.model[-1].f == [model.model[-1].f[0]]
    assert model.model[-1].stride.tolist() == [4.0]
    assert any(isinstance(layer, expected_module) for layer in model.model)
