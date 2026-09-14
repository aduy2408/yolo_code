"""Unit tests for the project-owned fixed-Haar resampling pair."""

from pathlib import Path

import torch

from project_ultralytics.modules.frequency_sampling import (
    FreqDown,
    FreqUp,
    haar_analysis,
    haar_synthesis,
)


ROOT = Path(__file__).resolve().parents[1]


def test_haar_reconstructs_even_tensor():
    x = torch.randn(2, 3, 16, 18)
    assert torch.allclose(haar_synthesis(haar_analysis(x)), x, atol=1e-6, rtol=1e-6)


def test_frequency_module_shapes_and_odd_downsample():
    down = FreqDown(32, 64)
    up = FreqUp(64)
    assert down(torch.randn(2, 32, 128, 128)).shape == (2, 64, 64, 64)
    assert up(torch.randn(2, 64, 64, 64)).shape == (2, 64, 128, 128)
    assert FreqDown(32, 64)(torch.randn(1, 32, 127, 129)).shape == (1, 64, 64, 65)


def test_frequency_path_has_finite_gradients():
    down = FreqDown(8, 16)
    up = FreqUp(16)
    x = torch.randn(2, 8, 15, 17, requires_grad=True)
    loss = up(down(x)).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in down.parameters() if parameter.requires_grad)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in up.parameters() if parameter.requires_grad)


def test_cross_band_mixers_start_as_identity():
    for module in (FreqDown(8, 16), FreqUp(16)):
        weight = module.cross_band.mix.weight.detach().reshape(module.c_band, 4, 4)
        expected = torch.eye(4).expand_as(weight)
        assert torch.equal(weight, expected)


def test_down_and_up_do_not_share_parameters():
    down, up = FreqDown(8, 16), FreqUp(16)
    assert not set(map(id, down.parameters())).intersection(map(id, up.parameters()))


def test_project_yaml_names_frequency_modules():
    paths = (
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_p2_freq_pair_v1.yaml",
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p3_freq_pair_v1.yaml",
    )
    for path in paths:
        text = path.read_text()
        assert text.count("FreqDown") == 3
        assert text.count("FreqUp") == 3
        assert "models_related" not in text


def test_project_yaml_parser_builds_frequency_layers():
    from project_ultralytics import load_project_model

    for path in (
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_p2_freq_pair_v1.yaml",
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p3_freq_pair_v1.yaml",
    ):
        model = load_project_model(str(path), task="detect", verbose=False)
        frequency_layers = [layer for layer in model.model.model if layer.__class__.__name__ in {"FreqDown", "FreqUp"}]
        assert len(frequency_layers) == 6
        assert model.model.stride.tolist() == [4.0, 8.0]
