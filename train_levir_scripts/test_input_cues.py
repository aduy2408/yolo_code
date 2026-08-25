"""Cheap correctness gates for the fixed input-cue stem."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics.nn.modules import InputCueBank, InputCueConv, INPUT_CUE_VARIANTS


EXPECTED_CHANNELS = {
    "sobel_xy": 2,
    "laplacian_split": 2,
    "log": 1,
    "haar": 3,
    "lab_ab": 2,
    "ycbcr_cbcr": 2,
    "chromatic_edge": 2,
    "local_zscore": 1,
    "structure_coherence": 1,
    "top_hat": 1,
}


@pytest.mark.parametrize("cue_type", INPUT_CUE_VARIANTS)
def test_output_shape_range_and_backward(cue_type):
    rgb = torch.rand(2, 3, 32, 32, requires_grad=True)
    bank = InputCueBank(cue_type)
    cue = bank(rgb)
    assert cue.shape == (2, EXPECTED_CHANNELS[cue_type], 32, 32)
    assert torch.isfinite(cue).all()
    assert cue.abs().amax() <= 1.0001
    InputCueConv(3, 16, 3, 2, cue_type)(rgb).mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()


@pytest.mark.parametrize("cue_type", INPUT_CUE_VARIANTS)
def test_identity_at_initialization(cue_type):
    torch.manual_seed(42)
    rgb_stem = torch.nn.Conv2d(3, 8, 3, 2, padding=1, bias=False)
    cue_stem = InputCueConv(3, 8, 3, 2, cue_type, act=False)
    with torch.no_grad():
        cue_stem.conv.weight.zero_()
        cue_stem.conv.weight[:, :3].copy_(rgb_stem.weight)
        cue_stem.bn.weight.fill_(1)
        cue_stem.bn.bias.zero_()
        cue_stem.bn.running_mean.zero_()
        cue_stem.bn.running_var.fill_(1)
    rgb_stem.eval()
    cue_stem.eval()
    rgb = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        expected = rgb_stem(rgb)
        actual = cue_stem(rgb)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)


def test_checkpoint_reload():
    source = InputCueConv(3, 8, 3, 2, "sobel_xy")
    target = InputCueConv(3, 8, 3, 2, "sobel_xy")
    target.load_state_dict(source.state_dict())
    x = torch.rand(1, 3, 16, 16)
    assert torch.equal(source(x), target(x))


def test_runner_defaults_and_resolved_configs():
    import train_all_levir_yolov8n_p2_input_cues as runner

    args = runner.parse_args([])
    assert args.variants == list(runner.VARIANTS)
    assert args.seeds == [42]
    assert set(runner.CONFIGS) == set(runner.VARIANTS)
    assert all(path.is_file() for path in runner.CONFIGS.values())
