"""Unit tests for the project-owned fixed-Haar resampling pair."""

from pathlib import Path

import torch

from project_ultralytics.modules.frequency_sampling import (
    BandRefineV2,
    FreqDown,
    FreqDownV2,
    FreqUp,
    FreqUpV2,
    IBSDown,
    IBSUp,
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


def test_cross_band_mixer_keeps_latent_channels_isolated():
    module = FreqDown(4, 4).eval()
    with torch.no_grad():
        module.cross_band.mix.weight.zero_()
        module.cross_band.mix.weight[0, 1, 0, 0] = 1.0  # LH_0 -> LL_0
        module.cross_band.mix.weight[1, 0, 0, 0] = 1.0  # LL_0 -> LH_0
        module.cross_band.mix.weight[2, 2, 0, 0] = 1.0
        module.cross_band.mix.weight[3, 3, 0, 0] = 1.0
    bands = torch.tensor([[[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]])
    mixed = module.cross_band(bands)
    assert torch.equal(mixed, torch.tensor([[[[[2.0]], [[1.0]], [[3.0]], [[4.0]]]]]))


def test_frequency_stats_include_pre_post_bands_and_betas():
    down = FreqDown(4, 8, record_stats=True).eval()
    up = FreqUp(8, record_stats=True).eval()
    down(torch.randn(1, 4, 8, 8))
    up(torch.randn(1, 8, 4, 4))
    for stats in (down.last_stats, up.last_stats):
        assert all(f"{prefix}_{band}" in stats for prefix in ("pre", "post") for band in ("LL", "LH", "HL", "HH"))
        assert all(f"{prefix}_{band}/LL" in stats for prefix in ("pre", "post") for band in ("LH", "HL", "HH"))
        assert all(f"beta_{band}" in stats for band in ("LL", "LH", "HL", "HH"))


def test_v2_band_refinement_starts_as_identity():
    refine = BandRefineV2(8).eval()
    x = torch.randn(2, 8, 12, 14)
    assert torch.equal(refine(x), x)


def test_v2_has_no_cross_band_mixer_and_records_refinement_ratios():
    down = FreqDownV2(4, 8, record_stats=True).eval()
    up = FreqUpV2(8, record_stats=True).eval()
    assert not hasattr(down, "cross_band")
    assert not hasattr(up, "cross_band")
    down(torch.randn(1, 4, 8, 8))
    up(torch.randn(1, 8, 4, 4))
    for stats in (down.last_stats, up.last_stats):
        assert all(f"refine_{band}_ratio" in stats for band in ("LL", "LH", "HL", "HH"))
        assert all(stats[f"refine_{band}_ratio"] == 0.0 for band in ("LL", "LH", "HL", "HH"))


def test_ibs_resampling_uses_explicit_output_relative_expansion():
    down = IBSDown(32, 64, record_stats=True)
    up = IBSUp(64, record_stats=True)
    assert down.c_mid == 128
    assert down.c_fold == 16
    assert up.c_mid == 128
    assert up.c_fold == 16
    assert down(torch.randn(1, 32, 127, 129)).shape == (1, 64, 64, 65)
    assert up(torch.randn(1, 64, 64, 65)).shape == (1, 64, 128, 130)
    assert down.last_stats["c_mid"] == 128.0
    assert up.last_stats["c_mid"] == 128.0


def test_down_and_up_do_not_share_parameters():
    down, up = FreqDown(8, 16), FreqUp(16)
    assert not set(map(id, down.parameters())).intersection(map(id, up.parameters()))


def test_project_yaml_names_frequency_modules():
    paths = {
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_freq_pair_v1.yaml": (5, 2),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_freq_pair_v1.yaml": (3, 3),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p3p5_freq_pair_v1.yaml": (5, 2),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p4_freq_pair_v1.yaml": (3, 3),
    }
    for path, (down_count, up_count) in paths.items():
        text = path.read_text()
        assert text.count("FreqDown") == down_count
        assert text.count("FreqUp") == up_count
        assert "models_related" not in text
    v2_paths = list((ROOT / "project_ultralytics/configs/frequency_sampling").glob("*_v2.yaml"))
    assert len(v2_paths) == 4
    for path in v2_paths:
        text = path.read_text()
        assert "FreqDownV2" in text and "FreqUpV2" in text
        assert "FreqDown," not in text and "FreqUp," not in text
    ibs_paths = list((ROOT / "project_ultralytics/configs/frequency_sampling").glob("*_ibs_v1.yaml"))
    assert len(ibs_paths) == 4
    for path in ibs_paths:
        text = path.read_text()
        assert "IBSDown" in text and "IBSUp" in text
        assert "FreqDown" not in text and "FreqUp" not in text
    assert "RepC2f" not in (ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_freq_pair_v1.yaml").read_text()
    assert "RepC2f" not in (ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_freq_pair_v1.yaml").read_text()


def test_project_yaml_parser_builds_frequency_layers():
    from project_ultralytics import load_project_model

    expected = {
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_freq_pair_v1.yaml": (7, [8.0, 16.0, 32.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_freq_pair_v1.yaml": (6, [4.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p3p5_freq_pair_v1.yaml": (7, [8.0, 16.0, 32.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p4_freq_pair_v1.yaml": (6, [4.0, 8.0, 16.0]),
    }
    for path, (layer_count, strides) in expected.items():
        model = load_project_model(str(path), task="detect", verbose=False)
        frequency_layers = [layer for layer in model.model.model if layer.__class__.__name__ in {"FreqDown", "FreqUp"}]
        assert len(frequency_layers) == layer_count
        assert model.model.stride.tolist() == strides


def test_project_yaml_parser_builds_v2_frequency_layers():
    from project_ultralytics import load_project_model

    expected = {
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_freq_pair_v2.yaml": (7, [8.0, 16.0, 32.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_freq_pair_v2.yaml": (6, [4.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p3p5_freq_pair_v2.yaml": (7, [8.0, 16.0, 32.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p4_freq_pair_v2.yaml": (6, [4.0, 8.0, 16.0]),
    }
    for path, (layer_count, strides) in expected.items():
        model = load_project_model(str(path), task="detect", verbose=False)
        frequency_layers = [layer for layer in model.model.model if layer.__class__.__name__ in {"FreqDownV2", "FreqUpV2"}]
        assert len(frequency_layers) == layer_count
        assert model.model.stride.tolist() == strides


def test_project_yaml_parser_builds_ibs_frequency_layers():
    from project_ultralytics import load_project_model

    expected = {
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p3p5_ibs_v1.yaml": (7, [8.0, 16.0, 32.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_levir_p2_only_ibs_v1.yaml": (6, [4.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p3p5_ibs_v1.yaml": (7, [8.0, 16.0, 32.0]),
        ROOT / "project_ultralytics/configs/frequency_sampling/yolov8n_tinyperson_p2p4_ibs_v1.yaml": (6, [4.0, 8.0, 16.0]),
    }
    for path, (layer_count, strides) in expected.items():
        model = load_project_model(str(path), task="detect", verbose=False)
        layers = [layer for layer in model.model.model if layer.__class__.__name__ in {"IBSDown", "IBSUp"}]
        assert len(layers) == layer_count
        assert model.model.stride.tolist() == strides
