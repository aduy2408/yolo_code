"""Correctness gates for the GAP plus FTAL GGCF runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from train_levir_scripts import train_all_levir_yolov8n_p2_gap_ftal_ggcf as runner


def test_runner_defaults_and_ftal_kappa():
    args = runner.parse_args([])
    assert args.variants == ["G1_field_only", "G2_ggcf", "G3_ggcf_refined_assign"]
    assert args.epochs == 100
    assert args.patience == 0
    assert args.hf_repo_id == "duyle2408/levir-yolov8n-p2-gap-ftal-ggcf"
    assert args.runner == Path(runner.__file__).resolve()
    data_yaml = Path("datasets/levir_ship_yolo_seed42/levir_ship.yaml")
    for variant in runner.workflow.VARIANTS:
        args._variant = variant
        kwargs = runner.train_kwargs(args, data_yaml, 42, True)
        assert kwargs["factorized_tal_kappa"] == 1.5
        assert kwargs["factorized_tal_lambda"] == 0.5
        assert kwargs["factorized_tal_target"] is (variant != "G4_ggcf_standard_tal")
        assert kwargs["ggcf_assign_refined"] is (variant in {"G3_ggcf_refined_assign", "G4_ggcf_standard_tal", "G5_field_only_refined_assign"})


@pytest.mark.skipif(not Path("yolov8n.pt").is_file(), reason="local pretrained checkpoint unavailable")
def test_model_topology_and_geometry_switches():
    for variant in runner.workflow.VARIANTS:
        model = runner.model_for(variant, "yolov8n.pt")
        layers = model.model.model
        head = layers[-1]
        assert layers[19].__class__.__name__ == "ChannelAttention"
        assert head.f == [19]
        assert head.stride.tolist() == [4.0]
        assert head.nl == 1
        assert head.ggcf_refine is True
        assert head.ggcf_geometry is not (variant in {"G1_field_only", "G5_field_only_refined_assign"})
