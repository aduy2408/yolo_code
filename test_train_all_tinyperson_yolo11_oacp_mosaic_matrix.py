from argparse import Namespace

import train_all_tinyperson_yolo11_oacp_mosaic_matrix as runner


def _args():
    return Namespace(
        data_root=__import__('pathlib').Path('/data/TinyPerson'),
        dataset_root=__import__('pathlib').Path('/repo/datasets'),
        project=__import__('pathlib').Path('/repo/runs/yolo11'),
        hf_repo_id='duyle2408/tinyperson-yolo11-oacp-mosaic-matrix',
        epochs=100, patience=0, imgsz=640, batch_size=8, workers=8,
        device='cuda', amp=True, split_seed=42,
        variants=list(runner.DEFAULT_VARIANTS), seeds=[42],
    )


def test_matrix_has_four_isolated_variants():
    assert tuple(runner.VARIANTS) == runner.DEFAULT_VARIANTS
    assert runner.MODEL == 'yolo11n.pt'
    assert runner.VARIANTS['legacy_default_oacp_mosaic']['legacy_double_oacp'] is True
    assert runner.VARIANTS['r2_frequent_mild_oacp_mosaic']['legacy_double_oacp'] is False
    assert runner.VARIANTS['legacy_default_oacp_mosaic']['mosaic'] == 1.0
    assert runner.VARIANTS['legacy_default_oacp_no_mosaic']['mosaic'] == 0.0


def test_effective_settings_encode_default_and_r2_parameters():
    args = _args()
    default = runner.effective_settings(args, 'legacy_default_oacp_mosaic', 42)
    r2 = runner.effective_settings(args, 'r2_frequent_mild_oacp_no_mosaic', 42)
    assert default['augmentation']['oacp'] == {'p': 0.20, 'expand': 3.0, 'strength': [0.20, 0.40], 'scale': [0.65, 0.85]}
    assert r2['augmentation']['oacp'] == {'p': 0.40, 'expand': 3.0, 'strength': [0.10, 0.25], 'scale': [0.80, 0.95]}
    assert r2['augmentation']['mosaic'] == 0.0
    runner.validate_settings(default)
    runner.validate_settings(r2)


def test_geometry_variants_are_explicit_and_recorded():
    args = _args()
    for name, expected in {
        'current_oacp_mosaic': 'current',
        'budget_oacp_mosaic': 'budget',
        'density_oacp_mosaic': 'density',
    }.items():
        settings = runner.effective_settings(args, name, 42)
        assert settings['augmentation']['oacp_variant'] == expected
        runner.validate_settings(settings)
        runner.configure_environment(name)
        assert __import__('os').environ['OACP_VARIANT'] == expected
