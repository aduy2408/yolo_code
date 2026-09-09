from __future__ import annotations

import train_all_tinyperson_yolov9_yolov10_legacy_oacp_mosaic as runner


def test_defaults_cover_both_models_and_legacy_oacp_mosaic() -> None:
    args = runner.parse_args([
        '--data-root', '/marimo/TinyPerson',
        '--dataset-root', '/marimo/yolo_code/datasets',
        '--project', '/marimo/yolo_code/runs/test',
        '--hf-repo-id', 'duyle2408/tinyperson-yolov9-yolov10-legacy-oacp-mosaic',
    ])
    assert args.models == ['yolov9', 'yolov10']
    assert args.seeds == [42]
    assert args.split_seed == 42
    assert args.patience == 0
    assert runner.STRICT_AUGMENTATION['context_augmentation'] == 'oacp'
    assert runner.STRICT_AUGMENTATION['legacy_double_oacp'] is True
    assert runner.TRAIN_AUGMENTATION['mosaic'] == 1.0
    assert runner.TRAIN_AUGMENTATION['close_mosaic'] == 10


def test_settings_validation_rejects_wrong_split_seed() -> None:
    args = runner.parse_args([
        '--data-root', '/marimo/TinyPerson',
        '--dataset-root', '/marimo/yolo_code/datasets',
        '--project', '/marimo/yolo_code/runs/test',
        '--hf-repo-id', 'duyle2408/tinyperson-yolov9-yolov10-legacy-oacp-mosaic',
        '--split-seed', '43',
    ])
    settings = runner.effective_settings(args, 'yolov9', 42)
    try:
        runner.validate_settings(settings)
    except ValueError as exc:
        assert 'split_seed' in str(exc)
    else:
        raise AssertionError('split seed 43 must be rejected')
