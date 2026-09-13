import random

import numpy as np

from project_ultralytics.scene_compatible_mosaic import (
    SceneCompatibleMosaic,
    build_reference_stats,
    compatibility_drift,
    feature_drift,
    scene_descriptor,
)


def raw_label(boxes, shape=(100, 100)):
    return {
        "shape": shape,
        "normalized": False,
        "bbox_format": "xyxy",
        "bboxes": np.asarray(boxes, dtype=np.float32),
    }


def test_scene_descriptor_computes_geometry_and_masks_single_box_spacing():
    descriptor = scene_descriptor(raw_label([[10, 10, 20, 20], [50, 10, 60, 20]]))
    assert descriptor["count"] == 2.0
    assert np.isclose(descriptor["relative_size"], 0.1)
    assert np.isclose(descriptor["spacing"], 4.0)
    assert np.isclose(descriptor["occupancy"], 0.02)

    single = scene_descriptor(raw_label([[10, 10, 20, 20]]))
    assert single["spacing"] is None


def test_reference_stats_and_drift_use_only_outside_support():
    reference = build_reference_stats(
        [
            raw_label([[10, 10, 20, 20]]),
            raw_label([[10, 10, 20, 20], [40, 10, 50, 20]]),
            raw_label([[10, 10, 30, 30], [60, 10, 80, 30]]),
        ]
    )
    assert reference["features"]["spacing"]["valid_count"] == 2
    assert np.isclose(feature_drift(0.5, {"q05": 1.0, "q50": 2.0, "q95": 3.0}), 0.5)
    assert np.isclose(feature_drift(2.0, {"q05": 1.0, "q50": 2.0, "q95": 3.0}), 0.0)
    drift, per_feature = compatibility_drift(
        {"count": 2.0, "relative_size": 0.1, "spacing": None, "occupancy": 0.02}, reference
    )
    assert drift >= 0.0
    assert per_feature["spacing"] is None


def test_scene_compatible_mosaic_keeps_geometry_transform_and_records_diagnostics(monkeypatch):
    dataset = type(
        "Dataset",
        (),
        {"labels": [raw_label([[10, 10, 20, 20]]), raw_label([[10, 10, 20, 20]])]},
    )()
    candidate = raw_label([[10, 10, 20, 20]])
    calls = {"count": 0}

    def base_mosaic(labels):
        calls["count"] += 1
        return candidate

    monkeypatch.setattr(random, "random", lambda: 0.0)
    transform = SceneCompatibleMosaic(dataset, base_mosaic, p=1.0)
    result = transform(raw_label([[1, 1, 2, 2]]))

    assert result is candidate
    assert calls["count"] == 1
    diagnostics = transform.diagnostics()
    assert diagnostics["stats"]["scm/candidates"] == 1
    assert diagnostics["stats"]["scm/accepted"] == 1
    assert diagnostics["stats"]["scm/accept_rate"] == 1.0
    assert "scm/reference_count_q05" in diagnostics["stats"]
    assert "scm/reference_size_q50" in diagnostics["stats"]
