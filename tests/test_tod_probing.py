from __future__ import annotations

import numpy as np
import torch

from analysis.probing.common import DatasetSpec, ProbeConfig, feature_summary, label_path, size_bucket
from analysis.probing.causal_intervention import _masked


def test_feature_summary_separates_object_and_context() -> None:
    feature = torch.ones(1, 2, 4, 4)
    feature[:, :, 1:3, 1:3] = 4
    values = feature_summary(feature, {"x1": 25, "y1": 25, "x2": 75, "y2": 75}, (100, 100))
    assert values["object_energy"] > values["context_energy"]
    assert values["object_fraction"] > 1


def test_mask_modes_are_deterministic() -> None:
    image = np.full((10, 10, 3), 10, dtype=np.uint8)
    image[2:5, 2:5] = 200
    boxes = [{"x1": 2, "y1": 2, "x2": 5, "y2": 5}]
    assert np.all(_masked(image, boxes, "object_only")[2:5, 2:5] == 200)
    assert np.all(_masked(image, boxes, "context_only")[2:5, 2:5] == 0)
    assert np.all(_masked(image, boxes, "object_removed")[2:5, 2:5] == 10)


def test_size_bucket_and_label_path() -> None:
    spec = DatasetSpec("x", __import__("pathlib").Path("/data"), "images/*")
    assert size_bucket({"x1": 0, "y1": 0, "x2": 4, "y2": 4}) == "tiny_lt8"
    assert label_path(__import__("pathlib").Path("/data/images/test/a.jpg"), spec).as_posix() == "/data/labels/test/a.txt"
    assert ProbeConfig().seed == 42
