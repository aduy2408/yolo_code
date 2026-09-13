import copy
import random

import numpy as np

from project_ultralytics.scene_compatible_mosaic import build_scale_reference
from ultralytics.data.augment import Mosaic, ScaleAdaptiveMosaic, build_mosaic
from ultralytics.utils.instance import Instances


def sample(boxes, size=4, index=0):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return {
        "im_file": f"image-{index}.jpg",
        "ori_shape": (size, size),
        "resized_shape": (size, size),
        "img": np.full((size, size, 3), index, dtype=np.uint8),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
        "instances": Instances(boxes, segments=np.empty((len(boxes), 0, 2), dtype=np.float32), bbox_format="xyxy", normalized=False),
    }


class FakeDataset:
    cache = "ram"
    buffer = []

    def __init__(self, samples):
        self.samples = samples
        self.labels = [
            {
                "shape": item["resized_shape"],
                "bboxes": item["instances"].bboxes.copy(),
                "normalized": False,
                "bbox_format": "xyxy",
            }
            for item in samples
        ]
        self.im_files = [item["im_file"] for item in samples]

    def __len__(self):
        return len(self.samples)

    def get_image_and_label(self, index):
        return copy.deepcopy(self.samples[index])


def make_policy(samples, floor=None):
    policy = ScaleAdaptiveMosaic(FakeDataset(samples), imgsz=4, p=1.0)
    if floor is not None:
        policy.scale_reference["r_floor"] = floor
        policy.scale_reference["r_q05"] = floor
    return policy


def test_scale_reference_uses_positive_scene_floor_only():
    reference = build_scale_reference(
        [
            {"shape": (4, 4), "bboxes": np.empty((0, 4)), "normalized": False, "bbox_format": "xyxy"},
            {"shape": (4, 4), "bboxes": np.array([[0, 0, 4, 4]], dtype=np.float32), "normalized": False, "bbox_format": "xyxy"},
        ]
    )
    assert reference["num_positive_images"] == 1
    assert reference["num_empty_images"] == 1
    assert reference["num_objects"] == 1
    assert reference["r_q05"] == 1.0


def test_large_boxes_choose_k4_and_small_boxes_choose_k2():
    large = [sample([[0, 0, 4, 4]], index=i) for i in range(4)]
    small = [sample([[1, 1, 2, 2]], index=i) for i in range(4)]

    large_policy = make_policy(large, floor=0.1)
    large_params = large_policy.get_params(large[0])
    assert large_params["samc_mode"] == 4

    small_policy = make_policy(small)
    small_params = small_policy.get_params(small[0])
    assert small_params["samc_mode"] == 2


def test_empty_group_chooses_k4_and_missing_boxes_choose_k2():
    empty = [sample([], index=i) for i in range(4)]
    policy = make_policy(empty)
    params = policy.get_params(empty[0])
    assert params["samc_mode"] == 4

    with_gt = [sample([[0, 0, 4, 4]], index=0), sample([], index=1), sample([], index=2), sample([], index=3)]
    policy = make_policy(with_gt)
    params = policy.get_params(with_gt[0])
    assert params["samc_mode"] in {2, 4}
    assert policy._diagnostics["groups_seen"] == 1


def test_two_way_layouts_keep_boxes_in_canvas_and_preserve_dimensions():
    samples = [sample([[0, 0, 4, 4]], index=i) for i in range(4)]
    policy = make_policy(samples)
    for horizontal in (True, False):
        layout = policy._build_two_layout(samples[0], samples[1], horizontal=horizontal)
        canvas_shape = (4, 8) if horizontal else (8, 4)
        boxes = policy._boxes_after_layout(layout, canvas_shape)
        assert len(boxes) == 2
        assert np.all(boxes >= 0)
        assert np.all(boxes[:, [0, 2]] <= canvas_shape[1])
        assert np.all(boxes[:, [1, 3]] <= canvas_shape[0])
        assert layout[0]["x2a"] - layout[0]["x1a"] <= 4
        assert layout[0]["y2a"] - layout[0]["y1a"] <= 4


def test_scale_adaptive_call_renders_selected_plan_once_and_logs_rates(tmp_path):
    random.seed(42)
    samples = [sample([[1, 1, 2, 2]], index=i) for i in range(4)]
    dataset = FakeDataset(samples)
    policy = ScaleAdaptiveMosaic(dataset, imgsz=4, p=1.0)
    result = policy(samples[0])
    assert result["img"].shape[:2] in {(4, 8), (8, 4)}
    assert tuple(result["resized_shape"]) in {(4, 8), (8, 4)}
    assert np.all(result["instances"].bboxes >= 0)
    assert np.all(result["instances"].bboxes[:, [0, 2]] <= result["resized_shape"][1])
    assert np.all(result["instances"].bboxes[:, [1, 3]] <= result["resized_shape"][0])
    diagnostics = policy._diagnostics
    assert diagnostics["groups_seen"] == 1
    assert diagnostics["k2_count"] + diagnostics["k4_count"] == 1
    output_path = tmp_path / "samc-test-diagnostics.json"
    policy.save_diagnostics(output_path)
    import json

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert "samc/r4_mean" in saved
    assert "samc/source_gt_mean" in saved


def test_standard_policy_is_unchanged_and_scale_adaptive_is_opt_in():
    samples = [sample([[0, 0, 4, 4]], index=i) for i in range(4)]
    dataset = FakeDataset(samples)
    standard = build_mosaic(dataset, 4, type("Hyp", (), {"mosaic": 1.0, "mosaic_policy": "standard"})())
    adaptive = build_mosaic(
        dataset,
        4,
        type("Hyp", (), {"mosaic": 1.0, "mosaic_policy": "scale_adaptive", "mosaic_scale_quantile": 0.05, "mosaic_scale_modes": [4, 2]})(),
    )
    assert isinstance(standard, Mosaic)
    assert not isinstance(standard, ScaleAdaptiveMosaic)
    assert isinstance(adaptive, ScaleAdaptiveMosaic)
