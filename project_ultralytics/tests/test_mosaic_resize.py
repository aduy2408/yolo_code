from __future__ import annotations

import numpy as np

from project_ultralytics.mosaic_resize import FullCanvasResize
from ultralytics.utils.instance import Instances


def test_full_canvas_resize_scales_image_and_boxes():
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    instances = Instances(
        np.asarray([[100, 200, 300, 400]], dtype=np.float32),
        np.zeros((1, 0, 2), dtype=np.float32),
        bbox_format="xyxy",
        normalized=False,
    )
    labels = {"img": image, "instances": instances, "cls": np.asarray([[0]], dtype=np.float32)}

    output = FullCanvasResize(512)(labels)

    assert output["img"].shape == (512, 512, 3)
    assert np.allclose(output["instances"].bboxes, [[50, 100, 150, 200]])
    assert output["resized_shape"] == (512, 512)
