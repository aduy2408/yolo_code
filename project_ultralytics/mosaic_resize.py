"""Full-canvas resize for Mosaic outputs before photometric augmentation."""
from __future__ import annotations

import cv2
import numpy as np


class FullCanvasResize:
    """Resize the complete square Mosaic canvas to the model input size.

    This intentionally does not crop, translate, rotate, shear, or apply a
    perspective transform. Bounding boxes and instance geometry receive the
    same isotropic resize as the image.
    """

    def __init__(self, size: int, fill: int = 114) -> None:
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, labels: dict) -> dict:
        image = labels["img"]
        source_h, source_w = image.shape[:2]
        if (source_h, source_w) != (self.size, self.size):
            labels["img"] = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        instances = labels.get("instances")
        if instances is not None and len(instances):
            instances.convert_bbox("xyxy")
            if instances.normalized:
                instances.denormalize(source_w, source_h)
            instances.scale(self.size / source_w, self.size / source_h)
            instances.clip(self.size, self.size)
            instances.normalized = False
            labels["instances"] = instances
        labels["resized_shape"] = (self.size, self.size)
        return labels


__all__ = ["FullCanvasResize"]
