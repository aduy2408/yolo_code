"""Measure object-region activation survival across hooked feature stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import DatasetSpec, ProbeConfig, feature_summary, image_paths, load_yolo, metadata, module_map, read_boxes, size_bucket, write_json, write_rows


def run(spec: DatasetSpec, checkpoint: Path, output: Path, config: ProbeConfig) -> dict[str, Any]:
    model = load_yolo(checkpoint, config.device)
    modules = module_map(model)
    requested = [name for name in config.feature_layers if name in modules]
    captures: dict[str, Any] = {}
    handles = [modules[name].register_forward_hook(lambda _m, _i, out, name=name: captures.__setitem__(name, out)) for name in requested]
    rows: list[dict[str, Any]] = []
    try:
        for image_path in image_paths(spec, config.max_images):
            image = cv2.imread(str(image_path))
            boxes = read_boxes(image_path, spec)
            if image is None or not boxes:
                continue
            captures.clear()
            model.predict(image, imgsz=config.imgsz, conf=config.conf, device=config.device, verbose=False)
            for index, box in enumerate(boxes):
                for layer in requested:
                    values = feature_summary(captures.get(layer), box, image.shape[:2])
                    rows.append({"dataset": spec.name, "image": image_path.name, "object_id": index, "layer": layer, "size_bucket": size_bucket(box), **values, "survival_score": values["object_energy"] / max(values["context_energy"], 1e-8)})
    finally:
        for handle in handles:
            handle.remove()
    write_rows(output / "object_survival.csv", rows)
    summary = {"metadata": metadata(spec, config, checkpoint, "object_survival"), "layers": requested, "rows": len(rows), "survival_curves": {}}
    for layer in requested:
        subset = [row for row in rows if row["layer"] == layer]
        summary["survival_curves"][layer] = {bucket: float(np.mean([row["survival_score"] for row in subset if row["size_bucket"] == bucket])) for bucket in sorted({row["size_bucket"] for row in subset})}
    write_json(output / "object_survival.json", summary)
    return summary
