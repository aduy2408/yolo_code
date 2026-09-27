"""Feature-level object/nuisance retention probe.

This is a lightweight proxy for information retention, not a mutual
information estimator. It compares foreground/background activation energy and
linear separability of object-present versus context-only pooled features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import DatasetSpec, ProbeConfig, feature_summary, image_paths, load_yolo, metadata, module_map, read_boxes, size_bucket, tensor_from_output, write_json, write_rows


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
                    rows.append({"dataset": spec.name, "image": image_path.name, "object_id": index, "layer": layer, "size_bucket": size_bucket(box), **values, "nuisance_proxy": values["context_energy"] / max(values["energy"], 1e-8)})
    finally:
        for handle in handles:
            handle.remove()
    write_rows(output / "information_probe.csv", rows)
    summary = {"metadata": metadata(spec, config, checkpoint, "information_probe"), "layers": requested, "rows": len(rows), "by_layer": {}}
    for layer in requested:
        subset = [row for row in rows if row["layer"] == layer]
        summary["by_layer"][layer] = {"object_fraction": float(np.mean([row["object_fraction"] for row in subset])), "nuisance_proxy": float(np.mean([row["nuisance_proxy"] for row in subset]))} if subset else {}
    write_json(output / "information_probe.json", summary)
    return summary
