"""Gradient stability proxy using per-object feature energy losses.

The probe backpropagates a scalar object-region activation objective through a
single image. It does not alter training or optimizer state. Repeated images
and objects provide gradient norm, variance, and signal-to-noise summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .common import DatasetSpec, ProbeConfig, image_paths, load_yolo, metadata, module_map, read_boxes, size_bucket, tensor_from_output, write_json, write_rows


def run(spec: DatasetSpec, checkpoint: Path, output: Path, config: ProbeConfig) -> dict[str, Any]:
    model = load_yolo(checkpoint, config.device)
    for parameter in model.model.parameters():
        parameter.requires_grad_(True)
    modules = module_map(model)
    requested = [name for name in config.feature_layers if name in modules]
    captures: dict[str, torch.Tensor] = {}
    handles = [modules[name].register_forward_hook(lambda _m, _i, out, name=name: captures.__setitem__(name, tensor_from_output(out))) for name in requested]
    rows: list[dict[str, Any]] = []
    try:
        for image_path in image_paths(spec, config.max_images):
            image = cv2.imread(str(image_path))
            boxes = read_boxes(image_path, spec)
            if image is None or not boxes:
                continue
            resized = cv2.resize(image, (config.imgsz, config.imgsz), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized[..., ::-1].copy()).to(config.device).permute(2, 0, 1).float()[None] / 255.0
            captures.clear()
            model.model.zero_grad(set_to_none=True)
            model.model(tensor)
            for index, box in enumerate(boxes):
                for layer in requested:
                    feature = captures.get(layer)
                    if feature is None or not feature.requires_grad:
                        continue
                    values = feature.float().abs()
                    if values.ndim == 4:
                        mask = torch.zeros(values.shape[-2:], dtype=torch.bool, device=values.device)
                        x1 = max(0, min(mask.shape[1] - 1, round(box["x1"] / image.shape[1] * mask.shape[1])))
                        x2 = max(x1 + 1, min(mask.shape[1], round(box["x2"] / image.shape[1] * mask.shape[1])))
                        y1 = max(0, min(mask.shape[0] - 1, round(box["y1"] / image.shape[0] * mask.shape[0])))
                        y2 = max(y1 + 1, min(mask.shape[0], round(box["y2"] / image.shape[0] * mask.shape[0])))
                        mask[y1:y2, x1:x2] = True
                        objective = values[0, :, mask].mean()
                    else:
                        objective = values.mean()
                    gradient = torch.autograd.grad(objective, feature, retain_graph=True, allow_unused=True)[0]
                    if gradient is None:
                        continue
                    values = gradient.detach().float().reshape(-1).cpu().numpy()
                    rows.append({"dataset": spec.name, "image": image_path.name, "object_id": index, "layer": layer, "size_bucket": size_bucket(box), "gradient_norm": float(np.linalg.norm(values)), "gradient_mean": float(values.mean()), "gradient_std": float(values.std()), "gradient_snr": float(abs(values.mean()) / (values.std() + 1e-8))})
    finally:
        for handle in handles:
            handle.remove()
    write_rows(output / "gradient_probe.csv", rows)
    summary = {"metadata": metadata(spec, config, checkpoint, "gradient_probe"), "layers": requested, "rows": len(rows), "by_layer": {}}
    for layer in requested:
        subset = [row for row in rows if row["layer"] == layer]
        summary["by_layer"][layer] = {key: float(np.mean([row[key] for row in subset])) for key in ("gradient_norm", "gradient_std", "gradient_snr")} if subset else {}
    write_json(output / "gradient_probe.json", summary)
    return summary
