"""Create paired visual explanations for two YOLO checkpoints.

This is intentionally image-first: each output is a montage containing detections,
intermediate activation heatmaps, an activation difference map, and occlusion maps.
The activation map is a Grad-CAM-like feature evidence proxy, not a claim about
true detector attention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_model(checkpoint: Path, device: str) -> Any:
    from analysis.probing.common import load_yolo

    return load_yolo(checkpoint, device)


def _module(model: Any, name: str) -> Any:
    modules = dict(model.model.named_modules())
    if name not in modules:
        raise ValueError(f"Layer {name!r} is not present. Available examples: {list(modules)[:12]}")
    return modules[name]


def _activation(model: Any, image: np.ndarray, layer: str, device: str, conf: float) -> np.ndarray:
    captured: dict[str, Any] = {}
    handle = _module(model, layer).register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("value", output)
    )
    try:
        model.predict(image, imgsz=max(image.shape[:2]), device=device, conf=conf, verbose=False)
    finally:
        handle.remove()
    value = captured.get("value")
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if hasattr(item, "ndim")), None)
    if value is None or not hasattr(value, "detach"):
        raise RuntimeError(f"Layer {layer!r} did not produce a tensor activation")
    tensor = value.detach().float().abs()
    if tensor.ndim == 4:
        tensor = tensor[0].mean(0)
    elif tensor.ndim == 3:
        tensor = tensor.mean(0)
    else:
        raise RuntimeError(f"Unsupported activation shape at {layer}: {tuple(value.shape)}")
    heat = tensor.cpu().numpy()
    heat -= heat.min()
    heat /= max(float(heat.max()), 1e-8)
    return cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)


def _score(model: Any, image: np.ndarray, device: str, conf: float) -> float:
    result = model.predict(image, imgsz=max(image.shape[:2]), device=device, conf=conf, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0
    return float(boxes.conf.detach().float().max().cpu())


def _occlusion(model: Any, image: np.ndarray, device: str, grid: int, conf: float) -> np.ndarray:
    baseline = _score(model, image, device, conf)
    height, width = image.shape[:2]
    heat = np.zeros((grid, grid), dtype=np.float32)
    for row in range(grid):
        for col in range(grid):
            y1, y2 = row * height // grid, (row + 1) * height // grid
            x1, x2 = col * width // grid, (col + 1) * width // grid
            masked = image.copy()
            fill = np.median(image.reshape(-1, 3), axis=0).astype(image.dtype)
            masked[y1:y2, x1:x2] = fill
            heat[row, col] = max(0.0, baseline - _score(model, masked, device, conf))
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
    return heat / max(float(heat.max()), 1e-8)


def _overlay(image: np.ndarray, heat: np.ndarray, title: str) -> np.ndarray:
    color = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
    output = cv2.addWeighted(image, 0.48, color, 0.52, 0)
    cv2.putText(output, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _prediction(model: Any, image: np.ndarray, device: str, title: str, conf: float) -> np.ndarray:
    result = model.predict(image, imgsz=max(image.shape[:2]), device=device, conf=conf, verbose=False)[0]
    plotted = result.plot()
    cv2.putText(plotted, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return plotted


def _make_montage(images: list[np.ndarray], columns: int = 3) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        resized = cv2.resize(image, (width, height))
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = resized
    return canvas


def render_pair(baseline_path: Path, oacp_path: Path, image_path: Path, output: Path, layer: str, device: str, grid: int, conf: float) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    baseline = _load_model(baseline_path, device)
    oacp = _load_model(oacp_path, device)
    baseline_activation = _activation(baseline, image, layer, device, conf)
    oacp_activation = _activation(oacp, image, layer, device, conf)
    difference = np.abs(oacp_activation - baseline_activation)
    difference /= max(float(difference.max()), 1e-8)
    panels = [
        image,
        _prediction(baseline, image, device, "baseline detections", conf),
        _prediction(oacp, image, device, "OACP detections", conf),
        _overlay(image, baseline_activation, "baseline activation"),
        _overlay(image, oacp_activation, "OACP activation"),
        _overlay(image, difference, "absolute activation difference"),
        _overlay(image, _occlusion(baseline, image, device, grid, conf), "baseline occlusion sensitivity"),
        _overlay(image, _occlusion(oacp, image, device, grid, conf), "OACP occlusion sensitivity"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), _make_montage(panels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oacp", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/oacp_visual_compare"))
    parser.add_argument("--layer", default="model.18")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--occlusion-grid", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if args.occlusion_grid < 2:
        parser.error("--occlusion-grid must be at least 2")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    for image_path in args.images:
        output = args.output_dir / f"{image_path.stem}_{args.layer.replace('.', '_')}.jpg"
        render_pair(args.baseline, args.oacp, image_path, output, args.layer, args.device, args.occlusion_grid, args.confidence)
        print(output)


if __name__ == "__main__":
    main()
