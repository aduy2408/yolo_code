#!/usr/bin/env python3
"""Build a concrete TinyPerson OACP visual resource pack."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from project_ultralytics.context_augment import _far_mask, _mask_from_boxes_per_expand, _resize_degrade

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/TinyPerson"
ANNOTATIONS = DATASET / "annotations/tiny_set_train.json"
IMAGE_ROOT = DATASET / "train"
PACK = ROOT / "docs/reports/augmentation_examples/oacp_tinyperson_resource_pack"
SCALE = 0.65
STRENGTH = 0.40


def choose_image():
    data = json.loads(ANNOTATIONS.read_text())
    grouped = {}
    for ann in data["annotations"]:
        if ann.get("ignore") or ann.get("uncertain") or ann.get("logo"):
            continue
        x, y, w, h = ann["bbox"]
        size = float(np.sqrt(max(1.0, w * h)))
        if size > 80:
            continue
        grouped.setdefault(ann["image_id"], []).append([x, y, x + w, y + h, size])
    candidates = []
    images = {item["id"]: item for item in data["images"]}
    for image_id, boxes in grouped.items():
        if len(boxes) >= 8:
            candidates.append((abs(len(boxes) - 20), float(np.median([b[4] for b in boxes])), image_id, boxes))
    if not candidates:
        raise RuntimeError("No suitable TinyPerson image found")
    _, _, image_id, boxes = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    image_info = images[image_id]
    return IMAGE_ROOT / image_info["file_name"], np.asarray(boxes, dtype=np.float32)[:, :4], image_info


def draw_boxes(image, boxes, colors=None, thickness=2):
    out = image.copy()
    colors = colors or [(0, 0, 255)] * len(boxes)
    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def crop(image, box, pad=80):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    right, bottom = min(w, x2 + pad), min(h, y2 + pad)
    return image[top:bottom, left:right].copy(), np.asarray([x1-left, y1-top, x2-left, y2-top])


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(path)


def make_flow(pack: Path):
    names = {
        "input": "input_with_gt_boxes.png",
        "expand": "expanded_protection_boxes.png",
        "mask": "protected_mask.png",
        "down": "downsampled_context.png",
        "up": "upsampled_context.png",
        "output": "oacp_output.png",
        "diff": "difference_map_black_red.png",
    }
    images = {key: cv2.imread(str(pack / value)) for key, value in names.items()}
    canvas = np.full((980, 1800, 3), 248, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def panel(x, y, key, title, size=(260, 155)):
        im = images[key]
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        canvas[y:y+size[1], x:x+size[0]] = im
        cv2.rectangle(canvas, (x, y), (x+size[0], y+size[1]), (45, 45, 45), 2)
        cv2.putText(canvas, title, (x, y-12), font, 0.65, (25, 25, 25), 2, cv2.LINE_AA)

    def arrow(x1, y1, x2, y2):
        cv2.arrowedLine(canvas, (x1, y1), (x2, y2), (55, 55, 55), 3, cv2.LINE_AA, tipLength=0.04)

    cv2.putText(canvas, "OACP: protect tiny objects, degrade far context", (44, 55), font, 1.05, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Protection branch", (450, 105), font, 0.78, (20, 90, 150), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Context degradation branch", (450, 530), font, 0.78, (150, 80, 35), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Fusion", (1280, 105), font, 0.78, (25, 110, 55), 2, cv2.LINE_AA)

    panel(45, 180, "input", "Input image + GT boxes")
    panel(380, 180, "expand", "Expanded protection boxes")
    panel(715, 180, "mask", "Soft protection mask")
    panel(380, 610, "down", "Downsample context")
    panel(715, 610, "up", "Upsample context")
    panel(1280, 180, "output", "OACP output")
    panel(1600, 180, "diff", "Difference: black to red", size=(155, 155))

    arrow(305, 258, 370, 258)
    arrow(640, 258, 705, 258)
    arrow(305, 310, 370, 688)
    arrow(640, 688, 705, 688)
    arrow(975, 258, 1265, 258)
    arrow(975, 688, 1180, 688)
    arrow(1180, 688, 1180, 340)
    arrow(1545, 258, 1590, 258)
    cv2.putText(canvas, "protect every GT object", (385, 390), font, 0.58, (20, 90, 150), 2, cv2.LINE_AA)
    cv2.putText(canvas, "degrade only far context", (745, 875), font, 0.58, (150, 80, 35), 2, cv2.LINE_AA)
    cv2.putText(canvas, "blend protected original with degraded context", (1000, 145), font, 0.58, (25, 110, 55), 2, cv2.LINE_AA)
    cv2.putText(canvas, "yellow/red boxes in the source are tiny people", (45, 910), font, 0.58, (65, 65, 65), 2, cv2.LINE_AA)
    save(pack / "oacp_tinyperson_flow.png", canvas)


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    image_path, boxes, info = choose_image()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    selected = boxes[int(np.argmin(np.sqrt((boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]))))]
    sizes = np.sqrt(np.maximum(1.0, (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])))
    factors = np.where(sizes < 32.0, 3.0, 1.2).astype(np.float32)
    protected = _mask_from_boxes_per_expand(boxes, factors, h, w).astype(bool)
    far_mask = _far_mask(protected.astype(np.uint8), h, w)
    downsampled = cv2.resize(image, (max(2, int(w*SCALE)), max(2, int(h*SCALE))), interpolation=cv2.INTER_AREA)
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = (STRENGTH * far_mask)[..., None]
    output = np.clip(image.astype(np.float32) * (1-alpha) + upsampled.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    diff = np.mean(np.abs(image.astype(np.float32)-output.astype(np.float32)), axis=2)
    positive = diff[diff > 1.0]
    low = float(np.percentile(positive, 5)) if len(positive) else 0.0
    high = float(np.percentile(positive, 98)) if len(positive) else 1.0
    high = max(high, low + 1e-6)
    robust = np.clip((diff - low) / (high - low), 0.0, 1.0)
    robust[diff <= low] = 0.0
    robust = np.power(robust, 0.55)
    norm = np.round(robust * 255.0).astype(np.uint8)
    heat = np.zeros_like(image)
    heat[..., 2] = norm

    expanded_boxes = np.asarray([
        [
            ((b[0] + b[2]) / 2 - (b[2] - b[0]) * f / 2),
            ((b[1] + b[3]) / 2 - (b[3] - b[1]) * f / 2),
            ((b[0] + b[2]) / 2 + (b[2] - b[0]) * f / 2),
            ((b[1] + b[3]) / 2 + (b[3] - b[1]) * f / 2),
        ]
        for b, f in zip(boxes, factors)
    ], dtype=np.float32)

    save(PACK / "input_with_gt_boxes.png", draw_boxes(image, boxes))
    gt_panel = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            gt_panel,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            255,
            thickness=-1,
        )
    save(PACK / "gt_boxes_black_white.png", cv2.cvtColor(gt_panel, cv2.COLOR_GRAY2BGR))
    selected_crop, local = crop(image, selected)
    save(PACK / "selected_tiny_person_zoom.png", draw_boxes(selected_crop, [local], [(0, 255, 255)], 3))
    expanded_panel = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in expanded_boxes:
        cv2.rectangle(
            expanded_panel,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            255,
            thickness=-1,
        )
    save(PACK / "expanded_protection_boxes.png", cv2.cvtColor(expanded_panel, cv2.COLOR_GRAY2BGR))

    # M follows the figure convention: M=0 protects the expanded object
    # regions, M=1 degrades far context, with a smooth transition between them.
    soft_mask = np.clip(far_mask * 255.0, 0.0, 255.0).astype(np.uint8)
    save(PACK / "protected_mask.png", cv2.cvtColor(soft_mask, cv2.COLOR_GRAY2BGR))
    save(PACK / "downsampled_context.png", downsampled)
    save(PACK / "upsampled_context.png", upsampled)
    save(PACK / "oacp_output.png", draw_boxes(output, boxes))
    save(PACK / "difference_map_black_red.png", heat)
    save(PACK / "difference_map_black_red_enhanced.png", heat)
    out_crop, _ = crop(output, selected)
    save(PACK / "selected_tiny_person_before_after.png", cv2.hconcat([selected_crop, out_crop]))
    make_flow(PACK)

    manifest = {
        "dataset": "TinyPerson",
        "source_image": str(image_path),
        "source_file_name": info["file_name"],
        "image_size": [int(w), int(h)],
        "num_valid_small_objects": int(len(boxes)),
        "selected_box_xyxy": selected.tolist(),
        "small_object_cutoff": 32.0,
        "effective_expand_factors": factors.tolist(),
        "degradation": {"scale": SCALE, "blend_strength": STRENGTH},
        "difference_map": {
            "colormap": "black_to_red",
            "lower_percentile": 5,
            "upper_percentile": 98,
            "gamma": 0.55,
            "noise_floor": 1.0,
        },
        "resources": sorted(p.name for p in PACK.glob("*.png")),
    }
    (PACK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (PACK / "README.md").write_text(
        "# TinyPerson OACP visual resource pack\n\n"
        "Concrete real-image resources for a clearer OACP flow diagram.\n\n"
        "The selected source contains multiple small people. Every valid object with size < 32 px uses a 3.0x protection expansion.\n"
        "The difference map uses robust 5th-98th percentile normalization, gamma 0.55 contrast lifting, a 1.0-pixel noise floor, and black for unchanged pixels.\n"
        "Use `oacp_tinyperson_flow.png` as the standalone replacement for the current draw.io flow draft.\n"
    )
    print(f"source={image_path}")
    print(f"objects={len(boxes)}")
    print(f"output={PACK}")


if __name__ == "__main__":
    main()
