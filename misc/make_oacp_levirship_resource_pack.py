#!/usr/bin/env python3
"""Build individual real-image resources for the OACP pipeline diagram."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from project_ultralytics.context_augment import (
    _far_mask,
    _mask_from_boxes_per_expand,
    _resize_degrade,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "datasets/LevirShipData/All Images"
LABEL_DIR = ROOT / "datasets/LevirShipData/All Annotations"
PACK = ROOT / "docs/reports/augmentation_examples/oacp_levirship_resource_pack"
SELECTED_SOURCE = ROOT / "datasets/LevirShipData/All Images/GF1_WFV1_E110.0_N17.9_20200703_L2A0004902374_8192_8704.png"
SELECTED_BOX = np.asarray([140.0, 400.0, 145.0, 412.0], dtype=np.float32)
STRENGTH = 0.30
SCALE = 0.75


def boxes_for(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    boxes = []
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            _, cx, cy, bw, bh = map(float, fields[:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
    return image, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def draw_boxes(image, boxes, color=(0, 0, 255), thickness=1):
    output = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return output


def crop_with_box(image, box, pad=16):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    right, bottom = min(w, x2 + pad), min(h, y2 + pad)
    crop = image[top:bottom, left:right].copy()
    local = np.asarray([x1 - left, y1 - top, x2 - left, y2 - top], dtype=np.int32)
    return crop, local


def save(path: Path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    selected, all_boxes = boxes_for(SELECTED_SOURCE)
    if len(all_boxes) == 0:
        all_boxes = SELECTED_BOX.reshape(1, 4)
    selected_box = SELECTED_BOX
    h, w = selected.shape[:2]

    effective_expand_factors = []
    for box in all_boxes:
        size = float(np.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])))
        effective_expand_factors.append(3.0 if size < 32.0 else 1.2)
    # Protect every GT object with its effective expansion. The selected object
    # is already covered by this union, so no selected-only mask is needed.
    protected = _mask_from_boxes_per_expand(
        all_boxes, np.asarray(effective_expand_factors), h, w
    ).astype(bool)
    far_mask = _far_mask(protected.astype(np.uint8), h, w)
    degraded = _resize_degrade(selected, SCALE)
    blend = np.clip(
        selected.astype(np.float32) * (1.0 - STRENGTH * far_mask[..., None])
        + degraded.astype(np.float32) * (STRENGTH * far_mask[..., None]),
        0,
        255,
    ).astype(np.uint8)

    save(PACK / "selected_small_object_source.png", draw_boxes(selected, all_boxes))
    crop, local_box = crop_with_box(selected, selected_box)
    crop_degraded, _ = crop_with_box(degraded, selected_box)
    save(PACK / "selected_object_crop_original.png", draw_boxes(crop, [local_box]))
    save(PACK / "selected_object_crop_degraded.png", draw_boxes(crop_degraded, [local_box]))
    save(PACK / "degraded_full_image.png", draw_boxes(degraded, all_boxes))
    save(PACK / "blend_output.png", draw_boxes(blend, all_boxes))
    save(PACK / "oacp_output.png", draw_boxes(blend, all_boxes))
    difference = np.mean(np.abs(selected.astype(np.float32) - blend.astype(np.float32)), axis=2)
    max_difference = float(np.max(difference))
    positive_difference = difference[difference > 0]
    contrast_reference = float(np.percentile(positive_difference, 99.0)) if len(positive_difference) else 0.0
    contrast_reference = max(contrast_reference, 1e-6)
    normalized_difference = np.clip(difference / contrast_reference * 255.0, 0, 255).astype(np.uint8)
    # Use a vivid perceptual colormap instead of a single red channel. The
    # percentile reference prevents a few extreme pixels from washing out the
    # rest of the difference structure.
    difference_heatmap = cv2.applyColorMap(normalized_difference, cv2.COLORMAP_TURBO)
    difference_heatmap[normalized_difference == 0] = 0
    difference_overlay = cv2.addWeighted(selected, 0.55, difference_heatmap, 0.45, 0.0)
    save(PACK / "oacp_normalized_difference_heatmap.png", difference_heatmap)
    save(PACK / "oacp_difference_overlay.png", difference_overlay)
    save(PACK / "oacp_original_output_pair.png", cv2.hconcat([selected, blend]))
    save(PACK / "protected_region.png", selected)
    region = selected.copy()
    region = draw_boxes(region, [selected_box], color=(0, 0, 255), thickness=2)
    sx1, sy1, sx2, sy2 = selected_box
    scx, scy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
    sw, sh = (sx2 - sx1) * 3.0, (sy2 - sy1) * 3.0
    selected_expand_box = np.asarray([scx - sw / 2.0, scy - sh / 2.0, scx + sw / 2.0, scy + sh / 2.0])
    all_expand_boxes = []
    for box, factor in zip(all_boxes, effective_expand_factors):
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        bw, bh = (box[2] - box[0]) * 1.2, (box[3] - box[1]) * 1.2
        all_expand_boxes.append([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0])
    effective_expand_boxes = []
    for box, factor in zip(all_boxes, effective_expand_factors):
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        bw, bh = (box[2] - box[0]) * factor, (box[3] - box[1]) * factor
        effective_expand_boxes.append([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0])
    region = draw_boxes(region, [selected_expand_box], color=(255, 180, 0), thickness=2)
    region = draw_boxes(region, np.asarray(all_expand_boxes), color=(0, 255, 0), thickness=1)
    region = draw_boxes(region, np.asarray(effective_expand_boxes), color=(255, 0, 255), thickness=1)
    save(PACK / "protected_region_boxes.png", region)
    original_boxes_view = np.zeros((h, w, 3), dtype=np.uint8)
    original_boxes_view = draw_boxes(original_boxes_view, all_boxes, color=(255, 255, 255), thickness=2)
    expanded_boxes_view = np.zeros((h, w, 3), dtype=np.uint8)
    expanded_boxes_view = draw_boxes(expanded_boxes_view, np.asarray(effective_expand_boxes), color=(255, 255, 255), thickness=2)
    save(PACK / "protected_original_boxes_B.png", original_boxes_view)
    save(PACK / "protected_expanded_boxes.png", expanded_boxes_view)
    save(PACK / "protected_effective_expansion.png", draw_boxes(selected, np.asarray(effective_expand_boxes), color=(255, 0, 255), thickness=1))

    protected_view = (protected.astype(np.uint8) * 255)
    far_view = np.clip(far_mask * 255.0, 0, 255).astype(np.uint8)
    save(PACK / "protected_map.png", cv2.cvtColor(protected_view, cv2.COLOR_GRAY2BGR))
    save(PACK / "far_context_mask.png", cv2.cvtColor(far_view, cv2.COLOR_GRAY2BGR))
    save(PACK / "protected_distance_transform_M.png", cv2.cvtColor(far_view, cv2.COLOR_GRAY2BGR))
    triptych = cv2.hconcat([original_boxes_view, expanded_boxes_view, cv2.cvtColor(far_view, cv2.COLOR_GRAY2BGR)])
    save(PACK / "protected_regions_triptych.png", triptych)
    small = cv2.resize(selected, (max(2, int(w * SCALE)), max(2, int(h * SCALE))), interpolation=cv2.INTER_AREA)
    save(PACK / "downsampled_context.png", small)
    save(PACK / "upsampled_context.png", degraded)
    save(PACK / "mask_M.png", cv2.cvtColor(far_view, cv2.COLOR_GRAY2BGR))

    input_names = [
        "GF1_WFV1_E110.0_N17.9_20200703_L2A0004902374_11264_7680.png",
        "GF1_WFV1_E110.0_N17.9_20200703_L2A0004902374_8192_4096.png",
        SELECTED_SOURCE.name,
    ]
    for index, name in enumerate(input_names, start=1):
        image, boxes = boxes_for(IMAGE_DIR / name)
        save(PACK / f"input_image_{index:02d}.png", draw_boxes(image, boxes))

    manifest = {
        "dataset": "LevirShipData",
        "selected_source": str(SELECTED_SOURCE),
        "selected_box_xyxy": selected_box.tolist(),
        "all_boxes_xyxy": all_boxes.tolist(),
        "num_gt": int(len(all_boxes)),
        "object_size": float(np.sqrt((selected_box[2] - selected_box[0]) * (selected_box[3] - selected_box[1]))),
        "small_object_cutoff": 32.0,
        "protected_expand_all_gt": 1.2,
        "protected_expand_selected_object": 3.0,
        "protected_effective_expansion": "per-GT union; every object with size < 32 uses 3.0x",
        "effective_expand_factors": effective_expand_factors,
        "all_objects_small": bool(all(factor == 3.0 for factor in effective_expand_factors)),
        "degradation": {"scale": SCALE, "blend_strength": STRENGTH},
        "difference": {
            "max_abs_mean_pixel": max_difference,
            "mean_abs_mean_pixel": float(np.mean(difference)),
            "contrast_reference_p99": contrast_reference,
            "colormap": "TURBO",
        },
        "resources": sorted(path.name for path in PACK.glob("*.png")),
    }
    (PACK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (PACK / "README.md").write_text(
        "# OACP LevirShip resource pack\n\n"
        "Individual real-image resources for rebuilding the OACP pipeline in draw.io.\n\n"
        "- `input_image_*.png`: real source images with red GT boxes.\n"
        "- `selected_small_object_source.png`: selected object with a red box.\n"
        "- `protected_region_boxes.png`: red selected object, cyan 3.0x protection, green 1.2x all-GT safety region.\n"
        "- `protected_original_boxes_B.png`: original GT boxes on black.\n"
        "- `protected_expanded_boxes.png`: effective expansion: every object with s<32 uses 3.0x; non-small objects use 1.2x.\n"
        "- `protected_effective_expansion.png`: effective expansion over the real source image.\n"
        "- `protected_distance_transform_M.png`: soft distance-transform mask.\n"
        "- `protected_regions_triptych.png`: original boxes, expanded boxes, and soft mask side by side.\n"
        "- `degraded_full_image.png`, `downsampled_context.png`, `upsampled_context.png`: far-context degradation resources.\n"
        "- `selected_object_crop_original.png`, `selected_object_crop_degraded.png`: original/degraded crop pair.\n"
        "- `blend_output.png`, `oacp_output.png`: blended OACP-style outputs.\n"
        "- `oacp_normalized_difference_heatmap.png`: percentile-contrast normalized absolute difference heatmap using the TURBO colormap; every GT object is protected in the source mask.\n"
        "- `oacp_difference_overlay.png`: original image overlaid with the difference heatmap.\n"
        "- `oacp_original_output_pair.png`: original and OACP output side by side.\n\n"
        "The resources use real LevirShip pixels and the selected object provenance recorded in `manifest.json`.\n"
    )
    print(f"resources={len(manifest['resources'])}")
    print(f"object_size={manifest['object_size']:.2f}")
    print(f"output={PACK}")


if __name__ == "__main__":
    main()
