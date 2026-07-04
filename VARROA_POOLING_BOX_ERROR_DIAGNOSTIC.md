# YOLOv8n Varroa Pooling Box Error Diagnostic

Short version. Full raw tables and older follow-ups moved to
[VARROA_POOLING_BOX_ERROR_DIAGNOSTIC_APPENDIX.md](VARROA_POOLING_BOX_ERROR_DIAGNOSTIC_APPENDIX.md).

## Current Diagnosis

Baseline finds mites, but boxes are not tight enough.

| Metric | Value |
|---|---:|
| Precision | 0.922 |
| Recall | 0.892 |
| mAP50 | 0.925 |
| mAP50-95 | 0.352 |

AP collapses as IoU threshold rises:

| IoU | AP |
|---:|---:|
| 0.50 | 0.92482 |
| 0.75 | 0.13778 |
| 0.90 | 0.00041 |
| 0.95 | 0.00002 |

Meaning: detection/classification is mostly fine; localization and ranking are
the bottlenecks.

## Main Evidence

Validation has `685` GT objects. Best matched IoU quantiles:

| Quantile | Best GT IoU |
|---:|---:|
| p25 | 0.690 |
| p50 | 0.753 |
| p75 | 0.809 |
| p90 | 0.854 |

The AP50-but-not-AP75 bucket is large:

| Best IoU bucket | Count |
|---|---:|
| < 0.50 | 8 |
| 0.50-0.75 | 327 |
| 0.75-0.90 | 336 |
| >= 0.90 | 14 |

In the `0.50-0.75` bucket, edge errors of only a few pixels are enough to break
high-IoU AP:

| Error | p50 | p75 | p90 |
|---|---:|---:|---:|
| center x | 2.30 | 3.44 | 4.48 |
| center y | 1.90 | 3.32 | 4.88 |
| left edge | 2.55 | 4.33 | 5.95 |
| top edge | 3.04 | 4.47 | 7.16 |
| right edge | 3.02 | 4.94 | 6.57 |
| bottom edge | 2.42 | 4.02 | 6.02 |
| width | 4.00 | 6.28 | 9.07 |
| height | 3.81 | 6.29 | 8.82 |

Confidence is weakly tied to box tightness. There are `599` predictions with:

```text
confidence >= 0.25
best IoU < 0.75
```

Median best IoU for those is only about `0.639`.

## Ranking Problem

Raw candidates often contain better boxes than final NMS keeps.

Oracle pre-NMS upper bound:

| Candidate source | Recall@0.50 | Recall@0.75 | Recall@0.90 | Median best IoU |
|---|---:|---:|---:|---:|
| stride8 only | 0.977 | 0.530 | 0.061 | 0.761 |
| stride16 only | 0.971 | 0.480 | 0.060 | 0.744 |
| stride8/16 oracle | 0.999 | 0.753 | 0.111 | 0.811 |
| stride4/8/16 oracle | 1.000 | 0.761 | 0.111 | 0.812 |

Key inversion counts:

- `357 / 685`: stride8 has better IoU than stride16, but stride16 has higher score.
- `180 / 685`: stride8 has an AP75-passing candidate while stride16 does not, but stride16 still has higher score.

Removing P4/P5 did not solve it. In P2/P3-only, mismatch moves inside P3:

- `349 / 685` GTs had a P3 candidate with IoU `>= 0.75`, but top-scoring P3 anchor had IoU `< 0.75`.
- `525 / 685` GTs had P3 best IoU at least `0.05` higher than top-score-anchor IoU.

## Failed Or Weak Fixes

These are not enough:

- Global stride score scaling.
- P2/P3-only Detect.
- Old `loc_quality` Gaussian center-map target.
- `cls_iou_target_set` alone.
- VFL loss-only change with current YOLOv8 classification branch.
- Soft-NMS alone.

Reason: they do not make final score reliably rank tight boxes above loose boxes.

## Correct Target

Do not reuse old Gaussian `loc_quality` center map as quality target.

Use decoded-box IoU:

```python
quality_target = IoU(decoded_pred_box, matched_gt_box).detach()
```

The quality score must mean "this anchor's decoded box is tight", not "this
grid point is near object center".

## Current Recommended Experiment

Use true IoU quality head:

- Detect predicts box, class, and quality `q`.
- Positives target `IoU(decoded_pred_box, assigned_gt).detach()`.
- Negatives target `0`.
- Final score uses one of:
  - `cls * q`
  - `sqrt(cls) * q`
  - `cls * q^2`

Current config:

```yaml
quality_head: true
quality_loss: bce_balanced
quality_gain: 0.5
quality_neg_gain: 0.10
quality_pos_iou_thr: 0.5
quality_hard_neg_iou_thr: 0.3
quality_hard_neg_score_thr: 0.05
quality_target_power: 2.0
quality_neg_mode: hard
quality_score_mode: sqrt_cls_mul_q
quality_detach_target: true
```

Use this YAML:

```text
models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling_quality_iou.yaml
```

Run name:

```text
yolov8_varroa_p3p4_edgepool_quality_iou_hardneg_power2
```

## Train Command Shape

```python
model_yaml = ROOT / "models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling_quality_iou.yaml"
data = "/marimo/data/datasets/varroa_yolo/varroa.yaml"

model = YOLO(str(model_yaml))
model.load("yolov8n.pt", smart_transfer=True)

model.train(
    data=data,
    epochs=200,
    imgsz=640,
    batch=16,
    workers=4,
    device="cuda",
    patience=25,
    bbox_iou_loss="wiou",
    wiou_monotonous=False,
    quality_head=True,
    quality_loss="bce_balanced",
    quality_gain=0.5,
    quality_neg_gain=0.10,
    quality_pos_iou_thr=0.5,
    quality_hard_neg_iou_thr=0.3,
    quality_hard_neg_score_thr=0.05,
    quality_target_power=2.0,
    quality_neg_mode="hard",
    quality_detach_target=True,
    quality_score_mode="sqrt_cls_mul_q",
    quality_debug=True,
    quality_debug_batches=5,
    project=str(ROOT / "runs/detect/yolo_related/runs/train"),
    name="yolov8_varroa_p3p4_edgepool_quality_iou_hardneg_power2",
)
```

## Eval After Train

```python
run_name = "yolov8_varroa_p3p4_edgepool_quality_iou_hardneg_power2"
best_path = f"/marimo/yolo_code/runs/detect/yolo_related/runs/train/{run_name}/weights/best.pt"
model2 = YOLO(best_path)

for mode in ["cls_mul_q", "sqrt_cls_mul_q", "cls_mul_q2"]:
    model2.val(
        data=data,
        imgsz=640,
        batch=16,
        workers=0,
        device="cuda",
        quality_score_mode=mode,
        quality_debug_export=True,
        quality_debug_max_preds=30000,
        project=str(ROOT / "runs/detect/yolo_related/runs/val"),
        name=f"{run_name}_{mode}_debug",
        exist_ok=True,
    )
```

Use `workers=0` for val in marimo to avoid `BrokenPipeError`.

## Pass Criteria

Training debug:

```text
num_q_pos vs num_tal_pos
num_neg / num_hard_neg
max_iou_to_gt min/mean/max for q positives
```

Validation debug:

```text
q_by_iou_bin monotonic increasing
Spearman(q, IoU) improves
Spearman(final, IoU) improves over cls
mean_best_minus_top_final < mean_best_minus_top_cls
```

## Known Gotcha

If marimo says `quality_neg_mode` or `quality_target_power` is not a valid YOLO
argument, it is using stale Ultralytics import path. Check:

```python
import ultralytics
from ultralytics.cfg import DEFAULT_CFG_DICT

print(ultralytics.__file__)
print(DEFAULT_CFG_DICT.get("quality_neg_mode"))
print(DEFAULT_CFG_DICT.get("quality_target_power"))
```

Notebook must import patched `models_related/ultralytics`, not stale package or
stale SSH checkout.
