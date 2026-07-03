# YOLOv8n Varroa Pooling Box Error Diagnostic

Run analyzed:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/train/yolov8n_varroa_compare_baseline_p2_api_boxgrad_edge_pooling/weights/best.pt
```

Dataset:

```text
/marimo/data/datasets/varroa_yolo/varroa.yaml
split: val
nc: 1
class: varroa
```

Validation was reproduced with Ultralytics from:

```text
/marimo/yolo_code/models_related/ultralytics
imgsz=640
conf=0.001
iou=0.7
nms_method=hard
```

## Metric Summary

The reproduced validation metrics are:

| Metric | Value |
|---|---:|
| Precision | 0.922 |
| Recall | 0.892 |
| mAP50 | 0.925 |
| mAP50-95 | 0.352 |

The key symptom is that `mAP50` is good, but `mAP50-95` is low. This means the
model usually finds the mite, but the predicted boxes are not tight enough for
strict IoU thresholds.

## AP Collapse By IoU Threshold

| IoU threshold | AP |
|---:|---:|
| 0.50 | 0.92482 |
| 0.55 | 0.83392 |
| 0.60 | 0.71707 |
| 0.65 | 0.53638 |
| 0.70 | 0.31545 |
| 0.75 | 0.13778 |
| 0.80 | 0.04457 |
| 0.85 | 0.00769 |
| 0.90 | 0.00041 |
| 0.95 | 0.00002 |

This is a localization-quality collapse. AP is still strong at IoU 0.50, but it
falls sharply once the metric requires tighter box alignment.

## Manual GT Matching

For each GT object, the best prediction was matched by IoU. There are 685 GT
objects in the val split.

| Match IoU | TP | FP | FN | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 677 | 3530 | 8 | 0.161 | 0.988 |
| 0.75 | 350 | 3857 | 335 | 0.083 | 0.511 |
| 0.90 | 14 | 4193 | 671 | 0.003 | 0.020 |

Interpretation:

- At IoU 0.50, almost every GT is found: only 8 false negatives.
- At IoU 0.75, about half of the GTs fail the tighter localization threshold.
- At IoU 0.90, almost all boxes fail.
- The high FP count is partly because predictions were collected at
  `conf=0.001` for AP/error analysis, but high-confidence bad-localization
  cases also exist.

Best-GT-IoU quantiles:

| Quantile | Best GT IoU |
|---:|---:|
| min | 0.000 |
| p10 | 0.625 |
| p25 | 0.690 |
| p50 | 0.753 |
| p75 | 0.809 |
| p90 | 0.854 |
| max | 0.967 |

The median best IoU is only about 0.753. That is exactly why `mAP50` looks fine
but `mAP50-95` is weak.

## DFL/Box Regression Error Shape

Model strides:

```text
[4, 8, 16, 32]
```

GT objects by best-IoU bucket:

| Best IoU bucket | Count |
|---|---:|
| < 0.50 | 8 |
| 0.50-0.75 | 327 |
| 0.75-0.90 | 336 |
| >= 0.90 | 14 |

The most important failure bucket is `0.50-0.75`: boxes are good enough to count
at AP50, but too loose or shifted for AP75+.

### Absolute Pixel Error, All GT

| Error | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| center x | 1.54 | 2.71 | 3.87 | 4.62 | 6.58 |
| center y | 1.39 | 2.43 | 3.83 | 5.13 | 7.41 |
| left edge | 1.83 | 3.37 | 5.20 | 6.58 | 9.96 |
| top edge | 1.98 | 3.62 | 5.40 | 7.56 | 10.31 |
| right edge | 2.00 | 3.69 | 5.73 | 6.93 | 10.97 |
| bottom edge | 1.87 | 3.30 | 4.91 | 6.32 | 10.38 |
| width | 2.84 | 5.07 | 7.62 | 9.29 | 15.53 |
| height | 2.82 | 4.96 | 7.27 | 8.94 | 13.71 |

### Absolute Pixel Error, IoU 0.50-0.75 Bucket

This bucket explains most of the AP50-to-AP75 collapse.

| Error | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| center x | 2.30 | 3.44 | 4.48 | 5.08 | 6.45 |
| center y | 1.90 | 3.32 | 4.88 | 5.59 | 7.36 |
| left edge | 2.55 | 4.33 | 5.95 | 7.57 | 10.38 |
| top edge | 3.04 | 4.47 | 7.16 | 8.54 | 10.63 |
| right edge | 3.02 | 4.94 | 6.57 | 7.48 | 11.14 |
| bottom edge | 2.42 | 4.02 | 6.02 | 7.72 | 10.06 |
| width | 4.00 | 6.28 | 9.07 | 10.71 | 15.75 |
| height | 3.81 | 6.29 | 8.82 | 10.69 | 14.07 |

Relative error in the same bucket:

| Error | p50 | p75 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| center x / GT width | 0.075 | 0.115 | 0.139 | 0.159 | 0.209 |
| center y / GT height | 0.067 | 0.107 | 0.148 | 0.182 | 0.220 |
| width / GT width | 0.129 | 0.210 | 0.296 | 0.347 | 0.544 |
| height / GT height | 0.119 | 0.209 | 0.287 | 0.344 | 0.490 |

For varroa boxes, which are often only tens of pixels wide/high, a 3-6 px edge
error is enough to destroy high-IoU AP. With P2 stride 4, those errors are often
around 0.75-1.5 stride units, and the p90 tail can exceed 2 stride units.

## Signed Bias

Signed errors for detected GTs with IoU >= 0.50:

| Error | Mean px | Median px |
|---|---:|---:|
| center x | +0.47 | +0.47 |
| center y | -0.27 | -0.18 |
| left edge | +0.40 | +0.37 |
| top edge | -0.52 | -0.50 |
| right edge | +0.55 | +0.61 |
| bottom edge | -0.03 | -0.02 |
| width | +0.16 | +0.31 |
| height | +0.50 | +0.60 |

There is no single catastrophic one-sided bias. The average box is slightly
shifted right and up, and slightly taller, but the main issue is regression
noise across all edges.

## Which Errors Hurt IoU Most

Correlation with GT best IoU:

| Feature | Corr with IoU |
|---|---:|
| abs top edge error | -0.573 |
| abs right edge error | -0.560 |
| abs center x error | -0.551 |
| abs center y error | -0.534 |
| abs bottom edge error | -0.533 |
| abs width error | -0.490 |
| abs left edge error | -0.474 |
| abs height error | -0.451 |
| confidence | +0.107 |
| GT area fraction | +0.153 |

The strongest IoU damage comes from top/right edge error and center error, but
all four edges contribute. Confidence has only weak correlation with IoU, so
better scoring alone will not fix the high-IoU AP collapse.

## High-Confidence Bad Boxes

There are 599 predictions with:

```text
confidence >= 0.25
best IoU < 0.75
```

Their median best IoU is about 0.639.

This means the model is often confident while the box is still too loose,
shifted, or wrong-sized. The issue is not only low-confidence noise from using
`conf=0.001`.

## Training Loss Context

Best epoch by mAP50-95:

| Field | Value |
|---|---:|
| epoch | 50 |
| mAP50 | 0.92838 |
| mAP50-95 | 0.35203 |
| val/box_loss | 1.79089 |
| val/dfl_loss | 1.74278 |
| val/cls_loss | 1.02995 |

Last logged epoch:

| Field | Value |
|---|---:|
| epoch | 90 |
| mAP50 | 0.91413 |
| mAP50-95 | 0.33139 |
| val/box_loss | 1.90182 |
| val/dfl_loss | 1.93549 |
| val/cls_loss | 0.96222 |

The final epoch has worse `val/box_loss` and `val/dfl_loss`, while
classification loss improves. That matches the diagnosis: localization quality,
not class recognition, is the limiting factor.

## DFL Bin Probe

The Detect head uses:

```text
reg_max = 16
bins = 0..15 per side
strides = [4, 8, 16, 32]
```

An exact DFL-bin probe was then run over the full val split. The custom
Ultralytics NMS supports `return_idxs=True`, so each post-NMS detection was
traced back to its original pre-NMS anchor index. This fixed an earlier
nearest-cell approximation and gives exact level/stride attribution for the
final selected boxes.

Important implementation note: the direct raw-forward probe must use
`Results.orig_img` from `model.predict()`. Feeding a PIL/RGB image directly into
`predictor.preprocess()` changes channel order and makes scores nearly zero,
which gives invalid attribution.

Exact post-NMS attribution:

| Level | Stride | Count |
|---:|---:|---:|
| 0 | 4 | 12 |
| 1 | 8 | 202 |
| 2 | 16 | 471 |
| 3 | 32 | 0 |

All post-NMS detections at `conf=0.001`:

| Level | Stride | Detection count |
|---:|---:|---:|
| 0 | 4 | 675 |
| 1 | 8 | 1483 |
| 2 | 16 | 2046 |
| 3 | 32 | 0 |

Matched GT level by IoU bucket:

| Best-IoU bucket | stride 4 | stride 8 | stride 16 |
|---|---:|---:|---:|
| < 0.50 | 0 | 4 | 4 |
| 0.50-0.75 | 10 | 100 | 212 |
| 0.75-0.90 | 2 | 96 | 243 |
| >= 0.90 | 0 | 2 | 12 |

Final selected boxes are heavily biased toward stride 16. This is not just a
debug artifact: exact NMS anchor indices show that most matched GTs are coming
from stride 16, even though the model has a P2/stride-4 head.

### DFL Bin Error Summary

Absolute error is measured in DFL bin units, where one bin corresponds to one
grid unit at that level.

All GTs:

| Side | Mean abs bin error | Median abs bin error |
|---|---:|---:|
| left | 0.469 | 0.332 |
| top | 0.517 | 0.355 |
| right | 0.483 | 0.357 |
| bottom | 0.478 | 0.312 |

IoU `0.50-0.75` bucket:

| Side | Mean abs bin error | Median abs bin error |
|---|---:|---:|
| left | 0.618 | 0.480 |
| top | 0.641 | 0.532 |
| right | 0.635 | 0.516 |
| bottom | 0.570 | 0.451 |

IoU `0.75-0.90` bucket:

| Side | Mean abs bin error | Median abs bin error |
|---|---:|---:|
| left | 0.311 | 0.255 |
| top | 0.320 | 0.230 |
| right | 0.317 | 0.248 |
| bottom | 0.309 | 0.235 |

IoU `>=0.90` bucket:

| Side | Mean abs bin error | Median abs bin error |
|---|---:|---:|
| left | 0.106 | 0.093 |
| top | 0.092 | 0.076 |
| right | 0.117 | 0.071 |
| bottom | 0.128 | 0.110 |

Interpretation:

- Bad boxes are not caused by DFL saturating at bin 0 or bin 15.
- Edge-bin probability is low, usually below 1-2%, so the range
  `reg_max=16` is not the immediate bottleneck.
- The main DFL issue is expected-bin offset error around 0.45-0.65 bins in the
  `0.50-0.75` bucket.
- At stride 16, 0.45-0.65 bins is about 7-10 px, which is large for small varroa
  boxes.
- High-IoU boxes have much lower bin error, around 0.07-0.11 median bins. The
  bin distribution can be accurate enough; the issue is that many selected
  boxes land at the wrong fractional side distance.

### DFL Distribution Sharpness

Median peak probability of the selected DFL side distributions is around
0.52-0.55:

| Bucket | left | top | right | bottom |
|---|---:|---:|---:|---:|
| all | 0.526 | 0.531 | 0.522 | 0.546 |
| IoU 0.50-0.75 | 0.518 | 0.532 | 0.521 | 0.546 |
| IoU 0.75-0.90 | 0.536 | 0.532 | 0.525 | 0.547 |

Entropy is around 1.0-1.1 nats, so distributions are moderately peaked, not
uniformly flat. Bad and good boxes have similar peak sharpness. The difference
is the expected bin location, not the number of bins or a completely flat DFL
distribution.

Correlation with matched IoU:

| DFL stat | Corr with IoU |
|---|---:|
| abs bin error top | -0.610 |
| abs bin error bottom | -0.561 |
| abs bin error right | -0.550 |
| abs bin error left | -0.466 |
| entropy right | -0.114 |
| entropy left | -0.102 |
| stride | +0.102 |

This confirms the box AP collapse is driven by side-distance error. Peakiness
and entropy matter much less than whether the expected bin is at the correct
fractional distance.

### P2/P3/P4 Ranking Check

Pre-NMS boxes were also compared per GT, before NMS suppression, to see whether
the best localizing level differs from the highest-scoring level.

Best-IoU level per GT:

| Level | Stride | GT count |
|---:|---:|---:|
| 0 | 4 | 15 |
| 1 | 8 | 360 |
| 2 | 16 | 310 |

Highest-score level per GT:

| Level | Stride | GT count |
|---:|---:|---:|
| 1 | 8 | 31 |
| 2 | 16 | 654 |

Per-level best-IoU quantiles:

| Level | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|
| stride 4 | 0.422 | 0.522 | 0.613 | 0.688 |
| stride 8 | 0.761 | 0.830 | 0.883 | 0.903 |
| stride 16 | 0.744 | 0.821 | 0.876 | 0.905 |
| stride 32 | 0.067 | 0.088 | 0.109 | 0.125 |

Median score at the best-IoU box for each level:

| Level | Median score |
|---|---:|
| stride 4 | 0.000011 |
| stride 8 | 0.130 |
| stride 16 | 0.432 |
| stride 32 | 0.000001 |

Median IoU at the top-scoring anchor of each level:

| Level | Median IoU at top score |
|---|---:|
| stride 4 | 0.287 |
| stride 8 | 0.657 |
| stride 16 | 0.667 |
| stride 32 | 0.027 |

Critical finding: in 253 GT cases, stride 4/8 had a better best-IoU box than
stride 16 by at least 0.05, but stride 16 still had the highest score.

For those 253 cases:

| Field | Mean | Median |
|---|---:|---:|
| best IoU stride 8 | 0.814 | 0.821 |
| best IoU stride 16 | 0.655 | 0.671 |
| top score stride 8 | 0.408 | 0.408 |
| top score stride 16 | 0.529 | 0.535 |
| IoU at top-score stride 16 | 0.543 | 0.626 |

This is the clearest evidence so far: **there is a score/localization mismatch
between levels**. P3/stride8 often has a tighter candidate, but P4/stride16 gets
the higher confidence and survives NMS. That directly hurts AP75+ while keeping
AP50 relatively high.

### Score Bias Ablations

Inference-only score scaling was tested to check whether a simple level penalty
fixes the problem:

| Policy | Pred | Recall@0.50 | Recall@0.75 | Recall@0.90 | stride4 det | stride8 det | stride16 det |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 4204 | 0.988 | 0.518 | 0.020 | 675 | 1483 | 2046 |
| stride16 x 0.85 | 4166 | 0.987 | 0.518 | 0.020 | 676 | 1620 | 1870 |
| stride16 x 0.70 | 4134 | 0.990 | 0.512 | 0.025 | 676 | 1893 | 1565 |
| stride16 x 0.50 | 4043 | 0.988 | 0.491 | 0.026 | 676 | 2037 | 1330 |
| stride8 x 1.15 | 4282 | 0.987 | 0.520 | 0.020 | 670 | 1658 | 1954 |
| stride8 x 1.25, stride16 x 0.75 | 4293 | 0.990 | 0.498 | 0.026 | 671 | 2117 | 1505 |

Simple global score scaling does not improve Recall@0.75. It changes which
level survives, but it does not reliably pick the tighter candidate. This means
the fix should not be a hard-coded constant level penalty; it needs a per-anchor
localization-quality score.

Oracle pre-NMS upper bound:

| Candidate source | Recall@0.50 | Recall@0.75 | Recall@0.90 | Median best IoU |
|---|---:|---:|---:|---:|
| stride4 only | 0.314 | 0.025 | 0.000 | 0.422 |
| stride8 only | 0.977 | 0.530 | 0.061 | 0.761 |
| stride16 only | 0.971 | 0.480 | 0.060 | 0.744 |
| stride4/8 oracle | 0.978 | 0.537 | 0.061 | 0.767 |
| stride8/16 oracle | 0.999 | 0.753 | 0.111 | 0.811 |
| stride4/8/16 oracle | 1.000 | 0.761 | 0.111 | 0.812 |

This is decisive: the raw candidates contain much better AP75-quality boxes.
The problem is selecting/ranking them, not the absence of a good box candidate.

Additional inversion counts:

- 357 / 685 GTs: stride8 has better IoU than stride16, but stride16 has higher
  top score.
- 180 / 685 GTs: stride8 has an AP75-passing candidate while stride16 does not,
  but stride16 still has higher top score.

For those 180 AP75-critical inversions:

| Field | Mean | Median |
|---|---:|---:|
| best IoU stride8 | 0.826 | 0.822 |
| best IoU stride16 | 0.655 | 0.676 |
| top score stride8 | 0.402 | 0.403 |
| top score stride16 | 0.523 | 0.526 |

This proves the failure is not just "P4 is coarse." It is specifically that
classification confidence is not calibrated to localization quality. P4 can
produce a lower-IoU box with a higher score than a tighter P3 box.

### What `loc_quality` Means In This Repo

`loc_quality` is an experimental auxiliary training loss in this codebase. It
adds a train-only 1x1 localization-quality map head (`Detect.loc_cv`) and
supervises it with a smooth Gaussian target over GT boxes on the first N detect
levels:

```text
loc_quality: 0.0
loc_quality_levels: 2
loc_quality_sigma: 0.45
loc_quality_loss: mse
```

Important caveat: the current implementation says this objective has **no
inference path**. It trains localization-quality maps as an auxiliary signal,
but detection ranking still uses class score unless inference is modified.

So there are two possible uses:

- Auxiliary-only: set `loc_quality > 0` to encourage P2/P3 features to learn
  better center/localization structure during training.
- Ranking-aware: modify inference/training so detection score becomes something
  like `cls_score * predicted_quality` or `sqrt(cls_score * predicted_quality)`.

The ablations above suggest the ranking-aware version is the more direct fix.
Auxiliary-only may help features, but it will not by itself guarantee that NMS
chooses the tighter P3 box over the overconfident P4 box.

### Follow-up: P2/P3-only Detect + `loc_quality` Ranking

A follow-up experiment tested a P2/P3-only Detect model with localization-quality
ranking enabled.

Run:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/train/yolov8n_varroa_p2p3_only_detect_dfl25_lqrank/weights/best.pt
```

Architecture/config:

```text
/marimo/yolo_code/models_related/models_config/yolov8/yolov8n_varroa_p2p3_only_detect_dfl25_lqrank.yaml
Detect strides: [4, 8]
```

Training settings:

```text
bbox_iou_loss=wiou
wiou_monotonous=False
box=10.0
dfl=2.5
cls=0.25
loc_quality=0.05
loc_quality_levels=2
loc_quality_sigma=0.45
loc_quality_loss=mse
loc_quality_rank=True
loc_quality_rank_weight=0.5
loc_quality_rank_levels=2
```

Inference ranking used:

```text
score = cls_score * quality_score ** 0.5
```

Test split results:

| NMS method | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| hard | 0.893768 | 0.848717 | 0.883971 | 0.332500 |
| soft-linear, min_score=0.001 | 0.893768 | 0.848717 | 0.885345 | 0.337002 |

Output folders:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/test_lqrank_nms/hard
/marimo/yolo_code/runs/detect/yolo_related/runs/test_lqrank_nms/soft_linear_min001
```

Interpretation:

- The current `loc_quality` head did not improve the result; it is a negative
  ablation.
- Soft-NMS linear gave only a small gain over hard NMS on the same checkpoint
  (`mAP50-95 +0.0045`), not a meaningful fix.
- The likely reason is that the current `loc_quality` target is a Gaussian
  center map, not a direct predicted-box IoU or assignment-quality target. It
  can learn "near object center" without learning "this anchor's decoded box is
  the tightest one."
- Future localization-quality work should use an IoU/alignment target for the
  quality score, or a Varifocal/QFL-style classification target, rather than
  the current center-map target.

### Follow-up: P2/P3-only Detect Error Shape Without `loc_quality`

The original P2/P3-only Detect run was probed without `loc_quality`,
`loc_quality_rank`, or `cls_iou_target`.

Run:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/train/yolov8n_varroa_p2p3_only_detect_dfl25/weights/best.pt
```

Config:

```text
/marimo/yolo_code/models_related/models_config/yolov8/yolov8n_varroa_p2p3_only_detect_dfl25.yaml
Detect strides: [4, 8]
```

Probe output:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/diagnostics/p2p3_only_detect_dfl25_error_probe_val.json
```

Post-NMS GT best-IoU buckets on val:

| Best IoU bucket | Count |
|---|---:|
| < 0.50 | 8 |
| 0.50-0.75 | 333 |
| 0.75-0.90 | 319 |
| >= 0.90 | 25 |

Best-IoU quantiles:

| Quantile | Best GT IoU |
|---:|---:|
| p10 | 0.6373 |
| p25 | 0.6866 |
| p50 | 0.7503 |
| p75 | 0.8051 |
| p90 | 0.8554 |
| p95 | 0.8892 |

Absolute post-NMS error in the AP50-but-not-AP75 bucket (`0.50-0.75`) was:

| Error | p50 | p75 | p90 |
|---|---:|---:|---:|
| center x | 5.38 | 8.15 | 10.93 |
| center y | 4.36 | 7.53 | 10.00 |
| left edge | 6.68 | 11.11 | 15.36 |
| top edge | 6.03 | 10.70 | 14.76 |
| right edge | 7.01 | 11.04 | 14.76 |
| bottom edge | 5.73 | 9.17 | 13.79 |
| width | 9.46 | 14.93 | 21.16 |
| height | 8.40 | 14.63 | 20.90 |

These values are in model input / letterbox pixels. For the current val images
(`160x280` letterboxed to the model input), divide by about `2.29` to compare
roughly to original-image pixels. That still leaves median edge errors around
`2.5-3.1` original pixels in the `0.50-0.75` bucket.

Pre-NMS best candidate by level:

| Level | Stride | Best-level GT count | Highest-score GT count |
|---:|---:|---:|---:|
| 0 | 4 | 114 | 0 |
| 1 | 8 | 571 | 685 |

Per-level pre-NMS candidate quality:

| Level | Stride | Best IoU p50 | Best IoU p75 | Top-score IoU p50 | Top-score IoU p75 | Best score p50 | Top score p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 0.6217 | 0.7324 | 0.5234 | 0.6546 | 0.0059 | 0.0755 |
| 1 | 8 | 0.8118 | 0.8628 | 0.6701 | 0.7437 | 0.0457 | 0.5326 |

Within P3/stride-8:

- `349 / 685` GTs had a P3 candidate with IoU `>= 0.75`, but the top-scoring
  P3 anchor for that GT had IoU `< 0.75`.
- `525 / 685` GTs had P3 best-candidate IoU at least `0.05` higher than
  P3 top-score-anchor IoU.
- This means removing P4/P5 did not solve the ranking problem; the mismatch
  now happens inside P3 itself.

DFL expected-bin error for the best pre-NMS candidate:

| Level | Stride | Side | abs p50 | abs p75 | abs p90 | signed median | signed mean |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 4 | left | 2.378 | 4.047 | 5.481 | -2.378 | -2.615 |
| 0 | 4 | top | 1.761 | 3.545 | 5.069 | -1.761 | -2.153 |
| 0 | 4 | right | 1.210 | 2.843 | 4.778 | -1.089 | -1.655 |
| 0 | 4 | bottom | 1.291 | 2.748 | 4.486 | -1.151 | -1.595 |
| 1 | 8 | left | 0.403 | 0.840 | 1.408 | +0.043 | +0.082 |
| 1 | 8 | top | 0.324 | 0.694 | 1.069 | +0.080 | +0.120 |
| 1 | 8 | right | 0.381 | 0.719 | 1.208 | +0.035 | +0.030 |
| 1 | 8 | bottom | 0.375 | 0.758 | 1.248 | +0.078 | +0.167 |

Interpretation:

- P2/stride-4 is not useful here despite higher spatial resolution. Its best
  boxes have low IoU and near-zero best-candidate scores. The DFL side-distance
  errors are very large in bin units and have a consistent negative signed
  bias, meaning the selected P2 boxes are badly mis-sized/mis-positioned.
- P3/stride-8 has genuinely good raw candidates. Median best IoU is about
  `0.812`, and the DFL expected-bin error is moderate (`~0.32-0.40` median
  bins).
- The failure after removing P4/P5 is mostly **ranking within P3**, not lack of
  a good P3 box. The top-scoring P3 anchor has median IoU only `0.670`, while
  the best P3 candidate has median IoU `0.812`.
- This supports an IoU/quality-aware classification target more than another
  global level penalty or P4/P5 removal.

### Follow-up: `cls_iou_target_set`

This experiment tested a Quality-Focal/Varifocal-style classification target
without adding a new inference head.

Run:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/train/yolov8n_varroa_p2p3_only_detect_dfl25_cls_iou_target_set/weights/best.pt
```

Probe output:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/diagnostics/cls_iou_target_set_p3_score_probe.json
```

What `cls_iou_target_set` changes:

- Default YOLOv8 classification target comes from the TaskAlignedAssigner
  target score. In this repo that target is already a soft alignment score,
  not just hard `1`.
- The first attempted version multiplied that target by IoU
  (`target_scores *= IoU`). That was too aggressive because it double-softened
  an already-soft target. That early run was stopped at epoch 6.
- The `set` version replaces the positive class target with the detached IoU
  between the currently decoded assigned box and the assigned GT:

```text
positive cls target = IoU(pred_box.detach(), gt_box)
```

- Negative anchors stay `0`.
- Box loss and DFL loss are unchanged.
- This does not change decoded box geometry at inference. It only tries to make
  class confidence mean "object exists and this anchor's box is tight."

Test split results:

| Inference | mAP50 | mAP50-95 |
|---|---:|---:|
| hard NMS, conf=0.001, iou=0.7 | 0.89304 | 0.35220 |
| soft-linear NMS, conf=0.001, iou=0.5, min_score=0.0001 | 0.91070 | 0.35750 |

P3/stride-8 score probe on val:

| P3 statistic | Best candidate | Top-score anchor |
|---|---:|---:|
| IoU p50 | 0.8046 | 0.6653 |
| IoU p75 | 0.8608 | 0.7470 |
| score p50 | 0.2217 | 0.5796 |
| score p75 | 0.4096 | 0.6269 |

Compared with the P2/P3-only baseline:

| P3 statistic | Baseline | `cls_iou_target_set` |
|---|---:|---:|
| best-candidate score p50 | 0.0457 | 0.2217 |
| top-score anchor score p50 | 0.5326 | 0.5796 |
| best-candidate IoU p50 | 0.8118 | 0.8046 |
| top-score anchor IoU p50 | 0.6701 | 0.6653 |
| P3 candidate >=0.75 but top-score <0.75 | 349 / 685 | 327 / 685 |
| P3 best IoU exceeds top-score IoU by >=0.05 | 525 / 685 | 508 / 685 |

Interpretation:

- The IoU-aware classification target did raise the score of the better P3
  candidates substantially (`0.0457 -> 0.2217` median).
- But it did not fix which P3 anchor becomes the top-scoring one. The top-score
  P3 IoU stayed roughly the same (`0.6701 -> 0.6653` median).
- AP75-critical mismatch improved only slightly (`349 -> 327` cases), so this
  is not a strong fix.
- The best observed test result was `mAP50-95 = 0.35750` with soft-linear NMS,
  only a small gain over the P2/P3-only baseline.

### Follow-up: VFL target correctness vs learned ranking

Latest VFL diagnostics separate target construction from learned ranking:

- BCE baseline scoring is not localization-aware. The measured relationship
  between `sigmoid(cls)` and assigned IoU is weakly negative, so the default
  class logit does not rank tighter anchors above looser anchors.
- VFL target construction is correct. Positive target bins exactly match the
  assigned IoU bins, which rules out a target-binning or assignment-label bug.
- VFL predicted class scores still remain weakly negative against assigned IoU.
  That means the failure is not that VFL receives the wrong target; it is that
  the classification branch representation/head is not learning an IoU-ranking
  signal strongly enough from the loss-only change.

Diagnostic added:

```text
misc/vfl_feature_iou_linear_probe.py
```

This script loads a checkpoint, extracts frozen per-anchor features immediately
before the final Detect box conv (`cv2[i][:-1]`) and class conv (`cv3[i][:-1]`),
reuses the existing TAL positives and assigned-IoU labels, and fits linear
probes from each feature type to IoU. If the regression-side feature predicts
IoU materially better than the classification-side feature, that supports the
current interpretation: VFL target construction is correct, but the YOLOv8
classification branch is not sufficiently localization-aware for IoU ranking.

Probe sanity checks and results:

- Hook/flatten alignment was verified by reconstructing Detect outputs from the
  hooked branch features:
  - `cv2[i][:-1] -> cv2[i][-1]` exactly reproduced `preds["boxes"]`.
  - `cv3[i][:-1] -> cv3[i][-1]` exactly reproduced `preds["scores"]`.
  - Max difference was `0.0` for all levels, so the weak probe result was not
    caused by a wrong layer or anchor flattening order.
- The actual YOLOv8n varroa cls feature dimensions are small:
  - cls pre-logit feature: `32` channels at every level.
  - reg pre-logit feature: `64` channels at every level.
- A larger probe with `12000` train positives and all `6850` val positives
  still showed weak linear IoU signal:

| Model | Branch | Train R2 | Train Pearson | Val R2 | Val Pearson | Val Spearman |
|---|---|---:|---:|---:|---:|---:|
| BASE diff | cls | 0.030 | 0.173 | -0.220 | 0.100 | 0.182 |
| BASE diff | reg | 0.035 | 0.187 | -0.163 | 0.181 | 0.187 |
| VFL stage2-2 | cls | 0.040 | 0.199 | -0.050 | 0.173 | 0.180 |
| VFL stage2-2 | reg | 0.054 | 0.232 | -0.082 | 0.149 | 0.203 |

Interpretation:

- The negative validation R2 is partly due to train/val target mean shift, but
  train R2 is also very low. A linear readout from pre-final cls/reg features
  does not strongly predict assigned IoU.
- Since decoded IoU appears only after `bbox logits -> DFL softmax/expected
  distance -> decoded box -> compare to assigned GT`, the failure of a linear
  `feature -> IoU` probe does not prove the reg branch lacks localization
  information. It does show that IoU is not linearly exposed in the pre-final
  feature.

### Follow-up: anchor-level Grad-CAM

An anchor-level Grad-CAM diagnostic was added in the running marimo notebook.
It selected TAL-positive anchors in three groups:

- high IoU + low cls
- high IoU + high cls
- low IoU + high cls

For each selected anchor, the notebook renders:

- original image with assigned GT and decoded anchor box
- Grad-CAM for the scalar class logit at that exact anchor
- Grad-CAM for the anchor-local regression objective, using CIoU between that
  decoded anchor box and assigned GT

Outputs were saved in marimo:

```text
/marimo/yolo_code/runs/diagnostics/anchor_gradcam/base_diff/
/marimo/yolo_code/runs/diagnostics/anchor_gradcam/vfl_stage2_2/
/marimo/yolo_code/runs/diagnostics/anchor_gradcam/metadata_all.json
/marimo/yolo_code/runs/diagnostics/anchor_gradcam/metadata_all.csv
```

The run generated `24` figures total: `12` for BASE diff and `12` for VFL
stage2-2, balanced as `4` examples per group per model. The first display cell
used base64 HTML and exceeded marimo's output limit, so it was replaced with a
matplotlib contact-sheet cell that reads the generated PNGs from disk.

Interpretation rule:

- If cls Grad-CAM is diffuse/off-object while reg Grad-CAM focuses on
  object/boundary, that supports the hypothesis that the classification branch
  is not localization-aware enough for VFL to learn IoU ranking.
- If both CAMs are similar or noisy, do not overclaim; keep the linear probe
  and score-vs-IoU bin correlation as the stronger evidence.

### Follow-up: geometry-aware classification head

Because VFL target construction is correct but loss-only VFL did not make cls
scores rank IoU well, a Muc 2 geometry-aware cls head was implemented as an
opt-in `Detect` experiment. This is not a separate quality head and does not
change NMS or output tensor shape.

New config flags:

```yaml
cls_geometry_fuse: false
cls_geometry_mode: add   # add or concat
cls_geometry_detach: true
cls_deform_geometry: false
```

Implementation:

- `cv2[i]` still predicts DFL bbox logits.
- The bbox logits are reshaped to `(B, 4, reg_max, H, W)`.
- Softmax over `reg_max` gives the DFL distribution for each side.
- Expected distances `[l, t, r, b]` are computed and normalized by
  `reg_max - 1`, so with `reg_max=16` the cue is approximately in `[0, 1]`.
- The normalized 4-channel geometry map is embedded with a per-level
  `Conv2d(4 -> cls_channels, 1x1)`.
- `cls_geometry_mode=add` uses:

```text
cls_feat + geometry_embed -> final cls conv
```

- `cls_geometry_mode=concat` uses:

```text
concat(cls_feat, geometry_embed) -> Conv(2C -> C, 1x1) -> final cls conv
```

The per-level geometry embed is intentional: even though distances are in DFL
bin-space, P2/P3/P4/P5 have different object-size and assignment
distributions. The geometry cue is detached by default so classification
gradients do not flow back into bbox logits through this auxiliary path during
the first experiment.

The current geometry-aware cls head is Muc 2 only. It does **not** use
deformable convolution, learned offsets, star-shaped sampling points, or a
VFNet-style feature sampler. `cls_deform_geometry` is reserved and raises
`NotImplementedError` if enabled.

Experiment YAMLs:

```text
models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_add.yaml
models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_concat.yaml
```

Both YAMLs were copied from the VFL pooling-edge config, then the P2 API
box-gradient layer was removed so the geometry-fusion ablation is not mixed
with the earlier API box-gradient block. The final Detect inputs are:

```yaml
[[28, 29, 24, 27], 1, Detect, [nc]]
```

Current cls head shape in the concat YAML:

```text
P2/stride 4:   Conv(32 -> 32, 3x3)  -> Conv(32 -> 32, 3x3) -> Conv2d(32 -> 1, 1x1)
P3/stride 8:   Conv(64 -> 32, 3x3)  -> Conv(32 -> 32, 3x3) -> Conv2d(32 -> 1, 1x1)
P4/stride 16:  Conv(128 -> 32, 3x3) -> Conv(32 -> 32, 3x3) -> Conv2d(32 -> 1, 1x1)
P5/stride 32:  Conv(256 -> 32, 3x3) -> Conv(32 -> 32, 3x3) -> Conv2d(32 -> 1, 1x1)
```

For `cls_geometry_mode=concat`, the actual cls path is:

```text
cls feature -> cv3[i][0] -> cv3[i][1]
bbox logits -> expected DFL [l,t,r,b] / (reg_max - 1) -> Conv2d(4 -> 32, 1x1)
concat -> Conv(64 -> 32, 1x1) -> cv3[i][2] -> cls logit
```

Validation:

- Python compile passed for the modified Ultralytics files.
- Build/forward smoke passed for baseline, `clsgeom_add`, and
  `clsgeom_concat`.
- Output shape remained unchanged:
  - training boxes: `(B, 64, A)`
  - training scores: `(B, 1, A)`
  - inference output: `(B, 5, A)` for `nc=1`
- Synthetic loss smoke passed for `add` and `concat`, both with GT and empty-GT
  batches, with finite losses.
- A real marimo smoke train completed for the concat YAML:

```text
Run name: yolov8n_varroa_vfl_clsgeom_concat_smoke3
Weights: /marimo/yolo_code/runs/detect/yolo_related/runs/train/yolov8n_varroa_vfl_clsgeom_concat_smoke3/weights/best.pt
Log: /marimo/yolo_code/runs/detect/yolo_related/runs/train_logs/clsgeom_concat_smoke3.log
Epochs: 3
Batch: 16
Device: cuda
```

Final smoke validation result:

| Metric | Value |
|---|---:|
| Precision | 0.638 |
| Recall | 0.753 |
| mAP50 | 0.705 |
| mAP50-95 | 0.226 |

This smoke result only verifies that the geometry-aware concat path trains
without crashing. It is too short to judge final accuracy.

### Follow-up: VFL and TAL assignment feedback

The current strongest hypothesis is that the VFL implementation is target-correct
but mismatched with YOLOv8's TaskAlignedAssigner.

Current YOLOv8 assignment path:

```text
assignment score = sigmoid(cls)^0.5 * IoU^6
```

This comes from `v8DetectionLoss`, which passes
`pred_scores.detach().sigmoid()` into `TaskAlignedAssigner`, and the assigner
computes:

```text
align_metric = bbox_scores^alpha * overlaps^beta
```

With VFL enabled, positive classification targets are then replaced with the
assigned box IoU. This creates a plausible optimization feedback loop:

```text
cls score low -> TAL metric low -> different/weaker positives -> changed cls
gradient -> cls score remains low
```

The `detach()` prevents direct assignment gradients through cls, but it does not
remove the cross-iteration feedback because the next assignment still uses the
current classification score.

A minimal ablation knob was added:

```yaml
tal_alpha: 0.5  # default, preserves current behavior
```

Setting `tal_alpha: 0.0` removes classification from the TAL ranking while
keeping TAL's top-k, in-GT, and IoU-weighted target logic:

```text
assignment score = IoU^beta
```

Initial ablation YAMLs:

```text
models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b6.yaml
models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b4.yaml
models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b2.yaml
```

These use the existing non-geometry VFL architecture. No quality head, quality
loss, inference multiplier, NMS change, or ATSS assigner has been added.

Interpretation rule:

- If `tal_alpha=0` improves positive cls logits, score-vs-IoU monotonicity, or
  AP75/AP50-95 versus VFL with default TAL, this supports the assignment
  feedback hypothesis.
- If `tal_alpha=0` still fails, test lower `tal_beta` values and then consider
  a true ATSS-style assigner using the `VarifocalNet` reference.

## Conclusion

The model is not mainly failing to detect varroa. It is detecting them at AP50,
but the boxes are not precise enough for AP75-AP95.

Concrete box failure pattern:

- Center error is usually small in pixels, but still large relative to small
  varroa boxes.
- Width/height errors around 4 px median in the `0.50-0.75` bucket are too high
  for small objects.
- Edge errors around 2.5-3.0 px median, with p90 around 6-7 px, explain the
  high-IoU collapse.
- Top and right edge errors correlate most strongly with IoU drop, but the
  problem is distributed across all edges.
- Confidence does not reliably indicate tight localization.

Recommended next checks:

- Visualize the 0.50-0.75 bucket, especially high-confidence boxes with
  `best_iou < 0.75`.
- Preserve pre-NMS anchor indices for matched predictions and rerun the DFL-bin
  probe exactly, not via nearest-cell heuristic.
- Check whether small objects are being scored from stride 16 too often. If yes,
  inspect P2/P3 assignment and classification logits for the same GTs.
- Inspect whether predicted boxes are systematically too loose around mite
  boundaries or shifted by bee texture/background.
- Try localization-focused changes before classification changes:
  higher DFL emphasis, different box loss settings, assignment tuning,
  stricter label quality checks, and NMS/duplicate analysis.
- Compare against a baseline model using the same diagnostic to see whether
  this architecture specifically worsens edge precision.

## 2026-07-03 Pairwise Ranking Loss Follow-Up

### Motivation

Pairwise ranking loss was tested because the current failure mode includes weak
score-vs-localization alignment: many detections are confident even when their
boxes are not tight enough for high-IoU AP.

The intended pairwise rule was:

```text
For positive candidates assigned to the same GT:
better IoU box should have higher class score than worse IoU box.
```

The suspected risk was that this objective can damage class-score calibration:
it ranks positives against positives, but it does not explicitly teach negatives
to be low. If the ranking loss is too strong or noisy, it can raise scores for
many positive candidates around each GT and reduce precision.

### Current Pairwise Ranking Implementation

The ranking loss lives in:

```text
models_related/ultralytics/ultralytics/utils/loss.py
```

Core behavior:

```python
rank_pred_bboxes = pred_bboxes.detach()[fg_mask]
rank_target_bboxes = target_bboxes.detach()[fg_mask]
rank_iou = bbox_iou(rank_pred_bboxes, rank_target_bboxes, xywh=False, CIoU=False).squeeze(-1).clamp(0)

class_indices = target_scores[fg_mask].argmax(-1)
assigned_logits = pred_scores[batch_indices, anchor_indices, class_indices]
group_ids = batch_indices * (target_gt_idx.max() + 1).clamp_min(1) + target_gt_idx[fg_mask]

for group_id in group_ids.unique():
    group_mask = group_ids == group_id
    group_iou = rank_iou[group_mask]
    group_logits = assigned_logits[group_mask]
    topk = min(self.rank_topk, int(group_iou.numel()))
    order = group_iou.argsort(descending=True)[:topk]
    group_iou = group_iou[order]
    group_logits = group_logits[order]

    iou_gap = group_iou[:, None] - group_iou[None, :]
    pair_mask = iou_gap > self.rank_iou_margin
    score_gap = group_logits[:, None] - group_logits[None, :]
    weights = iou_gap[pair_mask].detach()
    rank_losses.append(F.softplus(-score_gap[pair_mask] / tau) * weights)
```

For the follow-up tests, a runtime monkey patch added:

```python
pair_mask = (iou_gap > self.rank_iou_margin) & (group_iou[:, None] > rank_iou_hi_min)
```

so only pairs whose better box has IoU above the threshold are used.

### IoU Scale Debug

The debug print was added around the rank IoU calculation:

```python
print(
    "[rank debug]",
    "fg", int(fg_mask.sum()),
    "pred", float(rank_pred_bboxes.min()), float(rank_pred_bboxes.max()),
    "target", float(rank_target_bboxes.min()), float(rank_target_bboxes.max()),
    "iou_mean", float(rank_iou.mean()) if rank_iou.numel() else -1,
    "iou_max", float(rank_iou.max()) if rank_iou.numel() else -1,
    "iou_hi_min", float(rank_iou_hi_min),
)
```

Observed logs:

```text
[rank debug] fg 180 pred 13.9140625 120.5859375 target 17.762069702148438 116.6993637084961 iou_mean 0.6278674006462097 iou_max 0.9187124371528625 iou_hi_min 0.6
[rank debug] fg 370 pred 10.8671875 112.03125 target 12.5625 110.85714721679688 iou_mean 0.6693977117538452 iou_max 0.9366804957389832 iou_hi_min 0.6
[rank debug] fg 180 pred 13.97265625 119.1953125 target 17.762069702148438 116.6993637084961 iou_mean 0.6589277386665344 iou_max 0.8697048425674438 iou_hi_min 0.6
[rank debug] fg 370 pred 11.94921875 113.05859375 target 12.5625 110.85714721679688 iou_mean 0.7128806710243225 iou_max 0.9152914881706238 iou_hi_min 0.6
```

Interpretation:

- The IoU scale is not broken.
- Predicted and target boxes are in the same coordinate scale.
- `iou_max` is high enough; the ranking failure is not caused by all-zero or
  badly scaled IoU.

### Ablation 1: Baseline P2/P3 Edge Pooling Checkpoint

Seed checkpoint:

```text
/marimo/runs/detect/rank_loss_staged/baseline_p2p3_edge_pooling_rank0_40ep/weights/best.pt
```

Original source run:

```text
/marimo/runs/detect/rank_loss_staged/baseline_p2p3_edge_pooling_rank0_40ep
best epoch: 36
best precision: 0.88858
best recall: 0.89647
best mAP50: 0.90603
last epoch: 40
last precision: 0.88133
last recall: 0.87822
last mAP50: 0.89932
```

All ablations were run for 10 epochs from the same checkpoint with:

```text
rank_iou_margin=0.1
rank_tau=0.25
rank_topk=10
```

Results:

| Run | rank_loss | rank_iou_hi_min | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0 | 0.0 | 0.80190 | 0.76232 | 0.82516 | 0.28457 |
| B | 0.025 | 0.6 | 0.63715 | 0.67931 | 0.68362 | 0.25003 |
| C | 0.05 | 0.6 | 0.45912 | 0.51387 | 0.48777 | 0.17840 |

Run directories:

```text
/marimo/runs/detect/rank_loss_ablation_hi06/A_rank0_from_map90_10ep
/marimo/runs/detect/rank_loss_ablation_hi06/B_rank0025_iouhi06_from_map90_10ep
/marimo/runs/detect/rank_loss_ablation_hi06/C_rank005_iouhi06_from_map90_10ep
```

Result:

- `rank_loss=0.025` already hurts precision and mAP50 heavily.
- `rank_loss=0.05` is substantially worse.
- Filtering pairs with `rank_iou_hi_min=0.6` does not prevent the damage.

### Ablation 2: `/marimo/best_api_pooledge_p2p3.pt`

Available P2/P3 edge-pooling related checkpoints found under `/marimo`:

| Checkpoint | Notes |
|---|---|
| `/marimo/runs/detect/rank_loss_staged/baseline_p2p3_edge_pooling_rank0_40ep/weights/best.pt` | Best source mAP50 0.90603 |
| `/marimo/runs/detect/rank_loss_staged/baseline_p2p3_edge_pooling_rank0_40ep/weights/last.pt` | Last mAP50 0.89932 |
| `/marimo/runs/detect/rank_loss_staged/rank005_p2p3_edge_pooling_from_baseline40_80ep/weights/best.pt` | Best source mAP50 0.80299 |
| `/marimo/runs/detect/rank_loss_staged/rank005_p2p3_edge_pooling_from_baseline40_80ep/weights/last.pt` | Last mAP50 0.68852 |
| `/marimo/runs/detect/rank_loss_real/rank005_p2p3_edge_pooling_scratch/weights/best.pt` | Best source mAP50 0.71145 |
| `/marimo/best_api_pooledge_p2p3.pt` | No adjacent results.csv |
| `/marimo/best_api_pooledge_p2p3_vlf.pt` | No adjacent results.csv |

The requested checkpoint was tested:

```text
/marimo/best_api_pooledge_p2p3.pt
```

Same 10-epoch settings:

```text
rank_iou_margin=0.1
rank_tau=0.25
rank_topk=10
```

Results:

| Run | rank_loss | rank_iou_hi_min | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0 | 0.0 | 0.76916 | 0.72555 | 0.80460 | 0.28666 |
| B | 0.025 | 0.6 | 0.53296 | 0.62305 | 0.59009 | 0.22629 |
| C | 0.05 | 0.6 | 0.46191 | 0.65547 | 0.55932 | 0.21858 |

Run directories:

```text
/marimo/runs/detect/rank_loss_ablation_pooledge_hi06/A_rank0_from_pooledge_10ep
/marimo/runs/detect/rank_loss_ablation_pooledge_hi06/B_rank0025_iouhi06_from_pooledge_10ep
/marimo/runs/detect/rank_loss_ablation_pooledge_hi06/C_rank005_iouhi06_from_pooledge_10ep
```

Result:

- The same pattern repeats.
- A rank-free continuation is best.
- `rank_loss=0.025` and `rank_loss=0.05` both damage precision and mAP.
- The damage remains even with `rank_iou_hi_min=0.6`.

### Why `rank_iou_hi_min=0.6` Still Fails

The filter only says that the higher-IoU box in a pair must be reasonably good.
It does not solve the calibration problem.

Main failure reasons:

1. The objective ranks positives against positives only.
   It does not explicitly suppress negatives or duplicate low-quality
   candidates. BCE/VFL still does that separately, but the rank term can distort
   the positive score distribution enough to hurt precision.

2. A valid pair can still be noisy.
   With `rank_iou_hi_min=0.6`, pairs like `0.62 > 0.50` are valid. This is true
   in ranking terms, but it may not help NMS or thresholded detection quality.

3. The rank term is not tiny in practice.
   Observed rank losses were roughly:

   ```text
   rank_loss=0.025 run: train/rank_loss about 0.126, val/rank_loss about 0.133
   rank_loss=0.05 run:  train/rank_loss about 0.187, val/rank_loss about 0.209
   ```

   Even after multiplying by the gain, this is enough to push class logits and
   disrupt calibration.

4. It optimizes local ordering, not global confidence.
   The metric needs a calibrated score that separates true detections,
   duplicates, loose boxes, and background. Pairwise positive-only ranking does
   not directly optimize that global ordering.

5. The continuation LR schedule is itself destabilizing.
   Even `rank_loss=0.0` drops below the original checkpoint's best mAP50 after
   10 continuation epochs. However, the rank-loss runs drop much more than A,
   so rank loss remains the primary additional harm.

Conclusion:

```text
Current pairwise ranking loss should stay disabled for this model family.
```

If it is tested again, use a much smaller gain and safer scheduling:

```text
rank_loss: 0.005 or 0.01
enable only after warmup / late fine-tune
low LR continuation
possibly require iou_hi > 0.7 and iou_lo < 0.5, not just iou_gap > margin
```

### Current Quality Head Status

There is a train-only localization quality map auxiliary head, but no inference
quality head yet.

Detect head code:

```text
models_related/ultralytics/ultralytics/nn/modules/head.py
```

Current auxiliary head:

```python
# EXPERIMENTAL: localization quality map heads. These parameters live
# on the model so the optimizer can update them during LQM training.
self.loc_quality_enabled = False
self.loc_cv = nn.ModuleList(nn.Conv2d(x, 1, 1) for x in ch)
```

Forward path:

```python
out = dict(boxes=boxes, scores=scores, feats=x)
# EXPERIMENTAL: LQM maps are train-time auxiliary outputs only.
if self.training and getattr(self, "loc_quality_enabled", False):
    out["loc_maps"] = [self.loc_cv[i](x[i]) for i in range(self.nl)]
return out
```

Loss code:

```text
models_related/ultralytics/ultralytics/utils/loss.py
```

Current objective:

```python
class LocalizationQualityLoss(nn.Module):
    """Gaussian localization quality map supervision over Detect feature maps."""
```

It creates a smooth center-high Gaussian target inside each GT box and applies
MSE or SmoothL1 to `sigmoid(loc_map)`.

Config knobs:

```yaml
loc_quality: 0.0
loc_quality_levels: 2
loc_quality_sigma: 0.45
loc_quality_loss: mse
```

Important limitation:

```text
loc_quality is train-only auxiliary supervision.
It is not used at inference and it is not multiplied into class confidence.
```

`cls_iou_target` and `vfl` use assigned IoU as a classification target, but they
do not add a separate quality prediction head.

### Updated Recommendation

Given the ranking-loss results, the next more plausible direction is not
positive-only pairwise ranking. Better candidates:

- Keep rank loss off.
- If using quality, prefer a true IoU/quality prediction branch whose output is
  supervised by assigned IoU and used at inference as:

  ```text
  final_score = cls_score * quality_score
  ```

- Alternatively, test VFL/quality-target classification with careful TAL
  assignment tuning, especially `tal_alpha=0.0`, to avoid classification-score
  feedback loops during assignment.
- Continue focusing on localization: DFL/box loss, assignment scale, P2/P3
  positive distribution, and high-IoU box refinement.
