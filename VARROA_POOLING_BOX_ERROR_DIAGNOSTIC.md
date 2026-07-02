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
