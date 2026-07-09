# YOLOv8n Varroa Pooling Box Error Diagnostic

Short version. Full raw tables and older follow-ups moved to
[VARROA_POOLING_BOX_ERROR_DIAGNOSTIC_APPENDIX.md](VARROA_POOLING_BOX_ERROR_DIAGNOSTIC_APPENDIX.md).

Quick terms:

- `GT` means ground-truth annotation: the human-labeled mite box from the
  dataset label file. We can compute IoU/error because validation compares each
  prediction against these known label boxes.
- `Prediction` means one model-produced box plus its confidence score.
- `IoU` means intersection-over-union between prediction and GT. `1.0` is a
  perfect overlap; `0.5` is loose but usually counted by AP50; `0.75+` means
  the box must be much tighter.
- `AP` means average precision at one IoU threshold. It rewards both finding
  the object and ranking GT-matching detections above false positives or loose
  matches.
- `mAP50-95` averages AP from IoU `0.50` to `0.95`, so it punishes loose boxes
  much more than `mAP50`.

## Current Diagnosis

Why this section exists: pin down the actual failure mode before changing
architecture. If the model were missing mites, the fix would be recall/data
coverage; here AP50/recall show mites are usually found, while AP75/AP90 show
the boxes are too loose.

Baseline finds mites, but boxes are not tight enough.

How to read this table: precision and recall say whether the model finds mites
without too many false positives/misses. `mAP50` is the loose-box score.
`mAP50-95` is the stricter COCO-style score that falls when boxes are not tight.

| Metric | Value |
|---|---:|
| Precision | 0.922 |
| Recall | 0.892 |
| mAP50 | 0.925 |
| mAP50-95 | 0.352 |

AP collapses as IoU threshold rises:

How to read this table: each row changes the minimum IoU required for a
prediction to count as correct. A big drop from `0.50` to `0.75` means the
model usually finds the mite, but the predicted box edges are not aligned
tightly enough.

| IoU | AP |
|---:|---:|
| 0.50 | 0.92482 |
| 0.75 | 0.13778 |
| 0.90 | 0.00041 |
| 0.95 | 0.00002 |

Meaning: object discovery is not the dominant failure: precision `0.922`,
recall `0.892`, and AP50 `0.925` are high, but AP75 `0.13778` and AP90
`0.00041` expose the localization/ranking bottleneck.

Why this observation matters: AP50 is high while AP75/AP90 collapse, so a
change that only improves objectness/class confidence is unlikely to move the
metric that is failing.

## Capacity Check

Why this section exists: test whether the problem is simply "model too small"
before spending more runs on heavier backbones/heads.

A very small P3-only/no-P5 model is already close to the best known result:

```text
models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling_p3only_nop5.yaml
```

How to read this table: compare accuracy against compute/model size. The small
P3-only/no-P5 model is `0.005` mAP50-95 behind the current best while using far
less compute, so generic capacity is unlikely to be the main missing piece.

| Model | Approx cost | mAP50 | mAP50-95 |
|---|---:|---:|---:|
| P3-only no-P5 small model | 0.688M params / 5.9 GFLOPs | 0.907 | 0.352 |
| Current best SOTA variant | ~20 GFLOPs | 0.917 | 0.357 |

Interpretation:

- The measured gap is `0.352 -> 0.357` mAP50-95, so the current evidence does
  not support "make the model larger" as the main fix.
- More semantic depth / heavier heads are unlikely to be the main lever now.
- The active bottleneck is more likely data resolution/cropping, label
  consistency, localization target, or ranking/postprocess calibration.
- This is not a proof that architecture cannot help, but architecture-only
  changes should now be treated as low-priority unless they directly target
  edge localization.

Why this observation matters: when a tiny model reaches nearly the same
mAP50-95 as a much heavier variant, the next experiments should attack box
edge precision, score calibration, or postprocess selection directly instead
of adding generic capacity.

## Main Evidence

Why this section exists: quantify the box-error shape, not just the final mAP.
This tells whether failures are large misses, small edge shifts, or confidence
ordering mistakes.

Validation has `685` GT objects. Best matched IoU quantiles:

How to read this table: there are `685` labeled mite boxes in validation. For
each labeled mite, find the prediction that overlaps it the most, then record
that IoU. Now sort those `685` IoU values from low to high. `p50 = 0.753` is
the middle value: about half of the labeled mites have no prediction better
than `0.753` IoU. Since AP75 requires IoU `>= 0.75`, the median labeled mite is
only `0.003` IoU above the AP75 cutoff even when using its best-overlap
prediction. `p25 = 0.690` means 25% of mites cannot reach AP75 even with their
best prediction. `p90 = 0.854` means only the top 10% get above `0.854` IoU.

| Quantile | Best GT IoU |
|---:|---:|
| p25 | 0.690 |
| p50 | 0.753 |
| p75 | 0.809 |
| p90 | 0.854 |

The AP50-but-not-AP75 bucket is large:

How to read this table: each GT mite is placed into the bucket of its best
available IoU. The `0.50-0.75` bucket means "passes AP50 because IoU is at
least `0.50`, but fails AP75 because IoU is below `0.75`." That is the main
loose-box failure zone.

| Best IoU bucket | Count |
|---|---:|
| < 0.50 | 8 |
| 0.50-0.75 | 327 |
| 0.75-0.90 | 336 |
| >= 0.90 | 14 |

In the `0.50-0.75` bucket, edge errors of only a few pixels are enough to break
high-IoU AP:

How to read this table: values are absolute pixel errors between the matched
prediction box and GT box. `center x/y` are errors in the box center position.
`left/top/right/bottom edge` are errors for each side of the box. `width/height`
are size errors. `p50/p75/p90` are percentiles over examples in this bucket, so
`left edge p75 = 4.33` means 75% of these loose boxes have left-edge error at
or below `4.33` pixels, and 25% are worse.

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

Confidence is not a reliable tight-box signal. There are `599` predictions with:

```text
confidence >= 0.25
best IoU < 0.75
```

Median best IoU for those is only about `0.639`.

Why these observations matter:

- The bucket table shows most GTs have a nearby prediction, but many are just
  below AP75. That points to tightness, not object discovery.
- The pixel-error table shows the target is only a few pixels wide. Small edge
  regressions are enough to destroy AP75/AP90 on mite boxes.
- The confidence observation shows score is not a reliable proxy for tightness,
  so NMS can keep a looser box even when a better candidate exists.

## Ranking Problem

Why this section exists: separate "the model cannot produce an AP75-quality box" from
"the model produced one but postprocess/ranking discarded it".

Raw candidates often contain tighter boxes than final NMS keeps.

Oracle pre-NMS upper bound:

How to read this table: "oracle" means we look at all raw candidate boxes before
NMS and ask whether at least one AP75-quality box exists for each GT. It is not
a deployment method; it is an upper bound showing what ranking/NMS could
recover if it picked the highest-IoU candidate. `Recall@0.75` means the
fraction of GT mites that have at least one candidate with IoU `>= 0.75`.

| Candidate source | Recall@0.50 | Recall@0.75 | Recall@0.90 | Median best IoU |
|---|---:|---:|---:|---:|
| stride8 only | 0.977 | 0.530 | 0.061 | 0.761 |
| stride16 only | 0.971 | 0.480 | 0.060 | 0.744 |
| stride8/16 oracle | 0.999 | 0.753 | 0.111 | 0.811 |
| stride4/8/16 oracle | 1.000 | 0.761 | 0.111 | 0.812 |

Key inversion counts:

- `357 / 685`: stride8 has better IoU than stride16, but stride16 has higher score.
- `180 / 685`: stride8 has an AP75-passing candidate while stride16 does not, but stride16 still has higher score.

How to read these counts: an inversion means the model assigned a higher score
to a lower-IoU box. NMS keeps high-score boxes first, so these inversions turn
available AP75-quality boxes into discarded boxes.

Removing P4/P5 did not solve it. In P2/P3-only, mismatch moves inside P3:

- `349 / 685` GTs had a P3 candidate with IoU `>= 0.75`, but top-scoring P3 anchor had IoU `< 0.75`.
- `525 / 685` GTs had P3 best IoU at least `0.05` higher than top-score-anchor IoU.

Why these observations matter:

- The oracle rows prove higher-IoU boxes often exist before final selection.
- The stride8/stride16 inversions show cross-level scores are miscalibrated.
- The P2/P3-only result shows removing high strides does not remove the
  score-vs-IoU mismatch; the same issue remains inside the surviving feature
  level.

## Failed Or Weak Fixes

Why this section exists: avoid repeating experiments that already failed to
attack the measured bottleneck.

These did not meet the required condition: make the final score rank tighter
decoded boxes above looser decoded boxes for the same GT.

- Global stride score scaling.
- P2/P3-only Detect.
- Old `loc_quality` Gaussian center-map target.
- `cls_iou_target_set` alone.
- VFL loss-only change with current YOLOv8 classification branch.
- Box/DFL-side detail changes (`loc_assign`, `dfl_residual`, `box_detail_head`)
  as currently recorded: not counted as an improvement because no recorded row
  shows higher `mAP50-95`/AP75 than the current baselines.
- Soft-NMS alone improves AP75/mAP50-95, but still leaves a large gap to the
  pre-NMS oracle.
- True IoU quality head as currently implemented.
- Box voting / WBF-style coordinate averaging as currently implemented.
- Boundary contrast / box-detail feature experiments as currently recorded:
  not counted as helping the final result yet.
- API / AdversarialPerturbationInjection boxgrad regularization as currently
  recorded: not counted as helping the final result yet.
- SR-DGFE / FeatureDGFE feature-detail experiments as currently recorded: not
  counted as helping the final result yet.

Reason: they do not make the final score reliably rank tight boxes above loose
boxes.

Why this observation matters: these fixes may change recall, smooth
postprocess, or add auxiliary signals, but the core requirement is stricter:
the final score must prefer the tight decoded box.

Pairwise score-ranking loss exists in the loss code, but is still the missing
confirmed result: it explicitly penalizes cases where a looser candidate scores
above a tighter candidate for the same GT. Do not claim it improved results
or helped the final metric until a positive run exists.

## Latest Postprocess Results

Why this section exists: record the current deployable baseline and decide
which postprocess branch is still worth keeping.

Checkpoint:

```text
/marimo/best_api_pooledge_p2p3.pt
```

Data:

```text
/marimo/data/datasets/varroa_yolo/varroa.yaml
```

Run:

```text
runs/detect/box_voting_ablation_repo_patch/summary.json
```

| Method | mAP50 | mAP50-95 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| hard NMS | 0.91663 | 0.33392 | 0.11303 | 0.91357 | 0.90936 |
| Soft-NMS linear | 0.91417 | 0.35385 | 0.15676 | 0.91357 | 0.90936 |
| hard NMS + box voting `score_iou` | 0.91020 | 0.33926 | 0.12920 | 0.91681 | 0.91228 |
| Soft-NMS linear + box voting `score_iou` | 0.89862 | 0.32861 | 0.11521 | 0.91357 | 0.90936 |
| hard NMS + box voting `score_iou2` | 0.91040 | 0.34058 | 0.13100 | 0.91681 | 0.91228 |
| Soft-NMS linear + box voting `score_iou2` | 0.90079 | 0.33008 | 0.11733 | 0.91357 | 0.90936 |

How to read this table: each row changes only postprocess behavior after the
model has predicted boxes. `hard NMS` is the standard keep-highest-score and
suppress-overlaps method. `Soft-NMS` lowers scores of overlapping boxes instead
of deleting them immediately. `box voting` averages nearby box coordinates.
The row to keep is the one that raises `mAP50-95`/`AP75` while leaving
precision/recall close to hard NMS.

Interpretation:

- Soft-NMS linear is still the best postprocess result: `mAP50-95 +0.01993`,
  AP75 `+0.04374` vs hard NMS.
- Box voting helps hard NMS slightly, best case `0.33392 -> 0.34058`
  mAP50-95, while Soft-NMS reaches `0.35385`.
- Soft-NMS plus box voting is worse than Soft-NMS alone. Do not continue this
  combined direction unless the voting rule changes substantially.
- Current box voting averages coordinates from neighboring candidates. It can
  improve hard-NMS AP75 a little, but it can also move an AP75-quality box
  toward lower-IoU neighbors and hurt mAP50.

Why these observations matter:

- Soft-NMS linear is the best low-risk patch because it improves high-IoU AP
  without retraining.
- Box voting should only be retried with a different voting rule; this
  implementation averages coordinates and can move a tight box away from the
  GT.
- Combining Soft-NMS and current voting is worse, so the next postprocess run
  should not keep stacking them unchanged.

## Quality Head Result

Why this section exists: document what the current true-IoU quality-head attempt
proved, and when to stop cycling similar q-head configs.

Current true-IoU quality head attempt did not produce a final score that ranks
tight boxes above loose boxes reliably enough.

Observed outcome from quality-head validation/debug:

- `q` did not rank tight boxes reliably enough to beat class confidence.
- `cls * q`, `sqrt(cls) * q`, and `cls * q^2` did not close the gap to oracle
  pre-NMS ranking.
- If a simple post-hoc probe using `cls`, `q`, DFL entropy/peak, width, height,
  level, and area also fails to improve ranking, stop training new q heads with
  the same features/loss because the exported candidate features do not contain
  a usable tight-box ordering signal.
- If that probe succeeds, the issue is likely q-head training/loss calibration,
  not feature availability.

Practical guardrail:

- Do not keep cycling quality-head configs blindly.
- Next quality-head work must first show post-hoc probe signal on exported
  candidates.
- Keep Soft-NMS linear as the current deployable postprocess baseline.

Why these observations matter: a quality head is only useful if its output
orders tight boxes above loose boxes. If exported candidate features cannot do
that in a simple probe, another training run with the same signal is unlikely
to fix the bottleneck.

## Correct Target

Why this section exists: make the quality target match the failure mode.
Center proximity and decoded-box tightness are not the same target.

Do not reuse old Gaussian `loc_quality` center map as quality target.

Use decoded-box IoU:

```python
quality_target = IoU(decoded_pred_box, matched_gt_box).detach()
```

The quality score must mean "this anchor's decoded box is tight", not "this
grid point is near object center".

Why this observation matters: AP75/AP90 care about the final decoded box edges.
A center-map target can reward anchors near the object center even when their
decoded width/height/edges are still loose.

## Current Recommended Experiment

Why this section exists: keep the next training attempt narrow and testable
against the ranking diagnosis above.

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

Why these choices matter:

- `IoU(decoded_pred_box, assigned_gt).detach()` directly supervises box
  tightness without backpropagating through the target itself.
- Hard negatives focus q-learning on confident loose candidates, the exact
  cases that break NMS/ranking.
- Multiple score modes test calibration cheaply after training instead of
  requiring a new run for every score formula.

## Train Command Shape

Why this section exists: preserve the exact reproducible run shape so the next
result is comparable to the diagnostic numbers above.

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

Why this section exists: evaluate the same checkpoint under multiple q-score
formulas and export debug candidates needed to judge ranking, not just mAP.

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

Why this observation matters: marimo multiprocessing can fail independently of
model quality. Using `workers=0` removes that noise from validation.

## Pass Criteria

Why this section exists: define what would actually confirm the hypothesis
before looking at results.

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

Why these observations matter:

- Training debug checks whether q is being trained on the intended positives
  and hard negatives.
- Monotonic `q_by_iou_bin` and Spearman gains check whether q tracks box
  tightness.
- `mean_best_minus_top_final` checks the exact ranking gap: whether final score
  selects boxes closer to the best available candidate.

## Known Gotcha

Why this section exists: prevent wasting a run on the wrong Ultralytics import
path, where the new config fields silently do not exist.

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

Why this observation matters: if the notebook imports stale code, any q-head
result is invalid because the intended loss/config path was never active.
