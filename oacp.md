# OACP Report

**Date:** 2026-09-09  
**Dataset:** LEVIR-Ship  
**Fixed split seed:** `42`  
**Training seed for the comparison runs:** `42`

## Executive summary

The original OACP parameter sweep was not a single-pass OACP sweep. The training pipeline applied OACP twice for the normal training route:

```text
wrapper OACP -> normal_pipeline() -> v8_transforms() OACP
```

The seven-run aggressive sweep therefore selected parameters under the legacy double-OACP distribution. Its results cannot be used as a clean parameter selection for the corrected single-pass pipeline.

**Decision:** rerun all seven aggressive sweep configurations if the goal is to select OACP parameters for the corrected pipeline. R1-R5 must be rerun first because R6-R7 are selected from the best R1-R5 validation result.

## What changed on 2026-09-09

### 1. Corrected active training flow

Commit `c1deaf1` removed the unintended wrapper-level OACP call. The active default is now:

```text
regular training -> normal_pipeline() -> v8_transforms() -> OACP once
```

The legacy behavior is available only when explicitly enabled:

```bash
--legacy-double-oacp
```

or:

```bash
YOLO_LEGACY_DOUBLE_OACP=1
```

The option was added in commit `fe44e63`. The legacy call site and flow are documented inline in `models_related/ultralytics/ultralytics/data/augment.py`.

### 2. Matched standard versus approximation comparison

The corrected comparison runner was added in `2dc8b8b` and checkpoint support was finalized in `9a77f69`.

The completed Marimo comparison used the same fixed split, seed, model, schedule, and evaluation pipeline:

| Variant | Mosaic | OACP probability | Test mAP50 | Test mAP50-95 | Test AP75 |
|---|---:|---:|---:|---:|---:|
| Standard | 1.0 | 0.20 | 0.7428 | 0.2633 | 0.0784 |
| Approximation | 1.0 | 0.36 | 0.7656 | 0.2827 | 0.0890 |
| Standard | 0.0 | 0.20 | 0.6577 | 0.2222 | 0.0787 |
| Approximation | 0.0 | 0.36 | 0.8005 | 0.2936 | 0.1167 |

The `p=0.36` approximation recovers much of the legacy effect, but it is not pixel-equivalent to two sequential passes. It matches only:

```text
P(at least one pass) = 1 - (1 - 0.20)^2 = 0.36
```

### 3. Stronger one-pass experiment

Commit `ecaec0b` adds and launches two additional variants:

```text
p = 0.36
OACP strength = [0.40, 0.60]
```

Variants:

```text
oacp_double_approx_strong_mosaic
oacp_double_approx_strong_no_mosaic
```

HF repository:

```text
duyle2408/levir-oacp-strong-ecaec0b
```

These runs were still in progress when this report was written.

## The old seven-run sweep

The old runner is:

```text
train_levir_scripts/train_levir_yolov8n_p2_oacp_aggressive.py
```

It defines:

| Run | p | Strength | Scale | Protection expand |
|---|---:|---:|---:|---:|
| R1 sparse mild | 0.10 | [0.10, 0.25] | [0.80, 0.95] | 3.0 |
| R2 frequent mild | 0.40 | [0.10, 0.25] | [0.80, 0.95] | 3.0 |
| R3 frequent current | 0.40 | [0.20, 0.40] | [0.65, 0.85] | 3.0 |
| R4 strong | 0.20 | [0.40, 0.65] | [0.45, 0.70] | 3.0 |
| R5 very strong stress | 0.40 | [0.40, 0.65] | [0.45, 0.70] | 3.0 |
| R6 narrow protection | selected from best R1-R5 | selected | selected | 1.5 |
| R7 wide protection | selected from best R1-R5 | selected | selected | 5.0 |

The historical HF artifact is:

```text
duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758
```

Its manifests confirm that the runs were produced before the single-pass fix. The recorded commits are `9fb0758` for R1-R3 and `dfd3025` for R4-R7, both before `c1deaf1`.

Historical results:

| Run | Test mAP50 | Test mAP50-95 | Validation mAP50 |
|---|---:|---:|---:|
| R1 | 0.8188 | 0.3103 | 0.8261 |
| R2 | 0.8242 | 0.3164 | 0.8332 |
| R3 | 0.7116 | 0.2554 | 0.7489 |
| R4 | 0.8236 | 0.3209 | 0.8413 |
| R5 | 0.8147 | 0.3115 | **0.8470** |
| R6 | 0.7968 | 0.3019 | 0.8420 |
| R7 | 0.8050 | 0.3112 | 0.8428 |

These values are valid historical results for the legacy double-OACP pipeline, but not corrected single-pass parameter-selection results. The old sweep was also nondeterministic and used inconsistent worker settings across runs, so it should not be treated as a tightly controlled modern ablation.

## What must be rerun

### Required for corrected single-pass parameter selection

Rerun all seven:

```text
R1 -> R2 -> R3 -> R4 -> R5 -> select best by validation mAP50 -> R6 -> R7
```

R6 and R7 cannot be rerun independently because their parameters depend on the best R1-R5 result. Reusing the old winner `R5` would mix a legacy double-OACP selection with a corrected single-pass evaluation.

### Not required for historical reproduction

The old HF artifacts do not need to be overwritten. They should remain as the historical legacy record. If exact reproduction is needed, use the explicit legacy option:

```bash
--legacy-double-oacp
```

## Recommended corrected sweep contract

For the rerun, keep the data provenance fixed:

```text
split seed: 42
training seed: 42
model: YOLOv8n P2 LEVIR baseline
imgsz: 512
batch: 8
epochs: 100
patience: 0
Mosaic: disabled, matching the old aggressive sweep
NMS IoU: 0.5
OACP: single-pass default
```

Use one consistent worker count and deterministic mode for all seven runs. The runner was updated for this corrected sweep in commit `e5938dc` to default to `workers=4` and `deterministic=true`. Record the exact parameter grid, commit SHA, split seed, training seed, and selected R1-R5 winner in the new manifests.

## Final conclusion

The unexpectedly strong historical OACP results were not caused only by the nominal parameter values. They were produced under an accidental double-OACP training distribution. The corrected pipeline changes the experiment definition. Therefore the old seven-run sweep must be rerun if its purpose is to identify the best parameters for corrected single-pass OACP.
