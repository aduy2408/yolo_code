# Missing deployment results

**Updated:** 2026-11-08  
**Requested comparison:** test `AP50`, `AP50-Small`, and `mAP50-95`; models YOLOv8/YOLO11; OACP R2, Negative Copy-Paste R2/R4; datasets LEVIR-Ship, TinyPerson, Varroa; server `.pt`, Jetson `.pt`, TensorRT FP32, TensorRT FP16; seeds 43/44.  
**Protocol target:** `nms_iou: 0.5`.

This file records only values that are missing from the canonical recent artifacts or do not contain the requested metric field. It does not infer missing values from another evaluator or rename `mAP50-75` as `mAP50-95`.

## 1. Missing by dataset / variant
OACP Varroa is no longer missing after the seed-43/44 rerun. NC-R2/NC-R4 Varroa is intentionally out of scope. The remaining gaps are metric-field/protocol gaps listed below.


## 2. Missing TinyPerson metric field

The recent deployment summaries:

```text
DUY_DEPLOY/oacp_deploy_results.md
DUY_DEPLOY/negativeCP_deploy_results.md
```

report TinyPerson with TinyBenchmark fields:

```text
AP50
AP75
mAP50-75
AP50-tiny
AP50-small
```

The TinyPerson Jetson `.pt` predictions have now been merged and scored from the archived artifacts. No inference rerun was needed. All requested TinyPerson variants/models/seeds have `test_merged/AP50`, `test_merged/AP50-Small`, `test_merged/mAP50-95`, and `test_merged/mAP50-95-Small`.

| Dataset | Variants | Models | Backends | Seeds | Status |
|---|---|---|---|---|---|
| TinyPerson | OACP R2, NC R2, NC R4 | YOLOv8, YOLO11 | Jetson `.pt` | 43, 44 | Complete after offline merge + `test_merged/mAP50-95` |

The verified values are recorded in `DUY_DEPLOY/tinyperson_levirship_results.md` under `Verified TinyPerson Jetson .pt merge — mAP50-95`.

No `mAP50-75` value was copied into the requested `mAP50-95` field.

## 3. Summary matrix

`Present` means the requested metric family is available in a recent canonical artifact. `Missing` means it must remain `--` in the comparison table.

| Dataset | Variant | Server `.pt` | Jetson `.pt` | TRT FP32 | TRT FP16 |
|---|---|---|---|---|---|
| LEVIR-Ship | OACP R2 | Present | Present | Present | Present |
| TinyPerson | OACP R2 | Present, merged | Present, offline merged/scored | Present | Present |
| Varroa | OACP R2 | Present | Present | Present | Present |
| LEVIR-Ship | NC R2 | Present | Present | Present | Present |
| TinyPerson | NC R2 | Present, merged | Present, archived merged/scored | Present | Present |
| LEVIR-Ship | NC R4 | Present | Present | Present | Present |
| TinyPerson | NC R4 | Present, merged | Present, archived merged/scored | Present | Present |

## 4. Source/provenance notes

- OACP parity source: `duyle2408/oacp-r2-parity-rerun-seed43-44`.
- OACP merged TinyPerson backfill: `duyle2408/oacp-r2-seed44-merged-eval` and the size-evaluation repositories.
- Negative Copy-Paste parity sources and corrected backfills are documented in `docs/reports/copy_paste.md`.
- Server parity summaries: `docs/reports/oacp_r2_seed43_44_results.md`, `docs/reports/copy_paste.md`.
- Jetson deployment summaries: `DUY_DEPLOY/oacp_deploy_results.md`, `DUY_DEPLOY/negativeCP_deploy_results.md`.
- Saved archive: `DUY_DEPLOY/results/augmentation_runs_20261106.zip`
- PT prediction path inside archive: `augmentation_runs_20261106/pt/<job>/pt/best_test_corner_predictions.json`
- FP32/FP16 merged metric path inside archive: `augmentation_runs_20261106/{fp32,fp16}/<job>/engine_eval/best_test_merged_metrics.json`
- The TinyPerson Jetson `.pt` merge/evaluation completed for 12/12 jobs; no missing TinyPerson result remains.
