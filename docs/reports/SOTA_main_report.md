# SOTA Report: STW YOLO, TPH YOLO, FCOS-SET, FCOS-SR-TOD

**Updated:** 2026-10-05  
**Scope:** Varroa, LEVIR-Ship, and TinyPerson only. **VisDrone is intentionally excluded.**  
**Metric policy:** metrics are copied from remote HF artifacts. `--` means missing from the source artifact, not zero. Aggregate metrics without split labels are not relabeled as validation/test.

## 1. Scope and repositories

| Family | Repository | Scope/status |
|---|---|---|
| STW YOLO | `duyle2408/stw-yolo-runs` | STW YOLO runs for LEVIR-Ship, TinyPerson, Varroa. The same repo also contains Copy-Paste/variant subtrees, inventoried separately. |
| TPH YOLO | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs` | TPH-YOLOv5 runs for the three target datasets; VisDrone paths excluded. |
| TPH YOLO-n | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5n-runs` | Observed seed43 artifact for Varroa; requested comparison seed42 is not present in the inspected repo. |
| FCOS-SET | `duyle2408/set_fcos_runs` and `duyle2408/set-fcos-stw-protocol-runs` | FCOS-SET artifacts exist for target datasets; split-qualified and test-only artifacts are kept distinct. |
| FCOS-SR-TOD | `duyle2408/srtod-set-mosaic-matrix-runs` and `duyle2408/srtod-varroa-protocol-runs` | Requested seed43 artifacts exist for Varroa/LEVIR-Ship/TinyPerson, but split-qualified metrics are incomplete. |

## 2. Complete main-report metric table

This table is the requested main comparison slice: one selected run per family, across the three target datasets. Every split-qualified field is labelled explicitly. `--` means the source artifact did not expose that split/metric, not zero.

| Family | Dataset | Seed | Val AP50 | Val mAP50-95 | Val AP50-small | Test AP50 | Test mAP50-95 | Test AP50-small | Provenance/status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| STW YOLO | Varroa | 42 | -- | -- | 0.8978 | -- | -- | 0.8818 | Core artifact aggregate only: AP50 0.9166, mAP50-95 0.3534; size evaluator provides val/test AP50-small |
| STW YOLO | LEVIR-Ship | 42 | -- | -- | 0.8340 | -- | -- | 0.8408 | Core artifact aggregate only: AP50 0.7981, mAP50-95 0.2949; size evaluator provides val/test AP50-small |
| STW YOLO | TinyPerson | 42 | -- | -- | 0.5981 | -- | -- | 0.5418 | Core artifact aggregate tile AP50 0.5685, mAP50-95 0.2149; size evaluator provides val/test AP50-small |
| TPH YOLOv5 | Varroa | 42 | 0.8480 | 0.2930 | -- | 0.8220 | 0.2920 | -- | Native YOLO val/test artifact; AP50-small absent from HF artifact |
| TPH YOLOv5 | LEVIR-Ship | 42 | 0.7140 | 0.2590 | -- | 0.6740 | 0.2370 | -- | Native YOLO val/test artifact; AP50-small absent from HF artifact |
| TPH YOLOv5 | TinyPerson | 42 | 0.5420 | 0.1920 | -- | 0.5500 | 0.2010 | -- | Official corner-window val/test artifact; AP50-small absent from HF artifact |
| FCOS-SET | Varroa | 43 | 0.8950 | 0.3070 | 0.2820 | 0.8750 | 0.3180 | 0.2970 | Fresh `tools/test.py` evaluation of supplied HF checkpoint; predictions verified |
| FCOS-SET | LEVIR-Ship | 43 | 0.7656 | 0.2762 | 0.7683 | 0.7358 | 0.2708 | 0.7330 | Supplied HF `final_results.json`, cross-checked with fresh MMDetection evaluation |
| FCOS-SET | TinyPerson | 43 | 0.4084 | 0.1487 | 0.5366 | 0.4247 | 0.1523 | 0.5916 | Supplied HF `final_results.json`; val/test fields complete |
| FCOS-SR-TOD | Varroa | 43 | 0.9150 | 0.3100 | -- | 0.9030 | 0.3310 | 0.3100 | Supplied HF checkpoint/test artifacts; val AP50/mAP reported, val AP50-small absent |
| FCOS-SR-TOD | LEVIR-Ship | 43 | -- | -- | -- | 0.6520 | 0.2070 | 0.2050 | Supplied HF test artifact; split-qualified val metrics absent |
| FCOS-SR-TOD | TinyPerson | 43 | -- | -- | -- | 0.3730 | 0.1290 | 0.1150 | Supplied HF test artifact; split-qualified val metrics absent |

### Metric and provenance notes

- STW YOLO's `evaluation_metrics.json` exposes aggregate YOLO metrics, not explicit `val/*` and `test/*` keys. The table therefore does not relabel its aggregate AP50/mAP50-95 as validation or test. Its AP50-small values come from the separate split-aware size evaluator.
- TPH YOLO's HF artifacts expose val/test AP50 and mAP50-95, but the inspected `evaluation_metrics.json` files do not contain AP50-small. Those cells remain `--`.
- FCOS-SET seed43 Varroa was evaluated directly with `/marimo/mmdet_code/mmdetection/tools/test.py` using the supplied checkpoint. FCOS-SET Varroa and LEVIR prediction artifacts are stored under `/marimo/sota_eval/corrected_fcos_eval_results/`.
- FCOS-SR-TOD rows use the exact user-supplied HF prefixes. Existing HF artifacts provide complete test metrics for all three datasets, but not complete val metrics for LEVIR-Ship and TinyPerson.


The compact table below records the fields that are present in the remote artifacts. TPH and FCOS-SET rows are split-qualified where the source exposes val/test fields. STW YOLO rows expose aggregate YOLO evaluation metrics only. Existing rows are historical results and are not being marked for re-evaluation by the seed plan below.

| Family | Model/variant | Dataset | Protocol | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Status | Source |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| TPH YOLO | TPH-YOLOv5 | levirship | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.7140 | 0.2590 | 0.6740 | 0.2370 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/levirship/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | levirship | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.7550 | 0.2880 | 0.7430 | 0.2690 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/levirship/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | tinyperson | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.5420 | 0.1920 | 0.5500 | 0.2010 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/tinyperson/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | tinyperson | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.4920 | 0.1690 | 0.4890 | 0.1750 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/tinyperson/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | varroa | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.8500 | 0.2990 | 0.8270 | 0.2900 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/varroa/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | varroa | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.8410 | 0.3030 | 0.8390 | 0.3020 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/varroa/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5n | varroa | custom TPH-YOLOv5 yolov5n-xs-tph | 43 | 0.8820 | 0.3080 | 0.8620 | 0.3080 | observed seed43; requested seed42 missing | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5n-runs`: `runs/varroa/seed_43/experiment_manifest.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | LEVIR-Ship | standard STW run | 42 | 0.7981 | 0.2949 | 0.7981 | 0.2949 | aggregate only; no val/test labels | `duyle2408/stw-yolo-runs`: `runs/levirship/seed_42/evaluation_metrics.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | TinyPerson | standard STW run | 42 | 0.5685 | 0.2149 | 0.5685 | 0.2149 | aggregate only; no val/test labels | `duyle2408/stw-yolo-runs`: `runs/tinyperson/seed_42/evaluation_metrics.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | Varroa | standard STW run | 42 | 0.9166 | 0.3534 | 0.9166 | 0.3534 | aggregate only; no val/test labels | `duyle2408/stw-yolo-runs`: `runs/varroa/seed_42/evaluation_metrics.json` |
| FCOS-SET | FCOS_set STW protocol | LEVIR-Ship | no_mosaic | 42 | 0.7324 | 0.2598 | 0.7740 | 0.2785 | complete | `duyle2408/set-fcos-stw-protocol-runs`: `levirship/no_mosaic/patience15/seed42/fcos_set/final_results.json` |
| FCOS-SET | FCOS_set STW protocol | TinyPerson | mosaic | 42 | 0.4148 | 0.1494 | 0.4259 | 0.1519 | complete | `duyle2408/set-fcos-stw-protocol-runs`: `tinyperson/mosaic/patience15/seed42/fcos_set/final_results.json` |
| FCOS-SET | FCOS_set YOLO protocol final | Varroa | yolo_protocol | 42 | -- | -- | 0.8720 | 0.3080 | observed seed42; target seed43 not found | `duyle2408/set_fcos_runs`: `varroa_final/fcos_set/base/test_results/20260919_082947/20260919_082947.json` |
| FCOS-SR-TOD | FCOS-SR-TOD requested seed43 | Varroa | standard SRTOD | 43 | -- | -- | -- | -- | checkpoint/train/test artifacts; split metrics missing | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/varroa/fcos/seed43/` |


## 3. Requested seed alignment

| Family | Requested seed | Note |
|---|---:|---|
| STW YOLO | existing result uses seed42 | Keep the existing result. No new STW YOLO run is requested here. It is recorded simply as **STW YOLO**, not “main”. |
| TPH-YOLOv5n | 42 | Existing seed43 result remains historical. Run a new seed42 experiment. |
| FCOS-SET | 43 | Existing seed42 results remain historical. Run a new seed43 experiment. |
| FCOS-SR-TOD | 43 | Run/record the requested seed43 experiment. Existing artifacts remain historical. |

## 4. Historical artifact notes

The items below describe the old artifacts already present. They are retained as historical evidence and are not a request to rerun or re-evaluate them.

### 4.1 STW YOLO

- STW YOLO LEVIR-Ship, TinyPerson, and Varroa artifacts contain aggregate `metrics/mAP50(B)` and `metrics/mAP50-95(B)` plus precision/recall. They do not expose explicit val/test labels in `evaluation_metrics.json`, so this report does not relabel them.
- TinyPerson additionally has independent tile metrics and merged source-image metrics under `runs/tinyperson/seed_42/evaluation/`. These are test-side protocols and remain separately labeled.
- The STW repository contains additional Copy-Paste and LEVIR P2 variant subtrees. They are present on HF but are not expanded into the table because they are augmentation/variant branches rather than the requested STW YOLO result.

### 4.2 TPH YOLO

- TPH-YOLOv5 has split-qualified val/test AP50 and mAP50-95 for LEVIR-Ship seed42, TinyPerson seed42, Varroa seed42, plus LEVIR-Ship/TinyPerson seed43 where present.
- TPH-YOLOv5n has an observed Varroa seed43 result. This old result is retained. The new requested run seed is 42.
- The TPH manifests record `patience: 0`, so these historical results are not directly comparable to the baseline matrix patience15 protocol.

### 4.3 FCOS-SET

- LEVIR-Ship and TinyPerson FCOS-SET STW-protocol rows have complete split-qualified `final_results.json` artifacts at seed42.
- Varroa FCOS-SET has an observed seed42 checkpoint/job summary and test metric artifact.
- These seed42 results are retained as historical evidence. The new requested FCOS-SET run seed is 43.

### 4.4 FCOS-SR-TOD

- The existing SRTOD artifacts include Varroa, LEVIR-Ship, and TinyPerson train/checkpoint/test material under `duyle2408/srtod-set-mosaic-matrix-runs`.
- The old artifact set does not provide complete split-qualified `final_results.json` or equivalent metric files for all rows.
- The new requested FCOS-SR-TOD run seed is 43. Existing artifacts remain historical and should not be interpreted as a request to repair the old results.

## 5. Protocol cautions

- These four families are variants/SOTA comparisons, not the canonical MMDetection baseline control.
- Preserve each family's own model/config, image size, optimizer, epochs, patience, split seed, training seed, and NMS IoU.
- Do not compare aggregate STW metrics directly against split-qualified TPH/FCOS-SET metrics without marking the evaluation protocol.
- VisDrone is excluded from this report by scope, including any paths containing `visdrone`.

## 6. Requested future runs

This is the run note for the next execution. It does not invalidate or replace the historical results above.

| Model family | Training seed | Scope | Note |
|---|---:|---|---|
| TPH-YOLOv5n | 42 | Varroa, LEVIR-Ship, TinyPerson as applicable | New requested run. |
| FCOS-SET | 43 | Varroa, LEVIR-Ship, TinyPerson as applicable | New requested run. |
| FCOS-SR-TOD | 43 | Varroa, LEVIR-Ship, TinyPerson as applicable | New requested run. |

STW YOLO is not included in the new-run list. Keep its existing result only, and do not use the label “STW YOLO main”.

## 7. Recommended next steps

1. Upload or generate split-qualified evaluation artifacts for standard STW YOLO runs if val/test comparison is required.
2. Complete Varroa FCOS-SET validation evaluation.
3. Complete SRTOD evaluation and upload `final_results.json` or equivalent val/test metrics.
4. Keep TPH patience0 metadata visible when comparing against baseline patience15 runs.
5. Add the remaining STW variant subtrees only if a full augmentation/SOTA matrix is requested.
