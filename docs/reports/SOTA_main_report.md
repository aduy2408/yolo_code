# SOTA Main Report: STW YOLO, TPH YOLO, FCOS-SET, FCOS-SR-TOD

**Updated:** 2026-10-05  
**Scope:** Varroa, LEVIR-Ship, and TinyPerson only. **VisDrone is intentionally excluded.**  
**Metric policy:** metrics are copied from remote HF artifacts. `--` means missing from the source artifact, not zero. Aggregate metrics without split labels are not relabeled as validation/test.

## 1. Scope and repositories

| Family | Repository | Scope/status |
|---|---|---|
| STW YOLO | `duyle2408/stw-yolo-runs` | Main STW YOLO runs for LEVIR-Ship, TinyPerson, Varroa. The same repo also contains Copy-Paste/variant subtrees, inventoried separately. |
| TPH YOLO | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs` | TPH-YOLOv5 runs for the three target datasets; VisDrone paths excluded. |
| TPH YOLO-n | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5n-runs` | Additional TPH-YOLOv5n run found for Varroa seed43; VisDrone paths excluded. |
| FCOS-SET | `duyle2408/set_fcos_runs` and `duyle2408/set-fcos-stw-protocol-runs` | FCOS-SET artifacts exist for target datasets; split-qualified and test-only artifacts are kept distinct. |
| FCOS-SR-TOD | `duyle2408/srtod-varroa-protocol-runs` | Partial Varroa artifact set; no complete split-qualified metrics artifact found. |

## 2. Metrics inventory

The compact table below records the fields that are present in the remote artifacts. TPH and FCOS-SET rows are split-qualified where the source exposes val/test fields. STW main rows expose aggregate YOLO evaluation metrics only.

| Family | Model/variant | Dataset | Protocol | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Status | Source |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| TPH YOLO | TPH-YOLOv5 | levirship | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.7140 | 0.2590 | 0.6740 | 0.2370 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/levirship/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | levirship | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.7550 | 0.2880 | 0.7430 | 0.2690 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/levirship/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | tinyperson | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.5420 | 0.1920 | 0.5500 | 0.2010 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/tinyperson/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | tinyperson | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.4920 | 0.1690 | 0.4890 | 0.1750 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/tinyperson/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | varroa | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.8500 | 0.2990 | 0.8270 | 0.2900 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/varroa/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | varroa | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.8410 | 0.3030 | 0.8390 | 0.3020 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/varroa/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5n | varroa | custom TPH-YOLOv5 yolov5n-xs-tph | 43 | 0.8820 | 0.3080 | 0.8620 | 0.3080 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5n-runs`: `runs/varroa/seed_43/experiment_manifest.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | LEVIR-Ship | main run | 42 | 0.7981 | 0.2949 | 0.7981 | 0.2949 | aggregate only; no val/test labels | `duyle2408/stw-yolo-runs`: `runs/levirship/seed_42/evaluation_metrics.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | TinyPerson | main run | 42 | 0.5685 | 0.2149 | 0.5685 | 0.2149 | aggregate only; no val/test labels | `duyle2408/stw-yolo-runs`: `runs/tinyperson/seed_42/evaluation_metrics.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | Varroa | main run | 42 | 0.9166 | 0.3534 | 0.9166 | 0.3534 | aggregate only; no val/test labels | `duyle2408/stw-yolo-runs`: `runs/varroa/seed_42/evaluation_metrics.json` |
| FCOS-SET | FCOS_set STW protocol | LEVIR-Ship | no_mosaic | 42 | 0.7324 | 0.2598 | 0.7740 | 0.2785 | complete | `duyle2408/set-fcos-stw-protocol-runs`: `levirship/no_mosaic/patience15/seed42/fcos_set/final_results.json` |
| FCOS-SET | FCOS_set STW protocol | TinyPerson | mosaic | 42 | 0.4148 | 0.1494 | 0.4259 | 0.1519 | complete | `duyle2408/set-fcos-stw-protocol-runs`: `tinyperson/mosaic/patience15/seed42/fcos_set/final_results.json` |
| FCOS-SET | FCOS_set YOLO protocol final | Varroa | yolo_protocol | 42 | -- | -- | 0.8720 | 0.3080 | test artifact only; val missing | `duyle2408/set_fcos_runs`: `varroa_final/fcos_set/base/test_results/20260919_082947/20260919_082947.json` |
| FCOS-SR-TOD | SRTOD Faster | Varroa | protocol run | 42 | -- | -- | -- | -- | checkpoint/job_summary/test logs; final metrics missing | `duyle2408/srtod-varroa-protocol-runs`: `srtod_faster/` |

## 3. Missing metrics and re-evaluation backlog

### 3.1 STW YOLO

- Main LEVIR-Ship, TinyPerson, and Varroa artifacts contain aggregate `metrics/mAP50(B)` and `metrics/mAP50-95(B)` plus precision/recall. They do not expose explicit val/test labels in `evaluation_metrics.json`, so this report does not relabel them.
- TinyPerson additionally has independent tile metrics and merged source-image metrics under `runs/tinyperson/seed_42/evaluation/`. These are test-side protocols and must remain separately labeled.
- The STW repository contains additional Copy-Paste and LEVIR P2 variant subtrees. They are present on HF but are not expanded into the main table because they are augmentation/variant branches rather than the main STW control run.

### 3.2 TPH YOLO

- TPH-YOLOv5 has split-qualified val/test AP50 and mAP50-95 for LEVIR-Ship seed42, TinyPerson seed42, Varroa seed42, plus LEVIR-Ship/TinyPerson seed43 where present.
- TPH-YOLOv5n has a Varroa seed43 run with split-qualified metrics.
- The TPH manifests record `patience: 0`, so these are not directly comparable to the baseline matrix patience15 protocol.

### 3.3 FCOS-SET

- LEVIR-Ship and TinyPerson FCOS-SET STW-protocol rows have complete split-qualified `final_results.json` artifacts.
- Varroa FCOS-SET has a checkpoint/job summary and test metric artifact, but the source row used here has no validation metrics.
- Older FCOS-SET repositories contain other dataset/protocol variants. Preserve their manifest and do not merge metrics across protocol families.

### 3.4 FCOS-SR-TOD

- `duyle2408/srtod-varroa-protocol-runs` contains a Varroa SRTOD Faster checkpoint, job summary, and test log artifacts.
- No complete `final_results.json`, validation metrics, or split-qualified evaluation artifact was found in the inspected SRTOD protocol repo.
- The other SRTOD repositories for LEVIR/TinyPerson currently expose only placeholder-level content in the HF listing and need a separate access/artifact check.

## 4. Protocol cautions

- These four families are variants/SOTA comparisons, not the canonical MMDetection baseline control.
- Preserve each family's own model/config, image size, optimizer, epochs, patience, split seed, training seed, and NMS IoU.
- Do not compare aggregate STW metrics directly against split-qualified TPH/FCOS-SET metrics without marking the evaluation protocol.
- VisDrone is excluded from this report by scope, including any paths containing `visdrone`.

## 5. Recommended next steps

1. Upload or generate split-qualified evaluation artifacts for STW main runs if val/test comparison is required.
2. Complete Varroa FCOS-SET validation evaluation.
3. Complete SRTOD evaluation and upload `final_results.json` or equivalent val/test metrics.
4. Keep TPH patience0 metadata visible when comparing against baseline patience15 runs.
5. Add the remaining STW variant subtrees only if a full augmentation/SOTA matrix is requested.
