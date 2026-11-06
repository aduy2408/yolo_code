# SOTA Report: STW YOLO, TPH YOLO, FCOS-SET, FCOS-SR-TOD

**Updated:** 2026-11-06
**Scope:** Varroa, LEVIR-Ship, and TinyPerson only. **VisDrone is intentionally excluded.**  
**Metric policy:** metrics are copied from remote HF artifacts or directly verified rerun artifacts. `--` means missing or not verified for the stated protocol, not zero. Aggregate metrics without split labels are not relabeled as validation/test. Merged TinyPerson metrics are reported separately from native corner-window metrics.

## 1. Scope and repositories

| Family | Repository | Scope/status |
|---|---|---|
| STW YOLO | `duyle2408/stw-yolo-runs` | STW YOLO runs for LEVIR-Ship, TinyPerson, Varroa. The same repo also contains Copy-Paste/variant subtrees, inventoried separately. |
| TPH YOLO | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs` | TPH-YOLOv5 runs for the three target datasets; VisDrone paths excluded. |
| TPH YOLO-n | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5n-runs` | Observed seed43 artifact for Varroa; requested comparison seed42 is not present in the inspected repo. |
| FCOS-SET | `duyle2408/set_fcos_runs` and `duyle2408/set-fcos-stw-protocol-runs` | FCOS-SET artifacts exist for target datasets; split-qualified and test-only artifacts are kept distinct. |
| FCOS-SR-TOD | `duyle2408/srtod-set-mosaic-matrix-runs`, `duyle2408/srtod-levir-protocol-runs`, and `duyle2408/srtod-tinyperson-protocol-runs` | Seed43 standard-Mosaic artifacts are complete/uploaded. A matched standard-Mosaic seed42 LEVIR-Ship/TinyPerson backfill was trained, evaluated, and uploaded on 2026-11-05. Historical older-protocol seed42 artifacts remain distinct. |

## 1.1 FCOS-SR-TOD seed coverage audit (2026-11-06)

Seed42 now has a matched standard-Mosaic FCOS-SR-TOD backfill for both LEVIR-Ship and TinyPerson. The rows below use the recovered FCOS config, split seed 42, training seed 42, 640×640, 100 epochs, and the same standard-Mosaic protocol as the selected seed43 rows.

| Dataset | Training seed | Current artifact | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Native COCO AP-small | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| LEVIR-Ship | 42 | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/levir/fcos/standard_mosaic/seed42_backfill_20261105/` | 0.563 | 0.180 | 0.435 | 0.118 | 0.188 / 0.117 | Complete train + val/test evaluation + upload |
| TinyPerson | 42 | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/tinyperson/fcos/standard_mosaic/seed42_backfill_20261105/` | 0.480 | 0.159 | 0.371 | 0.128 | 0.132 / 0.118 | Complete train + val/test evaluation + upload |

The `Native COCO AP-small` values are `bbox_mAP_s` for val/test, averaged over IoU 0.50:0.95. They are not AP50-small and remain separate from the seed43 AP50-small columns.

Source evidence:

```text
/marimo/mmdet_code/runs/srtod_seed42_recovered/{levir,tinyperson}/train_retry/
/marimo/mmdet_code/runs/srtod_seed42_recovered/{levir,tinyperson}/test_seed42/predictions.pkl
duyle2408/srtod-set-mosaic-matrix-runs:
  srtod/{levir,tinyperson}/fcos/standard_mosaic/seed42_backfill_20261105/
```

No verified FCOS-SR-TOD seed44 result was found.

### FCOS-SR-TOD seed42 native area-bucket evaluation

Bucket evaluation used the native val/test predictions from the matched seed42 runs, TinyBenchmark area ranges, IoU thresholds 0.50:0.05:0.75, and the TinyPerson evaluator. Values labelled `AP50-*` are IoU=0.50 bucket AP; `AP-*` averages IoU 0.50:0.75.

| Dataset | Split | AP50 | mAP50-75 | AP50-Tiny1 | AP50-Tiny2 | AP50-Tiny3 | AP50-Small | AP50-Medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | Val | 0.6514 | 0.3870 | -- | 0.1376 | 0.4831 | 0.8226 | 0.6588 |
| LEVIR-Ship | Test | 0.4346 | 0.1961 | -- | 0.0000 | 0.3179 | 0.5622 | 0.8845 |
| TinyPerson | Val | 0.4801 | 0.2598 | 0.1655 | 0.3313 | 0.5814 | 0.6696 | 0.7880 |
| TinyPerson | Test | 0.4522 | 0.2589 | 0.1478 | 0.3827 | 0.5475 | 0.6289 | 0.6568 |

Bucket artifacts were uploaded under:

```text
duyle2408/srtod-set-mosaic-matrix-runs:
  srtod/{levir,tinyperson}/fcos/standard_mosaic/seed42_backfill_20261105/bucket_eval/
```

| Dataset | Training seed | Historical artifact | Test AP50 | Test mAP50-95 | Test small metric | Protocol/status |
|---|---:|---|---:|---:|---:|---|
| Varroa | 42 | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/varroa/fcos/seed42/` | 0.903 | 0.331 | 0.310 | COCO `bbox_mAP_s` (not AP50-small); FCOS, 640×640, complete historical artifact |
| LEVIR-Ship | 42 | `duyle2408/srtod-levir-protocol-runs`: `srtod_faster/` | 0.611 | 0.188 | 0.186 | COCO `bbox_mAP_s` (not AP50-small); SRTOD FasterRCNN, no-Mosaic, 512×512, complete test artifact |
| TinyPerson | 42 | `duyle2408/srtod-tinyperson-protocol-runs`: `fcos/` | 0.216 | 0.072 | 0.061 | COCO `bbox_mAP_s` (not AP50-small); FCOS corner-window Mosaic, 640×640, complete test artifact |
| Varroa | 43 | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/varroa/fcos/standard_mosaic/seed43_no_amp/` | 0.903 | 0.331 | 0.8759 | Selected seed43 rerun; split-qualified metrics in the main table |
| LEVIR-Ship | 43 | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/levir/fcos/standard_mosaic/seed43_recovery2_20261104/` | 0.652 | 0.207 | 0.6466 | Selected seed43 corrected full-split rerun |
| TinyPerson | 43 | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/tinyperson/fcos/standard_mosaic/seed43_recovery2_20261104/` | 0.373 | 0.129 | 0.3466 | Selected seed43 native corner-window rerun |


The historical seed42 files expose standard COCO `bbox_mAP_s` (AP-small averaged over IoU 0.50:0.95), not the AP50-small metric used by the corrected seed43 rows. Do not compare those small-object columns as if they were identical metrics.
The seed42 test metrics above are historical evidence and are not used to compute mean ± standard deviation with seed43 because LEVIR-Ship uses SRTOD FasterRCNN/no-Mosaic/512×512, while the selected seed43 LEVIR row uses FCOS/standard-Mosaic/640×640. The TinyPerson seed42 artifact is closer (FCOS/window Mosaic/640×640) but remains an older run/config; retain the provenance distinction.

Seed42 source files:

```text
duyle2408/srtod-set-mosaic-matrix-runs:
  srtod/varroa/fcos/seed42/evaluation_metrics.json
  srtod/varroa/fcos/seed42/test/20260919_043206/20260919_043206.json
duyle2408/srtod-levir-protocol-runs:
  srtod_faster/job_summary.json
  srtod_faster/test_results/20260923_092425/20260923_092425.json
duyle2408/srtod-tinyperson-protocol-runs:
  fcos/job_summary.json
  fcos/test_results/20260923_120432/20260923_120432.json
```

-No verified FCOS-SR-TOD seed44 result was found.

## 2. Complete main-report metric table

This table is the requested main comparison slice: one selected run per family, across the three target datasets. Every split-qualified field is labelled explicitly. `--` means the source artifact did not expose that split/metric, not zero.

| Family | Dataset | Seed | Val AP50 | Val mAP50-95 | Val AP50-small | Test AP50 | Test mAP50-95 | Test AP50-small | Provenance/status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| STW YOLO | Varroa | 42 | 0.9296 | 0.3491 | 0.8978 | 0.9166 | 0.3533 | 0.8818 | Existing split-aware STW evaluation artifact; native val/test metrics plus AP50-small verified |
| STW YOLO | LEVIR-Ship | 42 | 0.8321 | 0.3325 | 0.8340 | 0.7934 | 0.2861 | 0.8408 | Fresh split-qualified seed42 native evaluation from the HF checkpoint; bucket backfill recorded in the audit |
| STW YOLO | TinyPerson | 42 | 0.6190 | 0.2313 | 0.5982 | 0.5684 | 0.2149 | 0.5419 | Fresh split-qualified seed42 native evaluation from the HF checkpoint; bucket backfill recorded in the audit |
| TPH YOLOv5 | Varroa | 42 | 0.8480 | 0.2930 | 0.8486 | 0.8220 | 0.2920 | 0.8275 | Existing evaluation-only artifact from the supplied TPH checkpoint; native val/test metrics plus AP50-small verified |
| TPH YOLOv5 | LEVIR-Ship | 42 | 0.7140 | 0.2590 | 0.7862 | 0.6740 | 0.2370 | 0.7180 | Existing evaluation-only artifact from the supplied TPH checkpoint; native val/test metrics plus AP50-small verified |
| TPH YOLOv5 | TinyPerson | 42 | 0.5420 | 0.1920 | -- | 0.5350 | 0.1960 | -- | Verified native TPH-YOLOv5 rerun on Marimo; corrected merged val/test metrics are listed in the rerun addendum below |
| FCOS-SET | Varroa | 43 | 0.8950 | 0.3070 | 0.2820 | 0.8750 | 0.3180 | 0.2970 | Fresh `tools/test.py` evaluation of supplied HF checkpoint; predictions verified |
| FCOS-SET | LEVIR-Ship | 43 | 0.7656 | 0.2762 | 0.7683 | 0.7358 | 0.2708 | 0.7330 | Supplied HF `final_results.json`, cross-checked with fresh MMDetection evaluation |
| FCOS-SET | TinyPerson | 43 | 0.4084 | 0.1487 | 0.5366 | 0.4247 | 0.1523 | 0.5916 | Supplied HF `final_results.json`; val/test fields complete |
| FCOS-SR-TOD | Varroa | 43 | 0.9140 | 0.3120 | 0.8853 | 0.9030 | 0.3310 | 0.8759 | Evaluation-only rerun from supplied checkpoint; AP50-small recomputed with COCOeval IoU=0.50, area=small, maxDets=1000 |
| FCOS-SR-TOD | LEVIR-Ship | 43 | 0.7330 | 0.2670 | 0.7375 | 0.6520 | 0.2070 | 0.6466 | Corrected full 584-image val/test rerun from supplied checkpoint; AP50-small recomputed with COCOeval IoU=0.50, area=small |
| FCOS-SR-TOD | TinyPerson | 43 | 0.4180 | 0.1500 | 0.3712 | 0.3730 | 0.1290 | 0.3466 | Evaluation-only rerun from supplied checkpoint; AP50-small recomputed with COCOeval IoU=0.50, area=small, maxDets=1000 |
| FCOS-SR-TOD | Varroa | 42 | -- | -- | -- | 0.9030 | 0.3310 | -- | Historical HF test artifact; native COCO `bbox_mAP_s=0.310`, not AP50-small |
| FCOS-SR-TOD | LEVIR-Ship | 42 | 0.563 | 0.180 | -- | 0.435 | 0.118 | -- | Matched standard-Mosaic seed42 backfill; native COCO `bbox_mAP_s` is 0.188 val / 0.117 test, not AP50-small; uploaded under `srtod/levir/fcos/standard_mosaic/seed42_backfill_20261105/` |
| FCOS-SR-TOD | TinyPerson | 42 | 0.480 | 0.159 | -- | 0.371 | 0.128 | -- | Matched standard-Mosaic seed42 backfill; native COCO `bbox_mAP_s` is 0.132 val / 0.118 test, not AP50-small; uploaded under `srtod/tinyperson/fcos/standard_mosaic/seed42_backfill_20261105/` |
| FCOS-SR-TOD | LEVIR-Ship | 42 | -- | -- | -- | 0.611 | 0.188 | -- | Historical HF test artifact; older SRTOD FasterRCNN/no-Mosaic/512×512 protocol retained for provenance |
| FCOS-SR-TOD | TinyPerson | 42 | -- | -- | -- | 0.216 | 0.072 | -- | Historical HF test artifact; older FCOS corner-window Mosaic/640×640 protocol retained for provenance |

### Verified TinyPerson merged rerun addendum (2026-10-15)

These values come from the corrected GPU rerun on the authorized Marimo server using the exact TinyPerson corner-window protocol, split seed 42, image size 640, batch size 32, workers 8, NMS IoU 0.50, and the seed42 STW/TPH checkpoints. The evaluator used native TPH coordinate scaling, manifest corner `[x, y, width, height]` offsets, source-image merging, separate per-model artifacts, and class-agnostic NMS after merging. The TinyPerson COCO evaluator remained unchanged and ran on CPU. All values below are explicitly **merged** metrics.

| Model | Val merged AP50 | Val merged mAP50-95 | Val merged AP50-small | Test merged AP50 | Test merged mAP50-95 | Test merged AP50-small | Device/status |
|---|---:|---:|---:|---:|---:|---:|---|
| STW YOLO | 0.6167 | 0.2297 | 0.7192 | 0.5883 | 0.2237 | 0.6989 | `cuda:0`, complete |
| TPH YOLOv5 | 0.5368 | 0.1897 | 0.6507 | 0.5475 | 0.2042 | 0.6369 | `cuda:0`, complete |

The corrected artifacts and upload were verified in `duyle2408/stw-tph-tinyperson-corrected-20261015-runs` under `tinyperson/seed42/corrected_gpu_r4/`. Required files include `final_results.json`, both validation/test merged prediction files for each model, both merged ground-truth files, and `run_contract.json`. The TPH test artifact contains 2,288,316 detections before final report use. These values replace the earlier invalid TPH merged values produced by parsing saved normalized text predictions.

The source artifact records additional split-qualified values, including STW validation/test merged `mAP50-75` 0.3720/0.3596 and TPH validation/test merged `mAP50-75` 0.3074/0.3289.

### Latest Marimo merged rerun for STW and TPH (2026-10-24)

This is the latest verified test-only merged rerun for the seed42 STW-YOLO and TPH-YOLOv5 checkpoints. It uses the TinyPerson corner-window manifest, maps detections back to original-image IDs, applies NMS IoU `0.50`, and runs the project TinyBenchmark evaluator. These values supersede the older rounded test-only values above for this rerun, while the 2026-10-15 artifact remains preserved as historical provenance.

| Model | `test_merged/AP50` | `test_merged/mAP50-95` | `test_merged/AP50-Small` | `test_merged/mAP50-95-Small` |
|---|---:|---:|---:|---:|
| STW YOLO seed42 | 0.5892 | 0.2241 | 0.7004 | 0.2903 |
| TPH YOLOv5 seed42 | 0.5474 | 0.2041 | 0.6370 | 0.2565 |

The latest native size-qualified rerun verified `val/AP50-Small` / `test/AP50-Small` as `0.8340 / 0.8408` for STW on LEVIR-Ship and `0.5982 / 0.5419` for STW on TinyPerson. These are native split metrics and must not be relabeled as merged metrics.
### Verified FCOS TinyPerson merged rerun (2026-10-15)

These values are from the new GPU Marimo rerun using the exact seed43 checkpoints and configs recovered from the historical Hugging Face artifacts. The evaluator used the canonical MMDetection `tools/test.py` interface followed by `evaluate_tinyperson_metrics.py`, with split seed 42, 640x640 corner-window inference, source-image merging, and class-aware NMS. These are **merged** metrics and must not be substituted into the native corner-window columns above.

| Model | Val merged AP50 | Val merged mAP50-95 | Val merged AP50-small | Test merged AP50 | Test merged mAP50-95 | Test merged AP50-small | Checkpoint/provenance |
|---|---:|---:|---:|---:|---:|---:|---|
| FCOS-SET | 0.4084 | 0.1487 | 0.5366 | 0.4247 | 0.1523 | 0.5916 | `duyle2408/set_fcos_runs`, seed43, `best_coco_bbox_mAP_epoch_37.pth` |
| FCOS-SR-TOD | 0.4157 | 0.1467 | 0.5225 | 0.4616 | 0.1642 | 0.5954 | `duyle2408/srtod-set-mosaic-matrix-runs`, seed43, `best_coco_bbox_mAP_epoch_91.pth` |

The completed artifacts and verified upload are in `duyle2408/fcos-tinyperson-eval-20261015-runs` under `tinyperson/seed43/`, including per-model `final_results.json`, split `metrics.json`, and `upload_verified.json`.

### Complete merged TinyPerson seed mean ± sample std (2026-11-06)

All values below are source-image merged test metrics at NMS IoU `0.50`, averaged over training seeds `42` and `43`. `AP50-small` is the merged IoU=0.50 small-object field.

| Model family | AP50 | mAP50-95 | merged AP50-small |
|---|---:|---:|---:|
| STW-YOLO | 0.5924 ± 0.0046 | 0.2243 ± 0.0003 | 0.7050 ± 0.0065 |
| TPH-YOLOv5 | 0.5247 ± 0.0321 | 0.1945 ± 0.0136 | 0.6045 ± 0.0459 |
| FCOS-SET | 0.4253 ± 0.0009 | 0.1521 ± 0.0003 | 0.5961 ± 0.0064 |
| FCOS-SR-TOD | 0.4583 ± 0.0047 | 0.1632 ± 0.0015 | 0.5949 ± 0.0007 |

Per-seed inputs and reproducible evaluator: `docs/reports/sota_results_reproducible_20261106.json`, `evaluate_test/evaluate_sota_models.py`.

- The main table uses split-qualified native evaluation artifacts for STW YOLO and TPH YOLO, and corrected evaluation-only MMDetection artifacts for FCOS-SR-TOD. AP50-small is explicitly the IoU=0.50, area=small result, not COCO AP-small averaged over IoU=0.50:0.95. TPH TinyPerson uses the official corner-window protocol. FCOS-SR-TOD LEVIR uses the full 584-image val/test split.

The compact table below records the fields that are present in the remote artifacts. TPH, FCOS-SET, and both seed42/seed43 STW rows are split-qualified; the seed42 STW values were backfilled locally from the existing HF checkpoints and the full bucket audit is recorded separately.

| Family | Model/variant | Dataset | Protocol | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Status | Source |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| TPH YOLO | TPH-YOLOv5 | levirship | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.7140 | 0.2590 | 0.6740 | 0.2370 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/levirship/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | levirship | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.7550 | 0.2880 | 0.7430 | 0.2690 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/levirship/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | tinyperson | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.5420 | 0.1920 | 0.5500 | 0.2010 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/tinyperson/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | tinyperson | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.4920 | 0.1690 | 0.4890 | 0.1750 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/tinyperson/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | varroa | upstream TPH-YOLOv5 yolov5l-xs-tph | 42 | 0.8500 | 0.2990 | 0.8270 | 0.2900 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/varroa/seed_42/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5 | varroa | upstream TPH-YOLOv5 yolov5l-xs-tph | 43 | 0.8410 | 0.3030 | 0.8390 | 0.3020 | complete | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`: `runs/varroa/seed_43/experiment_manifest.json` |
| TPH YOLO | TPH-YOLOv5n | varroa | custom TPH-YOLOv5 yolov5n-xs-tph | 43 | 0.8820 | 0.3080 | 0.8620 | 0.3080 | observed seed43; requested seed42 missing | `duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5n-runs`: `runs/varroa/seed_43/experiment_manifest.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | LEVIR-Ship | standard STW run | 42 | 0.8321 | 0.3325 | 0.7934 | 0.2861 | complete split-qualified backfill; native bucket metrics in the audit | `duyle2408/stw-yolo-runs`: `runs/levirship/seed_42/weights/best.pt` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | LEVIR-Ship | standard STW run | 43 | 0.8295 | 0.3385 | 0.8205 | 0.3208 | complete split-qualified backfill | `duyle2408/stw-yolo-standard-seed43-backfill-runs`: `runs/levirship/seed_43/evaluation_metrics.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | TinyPerson | standard STW run | 42 | 0.6190 | 0.2313 | 0.5684 | 0.2149 | complete split-qualified backfill; native bucket metrics in the audit | `duyle2408/stw-yolo-runs`: `runs/tinyperson/seed_42/weights/best.pt` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | TinyPerson | standard STW run | 43 | 0.6241 | 0.2394 | 0.5792 | 0.2204 | complete split-qualified backfill | `duyle2408/stw-yolo-standard-seed43-backfill-runs`: `runs/tinyperson/seed_43/evaluation_metrics.json` |
| STW YOLO | STW-YOLO p2_rp5_yolo12s | Varroa | standard STW run | 42 | 0.9166 | 0.3534 | 0.9166 | 0.3534 | aggregate only; no val/test labels | `duyle2408/stw-yolo-runs`: `runs/varroa/seed_42/evaluation_metrics.json` |
| FCOS-SET | FCOS_set STW protocol | LEVIR-Ship | no_mosaic | 42 | 0.7324 | 0.2598 | 0.7740 | 0.2785 | complete | `duyle2408/set-fcos-stw-protocol-runs`: `levirship/no_mosaic/patience15/seed42/fcos_set/final_results.json` |
| FCOS-SET | FCOS_set STW protocol | TinyPerson | mosaic | 42 | 0.4148 | 0.1494 | 0.4259 | 0.1519 | complete | `duyle2408/set-fcos-stw-protocol-runs`: `tinyperson/mosaic/patience15/seed42/fcos_set/final_results.json` |
| FCOS-SET | FCOS_set YOLO protocol final | Varroa | yolo_protocol | 42 | -- | -- | 0.8720 | 0.3080 | observed seed42; target seed43 not found | `duyle2408/set_fcos_runs`: `varroa_final/fcos_set/base/test_results/20260919_082947/20260919_082947.json` |
| FCOS-SR-TOD | FCOS-SR-TOD requested seed43 | Varroa | standard SRTOD | 43 | -- | -- | -- | -- | checkpoint/train/test artifacts; split metrics missing | `duyle2408/srtod-set-mosaic-matrix-runs`: `srtod/varroa/fcos/seed43/` |


## 3. Historical requested-seed note

The requested-seed notes below are retained for provenance. They are not the current run queue. FCOS-SR-TOD seed42 for LEVIR-Ship and TinyPerson was completed on 2026-11-05; its matched standard-Mosaic metrics and verified HF upload are recorded in Section 1.1 and the main table.

| Family | Requested seed | Note |
|---|---:|---|
| STW YOLO | 43 | Standard seed43 LEVIR-Ship and TinyPerson backfill is complete and split-qualified; retain seed42 aggregate rows separately. |
| TPH-YOLOv5n | 42 | Existing seed43 result remains historical. Run a new seed42 experiment. |
| FCOS-SET | 43 | Existing seed42 results remain historical. Run a new seed43 experiment. |
| FCOS-SR-TOD | 43 | Seed43 artifacts and the 2026-11-04 recovery2 rerun are complete/uploaded. The matched seed42 LEVIR-Ship/TinyPerson backfill is also complete/uploaded; no FCOS-SR-TOD seed42 gap remains for those datasets. |

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

- The older SRTOD artifacts include Varroa, LEVIR-Ship, and TinyPerson train/checkpoint/test material under `duyle2408/srtod-set-mosaic-matrix-runs`.
- Those older artifacts do not provide complete split-qualified metric files for all rows and remain historical provenance only.
- The matched standard-Mosaic seed42 LEVIR-Ship/TinyPerson backfill is a separate completed artifact set, documented in Section 1.1; it must not be confused with the older rows.

## 5. Protocol cautions

- These four families are variants/SOTA comparisons, not the canonical MMDetection baseline control.
- Preserve each family's own model/config, image size, optimizer, epochs, patience, split seed, training seed, and NMS IoU.
- Do not compare aggregate STW metrics directly against split-qualified TPH/FCOS-SET metrics without marking the evaluation protocol.
- VisDrone is excluded from this report by scope, including any paths containing `visdrone`.

## 6. Historical future-run note

The following queue note is retained only to explain the original execution plan. It is not evidence that the runs are still pending. FCOS-SR-TOD seed42 LEVIR-Ship/TinyPerson is complete and must not be queued again.

| Model family | Training seed | Scope | Current status |
|---|---:|---|---|
| TPH-YOLOv5n | 42 | Varroa, LEVIR-Ship, TinyPerson as applicable | Historical requested run; verify separately before claiming complete. |
| FCOS-SET | 43 | Varroa, LEVIR-Ship, TinyPerson as applicable | Historical requested run; verify separately before claiming complete. |
| FCOS-SR-TOD | 43 | Varroa, LEVIR-Ship, TinyPerson as applicable | Seed43 complete/uploaded; seed42 LEVIR-Ship/TinyPerson also complete/uploaded. |

STW YOLO seed43 is backfilled for LEVIR-Ship and TinyPerson with split-qualified native val/test metrics. Do not merge it with the older seed42 aggregate-only rows.

## 7. Reproducible evaluation files

The evaluation contract and exact checkpoint/artifact mapping are recorded in:

- `evaluate_test/sota_evaluation_manifest.json`
- `docs/sota_evaluation_reproducible.md`
- `docs/reports/sota_results_reproducible_20261106.json`

These files are evaluation-only: they do not train or mutate weights. They keep native `test_size/AP50-Small` separate from official TinyPerson `test_merged/AP50-Small`; an absent merged scalar remains explicitly unavailable rather than inferred. The existing HF artifacts are the source of truth for previously completed runs.
