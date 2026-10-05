# MMDetection Baseline Runs: Varroa, LEVIR-Ship, TinyPerson

**Updated:** 2026-10-05  
**Base report:** `yolo_related/docs/reports/complete_main_report.md`  
**Scope:** YOLO-matched MMDetection baseline runs only. SOTA and architecture variants are excluded from the main matrix.  
**Metric policy:** values are copied from remote Hugging Face artifacts without inference. `--` means the requested field is absent from the source artifact and should be evaluated or backfilled on the server.

## 1. Executive status

| Dataset | HF repository | Verified jobs | Protocols | Seeds | Metric status |
|---|---|---:|---|---|---|
| Varroa | `duyle2408/varroa_mmdet_yolo_protocol_runs` | 24 | `no_mosaic`, `mosaic` | 42, 43 | Test metrics present. Validation metrics are absent from the uploaded aggregate artifact. |
| LEVIR-Ship | `duyle2408/levirship_mmdet_yolo_protocol_seed42_runs` | 12 planned baseline prefixes | `no_mosaic`, `mosaic` | 42 | Validation and test metrics are present for all standard rows except LEVIR no-mosaic RetinaNet, which has checkpoint/config/test artifacts but no `final_results.json`. |
| TinyPerson | `duyle2408/tinyperson_mmdet_yolo_protocol_seed42_runs` | 12 | `no_mosaic`, `mosaic` | 42 | Validation and test metrics are present for all six models. |

The remote configs inspected for the YOLO-matched matrix record **MuSGD, 100 epochs, early-stop patience 15, batch size 8, split seed 42, and NMS IoU 0.5**. Varroa records image size 640x640. LEVIR-Ship and TinyPerson retain their dataset-specific baseline resize/window protocols.

## 2. Complete validation/test metric table

Columns are AP50, AP75, mAP50-95, AP50-Small, and TinyPerson AP50-Tiny1/2/3, split by validation and test. TinyPerson Tiny1/2/3 are dataset-specific metrics. `--` is not a zero.

| Dataset | Protocol | Seed | Model | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Val T1 | Val T2 | Val T3 | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Test T1 | Test T2 | Test T3 | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LEVIR-Ship | no_mosaic | 42 | fcos | 0.7227 | 0.0945 | 0.2487 | 0.7222 | -- | -- | -- | 0.7602 | 0.0988 | 0.2872 | 0.7583 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | faster_rcnn | 0.7420 | 0.1318 | 0.2846 | 0.7459 | -- | -- | -- | 0.7344 | 0.1297 | 0.2749 | 0.7331 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | cascade_rcnn | 0.7265 | 0.0957 | 0.2568 | 0.7316 | -- | -- | -- | 0.6556 | 0.0959 | 0.2310 | 0.6512 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | rtmdet | 0.5875 | 0.1378 | 0.2435 | 0.5889 | -- | -- | -- | 0.5804 | 0.0802 | 0.2162 | 0.5857 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | atss | 0.7370 | 0.1029 | 0.2611 | 0.7372 | -- | -- | -- | 0.7714 | 0.1002 | 0.2795 | 0.7679 | -- | -- | -- | complete |
| LEVIR-Ship | no_mosaic | 42 | retinanet | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | missing `final_results.json`; re-eval/finalize |
| LEVIR-Ship | mosaic | 42 | fcos | 0.7306 | 0.1109 | 0.2726 | 0.7336 | -- | -- | -- | 0.6991 | 0.0837 | 0.2455 | 0.6947 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | faster_rcnn | 0.7262 | 0.1275 | 0.2776 | 0.7299 | -- | -- | -- | 0.7073 | 0.1100 | 0.2560 | 0.7056 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | cascade_rcnn | 0.7596 | 0.1151 | 0.2850 | 0.7614 | -- | -- | -- | 0.6479 | 0.0968 | 0.2347 | 0.6450 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | rtmdet | 0.7186 | 0.1205 | 0.2805 | 0.7272 | -- | -- | -- | 0.6341 | 0.0897 | 0.2313 | 0.6402 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | atss | 0.7486 | 0.0954 | 0.2647 | 0.7476 | -- | -- | -- | 0.7556 | 0.1212 | 0.2866 | 0.7563 | -- | -- | -- | complete |
| LEVIR-Ship | mosaic | 42 | retinanet | 0.6990 | 0.1022 | 0.2511 | 0.6994 | -- | -- | -- | 0.7378 | 0.1215 | 0.2829 | 0.7391 | -- | -- | -- | complete |
| TinyPerson | no_mosaic | 42 | fcos | 0.4154 | 0.0608 | 0.1411 | 0.5449 | 0.1215 | 0.3972 | 0.4781 | 0.4305 | 0.0672 | 0.1484 | 0.5666 | 0.1079 | 0.3299 | 0.4979 | complete |
| TinyPerson | no_mosaic | 42 | faster_rcnn | 0.3092 | 0.0557 | 0.1105 | 0.5954 | 0.0000 | 0.1604 | 0.4499 | 0.3688 | 0.0682 | 0.1361 | 0.6278 | 0.0000 | 0.1326 | 0.4649 | complete |
| TinyPerson | no_mosaic | 42 | cascade_rcnn | 0.5106 | 0.0923 | 0.1903 | 0.6034 | 0.2788 | 0.5493 | 0.5235 | 0.5139 | 0.0909 | 0.1896 | 0.6025 | 0.2639 | 0.4869 | 0.5501 | complete |
| TinyPerson | no_mosaic | 42 | rtmdet | 0.5061 | 0.0873 | 0.1818 | 0.6124 | 0.2627 | 0.5265 | 0.5154 | 0.5056 | 0.0953 | 0.1844 | 0.6118 | 0.2103 | 0.4712 | 0.5607 | complete |
| TinyPerson | no_mosaic | 42 | atss | 0.4392 | 0.0706 | 0.1525 | 0.5617 | 0.1674 | 0.4519 | 0.4698 | 0.4413 | 0.0718 | 0.1553 | 0.5768 | 0.1351 | 0.3593 | 0.5008 | complete |
| TinyPerson | no_mosaic | 42 | retinanet | 0.4790 | 0.0549 | 0.1514 | 0.5530 | 0.2278 | 0.5630 | 0.4833 | 0.4668 | 0.0463 | 0.1497 | 0.5383 | 0.2172 | 0.4686 | 0.5203 | complete |
| TinyPerson | mosaic | 42 | fcos | 0.4060 | 0.0557 | 0.1394 | 0.5862 | 0.1267 | 0.3710 | 0.4143 | 0.4415 | 0.0693 | 0.1523 | 0.5981 | 0.1252 | 0.3395 | 0.5013 | complete |
| TinyPerson | mosaic | 42 | faster_rcnn | 0.2904 | 0.0515 | 0.1084 | 0.5822 | 0.0000 | 0.1467 | 0.3820 | 0.3526 | 0.0681 | 0.1329 | 0.6185 | 0.0000 | 0.1229 | 0.4326 | complete |
| TinyPerson | mosaic | 42 | cascade_rcnn | 0.5272 | 0.0979 | 0.1981 | 0.5981 | 0.2737 | 0.5932 | 0.5218 | 0.4810 | 0.0944 | 0.1820 | 0.5865 | 0.1991 | 0.4270 | 0.5238 | complete |
| TinyPerson | mosaic | 42 | rtmdet | 0.5054 | 0.0861 | 0.1837 | 0.5813 | 0.2520 | 0.5368 | 0.5091 | 0.5191 | 0.0980 | 0.1902 | 0.6348 | 0.2179 | 0.4693 | 0.5640 | complete |
| TinyPerson | mosaic | 42 | atss | 0.4298 | 0.0660 | 0.1500 | 0.5995 | 0.1571 | 0.4366 | 0.4949 | 0.4469 | 0.0707 | 0.1550 | 0.5984 | 0.1356 | 0.3639 | 0.5135 | complete |
| TinyPerson | mosaic | 42 | retinanet | 0.4515 | 0.0532 | 0.1484 | 0.5020 | 0.2273 | 0.5134 | 0.4677 | 0.4344 | 0.0478 | 0.1405 | 0.4980 | 0.2059 | 0.4146 | 0.4925 | complete |
| Varroa | no_mosaic | 42 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.8987 | 0.0997 | 0.3257 | 0.8875 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8929 | 0.0919 | 0.3241 | 0.8750 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8802 | 0.1006 | 0.3128 | 0.8534 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9047 | 0.1088 | 0.3387 | 0.8930 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8892 | 0.1021 | 0.3169 | 0.8547 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 42 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8819 | 0.1028 | 0.3268 | 0.8727 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.9133 | 0.0983 | 0.3257 | 0.8985 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8854 | 0.0910 | 0.3108 | 0.8676 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8774 | 0.0879 | 0.3085 | 0.8408 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9191 | 0.1193 | 0.3427 | 0.8986 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8793 | 0.0859 | 0.3086 | 0.8588 | -- | -- | -- | test complete; validation missing |
| Varroa | no_mosaic | 43 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8923 | 0.1036 | 0.3303 | 0.8673 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.8939 | 0.1089 | 0.3207 | 0.8671 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8886 | 0.1013 | 0.3216 | 0.8764 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8929 | 0.1084 | 0.3157 | 0.8709 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9050 | 0.1257 | 0.3447 | 0.8795 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8778 | 0.0932 | 0.3218 | 0.8434 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 42 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8938 | 0.1105 | 0.3316 | 0.8630 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | fcos | -- | -- | -- | -- | -- | -- | -- | 0.9007 | 0.1162 | 0.3247 | 0.8623 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | faster_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8857 | 0.1036 | 0.3198 | 0.8741 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | cascade_rcnn | -- | -- | -- | -- | -- | -- | -- | 0.8913 | 0.1134 | 0.3267 | 0.8743 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | rtmdet | -- | -- | -- | -- | -- | -- | -- | 0.9134 | 0.1166 | 0.3289 | 0.8991 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | atss | -- | -- | -- | -- | -- | -- | -- | 0.8927 | 0.0837 | 0.3132 | 0.8615 | -- | -- | -- | test complete; validation missing |
| Varroa | mosaic | 43 | retinanet | -- | -- | -- | -- | -- | -- | -- | 0.8954 | 0.0930 | 0.3352 | 0.8757 | -- | -- | -- | test complete; validation missing |

## 3. Missing metrics and server re-evaluation backlog

### 3.1 Varroa validation metrics

The Varroa remote repository contains complete training/test artifacts and a consolidated `evaluation/20260924/varroa_metrics.json`, but that aggregate artifact is test-only. All 24 Varroa baseline rows therefore have `--` for validation metrics. Run validation-only evaluation on the uploaded best checkpoints and upload one split-qualified metrics artifact per existing prefix.

| Protocols | Seeds | Models | Missing fields |
|---|---|---|---|
| `no_mosaic`, `mosaic` | 42, 43 | fcos, faster_rcnn, cascade_rcnn, rtmdet, atss, retinanet | Val AP50, AP75, mAP50-95, AP50-Small |

### 3.2 LEVIR-Ship RetinaNet gap

The LEVIR-Ship no-mosaic RetinaNet prefix contains a checkpoint, patched config, and test artifacts, but no split-qualified `final_results.json` was found. Mark this row as pending until the existing checkpoint is evaluated and the result is uploaded.

### 3.3 Other fields

- LEVIR-Ship: all other core validation/test fields are present.
- TinyPerson: all core validation/test fields are present, including Tiny1/Tiny2/Tiny3.
- Large-object AP is not included in the main table because the source schema exposes `ap_small` and `ap_medium`, while this report focuses on AP50, AP75, mAP50-95, and AP50-Small. The raw source artifacts remain available for additional fields.

## 4. FCOS-SET special baseline runs

These are separate FCOS-SET runs, not part of the six-model baseline matrix. They are included because they were recorded under the STW protocol repository.

Repository: `duyle2408/set-fcos-stw-protocol-runs`

| Dataset | Protocol | Seed | Model | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LEVIR-Ship | no_mosaic | 42 | fcos_set | 0.7324 | 0.1057 | 0.2598 | 0.7339 | 0.7740 | 0.0926 | 0.2785 | 0.7714 | complete |
| TinyPerson | mosaic | 42 | fcos_set | 0.4148 | 0.0704 | 0.1494 | 0.5516 | 0.4259 | 0.0742 | 0.1519 | 0.5722 | complete |

## 5. Provenance and acceptance evidence

- Varroa manifests: `experiment_manifest.json`, `job_summary.json`, `patched_config.py`, checkpoints, and test results exist under each remote prefix.
- LEVIR-Ship and TinyPerson: `final_results.json`, `patched_config.py`, validation/test metrics, checkpoints, and prediction artifacts exist under the baseline prefixes, except the noted LEVIR no-mosaic RetinaNet `final_results.json` gap.
- Baseline matrix model registry: `fcos`, `faster_rcnn`, `cascade_rcnn`, `rtmdet`, `atss`, `retinanet`.
- Remote repositories were queried through authenticated Hugging Face Hub API on 2026-10-05. The token value is intentionally not recorded in this report.
- Local `mmdetection/work_dirs` is not the completion source of truth. The remote HF artifacts are the authoritative run evidence.

## 6. Recommended server work

1. Re-evaluate the 24 Varroa best checkpoints on validation with split seed 42 and the original dataset-specific protocol.
2. Evaluate the existing LEVIR-Ship no-mosaic RetinaNet checkpoint and upload its split-qualified `final_results.json`.
3. Upload validation metrics under the existing Varroa prefixes without overwriting the original test metrics.
4. Preserve checkpoint, source commit, model, protocol, seed, NMS IoU, and image-size metadata.
5. After upload, replace only the relevant `--` fields in this report with values from verified remote artifacts.

## 7. Excluded repositories

Repositories containing DynamicVis, PH-DETR, DGFE/API, SRTOD, DINO/DETR, or other architecture and augmentation variants are excluded from the baseline tables unless explicitly listed above. They should be reported separately as SOTA or variant runs to avoid mixing control and treatment experiments.
