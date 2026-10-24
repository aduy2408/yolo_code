# Validation Results Across Three Datasets

**Updated:** 2026-10-24  
**Scope:** Validation-only comparison table for the baseline YOLO family, selected SOTA models, and MMDetection baselines on **LEVIR-Ship, Varroa, and TinyPerson**.  
**Metric policy:** values are fractions, not percentages. Every metric is explicitly validation-labeled.

## 1. Protocol alignment

The table in the thesis screenshot uses these protocol slices, which are preserved here for alignment with its test table:

| Dataset | Selected protocol | Baseline mode | MMDetection mode | SOTA protocol |
|---|---|---|---|---|
| LEVIR-Ship | Native held-out validation split | No Mosaic | `no_mosaic`, seed 42 | Source-specific split-qualified artifact |
| Varroa | Native validation split | Mosaic | `mosaic`, seed 42 | Source-specific split-qualified artifact |
| TinyPerson | Native corner-window validation | Mosaic | `mosaic`, seed 42 | Native corner-window validation; merged metrics are separate |

All values below use the following three validation metrics:

- `val/AP50`
- `val/mAP50-95`
- `val/AP50-Small`

`--` means that the source artifact did not expose a split-qualified validation AP50-Small value. It is not zero.

## 2. LEVIR-Ship validation

### 2.1 YOLO baseline

Source: `docs/reports/baseline_main.md`, no-Mosaic mean across available seeds.

| Model | Seeds | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| YOLOv5 | 3 | 0.8252 ± 0.0071 | 0.3175 ± 0.0065 | 0.8291 ± 0.0100 |
| YOLOv8 | 3 | 0.8319 ± 0.0087 | 0.3253 ± 0.0067 | 0.8117 ± 0.0221 |
| YOLOv9 | 3 | 0.8311 ± 0.0107 | 0.3284 ± 0.0098 | 0.8476 ± 0.0273 |
| YOLOv10 | 3 | 0.7883 ± 0.0087 | 0.3243 ± 0.0076 | 0.7181 ± 0.0368 |
| YOLO11 | 2 | 0.8373 ± 0.0074 | 0.3317 ± 0.0043 | 0.8443 ± 0.0034 |

### 2.2 SOTA

Source: `docs/reports/SOTA_main_report.md`. Where seed 42 and seed 43 both expose the same split-qualified validation field, values are mean ± sample standard deviation. Single-seed or partially covered fields remain explicitly marked with `n=1` or `--`.

| Family | Seeds / n | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| STW YOLO | 42 / 1 | 0.8370 | 0.3413 | 0.8340 |
| TPH YOLOv5 | 42, 43 / 2 | 0.7345 ± 0.0290 | 0.2735 ± 0.0205 | 0.7862 (n=1) |
| FCOS-SET | 42, 43 / 2 | 0.7490 ± 0.0235 | 0.2680 ± 0.0116 | 0.7511 ± 0.0243 |
| FCOS-SR-TOD | 43 / 1 | 0.7330 | 0.2670 | 0.7375 |

### 2.3 MMDetection baseline

Source: `docs/reports/mmdetection_baseline_runs_report.md` plus the completed supplement in `docs/reports/baseline_main.md`, Section 8. `no_mosaic`, aggregate across seeds 42 and 43 using mean ± sample standard deviation. Per-seed values remain in the source reports.

| Model | Seeds / n | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| FCOS | 42, 43 / 2 | 0.7139 ± 0.0124 | 0.2486 ± 0.0001 | 0.7132 ± 0.0127 |
| Faster R-CNN | 42, 43 / 2 | 0.7484 ± 0.0091 | 0.2857 ± 0.0015 | 0.7509 ± 0.0070 |
| Cascade R-CNN | 42, 43 / 2 | 0.7365 ± 0.0141 | 0.2683 ± 0.0163 | 0.7415 ± 0.0141 |
| RTMDet | 42, 43 / 2 | 0.6469 ± 0.0839 | 0.2660 ± 0.0317 | 0.6492 ± 0.0852 |
| ATSS | 42, 43 / 2 | 0.7318 ± 0.0074 | 0.2603 ± 0.0011 | 0.7330 ± 0.0059 |
| RetinaNet | 42, 43 / 2 | 0.6899 ± 0.0404 | 0.2445 ± 0.0185 | 0.6909 ± 0.0412 |

## 3. Varroa validation

### 3.1 YOLO baseline

Source: `docs/reports/baseline_main.md`, Mosaic mean across available seeds.

| Model | Seeds | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| YOLOv5 | 2 | 0.9225 ± 0.0141 | 0.3310 ± 0.0037 | 0.8961 ± 0.0115 |
| YOLOv8 | 2 | 0.9109 ± 0.0117 | 0.3287 ± 0.0037 | 0.8922 ± 0.0040 |
| YOLOv9 | 2 | 0.8967 ± 0.0007 | 0.3259 ± 0.0035 | 0.8825 ± 0.0231 |
| YOLOv10 | 2 | 0.8519 ± 0.0594 | 0.3273 ± 0.0159 | 0.7791 ± 0.0967 |
| YOLO11 | 2 | 0.9131 ± 0.0044 | 0.3341 ± 0.0072 | 0.8866 ± 0.0033 |

### 3.2 SOTA

Source: `docs/reports/SOTA_main_report.md`. Aggregates use mean ± sample standard deviation only when both seeds have the requested split-qualified field.

| Family | Seeds / n | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| STW YOLO | 42 / 1 | 0.9296 | 0.3491 | 0.8978 |
| TPH YOLOv5 | 42, 43 / 2 | 0.8445 ± 0.0049 | 0.2980 ± 0.0071 | 0.8486 (n=1) |
| FCOS-SET | 43 / 1 | 0.8950 | 0.3070 | 0.2820 |
| FCOS-SR-TOD | 43 / 1 | 0.9140 | 0.3120 | 0.8853 |

### 3.3 MMDetection baseline

Source: `docs/reports/baseline_main.md`, Section 8 MMDetection supplement, Mosaic, aggregate across seeds 42 and 43 using mean ± sample standard deviation. Per-seed values remain in the source report.

| Model | Seeds / n | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| FCOS | 42, 43 / 2 | 0.8885 ± 0.0019 | 0.3117 ± 0.0022 | 0.8609 ± 0.0193 |
| Faster R-CNN | 42, 43 / 2 | 0.9141 ± 0.0060 | 0.3184 ± 0.0092 | 0.8940 ± 0.0106 |
| Cascade R-CNN | 42, 43 / 2 | 0.9047 ± 0.0057 | 0.3160 ± 0.0077 | 0.8853 ± 0.0092 |
| RTMDet | 42, 43 / 2 | 0.9143 ± 0.0127 | 0.3369 ± 0.0041 | 0.9086 ± 0.0011 |
| ATSS | 42, 43 / 2 | 0.8796 ± 0.0005 | 0.3087 ± 0.0063 | 0.8525 ± 0.0096 |
| RetinaNet | 42, 43 / 2 | 0.8918 ± 0.0028 | 0.3207 ± 0.0004 | 0.8814 ± 0.0048 |

## 4. TinyPerson validation

### 4.1 YOLO baseline

Source: `docs/reports/baseline_main.md`, Mosaic mean across available seeds. These are native corner-window validation metrics, not merged-original-image metrics.

| Model | Seeds | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| YOLOv5 | 3 | 0.5770 ± 0.1043 | 0.2211 ± 0.0556 | 0.5747 ± 0.1516 |
| YOLOv8 | 3 | 0.6033 ± 0.0955 | 0.2395 ± 0.0563 | 0.5927 ± 0.1616 |
| YOLOv9 | 3 | 0.5684 ± 0.0566 | 0.2177 ± 0.0373 | 0.5635 ± 0.1027 |
| YOLOv10 | 3 | 0.5751 ± 0.0607 | 0.2315 ± 0.0407 | 0.5604 ± 0.1028 |
| YOLO11 | 2 | 0.5918 ± 0.0669 | 0.2248 ± 0.0378 | 0.5316 ± 0.0642 |

### 4.2 SOTA

Source: `docs/reports/SOTA_main_report.md`, native corner-window validation rows. Aggregates use mean ± sample standard deviation only when both seeds have the requested split-qualified field.

| Family | Seeds / n | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| STW YOLO | 42 / 1 | 0.6262 | 0.2393 | -- |
| TPH YOLOv5 | 42, 43 / 2 | 0.5170 ± 0.0354 | 0.1805 ± 0.0163 | -- |
| FCOS-SET | 42, 43 / 2 | 0.4116 ± 0.0045 | 0.1491 ± 0.0005 | 0.5441 ± 0.0106 |
| FCOS-SR-TOD | 43 / 1 | 0.4180 | 0.1500 | 0.3712 |

### 4.3 MMDetection baseline

Source: `docs/reports/mmdetection_baseline_runs_report.md` plus the completed supplement in `docs/reports/baseline_main.md`, Section 8. Mosaic, aggregate across seeds 42 and 43 using mean ± sample standard deviation. These are native corner-window validation metrics. Merged validation metrics are maintained separately in the TinyPerson MMDetection re-evaluation section.

| Model | Seeds / n | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| FCOS | 42, 43 / 2 | 0.4054 ± 0.0008 | 0.1374 ± 0.0028 | 0.5772 ± 0.0127 |
| Faster R-CNN | 42, 43 / 2 | 0.2976 ± 0.0103 | 0.1102 ± 0.0025 | 0.5862 ± 0.0057 |
| Cascade R-CNN | 42, 43 / 2 | 0.5222 ± 0.0071 | 0.1946 ± 0.0049 | 0.5698 ± 0.0400 |
| RTMDet | 42, 43 / 2 | 0.4982 ± 0.0102 | 0.1823 ± 0.0020 | 0.5719 ± 0.0133 |
| ATSS | 42, 43 / 2 | 0.4268 ± 0.0043 | 0.1490 ± 0.0014 | 0.5595 ± 0.0565 |
| RetinaNet | 42, 43 / 2 | 0.4506 ± 0.0013 | 0.1464 ± 0.0028 | 0.5055 ± 0.0050 |

### 4.4 TinyPerson merged-image test protocol

TinyPerson validation is reported on the native corner-window split and includes `val/AP50-Small` where available. Merging corner-window predictions back to original images is a **test-only** protocol. The latest verified seed42 STW/TPH merged test artifacts are:

| Model | `test_merged/AP50` | `test_merged/mAP50-95` | `test_merged/AP50-Small` | `test_merged/mAP50-95-Small` |
|---|---:|---:|---:|---:|
| STW YOLO | 0.5892 | 0.2241 | 0.7004 | 0.2903 |
| TPH YOLOv5 | 0.5474 | 0.2041 | 0.6370 | 0.2565 |

These are test-only merged metrics. They must not be copied into the native validation columns above. The evaluator uses the TinyPerson corner manifest, original-image ID remapping, NMS IoU `0.50`, and the project TinyBenchmark evaluator. The source artifacts are `tph_test_merged_predictions.json` / `merged_summary.json` for TPH and the corresponding STW merged artifacts on the Marimo evaluation runs.

## 5. SOTA seed coverage audit

The SOTA rows in Sections 2.2, 3.2, and 4.2 are **selected comparison rows**, not a complete seed sweep. The available seed coverage is:

| Family | Dataset | Available validation seeds | Validation coverage | Notes |
|---|---|---|---|---|
| STW YOLO | LEVIR-Ship | 42, 43 | `val/AP50`, `val/mAP50-95`, `val/AP50-Small` | Include both STW training seeds in the comparison inventory |
| STW YOLO | Varroa | 42, 43 | `val/AP50`, `val/mAP50-95`, `val/AP50-Small` | Include both STW training seeds in the comparison inventory |
| STW YOLO | TinyPerson | 42, 43 | `val/AP50`, `val/mAP50-95`; historical `val/AP50-Small` available separately | Include both STW training seeds; merged metrics use a separate protocol |
| TPH YOLOv5 | LEVIR-Ship | 42, 43 | `val/AP50`, `val/mAP50-95`; per-seed AP50-Small is absent in the native JSON | Both seeds are present in the TPH repository |
| TPH YOLOv5 | Varroa | 42, 43 | `val/AP50`, `val/mAP50-95`; per-seed AP50-Small is absent in the native JSON | Both seeds are present in the TPH repository |
| TPH YOLOv5 | TinyPerson | 42, 43 | `val/AP50`, `val/mAP50-95`; per-seed AP50-Small is absent in the native JSON | Both seeds are present; merged metrics are separate |
| FCOS-SET | LEVIR-Ship | 42, 43 | Split-qualified validation available for both selected artifacts | Seed42 is the historical STW-protocol artifact; seed43 is the selected rerun |
| FCOS-SET | Varroa | 42, 43 | Seed42 validation is incomplete; seed43 validation is available | Seed42 source exposes test metrics but not complete validation fields |
| FCOS-SET | TinyPerson | 42, 43 | Split-qualified validation available for both selected artifacts | TinyPerson merged metrics are separate |
| FCOS-SR-TOD | LEVIR-Ship | 43 | Split-qualified validation available | Seed43 is the selected/corrected full-split evaluation; older seed42 material is incomplete |
| FCOS-SR-TOD | Varroa | 43 | Split-qualified validation available | Seed43 evaluation-only rerun is complete; older seed42 material is incomplete |
| FCOS-SR-TOD | TinyPerson | 43 | Split-qualified validation available | Seed43 native and merged rerun artifacts are complete; older seed42 material is incomplete |

The important distinction is **seed coverage versus selected comparison row**. The inventory now includes STW seeds 42 and 43 for all three datasets, alongside the existing TPH, FCOS-SET, and FCOS-SR-TOD seed coverage. FCOS-SR-TOD uses seed43 for all three datasets, with the corrected TinyPerson merged rerun also using seed43. The one-row tables above remain compact selected slices, not multi-seed means.

## 6. Interpretation and limitations

1. The screenshot test table aligns with the selected protocol slices above: LEVIR no-Mosaic YOLO baselines, Varroa Mosaic YOLO baselines, and TinyPerson Mosaic YOLO baselines.
2. Values are not pooled across unlike protocols. In particular, TinyPerson native corner-window metrics must not be mixed with `test_merged/*` metrics.
3. SOTA rows use the source-specific protocol and seed stated in `SOTA_main_report.md`; they are not multi-seed means unless explicitly marked.
4. MMDetection validation rows for Varroa come from the completed Section 8 supplement in `baseline_main.md`, because the older aggregate source artifact was test-only.
5. The screenshot uses percentages. To convert a report value such as `0.7956` to the screenshot format, multiply by 100 to obtain `79.56`.

## 6. Source reports

- [`docs/reports/baseline_main.md`](baseline_main.md)
- [`docs/reports/all_datasets_baseline_validation_test.md`](all_datasets_baseline_validation_test.md)
- [`docs/reports/baseline_validation_test_recovered.md`](baseline_validation_test_recovered.md)
- [`docs/reports/SOTA_main_report.md`](SOTA_main_report.md)
- [`docs/reports/mmdetection_baseline_runs_report.md`](mmdetection_baseline_runs_report.md)
