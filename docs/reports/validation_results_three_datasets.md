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

Source: `docs/reports/SOTA_main_report.md`, selected split-qualified rows.

| Family | Seed | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| STW YOLO | 42 | 0.8370 | 0.3413 | 0.8340 |
| TPH YOLOv5 | 42 | 0.7140 | 0.2590 | 0.7862 |
| FCOS-SET | 43 | 0.7656 | 0.2762 | 0.7683 |
| FCOS-SR-TOD | 43 | 0.7330 | 0.2670 | 0.7375 |

### 2.3 MMDetection baseline

Source: `docs/reports/mmdetection_baseline_runs_report.md`, `no_mosaic`, seed 42. The RetinaNet row is the completed supplement recorded in `docs/reports/baseline_main.md`, Section 8.

| Model | Seed | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| FCOS | 42 | 0.7227 | 0.2487 | 0.7222 |
| Faster R-CNN | 42 | 0.7420 | 0.2846 | 0.7459 |
| Cascade R-CNN | 42 | 0.7265 | 0.2568 | 0.7316 |
| RTMDet | 42 | 0.5875 | 0.2435 | 0.5889 |
| ATSS | 42 | 0.7370 | 0.2611 | 0.7372 |
| RetinaNet | 42 | 0.6613 | 0.2314 | 0.6618 |

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

Source: `docs/reports/SOTA_main_report.md`, selected split-qualified rows.

| Family | Seed | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| STW YOLO | 42 | 0.9296 | 0.3491 | 0.8978 |
| TPH YOLOv5 | 42 | 0.8480 | 0.2930 | 0.8486 |
| FCOS-SET | 43 | 0.8950 | 0.3070 | 0.2820 |
| FCOS-SR-TOD | 43 | 0.9140 | 0.3120 | 0.8853 |

### 3.3 MMDetection baseline

Source: `docs/reports/baseline_main.md`, Section 8 MMDetection supplement, Mosaic, seed 42.

| Model | Seed | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| ATSS | 42 | 0.8792 | 0.3043 | 0.8457 |
| Cascade R-CNN | 42 | 0.9007 | 0.3105 | 0.8788 |
| Faster R-CNN | 42 | 0.9183 | 0.3249 | 0.9015 |
| FCOS | 42 | 0.8899 | 0.3102 | 0.8745 |
| RetinaNet | 42 | 0.8937 | 0.3205 | 0.8780 |
| RTMDet | 42 | 0.9053 | 0.3398 | 0.9094 |

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

Source: `docs/reports/SOTA_main_report.md`, native corner-window validation rows.

| Family | Seed | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| STW YOLO | 42 | 0.6262 | 0.2393 | -- |
| TPH YOLOv5 | 42 | 0.5420 | 0.1920 | -- |
| FCOS-SET | 43 | 0.4084 | 0.1487 | 0.5366 |
| FCOS-SR-TOD | 43 | 0.4180 | 0.1500 | 0.3712 |

### 4.3 MMDetection baseline

Source: `docs/reports/mmdetection_baseline_runs_report.md`, Mosaic, seed 42. These are native corner-window validation metrics. Merged validation metrics are maintained separately in the TinyPerson MMDetection re-evaluation section.

| Model | Seed | `val/AP50` | `val/mAP50-95` | `val/AP50-Small` |
|---|---:|---:|---:|---:|
| FCOS | 42 | 0.4060 | 0.1394 | 0.5862 |
| Faster R-CNN | 42 | 0.2904 | 0.1084 | 0.5822 |
| Cascade R-CNN | 42 | 0.5272 | 0.1981 | 0.5981 |
| RTMDet | 42 | 0.5054 | 0.1837 | 0.5813 |
| ATSS | 42 | 0.4298 | 0.1500 | 0.5995 |
| RetinaNet | 42 | 0.4515 | 0.1484 | 0.5020 |

## 5. SOTA seed coverage audit

The SOTA rows in Sections 2.2, 3.2, and 4.2 are **selected comparison rows**, not a complete seed sweep. The available seed coverage is:

| Family | Dataset | Available validation seeds | Validation coverage | Notes |
|---|---|---|---|---|
| STW YOLO | LEVIR-Ship | 42 | `val/AP50`, `val/mAP50-95`, `val/AP50-Small` | Standard STW artifact is seed42; other seed paths in the repository are Copy-Paste/variant branches, not the same standard STW row |
| STW YOLO | Varroa | 42 | `val/AP50`, `val/mAP50-95`, `val/AP50-Small` | Standard STW artifact is seed42 |
| STW YOLO | TinyPerson | 42 | `val/AP50`, `val/mAP50-95`; historical `val/AP50-Small` available separately | Standard STW artifact is seed42; merged metrics use a separate protocol |
| TPH YOLOv5 | LEVIR-Ship | 42, 43 | `val/AP50`, `val/mAP50-95`; per-seed AP50-Small is absent in the native JSON | Both seeds are present in the TPH repository |
| TPH YOLOv5 | Varroa | 42, 43 | `val/AP50`, `val/mAP50-95`; per-seed AP50-Small is absent in the native JSON | Both seeds are present in the TPH repository |
| TPH YOLOv5 | TinyPerson | 42, 43 | `val/AP50`, `val/mAP50-95`; per-seed AP50-Small is absent in the native JSON | Both seeds are present; merged metrics are separate |
| FCOS-SET | LEVIR-Ship | 42, 43 | Split-qualified validation available for both selected artifacts | Seed42 is the historical STW-protocol artifact; seed43 is the selected rerun |
| FCOS-SET | Varroa | 42, 43 | Seed42 validation is incomplete; seed43 validation is available | Seed42 source exposes test metrics but not complete validation fields |
| FCOS-SET | TinyPerson | 42, 43 | Split-qualified validation available for both selected artifacts | TinyPerson merged metrics are separate |
| FCOS-SR-TOD | LEVIR-Ship | 43 | Split-qualified validation available | Older seed42 artifact set is incomplete |
| FCOS-SR-TOD | Varroa | 43 | Split-qualified validation available | Older seed42 artifact set is incomplete |
| FCOS-SR-TOD | TinyPerson | 43 | Split-qualified validation available | Older seed42 artifact set is incomplete |

Therefore, the selected one-row-per-family tables above must not be read as claiming that every SOTA family has only one seed. They are a compact comparison slice. The complete seed inventory is constrained by the artifacts actually exposing split-qualified validation fields.

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
