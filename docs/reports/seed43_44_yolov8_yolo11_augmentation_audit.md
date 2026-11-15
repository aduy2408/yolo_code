# YOLOv8/YOLO11 Augmentation Audit — Seeds 43/44

**Updated:** 2026-11-04  
**Scope:** OACP R2 và Negative Copy-Paste R2/R4 trên các artifact seed 43/44.  
**Primary metric:** `test/mAP50-95`. TinyPerson `test_merged/*` là protocol merged riêng và không thay thế native `test/*`.

## Sources

- OACP: [`duyle2408/levirship-tinyperson-yolov8-yolo11-oacp-seed43-44-runs`](https://huggingface.co/datasets/duyle2408/levirship-tinyperson-yolov8-yolo11-oacp-seed43-44-runs)
- Negative Copy-Paste: [`duyle2408/levirship-tinyperson-yolov8-yolo11-copy-paste-nc43-44-runs`](https://huggingface.co/datasets/duyle2408/levirship-tinyperson-yolov8-yolo11-copy-paste-nc43-44-runs)
- OACP LEVIR size backfill: [`duyle2408/oacp-r2-seed43-44-size-eval-runs`](https://huggingface.co/datasets/duyle2408/oacp-r2-seed43-44-size-eval-runs)

## Fixed protocol

- Training seeds: `43`, `44`
- Split seed: `42`
- Epochs: `100`
- Patience: `0`
- Batch: `8`
- Workers: `8`
- NMS IoU: `0.50`
- LEVIR-Ship: `imgsz=512`, Mosaic off
- TinyPerson: `imgsz=640`, Mosaic on
- Pretrained: `yolov8n.pt` hoặc `yolo11n.pt`

## OACP R2 — parity rerun

Artifact source: [`duyle2408/oacp-r2-parity-rerun-seed43-44`](https://huggingface.co/datasets/duyle2408/oacp-r2-parity-rerun-seed43-44). Valid coverage is **8/8**: two models × two datasets × two seeds.

The protocol is:

- Training seeds: `43`, `44`
- Split seed: `42`
- Epochs: `100`
- Patience: `0`
- Batch: `8`
- Workers: `8`
- NMS IoU: `0.50`
- LEVIR-Ship: `imgsz=512`, Mosaic off
- TinyPerson: `imgsz=640`, Mosaic on
- OACP variant: `R2`

### Per-seed native validation/test metrics

Values are fractions; `—` means that the artifact does not contain the metric.

| Model | Dataset | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test merged AP50-Small | Test merged mAP50-95-Small |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 | LEVIR-Ship | 43 | 0.8040 | 0.3078 | 0.7942 | 0.2945 | — | — |
| YOLOv8 | LEVIR-Ship | 44 | 0.8112 | 0.3184 | 0.7957 | 0.2936 | — | — |
| YOLO11 | LEVIR-Ship | 43 | 0.8333 | 0.3339 | 0.8115 | 0.3033 | — | — |
| YOLO11 | LEVIR-Ship | 44 | 0.8210 | 0.3144 | 0.8145 | 0.3018 | — | — |
| YOLOv8 | TinyPerson | 43 | 0.5684 | 0.2037 | 0.5235 | 0.1882 | 0.6761 | 0.2707 |
| YOLOv8 | TinyPerson | 44 | 0.5390 | 0.2016 | 0.5153 | 0.1861 | 0.6828 | 0.2761 |
| YOLO11 | TinyPerson | 43 | 0.5475 | 0.1948 | 0.5186 | 0.1880 | 0.6750 | 0.2718 |
| YOLO11 | TinyPerson | 44 | 0.5471 | 0.1999 | 0.5215 | 0.1873 | 0.6861 | 0.2739 |

Seed-44 merged evaluation was rerun evaluation-only from the original checkpoints on the new Marimo server with `pycocotools` installed. Native and merged metrics are now available.

### Two-seed mean ± sample SD

| Model | Dataset | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test merged AP50-Small | Test merged mAP50-95-Small |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8 | LEVIR-Ship | 0.8076 ± 0.0051 | 0.3131 ± 0.0075 | 0.7949 ± 0.0011 | 0.2940 ± 0.0006 | — | — |
| YOLO11 | LEVIR-Ship | 0.8271 ± 0.0087 | 0.3241 ± 0.0138 | 0.8130 ± 0.0021 | 0.3026 ± 0.0010 | — | — |
| YOLOv8 | TinyPerson | 0.5537 ± 0.0207 | 0.2027 ± 0.0015 | 0.5194 ± 0.0058 | 0.1872 ± 0.0015 | 0.6795 ± 0.0047 | 0.2734 ± 0.0038 |
| YOLO11 | TinyPerson | 0.5473 ± 0.0003 | 0.1973 ± 0.0036 | 0.5201 ± 0.0020 | 0.1877 ± 0.0005 | 0.6805 ± 0.0078 | 0.2729 ± 0.0015 |

Both seed-44 merged backfills are upload-verified in `duyle2408/oacp-r2-seed44-merged-eval`.

## Historical OACP size backfills

The previous size-evaluation backfills in `oacp-r2-seed43-44-size-eval-runs` and `oacp-r2-seed43-44-tiny-size-eval-runs` are retained as historical evaluation-only artifacts. They are not substituted into the parity-rerun table above.

## Negative Copy-Paste R2/R4 — LEVIR-Ship

| Model | Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8 | R2 | 43 | 0.7188 | 0.2572 | 0.6758 | 0.2419 |
| YOLOv8 | R2 | 44 | 0.7486 | 0.2773 | 0.6752 | 0.2481 |
| YOLOv8 | R4 | 43 | 0.7149 | 0.2536 | 0.7049 | 0.2511 |
| YOLOv8 | R4 | 44 | 0.7427 | 0.2635 | 0.6816 | 0.2419 |
| YOLO11 | R2 | 43 | 0.6842 | 0.2488 | 0.6751 | 0.2289 |
| YOLO11 | R4 | 43 | 0.6863 | 0.2530 | 0.6475 | 0.2300 |

## Negative Copy-Paste R2/R4 — TinyPerson

| Model | Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test merged mAP50-95 | Test merged AP50-Small |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 | R2 | 43 | 0.5189 | 0.1816 | 0.5049 | 0.1780 | 0.1866 | 0.6684 |
| YOLOv8 | R4 | 43 | 0.5139 | 0.1866 | 0.4975 | 0.1761 | 0.1846 | 0.6602 |
| YOLO11 | R2 | 43 | 0.5111 | 0.1885 | 0.5014 | 0.1778 | 0.1870 | 0.6678 |
| YOLO11 | R4 | 43 | 0.5159 | 0.1880 | 0.4913 | 0.1734 | 0.1820 | 0.6645 |

## Corrected YOLO11 baseline-parity rerun

The earlier YOLO11 seed43 rows above used `augmentation_matrix_current`
(`scale=0.0`, `translate=0.0`) and were not compatible with the seed42
protocol. Seed43 was rerun with `baseline_no_mosaic_parity`
(`scale=0.5`, `translate=0.1`), matching the seed42 workflow. Seed44
parity artifacts were retained. The corrected two-seed values are:

| Model | Dataset | Variant | Val AP50 | Val AP50-Small | Val mAP50-95 | Test AP50 | Test AP50-Small | Test mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| YOLO11 | LEVIR-Ship | NC-R2 | 0.823769 ± 0.001031 | 0.822394 ± 0.023921 | 0.325032 ± 0.002732 | 0.805516 ± 0.017653 | 0.831259 ± 0.045471 | 0.296434 ± 0.001918 |
| YOLO11 | LEVIR-Ship | NC-R4 | 0.825049 ± 0.002182 | 0.821418 ± 0.010948 | 0.327918 ± 0.004850 | 0.792255 ± 0.001478 | 0.799825 ± 0.009683 | 0.298528 ± 0.006432 |
| YOLO11 | TinyPerson | NC-R2 | 0.542617 ± 0.005962 | 0.510842 ± 0.001126 | 0.198432 ± 0.001516 | 0.518101 ± 0.006867 | 0.527061 ± 0.009468 | 0.186621 ± 0.002914 |
| YOLO11 | TinyPerson | NC-R4 | 0.545606 ± 0.008045 | 0.487605 ± 0.017529 | 0.199907 ± 0.001275 | 0.519385 ± 0.001113 | 0.524395 ± 0.002154 | 0.187623 ± 0.000963 |

TinyPerson corrected merged `test_merged/AP50-Small` is `0.685662 ±
0.009521` for NC-R2 and `0.684272 ± 0.001449` for NC-R4. The corrected
seed43 training artifacts are in
`duyle2408/yolo11-nc-seed43-baseline-parity-levir-runs`,
`duyle2408/yolo11-nc-seed43-baseline-parity-tinyperson-r2-runs`, and
`duyle2408/yolo11-nc-seed43-baseline-parity-tinyperson-r4-runs`;
their bucket backfills are in the corresponding `*-size-eval` repositories.

The previous YOLO11 rows remain as historical artifact values but MUST NOT
be combined with seed44 parity values for a multiseed claim.

## Coverage and interpretation

- OACP R2: core metrics are complete in the inspected seed43/44 repository for YOLOv8/YOLO11 on LEVIR-Ship and TinyPerson: **8/8**. Native LEVIR size metrics are now complete via the separate backfill repository; OACP TinyPerson native validation size metrics remain unavailable in the inspected source artifacts.
- The first Negative Copy-Paste repository inspected was incomplete; its **10/16** count must not be treated as the global run count.
- Cross-repository HF audit found additional completed NegativeCanvas artifacts:
  - `duyle2408/yolov8-cp-nc-seed43-44-runs`: YOLOv8 R2/R4, LEVIR/TinyPerson/Varroa, seeds 43/44;
  - `duyle2408/yolov8-cp-nc-recovery-tiny-seed43-r2-runs`: YOLOv8 TinyPerson R2 seed 43;
  - `duyle2408/yolov8-cp-nc-recovery-tiny-seed44-r2-runs`: YOLOv8 TinyPerson R2 seed 44;
  - `duyle2408/levirship-tinyperson-yolov8-yolo11-copy-paste-nc43-44-runs`: YOLO11 seed 43 on LEVIR/TinyPerson;
  - `duyle2408/yolov9-yolo11-cp-nc-seed44-runs`: YOLO11 seed 44 on LEVIR/TinyPerson.
- Therefore, YOLOv8 NegativeCanvas R2/R4 has cross-repository coverage for the three datasets, and YOLO11 NegativeCanvas R2/R4 has both seeds for LEVIR/TinyPerson. YOLO11 Varroa is not complete in the artifacts checked.
- The duplicate YOLOv8 LEVIR/TinyPerson artifacts have identical high-level manifests but different checkpoint hashes and metrics. They are separate runs, not interchangeable replicas. A canonical-source selection is required before combining them into one mean/std table.
- No Varroa OACP R2 run is included in the inspected seed43/44 OACP repository.
- YOLOv8 LEVIR NegativeCanvas has opposite per-seed ranking within each artifact family; do not claim a robust R2/R4 winner until the canonical duplicate is selected.
- TinyPerson headline comparisons should use `test_merged/mAP50-95` and `test_merged/AP50-Small`; native `test/*` values remain separate.
- Every inspected leaf contains the expected completion artifacts, including `evaluation_metrics.json`, `experiment_manifest.json`, and `upload_complete.json`.
- The manifests record protocol, seeds, and split seed, but do not consistently record the source Git SHA/model YAML. Exact source-code reproducibility is therefore not fully proven.
