# DETR-R18 and RT-DETR-R18 matrix results across four datasets

**Source:** [`duyle2408/detr_r18_rtdetr_r18_matrix_runs`](https://huggingface.co/datasets/duyle2408/detr_r18_rtdetr_r18_matrix_runs)

This report consolidates the matrix repository runs for **LEVIR-Ship, TinyPerson, Varroa, and VisDrone2019-DET**. Metrics are taken from the repository's uploaded `final_results.json`, `metrics.json`, or recorded MMDetection test JSON logs. Aggregate rows are arithmetic means over the available training seeds. They are not pooled COCO evaluations.

## Common protocol

- Split seed: `42`
- Training seeds: `42` and `43`, except where an artifact is missing
- Batch size: `8`
- Maximum epochs: `100`
- Patience: `15`
- AMP: off
- NMS IoU: `0.5`
- DETR config family: `detr_r18_8xb2-500e_coco.py`
- RT-DETR config family: `rtdetr_r18vd_8xb2-72e_coco.py`
- Source commit for the main matrix: `e069bcee9adba52fda5bf19f71ea0087b6b021e5`
- RT-DETR implementation commit: `66365c1553ffd121ca4ee2be9d091735faf3c182`

Image size and augmentation are dataset-specific in the manifests. LEVIR-Ship uses its native protocol, TinyPerson uses `640` native tiles with Mosaic, Varroa uses `640` with the recorded no-Mosaic variant, and VisDrone uses `1536×1536`.

## Aggregate results by dataset

Values are mean over the available seeds. `AP` means COCO AP or mAP50-95. `AP-small` and `AP-medium` use the source artifact's COCO area fields. A dash means the source artifact did not provide that field.

| Dataset | Model | n | Val AP | Val AP50 | Val AP75 | Val AP-small | Val AP-medium | Test AP | Test AP50 | Test AP75 | Test AP-small | Test AP-medium | Test AP-large |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | DETR-R18 | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | — |
| LEVIR-Ship | RT-DETR-R18 | 2 | 0.140738 | 0.379813 | 0.051126 | 0.142142 | 0.100317 | 0.121064 | 0.346799 | 0.045587 | 0.121345 | 0.142493 | — |
| TinyPerson | DETR-R18 | 2 | 0.000010 | 0.000069 | 0.000000 | 0.000000 | 0.000010 | 0.000000 | 0.000001 | 0.000000 | 0.000000 | 0.000000 | — |
| TinyPerson | RT-DETR-R18 | 2 | 0.196535 | 0.542484 | 0.096834 | 0.175609 | 0.315444 | 0.198731 | 0.543373 | 0.100272 | 0.183794 | 0.292956 | — |
| Varroa | DETR-R18 | 2 | — | — | — | — | — | 0.000500 | 0.002000 | 0.000000 | 0.000000 | 0.014000 | -1.000000 |
| Varroa | RT-DETR-R18 | 2 | — | — | — | — | — | 0.221094 | 0.770444 | 0.043468 | 0.208840 | 0.241332 | -1.000000 |
| VisDrone2019-DET | DETR-R18 | 2 | — | — | — | — | — | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| VisDrone2019-DET | RT-DETR-R18 | 2 | — | — | — | — | — | 0.000039 | 0.000139 | 0.000000 | 0.000000 | 0.000054 | 0.000016 |

## Per-seed details

### LEVIR-Ship

| Model | Seed | Val AP | Val AP50 | Val AP75 | Test AP | Test AP50 | Test AP75 | Test AP-small | Test AP-medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DETR-R18 | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| DETR-R18 | 43 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| RT-DETR-R18 | 42 | 0.281476 | 0.759624 | 0.102252 | 0.242074 | 0.693323 | 0.091174 | 0.242629 | 0.284983 |
| RT-DETR-R18 | 43 | 0.000000 | 0.000001 | 0.000000 | 0.000055 | 0.000275 | 0.000000 | 0.000062 | 0.000004 |

### TinyPerson

| Model | Seed | Val AP | Val AP50 | Val AP75 | Val AP-small | Test AP | Test AP50 | Test AP75 | Test AP-small | Test AP-medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DETR-R18 | 42 | 0.000012 | 0.000124 | 0.000000 | 0.000000 | 0.000000 | 0.000001 | 0.000000 | 0.000000 | 0.000000 |
| DETR-R18 | 43 | 0.000007 | 0.000013 | 0.000000 | 0.000000 | 0.000000 | 0.000001 | 0.000000 | 0.000000 | 0.000000 |
| RT-DETR-R18 | 42 | 0.196103 | 0.543565 | 0.098812 | 0.173183 | 0.199871 | 0.545406 | 0.100836 | 0.183680 | 0.275467 |
| RT-DETR-R18 | 43 | 0.196967 | 0.541402 | 0.094856 | 0.178034 | 0.197591 | 0.541340 | 0.099708 | 0.183907 | 0.270444 |

### Varroa

The Varroa matrix upload contains test JSON logs but no equivalent consolidated validation metric artifact. The `-1` large-object value is the COCO convention for an unavailable area bucket, not a measured negative AP.

| Model | Seed | Test AP | Test AP50 | Test AP75 | Test AP-small | Test AP-medium | Test AP-large |
|---|---:|---:|---:|---:|---:|---:|---:|
| DETR-R18 | 42 | 0.000000 | 0.001000 | 0.000000 | 0.000000 | 0.010000 | -1.000000 |
| DETR-R18 | 43 | 0.001000 | 0.003000 | 0.000000 | 0.000000 | 0.018000 | -1.000000 |
| RT-DETR-R18 | 42 | 0.224516 | 0.761552 | 0.050103 | 0.217442 | 0.237671 | -1.000000 |
| RT-DETR-R18 | 43 | 0.217672 | 0.779337 | 0.036834 | 0.200238 | 0.244993 | -1.000000 |

### VisDrone2019-DET

VisDrone uses the `1536×1536` matrix protocol. The two DETR logs record zero metrics. RT-DETR records near-zero metrics.

| Model | Seed | Test AP | Test AP50 | Test AP75 | Test AP-small | Test AP-medium | Test AP-large |
|---|---:|---:|---:|---:|---:|---:|---:|
| DETR-R18 | 42 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| DETR-R18 | 43 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| RT-DETR-R18 | 42 | 0.000004 | 0.000022 | 0.000000 | 0.000000 | 0.000007 | 0.000022 |
| RT-DETR-R18 | 43 | 0.000075 | 0.000256 | 0.000000 | 0.000000 | 0.000102 | 0.000010 |

## TPH-YOLOv5l-xs-tph cross-reference across four datasets

The intended TPH result is the upstream **TPH-YOLOv5l-xs-tph** model, not the vanilla YOLOv5n Mosaic row from the separate VisDrone repository. The multi-dataset repository records `variant: upstream TPH-YOLOv5 yolov5l-xs-tph` and uses NMS IoU `0.5`.

| Dataset | Seeds (n) | Image size | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | 42,43 (2) | 512 | 0.7345 | 0.2735 | 0.7085 | 0.2530 |
| TinyPerson | 42,43 (2) | 640 | 0.5170 | 0.1805 | 0.5195 | 0.1880 |
| Varroa | 42,43 (2) | 640 | 0.8455 | 0.3010 | 0.8330 | 0.2960 |
| VisDrone | 43 (1) | 1536 | 0.6120 | 0.3910 | 0.5040 | 0.3030 |

Source repository: [`duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs`](https://huggingface.co/datasets/duyle2408/varroa-tinyperson-levirship-visdrone-tph-yolov5-runs). The VisDrone TPH-YOLOv5l-xs-tph run currently has only seed 43. The repository's VisDrone manifest records the `tph_yolov5l-xs-tph_visdrone_recovery.yaml` model config, image size `1536`, batch size `8`, AMP enabled, Adam optimizer, split seed `42`, and training seed `43`.

| Dataset | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|
| LEVIR-Ship | 42 | 0.714 | 0.259 | 0.674 | 0.237 |
| LEVIR-Ship | 43 | 0.755 | 0.288 | 0.743 | 0.269 |
| TinyPerson | 42 | 0.542 | 0.192 | 0.550 | 0.201 |
| TinyPerson | 43 | 0.492 | 0.169 | 0.489 | 0.175 |
| Varroa | 42 | 0.850 | 0.299 | 0.827 | 0.290 |
| Varroa | 43 | 0.841 | 0.303 | 0.839 | 0.302 |
| VisDrone | 43 | 0.612 | 0.391 | 0.504 | 0.303 |

## Interpretation

- RT-DETR-R18 is substantially stronger than DETR-R18 on Varroa and TinyPerson in this matrix.
- LEVIR-Ship is unstable for RT-DETR-R18: seed 42 is strong, while seed 43 is effectively zero. The mean hides this failure mode, so the per-seed table should be cited.
- DETR-R18 is effectively non-functional in the uploaded LEVIR-Ship, TinyPerson, and VisDrone results. Varroa DETR is also near zero.
- VisDrone DETR/RT-DETR results should remain separate from the YOLO and five-model MMDetection baseline tables until prediction artifacts are independently re-evaluated. The values above are the metrics recorded by the uploaded test logs.
- These four datasets do not share identical image size or augmentation. Do not interpret the aggregate table as a cross-dataset leaderboard.

## Provenance

- HF repository: `duyle2408/detr_r18_rtdetr_r18_matrix_runs`
- Dataset prefixes: `detr_matrix/levirship`, `detr_matrix/tinyperson`, `detr_matrix/varroa/no_mosaic`, `detr_matrix/visdrone`
- VisDrone report cross-reference: [`visdrone_baseline_report.md`](visdrone_baseline_report.md)
- Matrix source commit: `e069bcee9adba52fda5bf19f71ea0087b6b021e5`
- RT-DETR source commit: `66365c1553ffd121ca4ee2be9d091735faf3c182`
