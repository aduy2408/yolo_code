# OACP R2 rerun — seed 43/44 results

- Protocol: OACP R2; LEVIR-Ship uses no mosaic; TinyPerson uses mosaic only.
- Models: YOLOv8n and YOLO11n.
- Seeds: 43, 44; split seed: 42.
- `nms_iou: 0.5`.
- Source commit: `bde5b46393c579934f7ac5f24a96c6cdd90020e4`.
- HF dataset: `duyle2408/oacp-r2-parity-rerun-seed43-44`.

## Per-seed metrics (percent)

| Model | Dataset | Seed | Val AP50 | Val mAP50:95 | Test AP50 | Test mAP50:95 | Test AP50 Small | Test mAP50:95 Small |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOV11 | levir | 43 | 83.33 | 33.39 | 81.15 | 30.33 | - | - |
| YOLOV11 | levir | 44 | 82.10 | 31.44 | 81.45 | 30.18 | - | - |
| YOLOV11 | tinyperson | 43 | 54.75 | 19.48 | 51.86 | 18.80 | 67.50 | 27.18 |
| YOLOV11 | tinyperson | 44 | 54.71 | 19.99 | 52.15 | 18.73 | 68.61 | 27.39 |
| YOLOV8 | levir | 43 | 80.40 | 30.78 | 79.42 | 29.45 | - | - |
| YOLOV8 | levir | 44 | 81.12 | 31.84 | 79.57 | 29.36 | - | - |
| YOLOV8 | tinyperson | 43 | 56.84 | 20.37 | 52.35 | 18.82 | 67.61 | 27.07 |
| YOLOV8 | tinyperson | 44 | 53.90 | 20.16 | 51.53 | 18.61 | 68.28 | 27.61 |

## Mean ± std across seeds 43/44 (percent)

| Model | Dataset | Val AP50 | Val mAP50:95 | Test AP50 | Test mAP50:95 | Test AP50 Small | Test mAP50:95 Small |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOV8 | levir | 80.76 ± 0.51 | 31.31 ± 0.75 | 79.49 ± 0.11 | 29.40 ± 0.06 | - | - |
| YOLOV11 | levir | 82.71 ± 0.87 | 32.41 ± 1.38 | 81.30 ± 0.21 | 30.26 ± 0.10 | - | - |
| YOLOV8 | tinyperson | 55.37 ± 2.07 | 20.27 ± 0.15 | 51.94 ± 0.58 | 18.72 ± 0.15 | 67.95 ± 0.47 | 27.34 ± 0.38 |
| YOLOV11 | tinyperson | 54.73 ± 0.03 | 19.73 ± 0.36 | 52.01 ± 0.20 | 18.77 ± 0.05 | 68.05 ± 0.78 | 27.29 ± 0.15 |
## Consolidated full metrics table

Core metrics are mean ± sample standard deviation across training seeds 43/44 from the parity rerun; values in this table are fractions. LEVIR `AP50-Small` uses the native size-evaluation backfill. TinyPerson `test_merged/AP50-Small` and `test_merged/mAP50-95-Small` now have two-seed evaluation-only backfill for seed44 with `pycocotools`.

| Model | Dataset | Val AP50 | Val AP50-Small | Val mAP50-95 | Test AP50 | Test AP50-Small | Test mAP50-95 | Test mAP50-95-Small | Small metric protocol |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| YOLOv8 | LEVIR-Ship | 0.8076 ± 0.0051 | 0.7790 ± 0.0280 | 0.3131 ± 0.0075 | 0.7949 ± 0.0011 | 0.7880 ± 0.0098 | 0.2940 ± 0.0006 | — | Native `test_size/AP50-Small` |
| YOLO11 | LEVIR-Ship | 0.8271 ± 0.0087 | 0.7696 ± 0.0250 | 0.3241 ± 0.0138 | 0.8130 ± 0.0021 | 0.7407 ± 0.0372 | 0.3026 ± 0.0010 | — | Native `test_size/AP50-Small` |
| YOLOv8 | TinyPerson | 0.5537 ± 0.0207 | 0.5432 ± 0.0111 | 0.2027 ± 0.0015 | 0.5194 ± 0.0058 | 0.6795 ± 0.0047 | 0.1872 ± 0.0015 | 0.2734 ± 0.0038 | Official merged `test_merged/*-Small` |
| YOLO11 | TinyPerson | 0.5473 ± 0.0003 | 0.4937 ± 0.0129 | 0.1973 ± 0.0036 | 0.5201 ± 0.0020 | 0.6805 ± 0.0078 | 0.1877 ± 0.0005 | 0.2729 ± 0.0015 | Official merged `test_merged/*-Small` |

The seed-44 merged evaluation was rerun evaluation-only from the original checkpoints on a Marimo server with `pycocotools` installed. The backfill artifacts are in `duyle2408/oacp-r2-seed44-merged-eval`, with both model prefixes upload-verified.


## Audit notes

- All 8 required model × dataset × seed artifacts were found in HF, with `evaluation_metrics.json` and `upload_complete.json` for each valid row.
- The initial TinyPerson seed 44 training artifacts had `test_merged/available = 0` because `pycocotools` was absent at training-time evaluation. The separate bucket backfill reran both native `val`/`test` buckets and merged `test` evaluation with `pycocotools`, so the bucket section below is complete for all four TinyPerson runs.
- TinyPerson no-mosaic artifacts were excluded because they are outside the agreed protocol.
- Bucket evaluation artifacts are uploaded and verified in the two size-evaluation HF repositories listed below.

## Bucket metrics — native validation/test and TinyPerson merged test

Bucket evaluation was rerun from each `weights/best.pt` checkpoint with the project evaluator:

- Native buckets: Tiny1 (`1–8 px²`), Tiny2 (`8–12 px²`), Tiny3 (`12–20 px²`), Small (`20–32 px²`), Medium (`32–96 px²`).
- IoU thresholds: `0.50:0.05:0.75`; NMS IoU: `0.50`.
- Values below are `AP50` bucket metrics, in percent, mean ± sample std across seeds 43/44.
- `—` means the evaluator returned no eligible ground-truth objects in both runs. `0.00` is a valid measured value.

| Model | Dataset | Val Tiny1 | Val Tiny2 | Val Tiny3 | Val Small | Val Medium | Test Tiny1 | Test Tiny2 | Test Tiny3 | Test Small | Test Medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 | LEVIR-Ship | — | 18.81 ± 4.20 | 63.04 ± 1.18 | 82.67 ± 0.18 | 74.70 ± 1.56 | 0.00 ± 0.00 | 22.61 ± 0.23 | 68.90 ± 2.42 | 82.67 ± 1.75 | 59.77 ± 3.78 |
| YOLOv11 | LEVIR-Ship | — | 16.83 ± 1.40 | 63.14 ± 0.04 | 81.25 ± 0.36 | 77.45 ± 3.89 | 0.00 ± 0.00 | 18.32 ± 2.10 | 64.82 ± 0.24 | 81.10 ± 1.28 | 73.34 ± 0.69 |
| YOLOv8 | TinyPerson | 26.93 ± 0.59 | 50.03 ± 2.00 | 53.69 ± 0.98 | 53.05 ± 3.38 | 61.51 ± 4.49 | 21.00 ± 0.17 | 43.67 ± 1.41 | 49.44 ± 1.02 | 52.46 ± 0.23 | 53.87 ± 1.48 |
| YOLOv11 | TinyPerson | 26.23 ± 1.42 | 48.52 ± 0.38 | 50.61 ± 0.93 | 48.82 ± 0.31 | 63.33 ± 0.23 | 21.32 ± 0.33 | 42.24 ± 0.43 | 48.00 ± 0.87 | 50.88 ± 1.19 | 54.42 ± 0.30 |

### TinyPerson merged test buckets

The merged columns use the official TinyPerson corner-window evaluator and are test-only.

| Model | AP50 Tiny1 | AP50 Tiny2 | AP50 Tiny3 | AP50 Small | AP50 Medium |
|---|---:|---:|---:|---:|---:|
| YOLOv8 | 33.66 ± 0.75 | 57.19 ± 1.43 | 63.53 ± 0.28 | 67.95 ± 0.47 | 65.76 ± 0.95 |
| YOLOv11 | 34.67 ± 0.16 | 56.07 ± 0.50 | 63.30 ± 0.57 | 68.05 ± 0.78 | 66.81 ± 0.09 |

Bucket artifacts are uploaded and verified in:

- `duyle2408/oacp-r2-parity-rerun-seed43-44-levir-size-eval` — 4/4 LEVIR runs.
- `duyle2408/oacp-r2-parity-rerun-seed43-44-tinyperson-size-eval` — 4/4 TinyPerson runs.
