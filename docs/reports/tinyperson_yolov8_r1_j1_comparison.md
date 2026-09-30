# TinyPerson recent baseline comparison: YOLOv8, R1, and J1

Updated: 2026-09-30

This report records the recent repository-backed comparison requested for the TinyPerson reranking experiments. It replaces the earlier comparison against the 3-epoch smoke control. The YOLOv8 baseline below is the recent standard-detector baseline aggregate, while R1 and J1 are the completed 100-epoch reranking runs.

## Source repositories and reports

### Standard, non-mosaic detector baselines

- HF dataset repository: [duyle2408/tinyperson-yolo-baselines](https://huggingface.co/datasets/duyle2408/tinyperson-yolo-baselines)
- Repository report: [`docs/reports/all_datasets_baseline_validation_test.md`](../reports/all_datasets_baseline_validation_test.md)
- Recovered merged-metric report: [`docs/reports/baseline_validation_test_recovered.md`](baseline_validation_test_recovered.md)

The YOLOv8n standard row is the mean ± sample standard deviation over seeds 42, 43, and 44. Standard detector metrics and merged corner-window metrics are kept separate.

### Mosaic baseline matrix

- HF dataset repository: [duyle2408/tinyperson-yolo11-oacp-mosaic-matrix](https://huggingface.co/datasets/duyle2408/tinyperson-yolo11-oacp-mosaic-matrix)
- Run protocol: [`docs/tinyperson_yolo11_oacp_mosaic_matrix_marimo.md`](../tinyperson_yolo11_oacp_mosaic_matrix_marimo.md)

This matrix is a separate YOLO11/OACP study. Its values should not be mixed into the YOLOv8/R1/J1 table below because the detector family and augmentation policy differ. The repository contains both mosaic and no-mosaic variants for that study.

### R1 and J1 reranking runs

- HF dataset repository: [duyle2408/tinyperson-reranking-runs](https://huggingface.co/datasets/duyle2408/tinyperson-reranking-runs)
- R1 prefix: `runs/localization/yolov8n_base/seed_42`
- J1 prefix: `runs/joint/yolov8n_base/seed_42`

Both runs use the same YOLOv8n P3/P4/P5 detector, TinyPerson split seed 42, training seed 42, 640px inputs, batch size 8, workers 8, 100 epochs, patience 0, and the official corner-window test split. The R1/J1 runs are non-mosaic reranking experiments.

## Standard split metrics

Values are the exact split-labeled metrics from each run's `evaluation_metrics.json`. The YOLOv8 baseline is a three-seed aggregate. R1 and J1 are single seed-42 runs, so no standard deviation is shown for them.

| Run | Training setup | Seeds | val/AP50 | val/mAP50-95 | test/AP50 | test/AP75 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline | Standard detector, non-mosaic | 42,43,44 | 0.6236 ± 0.0522 | 0.1795 ± 0.0153 | 0.4990 ± 0.0265 | 0.0829 ± 0.0122 | 0.1782 ± 0.0129 |
| R1 | TAL plus localization-only ranking loss | 42 | 0.515954 | 0.184284 | 0.496686 | not recorded | 0.177407 |
| J1 | TAL plus joint correction-only ranking loss | 42 | 0.526127 | 0.188514 | 0.498978 | not recorded | 0.178672 |

### Interpretation of the standard table

- R1 and J1 are not directly comparable to the baseline mean as a matched-seed claim because the baseline row aggregates three seeds while R1/J1 are seed 42 only.
- On the available R1/J1 runs, J1 is higher than R1 on all four reported AP/mAP fields:
  - val/AP50: `+0.010173`
  - val/mAP50-95: `+0.004230`
  - test/AP50: `+0.002293`
  - test/mAP50-95: `+0.001265`
- The YOLOv8 baseline row is from the recent standard baseline repository, not from the paper screenshot. The screenshot's non-mosaic and mosaic tables use a different reporting source and must not be silently merged with these artifacts.

## Merged corner-window and small-object metrics

The recent baseline repository exposes merged metrics for YOLOv8n seed 43 only. They are not a complete three-seed aggregate. R1 and J1 uploaded the official corner-window test protocol and source artifact, but the merged evaluator was unavailable for those runs, so their merged and small-object fields are reported as unavailable rather than inferred from standard metrics.

| Run | Merged protocol | AP50 | AP75 | mAP50-75 | AP-Tiny1 | AP-Tiny2 | AP-Tiny3 | AP-Small | AP-Medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline, seed 43 | Official TinyPerson corner-window merged test | 0.5369 | 0.0945 | 0.3176 | 0.1595 | 0.3084 | 0.3684 | 0.4390 | 0.4352 |
| R1, seed 42 | Protocol recorded, merged evaluator unavailable | — | — | — | — | — | — | — | — |
| J1, seed 42 | Protocol recorded, merged evaluator unavailable | — | — | — | — | — | — | — | — |

R1 and J1 source artifact for the protocol: `/marimo/yolo_code/datasets/tinyperson_test_corner_sw640_sh512/corner_manifest.json`. The upload manifests also record `test_merged/available = 0.0`. This is why the R1/J1 small-object cells are intentionally blank.

## Run configuration summary

| Parameter | YOLOv8n baseline aggregate | R1 | J1 |
|---|---|---|---|
| Detector | YOLOv8n P3/P4/P5 | YOLOv8n P3/P4/P5 | YOLOv8n P3/P4/P5 |
| Mosaic | Non-mosaic standard baseline | Non-mosaic | Non-mosaic |
| Training seed | 42, 43, 44 | 42 | 42 |
| Split seed | 42 | 42 | 42 |
| Epochs | 100 | 100 | 100 |
| Patience | 0 | 0 | 0 |
| Image size | 640 | 640 | 640 |
| Batch size | 8 | 8 | 8 |
| Workers | 8 | 8 | 8 |
| Responsibility/ranking | Standard TAL | Localization-only ranking, tiny-object gate `max_dim <= 16` | Joint correction-only ranking, tiny-object gate `max_dim <= 16` |
| HF repository | `duyle2408/tinyperson-yolo-baselines` | `duyle2408/tinyperson-reranking-runs` | `duyle2408/tinyperson-reranking-runs` |

## Reporting limitations

1. The standard YOLOv8 row is a three-seed aggregate, whereas R1 and J1 are single seed-42 runs.
2. The YOLOv8 AP-Small and AP-Tiny values are from the one available merged YOLOv8 seed-43 artifact, not the three-seed standard aggregate.
3. R1/J1 merged AP-Small and AP-Tiny metrics were not available because the merged evaluator did not complete. Their standard val/test metrics are still complete and split-labeled.
4. The YOLO11 mosaic matrix is linked above as a separate baseline study, but is not included in the direct YOLOv8/R1/J1 table because it changes both detector family and augmentation pipeline.
