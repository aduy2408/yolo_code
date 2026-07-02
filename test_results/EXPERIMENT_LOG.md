# Varroa YOLO Test Results Log

Generated from the CSV summaries currently in `test_results/`.

## Data Sources

- `summary.csv`: baseline, P2 baseline, and boundary/API comparison.
- `summary-1.csv`: API box-gradient with boundary-ring contrastive test runs.
- `placement_summary_shard1_of_2.csv`: KVCA placement shard 1/2 test runs.
- Manual screenshot note from user: new split dataset results plus Soft-NMS mAP50-95 for selected runs.

All listed metrics are test-split Ultralytics `model.val(...)` outputs with `conf=0.001`, `iou=0.5`, `imgsz=640` as used by the training/evaluation scripts.

## Overall Ranking

| Rank | Run | Source | mAP50 | mAP50-95 | Precision | Recall |
|---:|---|---|---:|---:|---:|---:|
| 1 | `yolov8n_varroa_compare_baseline_p2_api_boxgrad_bcon020_ring1` | `summary-1.csv` | 0.9228 | 0.3360 | 0.9117 | 0.9062 |
| 2 | `yolov8n_varroa_compare_baseline_p2_baseline_p2` | `summary.csv` | 0.9084 | 0.3293 | 0.9058 | 0.8997 |
| 3 | `yolov8_kvca_placement_c_n` | `placement_summary_shard1_of_2.csv` | 0.9073 | 0.3318 | 0.8871 | 0.8918 |
| 4 | `yolov8_kvca_placement_b_n` | `placement_summary_shard1_of_2.csv` | 0.9058 | 0.3293 | 0.8948 | 0.9064 |
| 5 | `yolov8n_varroa_compare_baseline_p2_api_boxgrad_ensimam_bcon020_ring1` | `summary-1.csv` | 0.9043 | 0.3298 | 0.9171 | 0.8894 |
| 6 | `yolov8_kvca_placement_a_sc_n` | `placement_summary_shard1_of_2.csv` | 0.9037 | 0.3295 | 0.8975 | 0.9123 |
| 7 | `yolov8n_varroa_compare_baseline_baseline` | `summary.csv` | 0.9019 | 0.3285 | 0.9040 | 0.9081 |
| 8 | `yolov8n_varroa_compare_baseline_p2_boundary_api_baseline_p2_boundary_api_bcon005` | `summary.csv` | 0.6985 | 0.2114 | 0.6542 | 0.7018 |

## Run-To-Config Mapping

| Run | Config / origin | What was tried |
|---|---|---|
| `yolov8n_varroa_compare_baseline_baseline` | `models_related/models_config/yolov8/tried/yolov8_varroa_compare_baseline.yaml` via `misc/train_boundary_api_compare.py` | Baseline YOLOv8n-style model, Detect on P3/P4/P5, no P2 Detect head. |
| `yolov8n_varroa_compare_baseline_p2_baseline_p2` | `models_related/models_config/yolov8/tried/yolov8_varroa_compare_baseline_p2.yaml` via `misc/train_boundary_api_compare.py` | Adds P2 Detect head and keeps KVCompressedAttention on P2/P3 detect inputs. |
| `yolov8n_varroa_compare_baseline_p2_boundary_api_baseline_p2_boundary_api_bcon005` | `models_related/models_config/yolov8/yolov8n_varroa_compare_baseline_p2_boundary_api.yaml` / generated scale copy | Adds train-time BoundaryFeatureBlock plus AdversarialPerturbationInjection after P2. Test suffix says boundary contrast `0.005`. |
| `yolov8n_varroa_compare_baseline_p2_api_boxgrad_bcon020_ring1` | `models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml` via `misc/train_api_boxgrad_boundary_ring.py` | P2 API uses localization box/dfl gradients: `AdversarialPerturbationInjection [0.005, 0.05, boxgrad]`; final P2/P3 use KVCompressedAttention. Training run used boundary contrast `0.020`, ring `1`. |
| `yolov8n_varroa_compare_baseline_p2_api_boxgrad_ensimam_bcon020_ring1` | `models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_ensimam.yaml` via `misc/train_api_boxgrad_boundary_ring.py` | Same API box-gradient setup, but final P2/P3 attention is EnSimAM instead of KVCompressedAttention. Training run used boundary contrast `0.020`, ring `1`. |
| `yolov8_kvca_placement_c_n` | `models_related/models_config/yolov8/tried/yolov8_kvca_placement_c.yaml` via `misc/sweep.py` | KVCA placement C, YOLOv8n scale, trained in placement sweep. |
| `yolov8_kvca_placement_b_n` | `models_related/models_config/yolov8/tried/yolov8_kvca_placement_b.yaml` via `misc/sweep.py` | KVCA placement B, YOLOv8n scale, trained in placement sweep. |
| `yolov8_kvca_placement_a_sc_n` | `models_related/models_config/yolov8/tried/yolov8_kvca_placement_a_sc.yaml` via `misc/sweep.py` | KVCA placement A with shortcut/channel variant, YOLOv8n scale, trained in placement sweep. |

## New Split / Soft-NMS Notes

These rows came from the manual table shared after the CSV summaries. Treat them as separate from the CSV ranking above because the columns are labeled "new split dataset" and not all runs have complete metric columns.

| Model | test_mAP50, new split | test_mAP50-95, new split | mAP50-95 with Soft-NMS | Soft-NMS delta |
|---|---:|---:|---:|---:|
| YOLOv8x base | 0.8970 | 0.3280 | - | - |
| KVCA before detection head | 0.9120 | 0.3420 | - | - |
| KVCA before detection head + boundary contrast | 0.9320 | 0.3420 | - | - |
| KVCA + API | - | 0.3380 | 0.3557 | +0.0177 |
| `yolov8n_varroa_compare_baseline_p2_api_boxgrad_bcon020_ring1` | 0.9228 | 0.3360 | 0.3523 | +0.0163 |

Manual-table inference:

- On the new split numbers, "KVCA before detection head + boundary contrast" has the best mAP50 shown: `0.9320`.
- Soft-NMS improves mAP50-95 for both rows where Soft-NMS is reported:
  - KVCA + API: `0.3380 -> 0.3557`, `+0.0177`.
  - `p2_api_boxgrad_bcon020_ring1`: `0.3360 -> 0.3523`, `+0.0163`.
- Soft-NMS is therefore worth keeping in the evaluation comparison, especially for API/KVCA variants where duplicate or overlapping small-object predictions may be hurting mAP50-95.

## Inference From Results

- Best current result is `p2_api_boxgrad_bcon020_ring1`: mAP50 `0.9228`, mAP50-95 `0.3360`.
- Against plain baseline, `p2_api_boxgrad_bcon020_ring1` improves mAP50 by `+0.0209` and mAP50-95 by `+0.0075`.
- Against P2 baseline, `p2_api_boxgrad_bcon020_ring1` improves mAP50 by `+0.0144` and mAP50-95 by `+0.0067`.
- Adding a P2 head alone helps modestly over baseline: mAP50 `+0.0065`, but recall drops from `0.9081` to `0.8997`.
- API box-gradient recovers the P2 recall drop while also improving precision: P `0.9117`, R `0.9062`.
- EnSimAM replacement in the API box-gradient run does not help here. It has higher precision (`0.9171`) but lower recall (`0.8894`), and mAP50 drops `0.0185` behind the KVCompressedAttention version.
- KVCA placement C is the best placement-shard result by mAP50-95 among the three shard rows, but it is still slightly below the P2 baseline on mAP50 and well below the API box-gradient result.
- The boundary/API bcon005 run is a clear failure case in this result set. Its mAP50 `0.6985` is about `-0.2099` below P2 baseline. This matches the earlier collapse notes in `BOUNDARY_API_COLLAPSE_NOTES.md`, where train-only boundary/API behavior was suspected to create train/eval mismatch.

## Current Recommendation

Use `yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml` as the strongest current direction. Keep the API box-gradient and boundary-ring setup, but do not replace the final P2/P3 KVCompressedAttention with EnSimAM based on this run.

Next useful checks:

1. Re-run `p2_api_boxgrad_bcon020_ring1` with another seed to verify the `+0.0144` mAP50 gain over P2 baseline is stable.
2. Test API box-gradient with KVCA placement C, because placement C has the strongest mAP50-95 among the KVCA placement shard rows.
3. Include Soft-NMS in the next evaluation table; the manual results show roughly `+0.016` to `+0.018` mAP50-95 for API/KVCA rows.
4. Avoid the older foreground/boundary API setup unless the train/eval mismatch is fixed or explicitly measured.
