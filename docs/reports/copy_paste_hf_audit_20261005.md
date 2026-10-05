# Copy-Paste Hugging Face audit

**Audit date:** 2026-10-05  
**Scope:** Public Hugging Face repositories named in the Copy-Paste comparison request.  
**Credentials:** No token is stored in this report. No new training or evaluation job was launched.

## Executive finding

The historical LEVIR Copy-Paste results in `docs/reports/augmentation_report.md` are real Copy-Paste results, not an unrelated experiment. However, they are **not a like-for-like replication** of the current seed 43/44 matrix.

The verified configuration differences are:

| Parameter | Historical CP/post-hoc family | Current seed 43/44 matrix |
|---|---|---|
| LEVIR data YAML | `levir_ship_copy_paste_split_42/levir_ship.yaml` | `levir_ship_augmentation_matrix_split_42/levir_ship.yaml` |
| Image size | 640 | 512 |
| Translate | 0.1 | 0.0 |
| Scale | 0.5 | 0.0 |
| Mosaic | 0.0 | 0.0 |
| Detector | Canonical YOLOv8 P3/P4/P5 | Canonical YOLOv8 P3/P4/P5 |
| Split seed | 42 | 42 |
| Source commit | `32f5899cdec987a0b07dbb0e5d558e8b971fed5f` | `53686d0364de3c720672c2fc05a3e0b6f4a5f` |

Therefore, the low current `AP50-Small` values must not be presented as evidence that the historical CP family was low-performing. The current matrix needs a parity rerun before a fair comparison. Historical post-hoc artifacts also do not contain native `AP50-Small`; that requires a separate size-bucket evaluation of the historical checkpoints.

## Repository inventory

Counts below are read-only artifact counts from the public repository trees. `metrics` counts metric JSON files, `manifests` counts manifest/argument files, and `upload_markers` counts `upload_complete.json` files.

| Repository | Files | Metrics | Manifests/args | Upload markers | Assessment |
|---|---:|---:|---:|---:|---|
| `duyle2408/augmentation-copy-paste-runs` | 361 | 48 | 96 | 48 | Current 43/44 matrix family; 48 seed-qualified metric artifacts |
| `duyle2408/stcp-copy-paste-3datasets-runs` | 43 | 5 | 10 | 5 | Separate STCP 3-dataset family |
| `duyle2408/visdrone-copy-paste-r1-r4-runs` | 64 | 9 | 18 | 9 | Separate VisDrone family |
| `duyle2408/levir-adaptive-copy-paste-adamw-runs` | 8 | 1 | 2 | 1 | Separate adaptive optimizer family |
| `duyle2408/levir-adaptive-copy-paste-sgd-runs` | 8 | 1 | 2 | 1 | Separate adaptive optimizer family |
| `duyle2408/levir-adaptive-copy-paste-runs` | 48 | 6 | 12 | 6 | Separate adaptive Copy-Paste family |
| `duyle2408/tinyperson-copy-paste-canonical-matrix-runs` | 65 | 8 | 16 | 8 | Separate TinyPerson canonical matrix |
| `duyle2408/levir-copy-paste-detector-matrix-runs` | 1 | 0 | 0 | 0 | Empty/incomplete for metrics |
| `duyle2408/levir-copy-paste-posthoc-test-runs` | 39 | 19 | 0 | 19 | Historical post-hoc test family |
| `duyle2408/tinyperson-copy-paste-mosaic-runs` | 33 | 4 | 8 | 4 | Separate TinyPerson Mosaic family |
| `duyle2408/tinyperson-copy-paste-runs` | 1 | 0 | 0 | 0 | Empty/incomplete for metrics |
| `duyle2408/levir-ship-copy-paste-load-adaptive-mosaic` | 8 | 1 | 2 | 1 | Separate LEVIR adaptive Mosaic artifact |
| `duyle2408/tinyperson-copy-paste-load-adaptive-mosaic` | 8 | 1 | 2 | 1 | Separate TinyPerson adaptive Mosaic artifact |
| `duyle2408/tinyperson-copy-paste-oacp-mass-cp3-mosaic` | 8 | 1 | 2 | 1 | Separate TinyPerson OACP/Mosaic artifact |
| `duyle2408/tinyperson-copy-paste-mosaic` | 6 | 0 | 2 | 0 | Manifests present, no metric/upload marker pair |
| `duyle2408/tinyperson-copy-paste` | 21 | 0 | 8 | 0 | Partial metadata, no metric/upload marker pair |
| `duyle2408/levir-ship-copy-paste-mosaic` | 6 | 0 | 2 | 0 | Manifests present, no metric/upload marker pair |
| `duyle2408/levir-ship-copy-paste` | 21 | 0 | 8 | 0 | Partial metadata, no metric/upload marker pair |

## Historical family versus current matrix

The historical report's corrected post-hoc table is the source of truth for its own test protocol. It reports, among others:

- CP0 test `mAP50-95`: 31.47
- CP1: 31.09
- CP2: 31.67
- CP3: 30.46
- Negative Copy-Paste: 31.01

Those values came from already-trained checkpoints evaluated later with the historical post-hoc protocol. “Post-hoc” means checkpoint re-evaluation, not retraining.

The current 43/44 matrix instead uses the current matrix data layout and LEVIR image size 512. Its Copy-Paste `AP50-Small` values are therefore measuring a different training/evaluation configuration. Changing the report cannot reconcile the two families.

## Concrete implementation finding: evaluator runtime mismatch

The baseline and current 43/44 size metrics are not evaluated by the same Ultralytics runtime:

- `evaluate_test/evaluate_yolo_baseline_matrix.py` calls `pinned_upstream_ultralytics()`, removes `models_related/ultralytics` from `sys.path`, and inserts `vendor/ultralytics_upstream` before evaluating the baseline checkpoints.
- `evaluate_test/evaluate_augmentation_size_metrics.py` calls `train_scripts.train_all_yolo_baselines_no_mosaic.local_ultralytics()`. That helper is documented as upstream but actually inserts `models_related/ultralytics` (`PROJECT_ULTRALYTICS`) into `sys.path`.
- The 76-job training runner also inserts `models_related/ultralytics` before importing YOLO and the project Copy-Paste transforms.
- The two runtimes differ materially in `ultralytics/data/augment.py` and other core files. The legacy/project fork adds custom transform plumbing and changes Mosaic/augmentation behavior.

This is a concrete comparison bug. The baseline table's approximately 0.80 LEVIR `AP50-Small` values come from the pinned upstream evaluator, while the current Copy-Paste `AP50-Small` values were produced through the project fork. Therefore the current 43/44 Copy-Paste size values cannot be compared to `baseline_main.md` until the same evaluator runtime is used. The immediate corrective experiment is to reevaluate representative baseline and Copy-Paste checkpoints through one pinned runtime, without retraining first.


1. Rerun the LEVIR Copy-Paste rows with historical parity settings: image size 640, translate 0.1, scale 0.5, Mosaic off, split seed 42, and a recorded parity manifest.
2. Evaluate the historical CP checkpoints with the native size-bucket evaluator if historical `AP50-Small` is required.
3. Keep the current 43/44 results labeled as the 512/zero-translate/zero-scale matrix until that rerun is complete.

No rerun was started by this audit.
