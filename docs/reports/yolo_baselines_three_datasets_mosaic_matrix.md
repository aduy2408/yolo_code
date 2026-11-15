# YOLO Baseline Matrix: LEVIR-Ship, TinyPerson, Varroa

**Updated:** 2026-10-04
**Scope:** YOLOv5, YOLOv8, YOLOv9, YOLOv10, and YOLO11 on LEVIR-Ship, TinyPerson, and Varroa, separated by explicit Mosaic and no-Mosaic provenance.

## Important correction

The earlier version of this report was too conservative because it did not inspect the Hugging Face **MuSGD/no-Mosaic repositories** and several dataset-specific baseline repositories. The claim that Mosaic was available only for YOLOv8 was incorrect.

The Hugging Face audit found complete or nearly complete five-family baseline archives for both augmentation settings:

- **Mosaic:** complete five-family coverage on LEVIR-Ship and Varroa, and four complete families plus partial YOLO11 coverage on TinyPerson.
- **No-Mosaic MuSGD:** complete five-family coverage on TinyPerson and Varroa, and one missing LEVIR-Ship seed cell.

A run is counted only when its repository contains the model/seed artifact and a metric artifact. The augmentation setting is taken from the uploaded `args.yaml`, not inferred from the repository name.

## Coverage matrix

The requested seed set is `42, 43, 44`.

| Dataset | Mosaic coverage | No-Mosaic MuSGD coverage | Remaining gap |
|---|---|---|---|
| **LEVIR-Ship** | YOLOv5/v8/v9/v10/YOLO11, all 3 seeds | YOLOv5/v8/v9/v10 all 3 seeds; YOLO11 seeds 42 and 44 | No-Mosaic YOLO11 seed 43 |
| **TinyPerson** | YOLOv5/v8/v9/v10 all 3 seeds; YOLO11 seeds 42 and 43 | YOLOv5/v8/v9/v10/YOLO11, all 3 seeds | Mosaic YOLO11 seed 44 in the standard baseline archive |
| **Varroa** | YOLOv5/v8/v9/v10/YOLO11, all 3 seeds, split across three repositories | YOLOv5/v8/v9/v10/YOLO11, all 3 seeds | No confirmed missing requested cell |

Therefore the corrected status is:

- **Mosaic:** 44 of 45 requested model-seed cells are directly evidenced in the standard baseline archives.
- **No-Mosaic MuSGD:** 44 of 45 requested model-seed cells are directly evidenced.
- The only two missing cells are **different cells**: `TinyPerson/Mosaic/YOLO11/seed_44` and `LEVIR-Ship/no-Mosaic/YOLO11/seed_43`.

This is a much stronger result than the previous report's statement that the Mosaic matrix was YOLOv8-only.

## Hugging Face sources

### Mosaic archives

#### LEVIR-Ship

- Repository: `duyle2408/levir-ship-yolo-baselines`
- Main archive: `evaluation/coco_fixedsplit42`
- Models: `yolov5nu`, `yolov8n`, `yolov9t`, `yolov10n`, `yolo11n`
- Seeds: `42, 43, 44` for every model
- Uploaded training arguments confirm:
  - `mosaic: 1.0`
  - `close_mosaic: 10`
  - `imgsz: 512`
  - fixed split dataset `levir_ship_yolo_seed42`
- Metric artifacts include per-seed `metrics.json`, aggregate `results.json`, and `summary.json`.

#### TinyPerson

- Repository: `duyle2408/tinyperson-yolo-baselines`
- Paths: `runs/{yolov5,yolov8,yolov9,yolov10,yolov11}/seed_{42,43,44}`
- Uploaded arguments for YOLOv5/v8/v9/v10 confirm:
  - `mosaic: 1.0`
  - `close_mosaic: 10`
  - corner-window split and `imgsz: 640`
- YOLO11 has metric artifacts for seeds 42 and 43. The expected `runs/yolov11/seed_44` artifact was not present in this standard baseline repository.
- TinyPerson test results must preserve the merged corner-window protocol. Do not mix `test_merged/*` fields with native test fields.

#### Varroa

The standard Mosaic baseline archive is split across:

- `duyle2408/varroa-yolo-baselines-part1-full`
  - YOLOv5 and YOLOv8/YOLOv9 families needed for the requested matrix
- `duyle2408/varroa-yolo-baselines-part2-full`
  - YOLOv10 and additional YOLOv9/YOLO11 artifacts
- `duyle2408/varroa-yolo-baselines-missing-part3-missing`
  - YOLO11 artifacts

The uploaded arguments confirm `mosaic: 1.0` and `close_mosaic: 10`. The requested nano-family cells for YOLOv5, YOLOv8, YOLOv9, YOLOv10, and YOLO11 are present for seeds 42, 43, and 44 across the split repositories.

### No-Mosaic MuSGD archive

- Repository: `duyle2408/yolo-baselines-no-mosaic-musgd-runs`
- Paths: `runs/{levirship,tinyperson,varroa}/{yolov5,yolov8,yolov9,yolov10,yolov11}/seed_{42,43,44}`
- Uploaded arguments confirm:
  - `mosaic: 0.0`
  - `close_mosaic: 0`
  - fixed split seed `42`
  - explicit `evaluation_metrics.json`
- Artifact count by dataset/model:
  - TinyPerson: 5 models x 3 seeds
  - Varroa: 5 models x 3 seeds
  - LEVIR-Ship: YOLOv5/v8/v9/v10 x 3 seeds, YOLO11 seeds 42 and 44

The older repository `duyle2408/yolo-baselines-no-mosaic-runs` is incomplete for LEVIR-Ship and should not be used as the sole source when the MuSGD repository is available.

## Direct metric examples from the audited artifacts

The LEVIR-Ship Mosaic archive's seed-42 `metrics.json` contains both validation and test metrics for all five model families. Examples:

| Model | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv5n | 0.5394 | 0.2126 | 0.6289 | 0.2413 |
| YOLOv8n | 0.8322 | 0.3320 | 0.8136 | 0.3101 |
| YOLOv9t | 0.8362 | 0.3372 | 0.7885 | 0.3024 |
| YOLOv10n | 0.8002 | 0.3484 | 0.7692 | 0.3434 |
| YOLO11n | 0.8341 | 0.3574 | 0.8306 | 0.3516 |

These values are from the uploaded LEVIR-Ship Mosaic archive and are not the same as the older report rows that used a different run/protocol. The report must preserve source repository and protocol alongside any table of numbers.

## What should not be mixed

- Do not mix the LEVIR-Ship Mosaic archive with the no-Mosaic MuSGD archive as if they were the same run.
- Do not use the older incomplete `yolo-baselines-no-mosaic-runs` repository to conclude that LEVIR YOLOv10 or YOLO11 coverage is absent when the MuSGD repository has additional cells.
- Do not fill `TinyPerson/Mosaic/YOLO11/seed_44` from an OACP or augmentation-policy experiment. The missing cell is specifically the standard YOLO11 baseline archive cell.
- Do not mix TinyPerson native test metrics with the official merged corner-window test protocol.
- Do not use VisDrone repositories to fill any of the three requested datasets.
- Do not merge P2/P3/P4 architecture experiments or OACP policy variants into this five-family baseline matrix.

## Related local reports and sessions

- `docs/reports/all_datasets_baseline_validation_test.md`
- `docs/reports/baseline_validation_test_recovered.md`
- `docs/reports/report_yolo.md`
- `docs/reports/augmentation_report.md`
- `docs/reports/complete_main_report.md`
- Historical no-Mosaic setup: `session_raccoon_1789652895716_28be419a594cf58f`
- Historical aggregation: `session_rose_1789871181047_e99443fd4686d466`
- Three-dataset reference: `session_maple_1789871730746_d53640c5cea4f38b`
- Canonical Mosaic matrix session: `session_llama_1789528731515_b75b072ce78bd214`

## Bottom line

The previous missing-cell answer should be corrected to:

1. **Mosaic is not YOLOv8-only.** The Hugging Face audit found five-family Mosaic baselines for LEVIR-Ship and Varroa, plus YOLOv5/v8/v9/v10 Mosaic baselines for TinyPerson.
2. **No-Mosaic MuSGD is almost complete.** Only `LEVIR-Ship / YOLO11 / seed_43` is missing from the requested 45 model-seed cells.
3. **The standard Mosaic archive has one remaining gap.** `TinyPerson / YOLO11 / seed_44` is missing.
4. The two gaps are artifact/seed gaps, not evidence that the corresponding models were never run under the other augmentation setting.
