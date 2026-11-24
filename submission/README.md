# Submission code map

Branch: `submission`. This folder is the self-contained YOLO submission payload. The repository's unrelated research/runtime files remain outside it.

## Included payload

| Area | Submission source |
|---|---|
| Ultralytics project base | `submission/models_related/ultralytics/` |
| Upstream Ultralytics base | `submission/vendor/ultralytics_upstream/` |
| Custom model configs | `submission/models_related/models_config/` |
| Dataset preparation | `submission/misc/prepare_dataset.py`, `submission/misc/prepare_levir_ship.py`, `submission/train_scripts/train_all_tinyperson.py` |
| Baseline training | `submission/train_scripts/train_all_yolo_baselines_no_mosaic.py` |
| Augmentation/OACP/Copy-Paste training | `submission/train_scripts/train_all_augmentation_matrix.py`, `submission/project_ultralytics/`, `submission/copy_paste_protocol.py` |
| STW/TPH training | `submission/misc/train_stw_yolo_marimo.py`, `submission/train_scripts/train_tph_yolov5_multidataset.py` |
| Baseline/SOTA evaluation | `submission/evaluate_test/` |
| Python entrypoints | `submission/train/*.py`, `submission/evaluate/*.py` |
| Marimo workflow | `submission/utils/`, `submission/.agents/workflows/`, `submission/marimo/` |
| CI/CD and Jetson workflow | `submission/.github/workflows/`, `submission/scripts/`, `submission/ci_cd/` |

The three dataset roots are runtime inputs: Varroa, LEVIR-Ship, and TinyPerson. Split seed is `42`; training defaults are 100 epochs, patience `0`, and seeds `42,43,44` in the full matrix runners. Runtime datasets, checkpoints, `runs/`, caches, and generated reports are excluded.

Run from the submission root so the copied scripts resolve `models_related/` and `vendor/` locally. Do not copy a second implementation into the payload.
