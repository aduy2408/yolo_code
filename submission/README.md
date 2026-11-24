# Submission code map

Branch: `submission`. This repository remains the Ultralytics base. The submission folder is a small, stable entrypoint map; canonical implementations stay in their existing paths so imports and history remain intact.

## Scope

Datasets: Varroa, LEVIR-Ship, TinyPerson. Split seed: `42`; training seeds are selected by each runner. Datasets, checkpoints, `runs/`, `weights/`, caches, and generated reports are runtime artifacts and are not submission source.

| Area | Canonical source |
|---|---|
| Varroa conversion | `misc/prepare_dataset.py` |
| LEVIR-Ship conversion | `misc/prepare_levir_ship.py` |
| TinyPerson conversion and merged test evaluation | `train_scripts/train_all_tinyperson.py` |
| YOLO baselines | `train_scripts/train_all_yolo_baselines_no_mosaic.py` |
| YOLO baseline evaluation | `evaluate_test/evaluate_yolo_baseline_matrix.py` |
| OACP implementation | `project_ultralytics/context_augment.py` |
| Copy-Paste protocol registry | `copy_paste_protocol.py` |
| Copy-Paste implementation | `project_ultralytics/copy_paste.py`, `negative_canvas_copy_paste.py`, `adaptive_negative_canvas.py`, `stcp_copy_paste.py` |
| Augmentation matrix | `train_scripts/train_all_augmentation_matrix.py` |
| Marimo preflight/launch/upload verification | `utils/marimo_ops.py` |
| CI/CD workflow and dispatch | `submission/ci_cd/README.md`, `.github/workflows/ci.yml`, `.github/workflows/mlops.yml` |
| STW-YOLO | `misc/train_stw_yolo_marimo.py` |
| TPH-YOLO | `train_scripts/train_tph_yolov5_multidataset.py` |

OACP `r2` is implemented in `project_ultralytics/context_augment.py`. Copy-Paste `r2` and `r4` are registered as `negative_canvas_r2`, `negative_canvas_r4`, `all_canvas_r2`, and `all_canvas_r4` in `copy_paste_protocol.py`. Do not copy a second implementation into this folder.

Use the launchers under `train/` and `evaluate/`; they delegate to the canonical source above. Training/evaluation through Marimo must use `utils.marimo_ops` and the live Marimo kernel. Never place tokens in commands or files.
