# ECF-YOLO setup for Varroa experiments

This directory contains a pinned clone of [YuIO26/ECF-YOLO](https://github.com/YuIO26/ECF-YOLO).
The clone is kept as a separate Git submodule at `ECF-YOLO/` so its upstream source
is not mixed with the project-owned runner.

## Effective model and defaults

The upstream repository's `train.py` uses 640px images, batch 8, four workers,
`close_mosaic=0`, SGD, `lr0=0.01`, and `momentum=0.949`. The project runner
`train_ecf_yolo.py` resolves the actual ECF model to:

```text
ECF-YOLO/ultralytics/cfg/models/11/yolo11-EPAN.yaml
```

That YAML uses the upstream custom `Fusion(fusion_mode=bifpn)` neck, a YOLO11
C3k2/C2PSA backbone, and a three-scale Detect head. The only intentional schedule
changes are `epochs=100` and `patience=0`, which prevents early stopping before
all 100 epochs. Validation and test NMS IoU are explicitly `0.5`.

The runner records both the parent checkout SHA and the ECF-YOLO submodule SHA in
every `experiment_manifest.json`. Dataset split seed and training seed remain
separate. The default split seed is `42`.

## Dataset commands

The existing prepared Varroa and scene-grouped LEVIR-Ship YAMLs are used by default.
TinyPerson is prepared through the existing project converter because its official
archive requires corner-window preprocessing. Use `--prepare` when a prepared YAML
is not already present.

```bash
# Print the exact effective configuration without training
python train_ecf_yolo.py \
  --dataset varroa \
  --hf-repo-id <your-dataset-repo> \
  --print-effective-config

# Prepare a fixed split and print its effective configuration
python train_ecf_yolo.py \
  --dataset levirship \
  --data-root /marimo/LevirShip/LevirShipData \
  --dataset-root /marimo/yolo_code/datasets \
  --hf-repo-id <your-dataset-repo> \
  --prepare --print-effective-config

python train_ecf_yolo.py \
  --dataset tinyperson \
  --data-root /marimo/TinyPersonData \
  --dataset-root /marimo/yolo_code/datasets \
  --hf-repo-id <your-dataset-repo> \
  --prepare --print-effective-config
```

`--data-yaml` can override a prepared dataset path when a Marimo workspace uses a
non-default location. Do not change the split seed for matched runs unless the
experiment is explicitly a split-sensitivity study.

## Marimo launch

Read `.agents/workflows/marimo-train.md` before execution. Push the setup commit,
check out that exact SHA on Marimo, and run the shared preflight before launching.
Do not run the training script directly. The detached launch injects
`MARIMO_TRAIN_WORKFLOW=1`; the runner also requires `HF_TOKEN` and a target HF
repository, uploads each run, and verifies the remote artifact paths before
writing `upload_complete.json`.

```bash
/marimo/mmdet-venv/bin/python -m utils.marimo_ops preflight \
  --repo /marimo/yolo_code \
  --expected-sha <project-commit> \
  --python /marimo/mmdet-venv/bin/python \
  --epochs 100 --patience 0 --upload-required \
  --hf-repo-id <your-dataset-repo>

/marimo/mmdet-venv/bin/python -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/ecf_yolo_varroa \
  -- /marimo/mmdet-venv/bin/python train_ecf_yolo.py \
  --dataset varroa \
  --project /marimo/yolo_code/runs/ecf_yolo \
  --hf-repo-id <your-dataset-repo>
```

For Ultralytics-only ECF-YOLO, record the actual Marimo base interpreter if it is
not `/marimo/mmdet-venv/bin/python`; the command above is an example and must not
silently substitute an interpreter after preflight.

Required local artifacts per run are `weights/best.pt`, `weights/last.pt`,
`results.csv`, `evaluation_metrics.json`, `experiment_manifest.json`, and the
post-verification `upload_complete.json` marker.
