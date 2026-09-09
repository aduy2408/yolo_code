# TinyPerson YOLO baselines via Marimo

Runner: `train_all_tinyperson_yolo_baselines.py`

## Required settings review before training

For the YOLOv8n TinyPerson P2/P3/P4 strict baseline, use
`train_all_tinyperson.py`. It now fails closed until the complete effective
configuration is printed and explicitly reviewed. The strict baseline has no
OACP/context augmentation and no Mosaic:

```bash
/tmp/uv-venv/bin/python train_all_tinyperson.py \
  --data-root /marimo/TinyPerson \
  --dataset-root /marimo/yolo_code/datasets \
  --project /marimo/yolo_code/runs/tinyperson_yolov8n_p2p4_nomosaic \
  --epochs 100 --patience 0 --imgsz 640 --batch-size 8 --workers 4 \
  --device cuda --seeds 42 --split-seed 42 \
  --variants yolov8n_p2p3p4_plain \
  --hf-repo-id duyle2408/tinyperson-yolov8n-baselines \
  --print-effective-config
```

Review every printed field, especially `variant`, `context_augmentation`,
`augmentation`, seeds, optimizer schedule, NMS IoU, dataset paths, upload
repository, and resource settings. Only after confirmation should the same
command be rerun with `--confirm-settings` instead of
`--print-effective-config`. OACP variants are rejected by this strict gate.

This matrix uses the existing TinyPerson preprocessing and evaluation code in
`train_all_tinyperson.py`. Each run uses the official corner windows, a
source-image-grouped 90/10 split, validation plus corner-window merged test,
and explicit NMS IoU `0.5`.

## Matrix

- Models: `yolov5nu.pt` (YOLOv5), `yolov8n.pt`, `yolov9t.pt`, `yolov10n.pt`, `yolo11n.pt`
- Seeds: `42 43 44`
- Total: 15 sequential runs
- Required artifacts per run: `weights/best.pt`, `weights/last.pt`, `results.csv`,
  `args.yaml`, `evaluation_metrics.json`, `config.yaml`, and
  `experiment_manifest.json`

## Two-machine split

Run the same command on both machines, changing only the index:

```bash
# machine 0
/marimo/mmdet-venv/bin/python -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/tinyperson_yolo_baselines_machine0 \
  -- \
  /marimo/mmdet-venv/bin/python train_all_tinyperson_yolo_baselines.py \
  --data-root /marimo/TinyPersonData \
  --dataset-root /marimo/yolo_code/datasets \
  --project /marimo/yolo_code/runs/tinyperson_yolo_baselines \
  --epochs 100 --patience 0 --imgsz 640 --batch-size 8 \
  --machine-index 0 --machine-count 2 \
  --hf-repo-id duyle2408/tinyperson-yolo-baselines

# machine 1: use --machine-index 1 and a different --run-dir
```

Run the complete Marimo preflight with the exact commit SHA before launch. The
runner deliberately rejects direct execution without the shared
`MARIMO_TRAIN_WORKFLOW=1` context and `HF_TOKEN`. It uploads and verifies each
model/seed before advancing to the next job.

## Prepare and post-hoc test

Dataset preparation is bounded and can be checked first:

```bash
/marimo/mmdet-venv/bin/python train_all_tinyperson_yolo_baselines.py \
  --data-root /marimo/TinyPersonData --prepare-only \
  --seeds 42 43 44
```

To rerun evaluation without retraining, use:

```bash
/marimo/mmdet-venv/bin/python evaluate_all_tinyperson_yolo_baselines.py \
  --data-root /marimo/TinyPersonData \
  --dataset-root /marimo/yolo_code/datasets \
  --project /marimo/yolo_code/runs/tinyperson_yolo_baselines \
  --machine-index 0 --machine-count 2
```

The evaluator calls the same `val`, `test`, merged-corner prediction,
coordinate translation, TinyBenchmark AP, and NMS IoU `0.5` flow used by the
previous TinyPerson training runner.

Merged-test metrics include AP50, AP75, mAP50-75, AP-Tiny1/2/3, AP-Small,
and AP-Medium. AP-Medium uses the standard COCO area range `32^2` to `96^2`
pixels and is averaged over IoU thresholds `0.50` through `0.75`.
