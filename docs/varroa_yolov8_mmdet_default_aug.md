# Varroa YOLOv8 baseline with MMDetection-style default augmentation

Runner: `train_varroa_yolov8n_mmdet_default_aug.py`

This is a single-control baseline: pretrained `yolov8n.pt`, seed `42`, 640px,
100 epochs, batch 8, patience 0, and NMS IoU `0.5`. The augmentation map is
explicit in the runner and mirrors `rtmdet_tiny_8xb32-300e_coco.py`:

- CachedMosaic -> `mosaic=1.0`
- RandomResize ratio range `(0.5, 2.0)` -> `scale=0.5` approximation
- YOLOX HSV -> `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`
- RandomFlip -> `fliplr=0.5`
- CachedMixUp -> `mixup=0.5`
- rotation, translation, shear, perspective, vertical flip disabled

Ultralytics has no exact cached-transform or RandomCrop equivalent. The
approximations are recorded in `experiment_manifest.json` instead of being
hidden. This keeps the baseline reproducible while making the cross-framework
difference visible.

## Local validation

```bash
cd yolo_related
conda run -n ml2 python -m py_compile train_varroa_yolov8n_mmdet_default_aug.py
conda run -n ml2 pytest -q tests/test_train_varroa_yolov8n_mmdet_default_aug.py
```

A bounded local smoke train can be run with `--epochs 1 --device cpu
--batch-size 2 --workers 0 --no-amp`. It still evaluates both validation and
test splits, so use the full command only when local compute is available.

## Marimo launch

After committing and synchronizing the exact SHA to `/marimo/yolo_code`, run
preflight in the live notebook, then launch through the shared helper. The
helper injects `MARIMO_TRAIN_WORKFLOW=1`; the runner refuses upload otherwise.

```bash
/marimo/mmdet-venv/bin/python -m utils.marimo_ops preflight \
  --repo /marimo/yolo_code \
  --expected-sha "$EXPECTED_SHA" \
  --python /marimo/mmdet-venv/bin/python \
  --required-path train_varroa_yolov8n_mmdet_default_aug.py \
  --required-path datasets/varroa_yolo/varroa.yaml \
  --epochs 100 --patience 0 --upload-required \
  --hf-repo-id duyle2408/varroa-yolov8n-mmdet-default-aug

/marimo/mmdet-venv/bin/python -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/varroa_yolov8n_mmdet_default_aug \
  -- /marimo/mmdet-venv/bin/python train_varroa_yolov8n_mmdet_default_aug.py \
  --data-yaml /marimo/yolo_code/datasets/varroa_yolo/varroa.yaml \
  --project /marimo/yolo_code/runs/varroa_yolov8n_mmdet_default_aug \
  --epochs 100 --patience 0 --imgsz 640 --batch-size 8 --seed 42 \
  --hf-repo-id duyle2408/varroa-yolov8n-mmdet-default-aug --upload
```

Required per-run artifacts are `best.pt`, `last.pt`, `results.csv`, `args.yaml`,
`evaluation_metrics.json`, `config.yaml`, `experiment_manifest.json`, and the
verified `upload_complete.json` marker.
