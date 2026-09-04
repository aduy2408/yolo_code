# LevirShip YOLOv8 baseline with MMDetection-style default augmentation

Runner: `train_levir_yolov8n_mmdet_default_aug.py`

This runner ports the actual `yolo_pipeline()` added by commit `067e4c7f` in
`mmdetection/train_all_levir_baseline.py`, rather than claiming to reproduce a
native MMDetection default.

- CachedMosaic -> `mosaic=1.0`
- RandomAffine (`translate=.1`, scale range `.5..1.5`) -> `translate=0.1`, `scale=0.5`
- Final 10-epoch no-Mosaic stage -> `close_mosaic=10`
- YOLOX HSV -> `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`
- RandomFlip -> `fliplr=0.5`
- CachedMixUp -> `mixup=0.5`
- rotation, translation, shear, perspective, vertical flip disabled

The YOLOv8 model is initialized from `yolov8n.pt`, while the MMDetection runner
initializes the selected detector from its config and backbone `init_cfg`.
These are matched training-data augmentations, not a matched architecture or
optimizer experiment.

## Local validation

```bash
cd yolo_related
conda run -n ml2 python -m py_compile train_levir_yolov8n_mmdet_default_aug.py
conda run -n ml2 pytest -q tests/test_train_levir_yolov8n_mmdet_default_aug.py
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
  --required-path train_levir_yolov8n_mmdet_default_aug.py \
  --required-path datasets/levir_ship_yolo/levir_ship.yaml \
  --epochs 100 --patience 0 --upload-required \
  --hf-repo-id duyle2408/levir-ship-yolov8n-mmdet-default-aug

/marimo/mmdet-venv/bin/python -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/levir_yolov8n_mmdet_default_aug \
  -- /marimo/mmdet-venv/bin/python train_levir_yolov8n_mmdet_default_aug.py \
  --data-yaml /marimo/yolo_code/datasets/levir_ship_yolo/levir_ship.yaml \
  --project /marimo/yolo_code/runs/levir_yolov8n_mmdet_default_aug \
  --epochs 100 --patience 0 --imgsz 640 --batch-size 8 --seed 42 \
  --hf-repo-id duyle2408/levir-ship-yolov8n-mmdet-default-aug --upload
```

Required per-run artifacts are `best.pt`, `last.pt`, `results.csv`, `args.yaml`,
`evaluation_metrics.json`, `config.yaml`, `experiment_manifest.json`, and the
verified `upload_complete.json` marker.
