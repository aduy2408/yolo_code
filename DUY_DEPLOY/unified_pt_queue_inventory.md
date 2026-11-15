# Unified PT Evaluation Queue — Checkpoint Inventory

## Purpose

Inventory for the unified PyTorch/TensorRT-parity evaluation queue. This file records the checkpoint source repositories and exact Hugging Face paths before launch.

## Queue summary

| Group | Models | Datasets | Seeds | Jobs |
|---|---|---|---:|---:|
| YOLO baseline (planned) | YOLOv5nu, YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n | LEVIR-Ship, TinyPerson, Varroa | 42, 43 | 30 |
| YOLO baseline (currently available) | Same | Same, except LEVIR-Ship YOLO11n seed43 is unavailable | 42, 43 | 29 |
| Augmentation | YOLOv8, YOLO11 | OACP R2: all three datasets; Negative Canvas R2/R4: LEVIR-Ship and TinyPerson | 42 | 14 |
| **Current evaluable total** |  |  |  | **43** |

Queue generator:

```text
DUY_DEPLOY/queue_unified_pt_eval.py
```

## Evaluation protocol

```text
backend: unified_cloned_ultralytics
rect: false
batch_size: 1
half: false
augment: false
conf: 0.001
nms_iou: 0.5
max_det: 300
split_seed: 42
splits: val, test
```

The evaluator enforces `rect=False`, `half=False`, and `augment=False` so the PyTorch reference uses the same static-square validation geometry as the TensorRT engine.

## YOLO baseline checkpoints

All baseline rows below use the corresponding `seed_42` or `seed_43` path. Seed42 records, revisions, and SHA256 values are maintained in:

```text
DUY_DEPLOY/manifests/checkpoints_manifest.json
```

Seed43 files were verified as present except for LEVIR-Ship YOLO11n seed43. The missing source is not replaced with seed44.

### LEVIR-Ship

Hugging Face dataset repository:

```text
https://huggingface.co/datasets/duyle2408/yolo-baselines-no-mosaic-musgd-runs
```

| Model | Seed42 file | Seed43 file |
|---|---|---|
| YOLOv5nu | `runs/levirship/yolov5/seed_42/weights/best.pt` | `runs/levirship/yolov5/seed_43/weights/best.pt` |
| YOLOv8n | `runs/levirship/yolov8/seed_42/weights/best.pt` | `runs/levirship/yolov8/seed_43/weights/best.pt` |
| YOLOv9t | `runs/levirship/yolov9/seed_42/weights/best.pt` | `runs/levirship/yolov9/seed_43/weights/best.pt` |
| YOLOv10n | `runs/levirship/yolov10/seed_42/weights/best.pt` | `runs/levirship/yolov10/seed_43/weights/best.pt` |
| YOLO11n | `runs/levirship/yolov11/seed_42/weights/best.pt` | **Unavailable: `seed_43` path does not exist; `seed_44` exists but is not substituted** |

### TinyPerson

Hugging Face dataset repository:

```text
https://huggingface.co/datasets/duyle2408/tinyperson-yolo-baselines
```

| Model | Seed42 file | Seed43 file |
|---|---|---|
| YOLOv5nu | `runs/yolov5/seed_42/weights/best.pt` | `runs/yolov5/seed_43/weights/best.pt` |
| YOLOv8n | `runs/yolov8/seed_42/weights/best.pt` | `runs/yolov8/seed_43/weights/best.pt` |
| YOLOv9t | `runs/yolov9/seed_42/weights/best.pt` | `runs/yolov9/seed_43/weights/best.pt` |
| YOLOv10n | `runs/yolov10/seed_42/weights/best.pt` | `runs/yolov10/seed_43/weights/best.pt` |
| YOLO11n | `runs/yolov11/seed_42/weights/best.pt` | `runs/yolov11/seed_43/weights/best.pt` |

### Varroa

Hugging Face dataset repository:

```text
https://huggingface.co/datasets/duyle2408/varroa-yolo-baselines-mosaic-two-seed-runs
```

| Model | Seed42 file | Seed43 file |
|---|---|---|
| YOLOv5nu | `runs/varroa/yolov5/seed_42/weights/best.pt` | `runs/varroa/yolov5/seed_43/weights/best.pt` |
| YOLOv8n | `runs/varroa/yolov8/seed_42/weights/best.pt` | `runs/varroa/yolov8/seed_43/weights/best.pt` |
| YOLOv9t | `runs/varroa/yolov9/seed_42/weights/best.pt` | `runs/varroa/yolov9/seed_43/weights/best.pt` |
| YOLOv10n | `runs/varroa/yolov10/seed_42/weights/best.pt` | `runs/varroa/yolov10/seed_43/weights/best.pt` |
| YOLO11n | `runs/varroa/yolov11/seed_42/weights/best.pt` | `runs/varroa/yolov11/seed_43/weights/best.pt` |

## Augmentation checkpoints

All augmentation checkpoints use seed42. The queue intentionally excludes YOLOv9 augmentation after the queue scope was narrowed to YOLOv8 and YOLO11. Hugging Face uses the directory prefixes `yolov8` and `yolov11` for these files.

### OACP R2

Hugging Face dataset repository:

```text
https://huggingface.co/datasets/duyle2408/model-augmentation-seed42-oacp-runs
```

| Model | Dataset | Hugging Face file |
|---|---|---|
| YOLOv8, YOLO11 | LEVIR-Ship | `yolov8` or `yolov11`/`levir/oacp/r2/no_mosaic/seed_42/weights/best.pt` |
| YOLOv8, YOLO11 | TinyPerson | `yolov8` or `yolov11`/`tinyperson/oacp/r2/mosaic/seed_42/weights/best.pt` |
| YOLOv8, YOLO11 | Varroa | `yolov8` or `yolov11`/`varroa/oacp/r2/mosaic/seed_42/weights/best.pt` |

OACP job count: `2 models × 3 datasets = 6`.

### Negative Canvas R2/R4

Hugging Face dataset repository:

```text
https://huggingface.co/datasets/duyle2408/model-augmentation-seed42-copy-paste-runs
```

| Model | Dataset | Variant | Hugging Face file |
|---|---|---|---|
| YOLOv8, YOLO11 | LEVIR-Ship | `negative_canvas_r2` | `yolov8` or `yolov11`/`levir/copy_paste/negative_canvas_r2/no_mosaic/seed_42/weights/best.pt` |
| YOLOv8, YOLO11 | LEVIR-Ship | `negative_canvas_r4` | `yolov8` or `yolov11`/`levir/copy_paste/negative_canvas_r4/no_mosaic/seed_42/weights/best.pt` |
| YOLOv8, YOLO11 | TinyPerson | `negative_canvas_r2` | `yolov8` or `yolov11`/`tinyperson/copy_paste/negative_canvas_r2/mosaic/seed_42/weights/best.pt` |
| YOLOv8, YOLO11 | TinyPerson | `negative_canvas_r4` | `yolov8` or `yolov11`/`tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_42/weights/best.pt` |

Negative Canvas job count: `2 models × 2 datasets × 2 variants = 8`.

### Explicitly excluded

Varroa Negative Canvas R2/R4 is not in this queue. This follows the existing experiment policy in `train_scripts/model_augmentation_seed42_manifest.json`, which specifies OACP R2 and CP2 for Varroa and skips Varroa Negative Canvas R2/R4.

Current HF `main` does not expose the TinyPerson YOLOv8 Negative Canvas R2/R4 paths. The Jetson queue reuses the existing verified files from:

```text
DUY_DEPLOY/results/augmentation_onnx_trt_eval_20261103/checkpoints/tinyperson/yolov8/{nc2,nc4}/best.pt
```

Their local SHA256 values are recorded in the Jetson `checkpoint_inventory.json`.

## Launch behavior

Setup-only manifest generation does not download or evaluate checkpoints. To generate the queue files:

```bash
python DUY_DEPLOY/queue_unified_pt_eval.py \
  --output-root DUY_DEPLOY/results/unified_pt_queue_20261103 \
  --dataset-yaml levirship=/path/to/levirship.yaml \
  --dataset-yaml tinyperson=/path/to/tinyperson.yaml \
  --dataset-yaml varroa=/path/to/varroa.yaml
```

Actual download and sequential evaluation requires `--run` and `HF_TOKEN`.
