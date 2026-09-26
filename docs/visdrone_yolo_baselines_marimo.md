# VisDrone YOLO baseline matrix

Runner: `train_scripts/train_all_visdrone_yolo_baselines.py`

The runner is ready for the full **30-run** matrix and supports explicit
sub-queues, so a future model family can be launched without changing the
runner or the baseline registry.

The setup covers **30 runs**:

| Detector | Smallest scale | Training seeds | Augmentation policies |
|---|---:|---:|---|
| YOLOv5 | `yolov5n` | 42, 43, 44 | Mosaic, no Mosaic |
| YOLOv8 | `yolov8n` | 42, 43, 44 | Mosaic, no Mosaic |
| YOLOv9 | `yolov9t` | 42, 43, 44 | Mosaic, no Mosaic |
| YOLOv10 | `yolov10n` | 42, 43, 44 | Mosaic, no Mosaic |
| YOLO11 | `yolo11n` | 42, 43, 44 | Mosaic, no Mosaic |

Fixed settings are `optimizer=auto` (MuSGD when the pinned Ultralytics
threshold is reached), `batch=8`, `workers=8`, `imgsz=640`, and NMS IoU `0.5`.
Mosaic runs use `mosaic=1.0` and `close_mosaic=10`; no-Mosaic runs use
`mosaic=0.0` and `close_mosaic=0`. The official VisDrone train, val, and
test-dev split is reused without random reassignment, so there is no generated
split seed.

Every run writes the same split-qualified standard metrics:
`val/AP50`, `val/mAP50-95`, `test/AP50`, and `test/mAP50-95`. It also writes
the established native-test TinyBenchmark area protocol under the
`test_size/` namespace, including `test_size/AP50-Tiny1`,
`test_size/AP50-Tiny2`, `test_size/AP50-Tiny3`, `test_size/AP50-Small`, and
`test_size/AP50-Medium`. The protocol and its source artifacts are recorded in
the manifest and uploaded with each run. These size metrics are not relabeled
as standard validation metrics.

The runner requires an explicit data root and task-specific Hugging Face repo.
It prepares one YOLO-format dataset, processes one model/augmentation/seed job
at a time, evaluates both `val` and `test`, and verifies each upload before
continuing.

Example Marimo launch after the exact data mount and HF repo have been
confirmed by preflight:

```bash
/marimo/<python> -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/visdrone_yolo_baselines \
  -- \
  /marimo/<python> train_scripts/train_all_visdrone_yolo_baselines.py \
  --data-root <exact-visdrone-data-root> \
  --dataset-root /marimo/yolo_code/datasets/visdrone_baselines \
  --project /marimo/yolo_code/runs/visdrone_yolo_baselines \
  --epochs 100 --patience 0 --batch-size 8 --workers 8 --device cuda \
  --hf-repo-id <hf-user>/visdrone-yolo-baselines-runs
```

Do not substitute a dataset path or launch this upload-required runner
directly. Run the complete Marimo preflight first.

## Prepared queue for YOLOv9, YOLOv10, and YOLO11

The model registry already pins the upstream detector YAML and pretrained
checkpoint for each future family:

| Queue name | Checkpoint | Canonical YAML |
|---|---|---|
| `yolov9t` | `yolov9t.pt` | `vendor/ultralytics_upstream/ultralytics/cfg/models/v9/yolov9t.yaml` |
| `yolov10n` | `yolov10n.pt` | `vendor/ultralytics_upstream/ultralytics/cfg/models/v10/yolov10n.yaml` |
| `yolo11n` | `yolo11n.pt` | `vendor/ultralytics_upstream/ultralytics/cfg/models/11/yolo11.yaml` |

After the new Marimo server has passed the exact checkout, dataset, Python,
CUDA, and HF preflight, launch the future matrix with one sequential queue:

```bash
/marimo/<python> -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/visdrone_yolo_v9_v10_v11 \
  --artifact-root /marimo/yolo_code/runs/visdrone_yolo_v9_v10_v11/artifacts \
  -- \
  /marimo/<python> train_scripts/train_all_visdrone_yolo_baselines.py \
  --data-root <exact-visdrone-data-root> \
  --dataset-root /marimo/yolo_code/datasets/visdrone_baselines \
  --project /marimo/yolo_code/runs/visdrone_yolo_v9_v10_v11/artifacts \
  --epochs 100 --patience 0 --imgsz 640 --batch-size 8 --workers 8 \
  --device cuda \
  --hf-repo-id <hf-user>/visdrone-yolov9-yolov10-yolo11-runs \
  --augmentations mosaic no_mosaic \
  --job yolov9t:42,43,44 \
  --job yolov10n:42,43,44 \
  --job yolo11n:42,43,44
```

For a smaller smoke or recovery queue, keep the same contract and reduce only
the explicit jobs, for example:

```bash
--augmentations mosaic --job yolov9t:42,43,44 --job yolov10n:42,43,44 --job yolo11n:42,43,44
```

The runner skips a job only when its remote `upload_complete.json` prefix is
already verified. A checkpoint without split-qualified validation and test
metrics is not considered resumable completion evidence.
