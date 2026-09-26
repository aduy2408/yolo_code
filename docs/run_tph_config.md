# YOLOv5 baseline with the TPH input profile

Config: `train_scripts/run_tph_config.yaml`

This run is a control baseline. It uses the canonical YOLOv5n model from
`vendor/ultralytics_upstream/ultralytics/cfg/models/v5/yolov5.yaml`. It does not
use `yolov5l-xs-tph.yaml`, Transformer heads, CBAM, or another TPH architecture.

The official `tph-yolov5-upstream/README.md` training command contributes only:

- `imgsz=1536`
- `batch=4`

The project defaults remain in force for learning rate, optimizer, and the
augmentation pipeline. The baseline runner's normal defaults are preserved when
`--imgsz` is not supplied.

After Marimo preflight, the equivalent runner selection is:

```bash
python train_scripts/train_all_visdrone_yolo_baselines.py \
  --data-root <exact-visdrone-data-root> \
  --models yolov5n \
  --seeds 42 \
  --allow-subset \
  --augmentations mosaic \
  --imgsz 1536 \
  --batch-size 8 \
  --epochs 100 \
  --patience 0 \
  --workers 8 \
  --device cuda \
  --hf-repo-id <task-specific-hf-repo>
```

Training remains upload-required and must be launched through the Marimo
workflow. Dataset root, Python executable, source commit, and HF repository are
therefore resolved during preflight and recorded in each run manifest.

For a mixed model/seed queue, repeat `--job` in the desired order. The runner
trains, evaluates both splits, uploads, and verifies each job before moving to
the next one:

```bash
--job yolov5n:43,44 \
--job yolov8n:42,43,44 \
--batch-size 8
```
