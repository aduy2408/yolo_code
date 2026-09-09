# TinyPerson YOLOv9/YOLOv10 legacy OACP + Mosaic via Marimo

Runner: `train_all_tinyperson_yolov9_yolov10_legacy_oacp_mosaic.py`

This setup runs both `yolov9t.pt` and `yolov10n.pt` on the fixed TinyPerson
split seed `42`, with the historical two-call OACP path and standard Mosaic.
It uses a separate Hugging Face dataset repository to avoid changing the
existing baseline results:

```text
duyle2408/tinyperson-yolov9-yolov10-legacy-oacp-mosaic
```

The runner is fail-closed. First print and review the complete settings:

```bash
/tmp/uv-venv/bin/python train_all_tinyperson_yolov9_yolov10_legacy_oacp_mosaic.py \
  --data-root /marimo/TinyPerson \
  --dataset-root /marimo/yolo_code/datasets \
  --project /marimo/yolo_code/runs/tinyperson_yolov9_yolov10_legacy_oacp_mosaic \
  --hf-repo-id duyle2408/tinyperson-yolov9-yolov10-legacy-oacp-mosaic \
  --models yolov9 yolov10 --seeds 42 --split-seed 42 \
  --epochs 100 --patience 0 --imgsz 640 --batch-size 8 --workers 8 \
  --device cuda --print-effective-config
```

The printed settings must show `context_augmentation: oacp`,
`legacy_double_oacp: true`, `mosaic: 1.0`, `close_mosaic: 10`, fixed
`split_seed: 42`, and NMS IoU `0.5`. Only after review should the same
command be launched through `utils.marimo_ops launch` with
`--confirm-settings` instead of `--print-effective-config`.
