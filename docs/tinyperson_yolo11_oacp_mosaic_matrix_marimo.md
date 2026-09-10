# TinyPerson YOLO11 OACP/Mosaic matrix via Marimo

Runner: `train_all_tinyperson_yolo11_oacp_mosaic_matrix.py`

The matrix uses the fixed TinyPerson `split_seed=42`, training seed `42`,
YOLO11n (`yolo11n.pt`), and four isolated remote prefixes:

| Variant | OACP path | OACP parameters | Mosaic |
|---|---|---|---|
| `legacy_default_oacp_mosaic` | legacy double OACP | default: `p=.20`, strength `.20-.40`, scale `.65-.85`, expand `3.0` | on |
| `legacy_default_oacp_no_mosaic` | legacy double OACP | same default parameters | off |
| `r2_frequent_mild_oacp_mosaic` | new single-pass OACP | R2: `p=.40`, strength `.10-.25`, scale `.80-.95`, expand `3.0` | on |
| `r2_frequent_mild_oacp_no_mosaic` | new single-pass OACP | same R2 parameters | off |

The runner requires `--print-effective-config` first and refuses to train
without `--confirm-settings`. Each variant is trained, evaluated, uploaded,
and remotely verified before the next variant starts. Remote paths are
`runs/<variant>/seed_42/`, so variants cannot overwrite one another.

Example settings preview:

```bash
/tmp/uv-venv/bin/python train_all_tinyperson_yolo11_oacp_mosaic_matrix.py \
  --data-root /marimo/TinyPerson \
  --dataset-root /marimo/yolo_code/datasets \
  --project /marimo/yolo_code/runs/tinyperson_yolo11_oacp_mosaic_matrix \
  --hf-repo-id duyle2408/tinyperson-yolo11-oacp-mosaic-matrix \
  --epochs 100 --patience 0 --imgsz 640 --batch-size 8 --workers 8 \
  --device cuda --seeds 42 --split-seed 42 --print-effective-config
```

NMS IoU is fixed at `0.5` for validation and test evaluation.
