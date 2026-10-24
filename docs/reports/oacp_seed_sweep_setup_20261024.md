# OACP seed-sweep setup: YOLOv8/YOLOv9 on LEVIR-Ship and TinyPerson

**Prepared:** 2026-10-24
**Status:** setup only. No Marimo server has been supplied, so no training process was launched.

## Fixed experiment contract

- Models: YOLOv8n and YOLOv9t
- Datasets: LEVIR-Ship and TinyPerson
- Training seeds: 43 and 44
- **Fixed split seed: 42**
- Epochs: 100
- Patience: 0
- Workers: 8
- NMS IoU: 0.50
- Image size: LEVIR 512, TinyPerson 640
- OACP: current single-pass OACP, not legacy double-OACP
- Mosaic: LEVIR off; TinyPerson standard Mosaic on with `mosaic=1.0`, `close_mosaic=10`
- Each job gets an isolated project directory, manifest, artifact prefix, and task-specific HF repository/prefix.

## Job matrix

| Job ID | Model | Dataset | Train seed | Split seed | Mosaic |
|---|---|---|---:|---:|---|
| `yolov8_levir_s43` | YOLOv8n | LEVIR-Ship | 43 | 42 | off |
| `yolov8_levir_s44` | YOLOv8n | LEVIR-Ship | 44 | 42 | off |
| `yolov8_tiny_s43` | YOLOv8n | TinyPerson | 43 | 42 | standard |
| `yolov8_tiny_s44` | YOLOv8n | TinyPerson | 44 | 42 | standard |
| `yolov9_levir_s43` | YOLOv9t | LEVIR-Ship | 43 | 42 | off |
| `yolov9_levir_s44` | YOLOv9t | LEVIR-Ship | 44 | 42 | off |
| `yolov9_tiny_s43` | YOLOv9t | TinyPerson | 43 | 42 | standard |
| `yolov9_tiny_s44` | YOLOv9t | TinyPerson | 44 | 42 | standard |

## Required local artifacts per job

```text
weights/best.pt
weights/last.pt
results.csv
args.yaml
config.yaml
experiment_manifest.json
evaluation_metrics.json
upload_complete.json
```

## Required metric handoff

Every uploaded `evaluation_metrics.json` must explicitly label:

```text
val/AP50
val/mAP50-95
test/AP50
test/mAP50-95
```

Also record `AP75` when exposed by the evaluator. For size metrics, retain the exact evaluator key and protocol. TinyPerson must additionally record merged test metrics when the evaluator produces them:

```text
test_merged/AP50
test_merged/AP50-Small
test_merged/mAP50-75
```

No validation metric will be relabeled as a merged test metric. If an evaluator does not expose a size-bucket field, the field remains explicitly unavailable rather than being inferred.

## Range assignment after servers are supplied

The queue is intentionally not launched yet. Once live servers are provided, assign contiguous ranges without mixing mutable output paths:

- one server: sequential jobs `0..7`
- two servers: server A `0..3`, server B `4..7`
- four servers: one job per server, then backfill verified slots

Before each launch, inspect server checkout, executable, dataset paths, runner import, seed/split seed, state, PID, command, contract, artifact timestamps, HF authentication, and task-specific repository. Launch only through `python -m utils.marimo_ops launch`.

## Existing implementation files checked

- `train_scripts/train_all_yolov8_default_oacp_matrix.py`
- `train_scripts/train_all_tinyperson_yolov9_yolov10_legacy_oacp_mosaic.py`
- `train_scripts/train_levir_scripts/train_all_levir_yolov9t_oacp.py`
- `train_scripts/train_augmentation_baselines_seed42.py`
- `utils/marimo_ops.py`

The old runners contain seed-42-specific gates and one YOLOv9 LEVIR path incorrectly couples `split_seed` to the training seed. Those gates must be corrected before live launch. This setup note deliberately records the required provenance so the later server launch cannot silently reuse a training seed as the dataset split seed.
