# YOLO baseline matrix without Mosaic

Runner: `train_scripts/train_all_yolo_baselines_no_mosaic.py`

The matrix contains 45 jobs:

- Datasets: `varroa`, `tinyperson`, `levirship`
- Models: YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11
- Training seeds: `42 43 44`
- Fixed split seed: `42`
- Augmentation gate: `mosaic=0.0`, `close_mosaic=0`
- Detector architecture: pinned upstream baseline YAML only
- NMS IoU: `0.5`

## Two-server split

Run the same commit, dataset roots, project root, and task-specific HF repository
on both Marimo servers. Change only `--machine-index`:

```bash
# Server 0, jobs 0, 2, 4, ...
/marimo/<python> -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/yolo_baselines_no_mosaic_server0 \
  -- \
  /marimo/<python> train_scripts/train_all_yolo_baselines_no_mosaic.py \
  --data-root varroa=/marimo/Varroa \
  --data-root tinyperson=/marimo/TinyPersonData \
  --data-root levirship=/marimo/LevirShip/LevirShipData \
  --dataset-root /marimo/yolo_code/datasets/no_mosaic_baselines \
  --project /marimo/yolo_code/runs/yolo_baselines_no_mosaic \
  --epochs 100 --patience 0 --batch-size 8 --workers 8 --device cuda \
  --machine-index 0 --machine-count 2 \
  --hf-repo-id <hf-user>/yolo-baselines-no-mosaic-runs

# Server 1, jobs 1, 3, 5, ...
# Use the identical command, changing:
#   --run-dir .../yolo_baselines_no_mosaic_server1
#   --machine-index 1
```

Each server owns a disjoint deterministic shard of 23 and 22 jobs. The
`experiment_manifest.json` records `machine_index` and `machine_count` so the
results can be audited and merged. Both servers may prepare the same split
output independently, but they must use the same `split_seed=42` and commit.

Do not launch either command directly. Run the required Marimo preflight first,
then launch through `python -m utils.marimo_ops launch`. The runner uploads and
verifies each run before advancing to the next job.
