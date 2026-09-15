# Mosaic-only exploratory matrix: M2-M5

The four variants share the existing `visibility` composition path and keep the
canonical YOLOv8 P3/P4/P5 detector unchanged. Training is intentionally not
started by this change.

| Run | `mosaic_policy` | Extra behavior |
|---|---|---|
| M2 | `cluster_preserve` | Crop object/nearby cluster plus context before each tile |
| M3 | `post_scale` | Retry up to four centers and reject candidates with excessive post-scale shrink |
| M4 | `adaptive_geometry` | Keep 2x2 Mosaic but give larger quadrants to smaller-object sources |
| M5 | `hard_negative` | Replace at most one donor tile with a static mined hard-negative crop |

Common settings are `mosaic=1.0`, `close_mosaic=10`, `workers=8`,
`mosaic_visibility_thresh=0.70`, and no MixUp or Copy-Paste. The matrix runner
records the exact policy settings, split seed, training seed, canonical model
YAML, and commit in `experiment_manifest.json`.

## Preparation

Precompute M3 statistics from **training labels only**. Use `--imgsz 512` for
LEVIR-Ship and `--imgsz 640` for TinyPerson:

```bash
python tools/build_mosaic_scale_statistics.py \
  --labels /path/to/train/labels --imgsz 512 \
  --output /path/to/artifacts/levir_mosaic_scale_statistics.json
```

M5 uses a static bank mined once from the existing checkpoint. Review the bank
for unlabeled/ignore regions before training, especially on TinyPerson:

```bash
conda run -n ml2 python tools/mine_mosaic_hard_negatives.py \
  --weights /path/to/checkpoint.pt \
  --images /path/to/train/images --labels /path/to/train/labels \
  --output /path/to/artifacts/hard_negative_bank.json
```

The current matrix runner accepts `--hard-negative-bank` for M5. M3 accepts the
statistics path through the training kwargs when it is wired to a run; no test
or validation labels are used to derive thresholds.

## Later Marimo launch

Read `.agents/workflows/marimo-train.md` first and launch only through
`python -m utils.marimo_ops launch`. Use the existing dataset roots and fixed
`--split-seed 42`; vary only the requested training run. Example policy names:

```text
--policies M2_cluster_preserving M3_post_scale_constrained \
           M4_adaptive_geometry M5_hard_negative
```

Verify local artifacts and the remote upload after each dataset/policy run
before starting the next one.
