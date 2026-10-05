# LevirShip Mosaic Metrics Report

**Updated:** 2026-10-05
**Scope:** LevirShip Mosaic variants only.

## Status

This is a separate report so LevirShip Mosaic is not mixed with the 76-prefix
seed 43/44 size-evaluator table. The values below are the existing
split-qualified LevirShip Mosaic metrics already present in the main report.
They are not new seed 43/44 size-evaluator outputs.

The seed 43/44 size-evaluator queue did not contain `augmentation-mosaic-runs/levir/...`
prefixes. Therefore native `val_size/*`, `test_size/*`, and TinyPerson-style
merged metrics are not claimed for these LevirShip Mosaic rows.

## LevirShip Mosaic metrics

`--` means the source artifact did not expose that metric.

| Variant | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-Small | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-Small | Provenance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Standard | 0.8044 | -- | 0.3118 | -- | 0.7653 | -- | 0.2876 | -- | Existing split-qualified source artifact |
| M2 Cluster-preserving | 0.7863 | -- | 0.2735 | -- | 0.7410 | -- | 0.2634 | -- | Existing split-qualified source artifact |
| M3 Post-scale-constrained | 0.7577 | -- | 0.2735 | -- | 0.7278 | -- | 0.2546 | -- | Existing split-qualified source artifact |
| M4 Adaptive-geometry | 0.7887 | -- | 0.2819 | -- | 0.7765 | -- | 0.2767 | -- | Existing split-qualified source artifact |
| M5 Hard-negative | 0.8180 | -- | 0.3079 | -- | 0.7883 | -- | 0.2921 | -- | Existing split-qualified source artifact |

## Coverage gap for a matching size-metric evaluation

| Requested artifact family | Current LevirShip Mosaic coverage in this file |
|---|---:|
| Split-qualified core metrics | 5/5 variants |
| Native `val_size/*` | 0/5 variants |
| Native `test_size/*` | 0/5 variants |
| `val_size/AP50-Small` | 0/5 variants |
| `test_size/AP50-Small` | 0/5 variants |
| `val_size/AP75` | 0/5 variants |
| `test_size/AP75` | 0/5 variants |

## Required follow-up for seed 43/44 parity

To produce LevirShip Mosaic metrics under the same protocol as the completed
76-prefix evaluator, a separate queue must be prepared for the five Mosaic
variants and their available training seeds. That queue must record its exact
checkpoint prefixes, commit, split seed, image size, workers, NMS IoU, output
HF dataset repository, and the native size/area-bucket protocol. No new
LevirShip Mosaic size evaluation is claimed by this file until that queue is
actually run and remotely verified.

The main augmentation report remains at:
`docs/reports/complete_main_report.md`.
