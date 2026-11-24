# Model configuration index

This directory contains historical model and experiment YAMLs. It is intentionally
not the source of truth for the default detector used by current experiments.

## Canonical current baseline

For the Mosaic policy matrix and other default YOLOv8 comparisons, use:

```text
models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml
```

This is the upstream-style default YOLOv8 detector with P3/P4/P5 outputs. It is
not P2, CBAM, ChannelAttention, GAP, or FPN-only. The checkpoint is loaded
separately from `/marimo/yolov8n.pt`.

The custom Mosaic policies are augmentation policies layered onto that detector:
`standard`, `visibility`, `occupancy_match`, and `context_contrast`.

## Historical corpus

| Directory | Count | Purpose |
|---|---:|---|
| `yolov8/` | 272 | Historical YOLOv8 ablations and custom architecture experiments |
| `yolov9/` | 8 | Historical YOLOv9 experiments |
| `yolov5/` | 2 | Historical YOLOv5 experiments |
| `yolov10/` | 1 | Historical YOLOv10 experiment |

The large `yolov8/` tree contains many intentionally specialized files. Names
such as `plain`, `baseline`, `p2`, `cbam`, `gap`, and `fpn_only` describe the
historical experiment that created the file, not an automatic claim that it is
the upstream default model.

## Selection rules

1. For a default YOLOv8 run, use the canonical path above.
2. Use a file under this directory only when the experiment explicitly names the
   architecture or historical reproduction.
3. Do not delete or move historical YAMLs casually. Existing runners and tests
   reference them directly, and their paths are part of experiment provenance.
4. When adding a new custom architecture, place it under the appropriate
   historical family directory and add its purpose here or in a nearby README.
5. Before training, record the exact YAML path and SHA256 in the run manifest.

See the repository-level `AGENTS.md` for the enforced training and provenance
rules.
