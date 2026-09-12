# YOLOv8 configuration index

These YAMLs are retained for historical reproduction. They are not interchangeable
with the clean default detector.

## Use this for new default runs

```text
../ultralytics/ultralytics/cfg/models/v8/yolov8.yaml
```

That path is the upstream-style P3/P4/P5 YOLOv8 model used by the current Mosaic
policy matrix. Do not substitute a file from this directory unless the experiment
explicitly requests a historical architecture.

## Historical groups

| Directory | Count | Scope |
|---|---:|---|
| `levir/` | 165 | LEVIR-Ship architecture and detector ablations |
| `tinyperson/` | 4 | TinyPerson detector variants |
| `varroa/` | 36 | Varroa detector experiments |
| `visdrone/` | 3 | VisDrone detector experiments |
| `kvca_sweep/` | 5 | KVCA architecture sweep |
| `tried/` | 59 | Earlier exploratory configurations |

## Filename vocabulary

- `p2`, `p2p3`, `p2p3p4`: extra small-object detection scales, not default YOLOv8.
- `cbam`, `channel`, `attention`, `kvca`, `self_attention`: custom attention or
  feature-calibration modules.
- `gap`, `gccc`, `dmm`, `ftal`, `tal`: custom head, loss, or training variants.
- `plain` and `baseline`: local historical names. They do not prove upstream
  equivalence; inspect the YAML and use the canonical upstream path for a true
  baseline.

Do not delete or move these files without updating every runner and test that
references their paths. Their paths are part of historical experiment provenance.
