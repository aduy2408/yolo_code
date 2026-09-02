# Project Ultralytics boundary

This package is the project-owned boundary for custom Ultralytics modules, losses, detection-loss adapters, and training glue. It exists so the upstream implementation can stay clean while old experiments remain reproducible.

## Repository layout

```text
vendor/ultralytics_upstream/   clean pinned upstream submodule
project_ultralytics/            project modules, losses, registry, adapters
models_related/ultralytics/     legacy compatibility fork for old experiments
models_related/models_config/   historical model and experiment YAML corpus
tests/                          project and YAML regression tests
```

## Source-of-truth rules

- Never edit `vendor/ultralytics_upstream` directly.
- New custom code belongs in `project_ultralytics`, not in the upstream tree.
- Keep `models_related/ultralytics` unchanged unless an experiment explicitly requires a compatibility patch. It is the oracle for historical YAML behavior.
- Do not rewrite or delete old YAMLs merely because they use legacy custom modules. Record whether a problem is a YAML call error, a legacy runtime limitation, a server-specific dependency difference, or an unmigrated module.
- Record the upstream submodule commit, project source commit, YAML path, data YAML, split, seed, and protocol for every experiment.

## Current project API

The public project registry is in `project_ultralytics.registry`:

```python
from project_ultralytics.registry import CUSTOM_MODULES, CUSTOM_LOSSES
from project_ultralytics.training import install_loss_adapter

install_loss_adapter(model, "ftal")       # factorized TAL
install_loss_adapter(model, "upstream")   # restore upstream criterion
```

Migrated module families currently include evidence/cue fusion, feature calibration, CBAM, KV attention, routing attention, NAT wrappers, PConv, input-cue modules, raw-cue fusion, `WeightedAdd`, and dual-stream formation (`DualChannelFormationBackbone`, `DualDownsample`, `DualCollapse`). Losses include Wise-IoU, boundary contrastive, localization quality, and FTAL target/loss helpers.

The registry is deliberately separate from the upstream parser. Automatic loading of every historical custom YAML through a clean upstream `YOLO(...)` entrypoint is not yet promised. Use the legacy fork for historical reproduction and use project modules directly or through an explicit adapter while parser integration is being completed.

## Testing

Use the configured ML environment:

```bash
cd /mnt/data/varroa/yolo_related
PYTHONPATH=. conda run -n ml2 pytest -q tests/test_project_fusion.py tests/test_project_dual_stream.py tests/test_custom_yaml_corpus.py
```

The YAML corpus test checks all architecture YAMLs for valid structure and loads representative historical YAMLs through the legacy fork. It is a regression boundary, not a claim that all old YAMLs are clean-upstream compatible.

The historical sweep currently covers 262 architecture YAMLs. NAT/natten API differences are server-specific and intentionally excluded from fixes. Unusual empty argument lists such as `ChannelAttention []` are normal for the legacy parser because channel arguments are injected by parser branches.

## Migration workflow

1. Check Git status and active jobs before editing.
2. Identify the exact YAML call and its parser branch or constructor signature.
3. Prefer a small project-owned migration with a focused unit test.
4. Validate model construction and a dummy forward before training.
5. Keep runtime artifacts outside the repository.
6. Commit only scoped changes and verify the upstream submodule remains clean.
