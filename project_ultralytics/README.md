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

## How to train

### 1. Clean upstream model with project FTAL loss

Run from the repository root. Put the clean upstream package before the system
installation on `PYTHONPATH`, and put the repository root there so Python can
find `project_ultralytics`:

```bash
cd /mnt/data/varroa/yolo_related
PYTHONPATH=vendor/ultralytics_upstream:. conda run -n ml2 python - <<'PY'
from ultralytics import YOLO
from project_ultralytics.training import train_with_loss_adapter

model = YOLO(
    "vendor/ultralytics_upstream/ultralytics/cfg/models/v8/yolov8.yaml",
    task="detect",
)

results = train_with_loss_adapter(
    model,
    loss_adapter="ftal",
    data="datasets/varroa_yolo/varroa.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    workers=0,
    device="0",             # use "cpu" for a CPU smoke test
    project="/home/duylearch/.jcode/scratch/project_runs",
    name="varroa_yolov8_ftal",
)
print(results)
PY
```

The important part is that `install_loss_adapter()` runs **before**
`model.train()`. The helper `train_with_loss_adapter()` does both steps. To use
the clean upstream loss instead:

```python
from project_ultralytics.training import install_loss_adapter

install_loss_adapter(model, "upstream")
model.train(data="datasets/varroa_yolo/varroa.yaml", epochs=1)
```

### 2. Explicit version when more control is needed

```python
from ultralytics import YOLO
from project_ultralytics.training import install_loss_adapter

model = YOLO("vendor/ultralytics_upstream/ultralytics/cfg/models/v8/yolov8.yaml")
install_loss_adapter(model, loss_adapter="factorized_tal")

# Optional switches are read from model.model.args when the criterion is built.
args = model.model.args
if isinstance(args, dict):
    args["factorized_tal_target"] = True
    args["factorized_tal_p2_only"] = True
else:
    args.factorized_tal_target = True
    args.factorized_tal_p2_only = True

results = model.train(
    data="datasets/varroa_yolo/varroa.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    workers=0,
    project="/home/duylearch/.jcode/scratch/project_runs",
    name="varroa_yolov8_ftal_explicit",
)
```

### 3. Inference after training

```python
from ultralytics import YOLO

model = YOLO("/path/to/project_runs/varroa_yolov8_ftal/weights/best.pt")
predictions = model.predict(
    source="datasets/varroa_yolo/images/val",
    imgsz=640,
    conf=0.25,
    iou=0.50,
    save=True,
)
```

### 4. Reproduce an old experiment

Old YAMLs that use modules still living only in the legacy fork must be run
with the legacy package explicitly. Do not mix this with the clean upstream
environment:

```bash
cd /mnt/data/varroa/yolo_related
PYTHONPATH=models_related/ultralytics conda run -n ml2 python - <<'PY'
from ultralytics import YOLO

model = YOLO(
    "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml",
    task="detect",
)
model.train(
    data="datasets/varroa_yolo/varroa.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    workers=0,
    project="/home/duylearch/.jcode/scratch/legacy_runs",
    name="old_yaml_reproduction",
)
PY
```

`project_ultralytics.registry` is not a magic global import. A historical YAML
that names a custom class cannot be loaded by clean upstream `YOLO(...)` until
that class and its parser/runtime call contract have been migrated explicitly.

Migrated module families currently include evidence/cue fusion, feature calibration, CBAM, KV attention, routing attention, NAT wrappers, PConv, input-cue modules, raw-cue fusion, `WeightedAdd`, and dual-stream formation (`DualChannelFormationBackbone`, `DualDownsample`, `DualCollapse`). The first project-owned detection head is `DetectClsAttention` with `cbam`, `kvca`, and `kvca_block` variants. Losses include Wise-IoU, boundary contrastive, localization quality, and FTAL target/loss helpers.

The registry is deliberately separate from the upstream parser. Automatic loading of every historical custom YAML through a clean upstream `YOLO(...)` entrypoint is not yet promised. Use the legacy fork for historical reproduction and use project modules directly or through an explicit adapter while parser integration is being completed.

The project-owned parser bridge currently supports YAMLs using the migrated
`WeightedAdd`, attention/block, input-cue, evidence/cue, calibration, and
`DetectClsAttention` modules. For example:

```python
from project_ultralytics import load_project_model
from project_ultralytics.training import train_with_loss_adapter

model = load_project_model(
    "models_related/models_config/yolov8/levir/"
    "yolov8n_p2_fpn_only_kvca_clsonly.yaml",
    task="detect",
)

train_with_loss_adapter(
    model,
    loss_adapter="ftal",
    data="datasets/varroa_yolo/varroa.yaml",
    epochs=1,
    imgsz=64,
    batch=1,
    workers=0,
    device="cpu",
    project="/home/duylearch/.jcode/scratch/project_runs",
    name="project_detect_cls_attention_ftal_smoke",
)
```

The bridge is intentionally explicit and scoped. It does not modify the
upstream package or silently make every legacy YAML compatible. The remaining
historical heads, including NUDFL, HV-decoupled, and GCTS variants, are not yet
migrated. Use the legacy fork for those experiments until their project-owned
replacement is available.

## Testing

Use the configured ML environment:

```bash
cd /mnt/data/varroa/yolo_related
PYTHONPATH=. conda run -n ml2 pytest -q tests/test_project_fusion.py tests/test_project_dual_stream.py tests/test_custom_yaml_corpus.py
```

The YAML corpus test checks all architecture YAMLs for valid structure and loads representative historical YAMLs through the legacy fork. It is a regression boundary, not a claim that all old YAMLs are clean-upstream compatible.

The historical sweep currently covers 262 architecture YAMLs. NAT/natten API differences are server-specific and intentionally excluded from fixes. Unusual empty argument lists such as `ChannelAttention []` are normal for the legacy parser because channel arguments are injected by parser branches. The corpus regression test is not a claim that all 262 files are clean-upstream compatible.

## Migration workflow

1. Check Git status and active jobs before editing.
2. Identify the exact YAML call and its parser branch or constructor signature.
3. Prefer a small project-owned migration with a focused unit test.
4. Validate model construction and a dummy forward before training.
5. Keep runtime artifacts outside the repository.
6. Commit only scoped changes and verify the upstream submodule remains clean.
