# Standalone YOLO Experiments For Varroa

This folder is independent from `object_detection_related`. It reads the
original dataset layout directly:

```text
train|val|test/
  videos/<video-id>/*.png
  labels/<video-id>/*.txt
```

Original labels contain a first count line, then absolute pixel `xyxy` boxes.
The converter writes Ultralytics YOLO labels with class `0 = varroa`.

Activate an environment with `torch`, `ultralytics`, and `pillow` installed:

```bash
conda activate ml2
python --version
```

If your shell cannot use `conda activate`, use `conda run -n ml2` before the
`python` command. The training and evaluation scripts default to `device=cpu`.

## Prepare Dataset

Full conversion:

```bash
python yolo_related/prepare_dataset.py \
  --root . \
  --out-dir yolo_related/datasets/varroa_yolo
```

Small smoke dataset:

```bash
python yolo_related/prepare_dataset.py \
  --root . \
  --out-dir /tmp/varroa_yolo_smoke \
  --limit 2
```

## Train

YOLOv8 baseline:

```bash
python yolo_related/train.py \
  --root . \
  --weights yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch-size 4 \
  --device cpu \
  --name yolov8n_varroa_cpu
```

Wise-IoU bounding box loss can be enabled through the local Ultralytics patch:

```bash
python yolo_related/train.py \
  --root . \
  --model-yaml yolo_related/models/yolov8_varroa_custom.yaml \
  --pretrained yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch-size 4 \
  --device cpu \
  --bbox-iou-loss wiou \
  --name yolov8n_varroa_wiou_cpu
```

Use `--wiou-monotonous` for the monotonic Wise-IoU focusing mode. Without it,
Wise-IoU uses the non-monotonic focusing mode from the reference implementation.

YOLOv8 P2 four-head model with Wise-IoU:

```bash
python models_related/train_eval/train.py \
  --root data \
  --yolo-dir datasets/varroa_yolo \
  --model-yaml models_related/models_config/yolov8_varroa_p2_wiou.yaml \
  --pretrained yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch-size 4 \
  --device cpu \
  --bbox-iou-loss wiou \
  --name yolov8n_varroa_p2_wiou
```

YOLOv8 custom YAML with pretrained partial load:

```bash
python yolo_related/train.py \
  --root . \
  --model-yaml yolo_related/models/yolov8_varroa_custom.yaml \
  --pretrained yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch-size 4 \
  --device cpu \
  --name yolov8n_varroa_custom_cpu
```

YOLOv10 baseline, if your Ultralytics version supports the weight:

```bash
python yolo_related/train.py \
  --root . \
  --weights yolov10n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch-size 4 \
  --device cpu \
  --name yolov10n_varroa_cpu
```

## Evaluate

```bash
python yolo_related/eval.py \
  --root . \
  --weights yolo_related/runs/train/yolov8n_varroa_cpu/weights/best.pt \
  --split test \
  --device cpu
```

Outputs:

- `yolo_related/runs/eval/<name>/test_per_image.csv`
- `yolo_related/runs/eval/<name>/test_summary.csv`

## Local Ultralytics Clone

Custom YAML module names are resolved by Ultralytics internals, not by normal
project imports. This repo uses a patched local clone:

```text
yolo_related/ultralytics
```

`yolo_related/train.py` and `yolo_related/eval.py` call
`prefer_local_ultralytics()` before importing `ultralytics`, so they use this
clone automatically.

Manual one-liners do not use the clone unless you add it to `PYTHONPATH`:

```bash
PYTHONPATH=yolo_related/ultralytics \
python -c \
"import ultralytics; print(ultralytics.__file__)"
```

Expected path:

```text
<repo>/yolo_related/ultralytics/ultralytics/__init__.py
```

If the path contains `site-packages/ultralytics`, your manual command is not
using the patched clone and custom block names may fail with `KeyError`.

## Current Custom Blocks

These blocks are already registered in the patched clone:

- `VarroaConvBlock`
- `VarroaSEBlock`

Registration locations:

- `yolo_related/ultralytics/ultralytics/nn/modules/block.py`
- `yolo_related/ultralytics/ultralytics/nn/modules/__init__.py`
- `yolo_related/ultralytics/ultralytics/nn/tasks.py`

`yolo_related/custom_blocks.py` is only a readable reference copy. The YAML
resolver uses the classes registered inside the local Ultralytics clone.

`yolo_related/models/yolov8_varroa_custom.yaml` currently uses a real custom
block:

```yaml
- [-1, 3, VarroaConvBlock, [128, 3, True]]
```

## Step By Step: Replace A Block

Prefer replacement first because layer indices stay the same.

For YOLOv8, edit:

```text
yolo_related/models/yolov8_varroa_custom.yaml
```

For YOLOv10, edit:

```text
yolo_related/models/yolov10_varroa_custom.yaml
```

Original YOLOv8 line:

```yaml
- [-1, 3, C2f, [128, True]]
```

Custom replacement:

```yaml
- [-1, 3, VarroaConvBlock, [128, 3, True]]
```

Layer format:

```text
[from, repeats, module, args]
```

For `VarroaConvBlock`, YAML args are:

```text
[c2, kernel_size, shortcut]
```

Because `VarroaConvBlock` is in `parse_model()` `base_modules`, Ultralytics
injects `c1` from the previous layer and scales `c2` according to the selected
model scale.

### What `c1` And `c2` Mean

Ultralytics modules usually use:

- `c1`: input channels, inferred from the previous layer.
- `c2`: output channels, written as the first value in YAML `args`.

You do not write `c1` in the YAML for modules in `base_modules`. For this line:

```yaml
- [-1, 3, VarroaConvBlock, [128, 3, True]]
```

Ultralytics reads:

```text
from=-1, repeats=3, module=VarroaConvBlock, args=[128, 3, True]
```

Then `parse_model()` converts it to a constructor call like:

```python
VarroaConvBlock(c1=previous_layer_channels, c2=scaled_128, kernel_size=3, shortcut=True)
```

With YOLOv8 nano scale, `128` is scaled by width multiplier `0.25`, so the
actual `c2` becomes `32`. That is why model printout can show:

```text
Conv2d(32, 32, kernel_size=(3, 3), ...)
```

For blocks not in `base_modules`, Ultralytics does not inject `c1/c2`; you must
handle their args manually in `parse_model()`.

## Step By Step: Insert A Block

Insert only after a replacement build works. Example after `SPPF`:

Edit the model YAML you are training, usually:

```text
yolo_related/models/yolov8_varroa_custom.yaml
```

```yaml
- [-1, 1, SPPF, [1024, 5]]
- [-1, 1, VarroaSEBlock, [1024, 8]]
```

For `VarroaSEBlock`, YAML args are:

```text
[c2, reduction]
```

Warning: inserting a layer shifts later layer indices. Update downstream
`from` references such as `[-1, 9]` or `[15, 18, 21]` when needed.

## Step By Step: Add A New Block Class

For a new block named `MyBlock`:

1. Add the class to `yolo_related/ultralytics/ultralytics/nn/modules/block.py`.
2. Add `"MyBlock"` to `block.py` `__all__`.
3. Import `MyBlock` in `yolo_related/ultralytics/ultralytics/nn/modules/__init__.py`.
4. Add `"MyBlock"` to `ultralytics/nn/modules/__init__.py` `__all__`.
5. Import `MyBlock` in `yolo_related/ultralytics/ultralytics/nn/tasks.py`.
6. Add `MyBlock` to `base_modules` in `parse_model()` if its constructor starts
   with `(c1, c2, ...)`.
7. Add `MyBlock` to `repeat_modules` only if its constructor expects an
   internal repeat count `n`, like `C2f(c1, c2, n, ...)`.
8. Reference `MyBlock` in the model YAML you train, for example
   `yolo_related/models/yolov8_varroa_custom.yaml`.

Simple class shape:

```python
class MyBlock(nn.Module):
    def __init__(self, c1: int, c2: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)
```

YAML:

```yaml
- [-1, 1, MyBlock, [256, 3]]
```

Ultralytics constructs it as:

```python
MyBlock(c1=previous_layer_channels, c2=256, kernel_size=3)
```

## Step By Step: Test After Editing

1. Confirm the patched clone is active:

```bash
PYTHONPATH=yolo_related/ultralytics \
python -c \
"import ultralytics; print(ultralytics.__file__)"
```

2. Build the model and print the edited layer:

```bash
PYTHONPATH=yolo_related/ultralytics \
python -c \
"from ultralytics import YOLO; m=YOLO('yolo_related/models/yolov8_varroa_custom.yaml'); print(m.model.model[2]); m.info(detailed=True)"
```

3. Run a dummy forward:

```bash
PYTHONPATH=yolo_related/ultralytics \
python -c \
"import torch; from ultralytics import YOLO; m=YOLO('yolo_related/models/yolov8_varroa_custom.yaml'); y=m.model(torch.randn(1, 3, 640, 640)); print(type(y))"
```

4. Check pretrained partial loading:

```bash
PYTHONPATH=yolo_related/ultralytics \
python -c \
"from ultralytics import YOLO; m=YOLO('yolo_related/models/yolov8_varroa_custom.yaml'); m.load('yolov8n.pt')"
```

Expected: a line like `Transferred X/Y items from pretrained weights` with
`X > 0`. The exact numbers change when the architecture changes. Lower transfer
counts are normal after replacing layers or changing channel shapes. The detect
head often transfers partially or not at all because this dataset uses `nc: 1`
while COCO pretrained weights use `nc: 80`.

5. Run a CPU smoke train:

```bash
python yolo_related/train.py \
  --root . \
  --yolo-dir /tmp/varroa_yolo_smoke \
  --model-yaml yolo_related/models/yolov8_varroa_custom.yaml \
  --pretrained yolov8n.pt \
  --epochs 1 \
  --imgsz 160 \
  --batch-size 2 \
  --device cpu \
  --workers 0 \
  --limit 2 \
  --project /tmp/varroa_yolo_runs \
  --name smoke_custom_block_cpu
```

## Safe Editing Rules

- Replace first, insert later.
- Keep custom block output `c2` compatible with the layer being replaced.
- Change one block at a time, then run the test steps.
- If inserting layers, update later `from` indices.
- Keep `Detect` for YOLOv8 and `v10Detect` for YOLOv10 unchanged until simpler
  backbone/neck edits train cleanly.

## Original YAML Files

Local patched clone:

```text
yolo_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml
```

Installed `ml2` package:

```bash
python -c \
"import ultralytics, pathlib; root=pathlib.Path(ultralytics.__file__).parent; print(root/'cfg'/'models'/'v8'/'yolov8.yaml')"
```
