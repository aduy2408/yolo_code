# MMDetection ONNX and TensorRT export

MMDetection checkpoints are exported with **MMDeploy**, not the Ultralytics
exporter. `utils/export_mmdet.py` keeps the model config and checkpoint paired,
then invokes the target MMDeploy interpreter.

The MMDeploy checkout currently provides:

```text
configs/mmdet/detection/detection_onnxruntime_dynamic.py
configs/mmdet/detection/detection_tensorrt-fp16_dynamic-64x64-800x800.py
```


## Reproduce the environment

The pinned setup script clones MMDeploy `v1.3.1`, installs it into the target
interpreter, and installs ONNX Runtime plus TensorRT:

```bash
/marimo/mmdet-venv/bin/python tools/setup_mmdet_export.py \
  --mmdeploy-root /marimo/mmdeploy \
  --install
```

After obtaining a TensorRT SDK with `include/NvInfer.h` and
`lib/libnvinfer.so`, build the MMDeploy custom plugins:

```bash
/marimo/mmdet-venv/bin/python tools/setup_mmdet_export.py \
  --mmdeploy-root /marimo/mmdeploy \
  --tensorrt-sdk /marimo/tensorrt-sdk \
  --cudnn-dir /path/to/cudnn \
  --build-plugins
```

The command configures CMake with `BUILD_TENSORRT_OPS=ON`, installs
`libmmdeploy_tensorrt_ops.so` into the MMDeploy checkout, and verifies that
`TRTBatchedNMS` is registered.

## Reproduce the verified TensorRT engine

The verified Blackwell build uses NVIDIA's official TensorRT 10.15.1 CUDA 13.1
tar package and `trtexec`. It restricts CMake to two workers and TensorRT to
one tactic per layer:

```bash
python tools/build_mmdet_tensorrt.py \
  --mmdeploy-root /marimo/mmdeploy \
  --sdk-root /marimo/tensorrt-official \
  --cuda-root /usr/local/cuda-13.1 \
  --cudnn-root /marimo/mmdet-venv/lib/python3.11/site-packages/nvidia/cudnn \
  --onnx /marimo/mmdet_exports/levir_rtmdet_seed43/tensorrt/end2end.onnx \
  --download-sdk \
  --build-plugins \
  --build-engine \
  --build-workers 2
```

The script produces `end2end.engine` and `timing.cache`. The engine is built
for the fixed `1x3x800x800` input shape. `--maxTactics=1` is intentional: it
avoids the unbounded first-build tactic search on the RTX PRO 6000 Blackwell.

Run it again without `--install` to verify the environment. The script is
idempotent for an existing MMDeploy checkout and refuses a directory that is
not a valid checkout.
## Local artifacts

```bash
conda run -n ml2 python utils/export_mmdet.py \
  --checkpoint runs/mmdet/best.pth \
  --config runs/mmdet/patched_config.py \
  --deploy-config /path/to/mmdeploy/configs/mmdet/detection/<onnx-deploy-config>.py \
  --mmdeploy-root /path/to/mmdeploy \
  --mmdet-root /path/to/mmdetection \
  --backend onnx \
  --input datasets/LevirShipData/images/val/example.jpg \
  --work-dir exports/mmdet_onnx \
  --python /path/to/mmdet-venv/bin/python \
  --device cuda:0
```

Run TensorRT separately with the TensorRT deploy config:

```bash
conda run -n ml2 python utils/export_mmdet.py \
  --repo-id duyle2408/levirship_mmdet_yolo_protocol_seed43_runs \
  --hf-checkpoint-path levirship_yolo_protocol/no_mosaic/seed43/rtmdet/best_coco_bbox_mAP_epoch_59.pth \
  --hf-config-path levirship_yolo_protocol/no_mosaic/seed43/rtmdet/patched_config.py \
  --deploy-config /path/to/mmdeploy/configs/mmdet/detection/<tensorrt-deploy-config>.py \
  --mmdeploy-root /path/to/mmdeploy \
  --mmdet-root /path/to/mmdetection \
  --backend tensorrt \
  --input datasets/LevirShipData/images/val/example.jpg \
  --work-dir exports/mmdet_tensorrt \
  --python /marimo/mmdet-venv/bin/python \
  --device cuda:0
```

The script writes `export_manifest.json` and refuses to reuse a non-empty work
folder unless `--overwrite` is supplied. The deploy config controls input size,
static/dynamic shapes, precision, and backend-specific options. Use a matching
MMDeploy config from the same MMDeploy checkout; do not mix configs from a
second installation.

For an HF export, `--repo-id` may be either a dataset or model repository. If
paths are omitted, the script chooses `best.pth` and `patched_config.py` (or
`config.py`) when those names exist. Explicit paths are recommended because the
baseline repositories contain multiple runs.

## Baseline sources

The verified MMDetection baseline report is
[`docs/reports/mmdetection_baseline_runs_report.md`](reports/mmdetection_baseline_runs_report.md).
Its main source repositories are:

- `duyle2408/varroa_mmdet_yolo_protocol_runs`
- `duyle2408/levirship_mmdet_yolo_protocol_seed42_runs`
- `duyle2408/levirship_mmdet_yolo_protocol_seed43_runs`
- `duyle2408/tinyperson-mmdet-merged-eval-20261015-runs` (merged evaluation artifacts)

The report records the baseline matrix, checkpoint/config provenance, split seed
`42`, and evaluation NMS IoU `0.5`. It is the source of truth for metrics; an
export does not change or re-evaluate those results.

## Requirements and limitations

- The setup script pins ONNX exporter support for the server's modern Torch
  build and patches MMDeploy to call the legacy `torch.onnx` exporter.
- TensorRT detection exports using `mmdeploy::TRTBatchedNMS` require the
  MMDeploy TensorRT custom-plugin shared library. The Python TensorRT wheel
  alone does not provide that library or its development headers; build the
  MMDeploy `trt` backend against a TensorRT SDK before expecting an `.engine`.
- The utility performs export only. Validate numerical parity and detection
  metrics separately against the original MMDetection checkpoint.
- Export artifacts are intentionally not committed; `.onnx` and `.engine` are
  ignored by the repository.
