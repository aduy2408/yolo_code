# OACP R2 parity Jetson bundle

## Source and scope

TinyPerson and LEVIR-Ship checkpoints come from the recent parity rerun:

```text
duyle2408/oacp-r2-parity-rerun-seed43-44
```

Varroa checkpoints were downloaded from the authenticated HF repository:

```text
duyle2408/varroa-yolov8-yolo11-oacp-r2-seed43-44-runs
```

The four Varroa checkpoint files were SHA-256 verified after download. Their
seed43 and seed44 hashes are distinct. The previous Jetson staging folder had
duplicate copies; those are not used by this bundle.

The bundle contains twelve checkpoints:

```text
checkpoints/<dataset>/<model>/seed<43|44>/best.pt
```

Datasets: `tinyperson`, `levirship`, `varroa`. Models: `yolov8`, `yolov11`. Seeds: `43`, `44`.

## Contents

```text
.
├── export.py
├── evaluate.py
├── checkpoints/
│   ├── tinyperson/<model>/seed43|seed44/best.pt
│   ├── levirship/<model>/seed43|seed44/best.pt
│   └── varroa/<model>/seed43|seed44/best.pt
└── output/
```

Fixed settings: `conf=0.001`, `nms_iou=0.5`, `max_det=300`.

## Jetson prerequisites

```bash
cd /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/oacp_r2_parity_bundle_20261106
```

Required:

- Docker image `duy-yolo-pytorch:jetpack6`.
- Docker image `yolo-onnx-jetson:jp6`.
- Dataset `/home/ait01/jetson_seed42_benchmark/data`.
- `../runtime_deps/`.

## 1. Export ONNX

Export both precisions for all twelve checkpoints:

```bash
python3 export.py --dataset all --precision both
```

Or one dataset:

```bash
python3 export.py --dataset tinyperson --precision both
python3 export.py --dataset levirship --precision both
python3 export.py --dataset varroa --precision both
```

Outputs preserve provenance:

```text
output/onnx/<dataset>/<model>/seed<seed>/fp32.onnx
output/onnx/<dataset>/<model>/seed<seed>/fp16.onnx
```

The `.pt` files are read-only inputs.

## 2. Evaluate native PyTorch `.pt`

Run all three backends for one dataset:

```bash
python3 evaluate.py --dataset tinyperson --backend pt
python3 evaluate.py --dataset levirship --backend pt
python3 evaluate.py --dataset varroa --backend pt
```

TinyPerson writes crop predictions only; it does not merge or score them:

```text
output/predictions/tinyperson/pt/<model>/seed<seed>/*_test_corner_predictions.json
```

LEVIR-Ship and Varroa write native validation/test metrics:

```text
output/metrics/<dataset>/pt/<model>/seed<seed>/
```

## 3. Evaluate TensorRT FP32/FP16

Export ONNX first, then run:

```bash
python3 evaluate.py --dataset tinyperson --backend fp32
python3 evaluate.py --dataset tinyperson --backend fp16
python3 evaluate.py --dataset levirship --backend fp32
python3 evaluate.py --dataset levirship --backend fp16
python3 evaluate.py --dataset varroa --backend fp32
python3 evaluate.py --dataset varroa --backend fp16
```

TinyPerson writes predictions only:

```text
output/predictions/tinyperson/<backend>/<model>/seed<seed>/
```

LEVIR-Ship and Varroa write metrics:

```text
output/metrics/<dataset>/<backend>/<model>/seed<seed>/
```

TensorRT engine caches:

```text
output/trt_cache/<dataset>/<backend>/<model>/seed<seed>/
```

## 4. Run all three backends

```bash
python3 evaluate.py --dataset tinyperson --backend all
python3 evaluate.py --dataset levirship --backend all
python3 evaluate.py --dataset varroa --backend all
```

Backend meaning:

```text
pt    → native Ultralytics inference
fp32  → ONNX + TensorRT FP32
fp16  → ONNX + TensorRT FP16
```

## Final TinyPerson step

Keep PT/FP32/FP16 TinyPerson prediction files unchanged. Merge and score them later with the pinned TinyPerson evaluator. Do not substitute merged metrics into prediction generation.

Record with every artifact:

```text
checkpoint source: <source repo above>
nms_iou: 0.5
conf: 0.001
max_det: 300

## No-Docker x86/Colab flow

Use this variant when the host has `ultralytics`, `onnxruntime-gpu`, and the
ONNX Runtime `TensorrtExecutionProvider`, but does not have Docker or the
JetPack container images:

```bash
python3 export_nodocker.py --dataset tinyperson --precision both
python3 export_nodocker.py --dataset levirship --precision both

python3 evaluate_nodocker.py \
  --dataset tinyperson \
  --precision both \
  --data-root /path/to/prepared/data

python3 evaluate_nodocker.py \
  --dataset levirship \
  --precision both \
  --data-root /path/to/prepared/data
```

`export_nodocker.py` uses the same Ultralytics ONNX export settings as the
Docker exporter. `evaluate_nodocker.py` invokes the existing TensorRT
evaluators directly and requires the first active provider to be
`TensorrtExecutionProvider`; it does not silently fall back to CUDA or CPU.

This is an x86/Colab TensorRT flow, not a Jetson/JetPack measurement. Keep its
artifacts separate from Jetson results.
```
