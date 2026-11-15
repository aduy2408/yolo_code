# Jetson backend comparison — 2026-11-04

## Scope

Compared the real Jetson artifacts still present after removing the invalid `results/tensorrt/rerun_20261029` dry-run bundle:

- Unified PyTorch matrix: `results/unified_pt_jetson_seed42_43_20261103/metrics_all.json` and `results/unified_pt_jetson_seed44_20261103/metrics_all.json`.
- Legacy ONNX CUDA accuracy: `results/onnx_cuda_accuracy_legacy_20261020/onnx_cuda_accuracy/`.
- Shared PyTorch smoke matrix: `results/pt_shared_all5_levir_no_mosaic_seed42_20261031/`.
- Shared TensorRT attempt: `results/trt_shared_all5_levir_no_mosaic_seed42_20261031/`.

## Findings

1. `onnx_cuda_accuracy_legacy_20261020` contains 48 real JSON metrics, but the provider is `CUDAExecutionProvider`, not TensorRT. The files report `batch_size: 4` despite the `best_fp16_batch1` filename. Coverage is only YOLOv5/v8/v9; no YOLOv10 or YOLO11.
2. The shared TensorRT attempt contains only `failure.json` for the five LEVIR models; no successful TRT accuracy metric exists there.
3. The shared PyTorch path agrees closely with the new unified PyTorch path on LEVIR seed42 test for YOLOv5, YOLOv8, YOLOv9, and YOLO11 (absolute mAP50-95 difference <= 0.0010). Its YOLOv10 row is all zeros and is an invalid/unsupported result, not a counterexample.

## Legacy ONNX CUDA vs unified PyTorch test mAP50-95 means

| Dataset/model | Legacy ONNX CUDA | Unified PyTorch | Δ unified − legacy | Legacy batch/provider |
|---|---:|---:|---:|---|
| LEVIR YOLOv5 | 0.3039 | 0.3041 | +0.0002 | 4 / CUDA EP |
| LEVIR YOLOv8 | 0.3016 | 0.3051 | +0.0034 | 4 / CUDA EP |
| LEVIR YOLOv9 | 0.2944 | 0.2971 | +0.0027 | 4 / CUDA EP |
| TinyPerson YOLOv5 | 0.1798 | 0.1705 | -0.0093 | 4 / CUDA EP |
| TinyPerson YOLOv8 | 0.1830 | 0.1765 | -0.0065 | 4 / CUDA EP |
| TinyPerson YOLOv9 | 0.1875 | 0.1816 | -0.0060 | 4 / CUDA EP |
| Varroa YOLOv5 | 0.3359 | 0.3367 | +0.0008 | 4 / CUDA EP |
| Varroa YOLOv8 | 0.3327 | 0.3322 | -0.0005 | 4 / CUDA EP |
| Varroa YOLOv9 | 0.3200 | 0.3197 | -0.0003 | 4 / CUDA EP |

The means use the legacy files' available seeds: LEVIR/TinyPerson 42/43/44; Varroa 42/43. This is a sanity comparison, not a strict parity result, because the evaluator, provider, and batch size differ.

## Conclusion

- The current unified `.pt` Jetson metrics look structurally and numerically plausible.
- Independent shared-PyTorch results validate the current LEVIR v5/v8/v9/v11 test values closely.
- Varroa agrees closely with the legacy CUDA-EP path.
- TinyPerson has a repeatable moderate difference versus the legacy CUDA-EP path, especially on validation; it needs a same-contract rerun before attributing the gap to model/backend behavior.
- There is still no successful TensorRT accuracy result to compare. Do not label the legacy ONNX CUDA rows as TensorRT.
