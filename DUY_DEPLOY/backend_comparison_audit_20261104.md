# Backend comparison audit

## Finding

The current Jetson unified results cannot be numerically compared to the old TensorRT results because the old TensorRT accuracy run did not produce metric artifacts.

## Evidence

- Current PyTorch unified results:
  - `results/unified_pt_jetson_seed42_43_20261103/metrics_all.json`: 43 jobs, 86 split rows.
  - `results/unified_pt_jetson_seed44_20261103/metrics_all.json`: 9 jobs, 18 split rows.
- Historical TensorRT bundle inspected before cleanup: `results/tensorrt/rerun_20261029/`.
- It contained 29 `.log` files, all `DRY-RUN docker run ...` command records.
- `accuracy/` contained empty result directories.
- It contained **0 JSON metric files** and **0 observed accuracy files**.
- The invalid dry-run bundle was deleted locally and on Jetson after this audit.

## Protocol/path difference

- Unified matrix: `evaluate.py --model <best.pt>` → `evaluate_with_cloned_ultralytics()` → `model.val()`; `rect=False`, `batch=1`, `half=False`, `augment=False`, `conf=0.001`, `nms_iou=0.5`, `max_det=300`; val and native test.
- Old TensorRT accuracy commands were the legacy `--models-root` ONNX path using `onnx_cuda_fp16` and the custom NumPy scorer; the saved logs are dry-run only.

## Conclusion

- PyTorch Jetson FP32 metrics are real and saved.
- The old TensorRT FP16/FP32 accuracy numbers are **not present in this checkout**, so no metric delta or backend parity claim is currently grounded.
- The invalid TensorRT directory was deleted; no stale TensorRT result is retained.
- Definitive parity requires a real run using the same checkpoint, prepared YAML, split, and `evaluate.py` contract for `.pt` and `.engine`, then saving raw predictions and metric JSON for both.

The remaining real Jetson backend artifacts were compared in `DUY_DEPLOY/jetson_backend_comparison_20261104.md`. The legacy ONNX rows use `CUDAExecutionProvider`, not TensorRT; the shared TensorRT attempt has only failures.
