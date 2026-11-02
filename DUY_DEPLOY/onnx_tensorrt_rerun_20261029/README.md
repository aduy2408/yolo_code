# Local ONNX/TensorRT runner

This folder is at the repository root and contains runnable files only. It is not a result bundle.

```text
onnx_tensorrt_rerun_20261029/
├── rerun.sh
└── source/
    ├── benchmark.py
    ├── benchmark_end_to_end.py
    ├── evaluate.py
    ├── evaluate_pytorch.py
    ├── evaluate_tinyperson_merged.py
    ├── merge_tinyperson_predictions.py
    ├── prepare_cuda_models.py
    ├── preflight.py
    ├── build_tensorrt_fp16.sh
    ├── benchmark_ultralytics.py
    ├── Dockerfile
    └── requirements.txt
```

The ONNX models are not duplicated. The runner reads `DUY_DEPLOY/models` by default.

`evaluate_pytorch.py` uses the same `letterbox`, output decoding, confidence filtering, NMS, and AP metric implementation as `evaluate.py`; only the model backend differs. Run it in the PyTorch Jetson image for a direct `.pt` versus TensorRT comparison.

## Run

From this folder on the Jetson checkout:

```bash
./rerun.sh benchmark --dataset all
```

The default input/model/data paths are:

- models: `DUY_DEPLOY/models`
- dataset: `/home/ait01/jetson_seed42_benchmark/data`
- Docker image: `yolo-onnx-jetson:jp6`
- output: `DUY_DEPLOY/results/tensorrt/rerun_20261029`
- batch: `4`
- warmup: `50`
- measured iterations: `200`

For Ultralytics-style end-to-end timing, including preprocessing, TensorRT inference, decode, and NMS:

```bash
./rerun.sh e2e --dataset all
```
Override paths without editing the runner:

```bash
MODELS_ROOT=/path/to/models \
DATA_ROOT=/path/to/data \
RESULTS_ROOT=/path/to/results \
./rerun.sh benchmark --dataset levirship
```

Reuse an existing TensorRT cache without rebuilding missing engines:

```bash
REUSE_TRT_CACHE_ROOT=/path/to/existing/engines \
./rerun.sh e2e --dataset all
```

Print commands without starting Docker:

```bash
./rerun.sh benchmark --dataset levirship --dry-run
```

Modes:

```bash
./rerun.sh benchmark --dataset all
./rerun.sh e2e --dataset all
./rerun.sh accuracy --dataset all
./rerun.sh all --dataset all
```

## Seed-42/43 prediction matrix

`run_seed_fallback_predictions.sh` selects seed 42 and seed 43 for every model. If seed 43 is unavailable for a dataset/model, it selects seed 44 instead. The split remains `seed_42` for all datasets.

- LEVIR-Ship: `no_mosaic`
- TinyPerson: `mosaic`, prediction-only corner-window `.npz` output; no merge or metrics
- Varroa: `mosaic`

Run sequentially on Jetson:

```bash
RESULTS_ROOT=/path/to/results \
./run_seed_fallback_predictions.sh
```

Validate selection without launching containers:

```bash
RESULTS_ROOT=/path/to/results \
./run_seed_fallback_predictions.sh --dry-run
```

By default the runner gives every model a separate TensorRT cache under the result root. With `REUSE_TRT_CACHE_ROOT`, the existing cache is mounted read-only; models without a matching engine are recorded as failures instead of triggering a rebuild.
