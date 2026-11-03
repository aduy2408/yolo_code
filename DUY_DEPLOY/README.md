# DUY_DEPLOY

Jetson Orin NX ONNX deployment bundle.

## Layout

```text
DUY_DEPLOY/
├── docker/jetson_onnx/        # ARM64 Jetson image context
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── preflight.py
│   ├── benchmark.py
│   ├── evaluate.py
│   └── prepare_cuda_models.py
└── models/
    ├── onnx_fp16_dynamic/     # Original 38 dynamic-batch FP16 exports
    ├── onnx_cuda_fp16/        # CUDA-only raw-head graphs for AP evaluation
    └── onnx_cpu_fp32/         # 38 local FP32 exports for CPU execution
```

Datasets are intentionally not included. Mount them on the Jetson at runtime.
Benchmark results and TensorRT engine caches should also stay outside the image.

## Build on the development PC

```bash
docker buildx build \
  --platform linux/arm64 \
  --file DUY_DEPLOY/docker/jetson_onnx/Dockerfile \
  --tag yolo-onnx-jetson:jp6 \
  --output type=oci,dest=/tmp/jetson-images/yolo-onnx-jetson-jp6.tar \
  DUY_DEPLOY/docker/jetson_onnx
```

Copy the image and models to the Jetson over the USB SSH link:

```bash
scp /tmp/jetson-images/yolo-onnx-jetson-jp6.tar \
  ait01@192.168.55.1:/home/ait01/

rsync -avP DUY_DEPLOY/models/ \
  ait01@192.168.55.1:/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/models/
```

Load the image on the Jetson:

```bash
docker load -i /home/ait01/yolo-onnx-jetson-jp6.tar
```

## Run preflight

```bash
docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/models:/models:ro \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results:/results \
  yolo-onnx-jetson:jp6
```

## Run the benchmark

```bash
docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/models:/models:ro \
  -v /home/ait01/jetson_seed42_benchmark/data:/datasets:ro \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results:/results \
  --entrypoint python3 \
yolo-onnx-jetson:jp6 \
  /app/benchmark.py \
  --models-root /models/onnx_fp16_dynamic \
  --results-root /results/onnx_dynamic_fp16_full \
  --provider cuda \
  --batch-size 4 \
  --warmup 50 \
  --iterations 200
```
The first benchmark phase is pure ONNX Runtime CUDA: it does not configure TensorRT or CPU fallback. If CUDA cannot execute a model or operator, the run fails instead of producing a mixed-backend measurement.

## CPU benchmark and accuracy evaluation

The same container can run ONNX Runtime's CPU provider on the Jetson. The CPU
bundle uses the 38 FP32 exports under `models/onnx_cpu_fp32`; it does not use
CUDA, TensorRT, or CPU fallback from another execution provider.

The original 38 FP16 exports under `models/onnx_fp16_dynamic` are also CPU
runnable and were benchmarked separately. They are retained as a comparison
bundle; FP32 is the recommended CPU accuracy path because it was faster on the
Jetson CPU.

```bash
docker run --rm \
  --ipc=host \
  -v /home/ait01/jetson_cpu_onnx/models:/models:ro \
  -v /home/ait01/jetson_cpu_onnx/results:/results \
  -v /home/ait01/jetson_cpu_onnx/app/benchmark.py:/app/benchmark.py:ro \
  --entrypoint python3 \
  yolo-onnx-jetson:jp6 \
  /app/benchmark.py \
  --models-root /models/onnx_cpu_fp32 \
  --results-root /results/onnx_cpu_benchmark \
  --provider cpu \
  --batch-size 1 \
  --warmup 10 \
  --iterations 50
```

```bash
for dataset in levirship tinyperson varroa; do
  docker run --rm \
    --ipc=host \
    -v /home/ait01/jetson_cpu_onnx/models:/models:ro \
    -v /home/ait01/jetson_seed42_benchmark/data:/datasets:ro \
    -v /home/ait01/jetson_cpu_onnx/results:/results \
    -v /home/ait01/jetson_cpu_onnx/app/evaluate.py:/app/evaluate.py:ro \
    --entrypoint python3 \
    yolo-onnx-jetson:jp6 \
    /app/evaluate.py \
    --models-root /models/onnx_cpu_fp32 \
    --data-root /datasets \
    --dataset "$dataset" \
    --split val test \
    --results-root /results/onnx_cpu_accuracy \
    --provider cpu \
    --batch-size 1 \
    --conf 0.001 \
    --nms-iou 0.5 \
    --max-det 300
done

## Reproducible accuracy evaluation

`prepare_cuda_models.py` creates CUDA-compatible raw-head ONNX graphs with static spatial dimensions and dynamic batch. The evaluator decodes the raw YOLO heads, applies NMS IoU `0.5`, and writes one JSON artifact per model and split.

```bash
docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/models:/models:ro \
  -v /home/ait01/jetson_seed42_benchmark/data:/datasets:ro \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results:/results \
  --entrypoint python3 \
  yolo-onnx-jetson:jp6 \
  /app/evaluate.py \
  --models-root /models/onnx_cuda_fp16/levirship/no_mosaic/yolov8/seed_42 \
  --data-root /datasets \
  --dataset levirship \
  --split val test \
  --results-root /results/onnx_cuda_accuracy \
  --provider cuda \
  --batch-size 4 \
  --conf 0.001 \
  --nms-iou 0.5 \
  --max-det 300
```

Each JSON records the model SHA-256, dataset, split, image count, provider, NMS IoU, `AP50-all`, `AP50-Small`, `mAP50-95-all`, and `mAP50-95-Small`. The evaluator refuses CPU fallback. YOLO10 and YOLO11 attention graphs currently contain CPU-preferred attention nodes under ONNX Runtime CUDA and are intentionally not included in the CUDA-only matrix until exported without those nodes.

The persistent `/results` mount stores benchmark JSON. TensorRT engine caching is reserved for the later TensorRT phase.

`rerun.sh accuracy` intentionally remains the legacy raw-head ONNX/TensorRT matrix and custom AP path. It does not invoke `model.val()`. For a unified `.pt`/Ultralytics-engine comparison, call the cloned-validator path explicitly with the same prepared YAML:

```bash
python onnx_tensorrt_rerun_20261029/source/evaluate.py \
  --model /path/to/model.pt \
  --dataset levirship \
  --dataset-yaml /path/to/prepared/levirship.yaml \
  --split test \
  --results-root /path/to/results \
  --conf 0.001 --nms-iou 0.5 --max-det 300
```

## TensorRT export and evaluation

TensorRT engines are built natively by ONNX Runtime's `TensorrtExecutionProvider` on the Jetson. The provider list contains only TensorRT, so unsupported operators fail instead of silently falling back to CUDA or CPU. The cache is the exported engine store and must remain on the Jetson because TensorRT engines are device/runtime-specific.

An ONNX Runtime TensorRT provider cache is not an Ultralytics `.engine` artifact; do not pass it to `YOLO(...)` unless it was exported and validated in the Ultralytics engine format.

Transfer the updated scripts, then benchmark all 38 dynamic ONNX models:

```bash
scp DUY_DEPLOY/docker/jetson_onnx/{benchmark.py,evaluate.py} \
  ait01@192.168.55.1:/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/

docker run --rm \
  --runtime=nvidia \
  --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/models/onnx_fp16_dynamic:/models:ro \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/tensorrt/matrix:/results \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/tensorrt/engines:/trt_engines \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/benchmark.py:/app/benchmark.py:ro \
  --entrypoint python3 \
  yolo-onnx-jetson:jp6 \
  /app/benchmark.py \
  --models-root /models \
  --results-root /results \
  --provider tensorrt \
  --trt-cache-root /trt_engines \
  --batch-size 4 \
  --warmup 50 \
  --iterations 200
```

Evaluate all 38 raw-head models on both validation and test splits for each dataset:

```bash
for dataset in levirship tinyperson varroa; do
  docker run --rm \
    --runtime=nvidia \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/models/onnx_cuda_fp16:/models:ro \
    -v /home/ait01/jetson_seed42_benchmark/data:/datasets:ro \
    -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/tensorrt/accuracy_final:/results \
    -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/tensorrt/engines:/trt_engines \
    -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/evaluate.py:/app/evaluate.py:ro \
    --entrypoint python3 \
    yolo-onnx-jetson:jp6 \
    /app/evaluate.py \
    --models-root /models \
    --data-root /datasets \
    --dataset "$dataset" \
    --split val test \
    --results-root /results \
    --provider tensorrt \
    --trt-cache-root /trt_engines \
    --batch-size 4 \
    --conf 0.001 \
    --nms-iou 0.5 \
    --max-det 300
done
```

The accuracy JSON files contain `AP50-all`, `AP50-Small`, `mAP50-95-all`, and `mAP50-95-Small`; `nms_iou` is `0.5`. TensorRT engine files are generated under the mounted `results/tensorrt/engines` directory.

TinyPerson test evaluation uses the required corner-window protocol. Run TensorRT inference once and retain crop-level predictions:

```bash
docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/models:/models:ro \
  -v /home/ait01/jetson_seed42_benchmark/data:/datasets:ro \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/tensorrt/accuracy_final:/results \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/tensorrt/engines_tinyperson_merged:/trt_engines \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/evaluate.py:/app/evaluate.py:ro \
  -v /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/evaluate_tinyperson_merged.py:/app/evaluate_tinyperson_merged.py:ro \
  --entrypoint python3 yolo-onnx-jetson:jp6 /app/evaluate_tinyperson_merged.py \
  --models-root /models/onnx_cuda_fp16/tinyperson \
  --data-root /datasets --results-root /results \
  --predictions-root /results/tinyperson_predictions \
  --trt-cache-root /trt_engines --batch-size 4 --conf 0.001 --nms-iou 0.5
```

The `.npz` files under `results/tensorrt/accuracy_final/tinyperson_predictions` contain crop IDs, boxes, and scores. Merge and evaluate them later without TensorRT:

```bash
python3 /app/merge_tinyperson_predictions.py \
  --predictions-root /results/tinyperson_predictions \
  --data-root /datasets --results-root /results --nms-iou 0.5
```
