#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ait01/jetson_deployment_benchmark}
BUNDLE=${BUNDLE:-$ROOT/onnx_tensorrt_rerun_20261029}
OUT=${OUT:-$ROOT/DUY_DEPLOY/results/tinyperson_build_evaluate_20261102}
IMAGE=${IMAGE:-yolo-onnx-jetson:jp6}
BUILD_TIMEOUT=${BUILD_TIMEOUT:-1800}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-7200}

mkdir -p "$OUT/build_logs" "$OUT/eval_logs" "$OUT/status"

for spec in "yolov5 42" "yolov5 43" "yolov8 42" "yolov8 43" "yolov9 42" "yolov9 43" "yolov10 42" "yolov10 43" "yolo11 42" "yolo11 43"; do
    read -r model seed <<< "$spec"
    model_root="$ROOT/models/onnx_cuda_fp16/tinyperson/mosaic/$model/seed_$seed"
    job="$OUT/$model/seed_$seed"
    cache="$job/trt_cache"
    mkdir -p "$job" "$cache"
    log="$OUT/build_logs/${model}_seed_${seed}.log"
    echo "BUILD $model seed_$seed"
    status=0
    timeout --signal=TERM --kill-after=30 "$BUILD_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
        -v "$model_root:/models:ro" \
        -v "$BUNDLE/source/evaluate.py:/app/evaluate.py:ro" \
        -v "$BUNDLE/source/build_engine.py:/app/build_engine.py:ro" \
        -v "$cache:/cache" \
        --entrypoint python3 "$IMAGE" /app/build_engine.py \
        --model /models/best_fp16_batch1.onnx --cache /cache >"$log" 2>&1 || status=$?
    if (( status != 0 )); then
        printf '%s\n' "$status" > "$OUT/status/${model}_seed_${seed}.build_failed"
        echo "BUILD_FAILED $model seed_$seed status=$status"
    else
        touch "$OUT/status/${model}_seed_${seed}.build_ok"
        echo "BUILD_OK $model seed_$seed"
    fi
done

for spec in "yolov5 42" "yolov5 43" "yolov8 42" "yolov8 43" "yolov9 42" "yolov9 43" "yolov10 42" "yolov10 43" "yolo11 42" "yolo11 43"; do
    read -r model seed <<< "$spec"
    if [[ ! -f "$OUT/status/${model}_seed_${seed}.build_ok" ]]; then
        echo "SKIP_EVAL $model seed_$seed: build failed"
        continue
    fi
    model_root="$ROOT/models/onnx_cuda_fp16/tinyperson/mosaic/$model/seed_$seed"
    job="$OUT/$model/seed_$seed"
    log="$OUT/eval_logs/${model}_seed_${seed}.log"
    echo "EVAL $model seed_$seed"
    status=0
    timeout --signal=TERM --kill-after=30 "$EVAL_TIMEOUT" docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
        -v "$model_root:/models:ro" \
        -v /home/ait01/jetson_seed42_benchmark/data:/datasets:ro \
        -v "$BUNDLE/source/evaluate.py:/app/evaluate.py:ro" \
        -v "$BUNDLE/source/evaluate_tinyperson_merged.py:/app/evaluate_tinyperson_merged.py:ro" \
        -v "$job:/results" \
        -v "$job/trt_cache:/cache" \
        --entrypoint python3 "$IMAGE" /app/evaluate_tinyperson_merged.py \
        --models-root /models --model-path /models/best_fp16_batch1.onnx \
        --data-root /datasets --results-root /results/metrics \
        --predictions-root /results/predictions --trt-cache-root /cache \
        --batch-size 1 --conf 0.001 --nms-iou 0.5 --max-det 300 --save-only >"$log" 2>&1 || status=$?
    if (( status != 0 )); then
        printf '%s\n' "$status" > "$OUT/status/${model}_seed_${seed}.eval_failed"
        echo "EVAL_FAILED $model seed_$seed status=$status"
    else
        touch "$OUT/status/${model}_seed_${seed}.eval_ok"
        echo "EVAL_OK $model seed_$seed"
    fi
done
