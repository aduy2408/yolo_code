#!/usr/bin/env bash
set -euo pipefail

: "${HF_REPO_ID:?HF_REPO_ID is required}"
: "${HF_PATH:?HF_PATH is required}"
: "${JETSON_HOST:?JETSON_HOST is required}"
: "${JETSON_USER:?JETSON_USER is required}"
: "${JETSON_SSH_KEY:?JETSON_SSH_KEY is required}"
: "${JETSON_REPO:?JETSON_REPO is required}"
: "${JETSON_DATA_ROOT:?JETSON_DATA_ROOT is required}"
: "${JETSON_IMAGE:?JETSON_IMAGE is required}"
: "${RUN_DIR:?RUN_DIR is required}"

operation=${OPERATION:?OPERATION is required}
precision=${PRECISION:-fp16}
model_kind=${MODEL_KIND:-engine}
if [[ "$operation" == "evaluate_jetson" ]]; then
  if [[ -n "${MODEL_NAME:-}" && "$MODEL_NAME" != "best.pt" ]]; then
    echo "Standalone evaluate_jetson only has transferred best.pt; use jetson_pipeline for an exported engine." >&2
    exit 2
  fi
  model_name=best.pt
elif [[ -n "${MODEL_NAME:-}" ]]; then
  model_name=$MODEL_NAME
else
  case "$model_kind" in
    pt) model_name=best.pt ;;
    onnx) model_name="best_${precision}.onnx" ;;
    engine) model_name="best_${precision}.engine" ;;
    *) echo "Unsupported MODEL_KIND: $model_kind" >&2; exit 2 ;;
  esac
fi
dataset=${DATASET:-levirship}
image_size=${IMAGE_SIZE:-512}
splits=${SPLITS:-"val test"}
local_dir="${GITHUB_WORKSPACE:-.}/${RUN_DIR}"
remote_dir="${JETSON_REPO}/runs/github-${GITHUB_RUN_ID:-manual}"
mkdir -p "$local_dir"

curl_args=(-fsSL)
if [[ -n "${HF_TOKEN:-}" ]]; then
  curl_args+=(-H "Authorization: Bearer ${HF_TOKEN}")
fi
curl "${curl_args[@]}" \
  "https://huggingface.co/datasets/${HF_REPO_ID}/resolve/main/${HF_PATH}?download=true" \
  -o "$local_dir/best.pt"
test -s "$local_dir/best.pt"

ssh_opts=(-i "$JETSON_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
target="${JETSON_USER}@${JETSON_HOST}"
ssh "${ssh_opts[@]}" "$target" "mkdir -p '$remote_dir/checkpoints' '$remote_dir/output' && test -f '$JETSON_REPO/DUY_DEPLOY/docker/pytorch_jetson/export_onnx.py' && test -f '$JETSON_REPO/DUY_DEPLOY/docker/pytorch_jetson/export_engine.py' && if [[ '$dataset' == tinyperson ]]; then test -f '$JETSON_REPO/DUY_DEPLOY/docker/pytorch_jetson/evaluate_ultralytics_tinyperson.py' && test -f '$JETSON_REPO/vendor/tinyperson_cocoeval.py'; else test -f '$JETSON_REPO/DUY_DEPLOY/docker/pytorch_jetson/evaluate_ultralytics.py'; fi"
scp "${ssh_opts[@]}" "$local_dir/best.pt" "$target:$remote_dir/checkpoints/best.pt"
scp "${ssh_opts[@]}" "$local_dir/best.pt" "$target:$remote_dir/output/best.pt"

if [[ "$operation" == "export_jetson" || "$operation" == "jetson_pipeline" ]]; then
  ssh "${ssh_opts[@]}" "$target" bash -s -- "$remote_dir" "$JETSON_REPO" "$JETSON_IMAGE" "$precision" "$image_size" <<'REMOTE'
set -euo pipefail
remote_dir=$1
jetson_repo=$2
image=$3
precision=$4
image_size=$5
common=(docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all
  -v "$remote_dir/checkpoints/best.pt:/models/best.pt:ro"
  -v "$remote_dir/output:/out"
  -v "$jetson_repo/DUY_DEPLOY/docker/pytorch_jetson/export_onnx.py:/app/export_onnx.py:ro"
  -v "$jetson_repo/DUY_DEPLOY/docker/pytorch_jetson/export_engine.py:/app/export_engine.py:ro")
"${common[@]}" --entrypoint python3 "$image" /app/export_onnx.py \
  --checkpoint /models/best.pt --output "/out/best_${precision}.onnx" --imgsz "$image_size" --precision "$precision"
"${common[@]}" --entrypoint python3 "$image" /app/export_engine.py \
  --checkpoint /models/best.pt --output "/out/best_${precision}.engine" --imgsz "$image_size" --precision "$precision"
REMOTE
fi
if [[ "$operation" == "evaluate_jetson" || "$operation" == "jetson_pipeline" ]]; then
  ssh "${ssh_opts[@]}" "$target" bash -s -- "$remote_dir" "$JETSON_REPO" "$JETSON_DATA_ROOT" "$JETSON_IMAGE" "$model_kind" "$model_name" "$splits" "$dataset" "$precision" <<'REMOTE'
set -euo pipefail
remote_dir=$1
jetson_repo=$2
data_root=$3
image=$4
model_kind=$5
model_name=$6
splits=$7
dataset=$8
precision=$9
read -r -a split_args <<< "$splits"
if [[ "$dataset" == tinyperson ]]; then
  docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$remote_dir/output:/models:ro" -v "$data_root:/datasets:ro" -v "$remote_dir/eval:/results" \
    -v "$jetson_repo/DUY_DEPLOY/docker/pytorch_jetson/evaluate_ultralytics_tinyperson.py:/app/evaluate_tinyperson.py:ro" \
    -v "$jetson_repo/DUY_DEPLOY/docker/pytorch_jetson/checkpoint_compat.py:/app/checkpoint_compat.py:ro" \
    -v "$jetson_repo/vendor/tinyperson_cocoeval.py:/app/tinyperson_cocoeval.py:ro" \
    --entrypoint python3 "$image" /app/evaluate_tinyperson.py \
    --model "/models/$model_name" --data-root /datasets --results-root /results \
    --evaluator /app/tinyperson_cocoeval.py --device 0 --precision "$precision" --conf 0.001 --iou 0.5 --max-det 300
else
  docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$remote_dir/output:/models:ro" -v "$data_root:/datasets:ro" -v "$remote_dir/eval:/results" \
    -v "$jetson_repo/DUY_DEPLOY/docker/pytorch_jetson/evaluate_ultralytics.py:/app/evaluate.py:ro" \
    -v "$jetson_repo/DUY_DEPLOY/docker/pytorch_jetson/checkpoint_compat.py:/app/checkpoint_compat.py:ro" \
    --entrypoint python3 "$image" /app/evaluate.py \
    --model "/models/$model_name" --data-root /datasets --dataset "$dataset" \
    --split "${split_args[@]}" --results-root /results --device 0 --precision "$precision" --conf 0.001 --iou 0.5 --max-det 300
fi
REMOTE
elif [[ "$operation" != "export_jetson" && "$operation" != "jetson_pipeline" ]]; then
  echo "Unsupported OPERATION: $operation" >&2
  exit 2
fi

scp -r "${ssh_opts[@]}" "$target:$remote_dir/output" "$local_dir/"
if [[ "$operation" == "evaluate_jetson" || "$operation" == "jetson_pipeline" ]]; then
  scp -r "${ssh_opts[@]}" "$target:$remote_dir/eval" "$local_dir/"
fi
printf 'JETSON_COMPLETE operation=%s local_dir=%s\n' "$operation" "$local_dir"
