#!/usr/bin/env bash
set -euo pipefail

JETSON=${JETSON:-jetson}
REMOTE_ROOT=${REMOTE_ROOT:-/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/augmentation_onnx_trt_eval_20261103}
STAGE=${STAGE:-/tmp/augmentation_pt_stage_20261103}
OACP_REPO=duyle2408/model-augmentation-seed42-oacp-runs
CP_REPO=duyle2408/model-augmentation-seed42-copy-paste-runs
CP_RERUN_REPO=duyle2408/model-augmentation-seed42-missing-copy-paste-rerun-runs

rm -rf "$STAGE"
mkdir -p "$STAGE"
for spec in \
  "levir yolov8 oacp r2 no_mosaic" "levir yolov8 nc2 negative_canvas_r2 no_mosaic" "levir yolov8 nc4 negative_canvas_r4 no_mosaic" \
  "levir yolov11 oacp r2 no_mosaic" "levir yolov11 nc2 negative_canvas_r2 no_mosaic" "levir yolov11 nc4 negative_canvas_r4 no_mosaic" \
  "tinyperson yolov8 oacp r2 mosaic" "tinyperson yolov8 nc2 negative_canvas_r2 mosaic" "tinyperson yolov8 nc4 negative_canvas_r4 mosaic" \
  "tinyperson yolov11 oacp r2 mosaic" "tinyperson yolov11 nc2 negative_canvas_r2 mosaic" "tinyperson yolov11 nc4 negative_canvas_r4 mosaic"; do
  read -r dataset model aug variant mosaic <<< "$spec"
  repo="$CP_REPO"; method=copy_paste
  if [[ "$dataset" == tinyperson && "$model" == yolov8 && "$aug" == nc2 ]]; then
    repo="$CP_RERUN_REPO"
  elif [[ "$dataset" == tinyperson && "$model" == yolov8 && "$aug" == nc4 ]]; then
    repo="$CP_RERUN_REPO"
  fi
  [[ "$aug" == oacp ]] && repo="$OACP_REPO" && method=oacp
  relative="$dataset/$model/$aug/best.pt"
  url="https://huggingface.co/datasets/$repo/resolve/main/$model/$dataset/$method/$variant/$mosaic/seed_42/weights/best.pt"
  target="$STAGE/$relative"
  mkdir -p "$(dirname "$target")"
  echo "DOWNLOAD $relative"
  curl --fail --location --retry 3 --retry-delay 5 "$url" -o "$target"
done
tar -C "$STAGE" -cf - . | ssh "$JETSON" "tar -C '$REMOTE_ROOT/checkpoints' -xf -"
ssh "$JETSON" "find '$REMOTE_ROOT/checkpoints' -type f -name best.pt | wc -l"
