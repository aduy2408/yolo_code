#!/usr/bin/env bash
set -eu
fp32_pid=188743
while kill -0 "$fp32_pid" 2>/dev/null; do sleep 30; done
fp32_root=/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/augmentation_pt_engine_20261105
engines=$(find "$fp32_root" -mindepth 3 -maxdepth 3 -type f -name best.engine | wc -l)
statuses=$(find "$fp32_root" -mindepth 2 -maxdepth 2 -type f -name status.json | wc -l)
log=/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/logs/fp16_after_fp32.log
if [ "$engines" -ne 24 ] || [ "$statuses" -ne 24 ]; then
  printf 'FP32 queue incomplete: engines=%s statuses=%s\n' "$engines" "$statuses" >> "$log"
  exit 1
fi
printf 'FP32 complete; launching augmentation FP16 engine queue\n' >> "$log"
fp16_root=/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/augmentation_fp16_engine_20261105
/home/ait01/jetson_orin_batch16/.venv/bin/python /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/queue_augmentation_pt_engine.py --manifest /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/manifests/augmentation_pt_engine_manifest.json --checkpoint-root /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/augmentation_seed43_44_checkpoints --data-root /home/ait01/jetson_seed42_benchmark/data --results-root "$fp16_root" --image duy-yolo-pytorch:jetpack6 --max-seeds 2 --skip-pt --tinyperson-save-only --precision fp16 >> "$log" 2>&1
printf 'Augmentation FP16 complete; launching Varroa FP32 engine queue\n' >> "$log"
varroa_fp32_root=/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/varroa_oacp_fp32_engine_20261105
/home/ait01/jetson_orin_batch16/.venv/bin/python /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/queue_augmentation_pt_engine.py --manifest /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/manifests/varroa_oacp_pt_engine_manifest.json --checkpoint-root /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/augmentation_seed43_44_checkpoints --data-root /home/ait01/jetson_seed42_benchmark/data --results-root "$varroa_fp32_root" --image duy-yolo-pytorch:jetpack6 --max-seeds 2 --skip-pt --precision fp32 >> "$log" 2>&1
printf 'Varroa FP32 complete; launching Varroa FP16 engine queue\n' >> "$log"
varroa_fp16_root=/home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/varroa_oacp_fp16_engine_20261105
exec /home/ait01/jetson_orin_batch16/.venv/bin/python /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/queue_augmentation_pt_engine.py --manifest /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/manifests/varroa_oacp_pt_engine_manifest.json --checkpoint-root /home/ait01/jetson_deployment_benchmark/DUY_DEPLOY/results/augmentation_seed43_44_checkpoints --data-root /home/ait01/jetson_seed42_benchmark/data --results-root "$varroa_fp16_root" --image duy-yolo-pytorch:jetpack6 --max-seeds 2 --skip-pt --precision fp16 >> "$log" 2>&1
