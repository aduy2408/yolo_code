#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_marimo_workflow.sh \
    --environment NAME \
    --marimo-url URL \
    [--data-root PATH] [--dataset-yaml PATH] \
    [--train-script PATH] [--train-args JSON] [--artifact-root PATH] \
    [--operation train|evaluate|export] \
    [--marimo-repo PATH] [--dataset-root PATH] [--ref BRANCH_OR_SHA] [--watch]
EOF
  echo
  echo "The Marimo URL and token are updated only in the selected GitHub Environment."
  echo "The token is read silently from stdin or the interactive prompt."
}

operation=train
marimo_repo=/marimo/yolo_code
dataset_root=/marimo/yolo_code/datasets/ci
marimo_environment=
marimo_url=
data_root=/marimo/LevirShip/LevirShipData
dataset_yaml=
hf_repo_id=duyle2408/ci-oacp-smoke-runs
train_script=train_scripts/train_levir_scripts/train_all_levir_yolov9t_oacp.py
train_args='[]'
artifact_root='project/yolov9t_oacp_seed{seed}'
epochs=100
watch=false
workflow_ref="$(git branch --show-current 2>/dev/null || true)"
workflow_ref="${workflow_ref:-main}"

while (($#)); do
  case "$1" in
    --operation) operation=$2; shift 2 ;;
    --environment) marimo_environment=$2; shift 2 ;;
    --marimo-url) marimo_url=$2; shift 2 ;;
    --marimo-repo) marimo_repo=$2; shift 2 ;;
    --data-root) data_root=$2; shift 2 ;;
    --dataset-root) dataset_root=$2; shift 2 ;;
    --dataset-yaml) dataset_yaml=$2; shift 2 ;;
    --train-script) train_script=$2; shift 2 ;;
    --train-args) train_args=$2; shift 2 ;;
    --artifact-root) artifact_root=$2; shift 2 ;;
    --hf-repo-id) hf_repo_id=$2; shift 2 ;;
    --epochs) epochs=$2; shift 2 ;;
    --ref) workflow_ref=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --watch) watch=true; shift ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

: "${marimo_environment:?--environment is required}"
: "${marimo_url:?--marimo-url is required}"
[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || { echo "--epochs must be a positive integer" >&2; exit 2; }


if [[ -t 0 ]]; then
  read -r -s -p "Marimo token (not echoed): " marimo_token
  echo
else
  IFS= read -r marimo_token
fi
if [[ -z "$marimo_token" ]]; then
  echo "Marimo token is empty" >&2
  exit 2
fi

# Keep each user's/server's credentials isolated from other environments.
gh variable set MARIMO_URL --env "$marimo_environment" --body "$marimo_url"
printf '%s' "$marimo_token" | gh secret set MARIMO_TOKEN --env "$marimo_environment"
unset marimo_token

dispatch_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gh workflow run mlops.yml --ref "$workflow_ref" \
  -f operation="$operation" \
  -f marimo_environment="$marimo_environment" \
  -f marimo_repo="$marimo_repo" \
  -f data_root="$data_root" \
  -f dataset_root="$dataset_root" \
  -f dataset_yaml="$dataset_yaml" \
  -f train_script="$train_script" \
  -f train_args="$train_args" \
  -f artifact_root="$artifact_root" \
  -f dataset=levirship \
  -f hf_repo_id="$hf_repo_id" \
  -f split_seed=42 \
  -f epochs="$epochs" \
  -f image_size=512 \
  -f batch_size=8 \
  -f workers=4

run_json=
for _ in {1..15}; do
  run_json="$(gh run list --workflow mlops.yml --event workflow_dispatch --created ">=${dispatch_started}" --limit 1 --json databaseId,url,status,conclusion,createdAt 2>/dev/null || true)"
  if [[ "$run_json" != "[]" && -n "$run_json" ]]; then
    break
  fi
  sleep 2
done

if [[ "$run_json" == "[]" || -z "$run_json" ]]; then
  echo "Workflow dispatched, but run metadata is not visible yet." >&2
  echo "Use: gh run list --workflow mlops.yml --limit 5" >&2
  exit 0
fi

printf '%s\n' "$run_json" | python -c '
import json, sys
row = json.load(sys.stdin)[0]
print("run_id={}".format(row["databaseId"]))
print("url={}".format(row["url"]))
print("status={}".format(row["status"]))
print("conclusion={}".format(row["conclusion"]))
'

if [[ "$watch" == true ]]; then
  run_id="$(printf '%s\n' "$run_json" | python -c 'import json,sys; print(json.load(sys.stdin)[0]["databaseId"])')"
  gh run watch "$run_id" --exit-status --interval 20
fi
