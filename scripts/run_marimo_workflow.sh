#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_marimo_workflow.sh \
    --environment NAME \
    --marimo-url URL \
    --expected-sha SHA \
    --data-root PATH \
    [--dataset-yaml PATH] \
    [--hf-repo-id OWNER/REPO] \
    [--operation train|evaluate|export] \
    [--marimo-repo PATH] [--dataset-root PATH] [--run-dir PATH]

The Marimo URL and token are updated only in the selected GitHub Environment.
The token is read silently from stdin or the interactive prompt.
EOF
}

operation=train
marimo_repo=/marimo/yolo_code
dataset_root=/marimo/yolo_code/datasets/ci
run_dir=
marimo_environment=
marimo_url=
expected_sha=
data_root=/marimo/LevirShip/LevirShipData
dataset_yaml=
hf_repo_id=duyle2408/ci-oacp-smoke-runs

while (($#)); do
  case "$1" in
    --operation) operation=$2; shift 2 ;;
    --environment) marimo_environment=$2; shift 2 ;;
    --marimo-url) marimo_url=$2; shift 2 ;;
    --expected-sha) expected_sha=$2; shift 2 ;;
    --marimo-repo) marimo_repo=$2; shift 2 ;;
    --run-dir) run_dir=$2; shift 2 ;;
    --data-root) data_root=$2; shift 2 ;;
    --dataset-root) dataset_root=$2; shift 2 ;;
    --dataset-yaml) dataset_yaml=$2; shift 2 ;;
    --hf-repo-id) hf_repo_id=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

: "${marimo_environment:?--environment is required}"
: "${marimo_url:?--marimo-url is required}"
: "${expected_sha:?--expected-sha is required}"
: "${data_root:?--data-root is required}"
: "${hf_repo_id:?--hf-repo-id is required}"

if [[ -z "$run_dir" ]]; then
  run_dir="$marimo_repo/runs/ci/${operation}-$(date -u +%Y%m%dT%H%M%SZ)-${GITHUB_RUN_ID:-local}"
fi

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

gh workflow run mlops.yml \
  -f operation="$operation" \
  -f marimo_environment="$marimo_environment" \
  -f marimo_repo="$marimo_repo" \
  -f expected_sha="$expected_sha" \
  -f run_dir="$run_dir" \
  -f data_root="$data_root" \
  -f dataset_root="$dataset_root" \
  -f dataset_yaml="$dataset_yaml" \
  -f dataset=levirship \
  -f hf_repo_id="$hf_repo_id" \
  -f seed=43 \
  -f split_seed=42 \
  -f epochs=1 \
  -f patience=0 \
  -f image_size=512 \
  -f batch_size=8 \
  -f workers=4 \
  -f device=0

echo "Workflow dispatched in environment=$marimo_environment run_dir=$run_dir"
