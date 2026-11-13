# CI/CD / MLOps workflow

Workflow: `.github/workflows/mlops.yml`.

Workflow này chỉ kích hoạt bằng `workflow_dispatch`. GitHub Actions đóng vai trò client; code train/evaluate/export chạy trên checkout đã có sẵn ở Marimo thông qua `utils.marimo_run` và `utils.marimo_ops`.

## GitHub Environments

Server Marimo chỉ sống khoảng 12 giờ, nên mỗi user/server dùng một GitHub
Environment riêng:

```text
marimo-duy
marimo-user-2
```

Trong mỗi Environment tạo:

```text
Variable:
  MARIMO_URL=https://<server-host>

Secret:
  MARIMO_TOKEN=<token hiện tại của server>
```

Khi server mới được tạo, chỉ cần cập nhật `MARIMO_URL` và `MARIMO_TOKEN`
trong Environment tương ứng. User khác không bị ảnh hưởng.

Không nhập token vào form GitHub Actions. Chạy launcher local:

```bash
gh auth login
printf '%s\n' '<token mới>' \
  | scripts/run_marimo_workflow.sh \
      --environment marimo-duy \
      --marimo-url 'https://<server-host>' \
      --expected-sha '<SHA-của-checkout-trên-Marimo>' \
      --epochs 1 \
      --data-root '<LEVIR-Ship-root-trên-Marimo>' \
      --hf-repo-id '<user>/ci-oacp-smoke-runs' \
      --watch
```

`--epochs` truyền số epoch vào workflow. `--watch` chờ run hoàn tất và trả
exit code theo conclusion của GitHub Actions. Launcher in:

```text
run_id
url
status
conclusion
```

`dataset_yaml` là optional cho train; khi để trống workflow tự prepare. Nếu
không truyền `--run-dir`, launcher tự tạo path unique.

Launcher sẽ:

1. cập nhật `MARIMO_URL` trong Environment được chọn;
2. cập nhật `MARIMO_TOKEN` trong Environment đó;
3. tạo `run_dir` mới;
4. dispatch workflow với `marimo_environment`;
5. không in token.

Workflow dùng:

```yaml
environment: ${{ inputs.marimo_environment }}
MARIMO_URL: ${{ vars.MARIMO_URL }}
MARIMO_TOKEN: ${{ secrets.MARIMO_TOKEN }}
```

Vì vậy một repository có thể chạy đồng thời cho nhiều user mà không có
shared-secret race và không cần fork.

Training OACP cũng cần `HF_TOKEN` được cấu hình an toàn trong environment của
server Marimo, vì process training được launch trên Marimo chứ không chạy
trong GitHub runner.

## Contract của runner Marimo

`marimo_repo` phải là một checkout tồn tại trên server Marimo. Trước khi chạy,
checkout đó phải ở đúng `expected_sha` và clean. Với `operation=train`,
workflow tự gọi `misc.prepare_levir_ship.prepare` khi `dataset_yaml` để trống,
ghi YAML vào `dataset_root`, rồi mới chạy preflight. Python environment và
custom Ultralytics code phải tồn tại trên server.

Workflow không cài package, không sửa environment và không tự chọn mount
dataset. Default LEVIR mount là `/marimo/LevirShip/LevirShipData`.

## Pipeline gộp

Workflow chỉ có hai nhóm chính:

1. **Validate and run on Marimo**
   - kết nối Marimo;
   - prepare dataset/YAML nếu train và chưa truyền YAML;
   - ghi immutable `run_contract.json`;
   - kiểm tra exact SHA, Python, dataset YAML và artifact context bằng `marimo_ops`;
   - launch detached operation;
   - theo dõi process tới khi kết thúc.
2. **Upload workflow evidence**
   - lưu evidence của invocation GitHub Actions.

`utils/marimo_ops` vẫn là implementation của preflight, launch và status; workflow không tạo launcher thay thế.

## Ví dụ chạy OACP smoke

Dùng các input sau để chạy một job OACP rất nhỏ trên LEVIR-Ship:

```text
operation: train
marimo_environment: marimo-duy
marimo_repo: /marimo/yolo_code
expected_sha: <SHA-của-checkout-trên-Marimo>
run_dir: /marimo/yolo_code/runs/ci/oacp-smoke-<unique-id>
data_root: /marimo/LevirShip/LevirShipData
dataset_root: /marimo/yolo_code/datasets/ci
dataset_yaml: ""
dataset: levirship
hf_repo_id: <user>/ci-oacp-smoke-runs
seed: 43
split_seed: 42
epochs: 1
patience: 0
image_size: 512
batch_size: 8
workers: 4
device: 0
```

Entrypoint mặc định là:

```text
train_scripts/train_levir_scripts/train_all_levir_yolov9t_oacp.py
```

Run directory phải mới và collision-free. Không dùng lại run directory chưa được remote verify.

## Evaluation và export

`operation=evaluate` yêu cầu `checkpoint` trên Marimo và dùng evaluator hiện có với NMS IoU `0.5`.

`operation=export` yêu cầu `checkpoint` trên Marimo. Mặc định export ONNX; TensorRT chỉ dùng trên server có CUDA/TensorRT tương ứng.

## Giới hạn xác nhận

Job OACP một epoch là smoke/runtime check, không phải training result nghiên cứu. Kết quả chính thức vẫn phải kiểm tra checkpoint, manifest, metrics, split provenance và artifact upload sau run.
