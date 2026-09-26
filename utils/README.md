# LEVIR-Ship YOLO Automation Utilities

Thư mục này chứa các công cụ tự động hóa quá trình đánh giá, trích xuất thông số mô hình và cập nhật báo cáo thực nghiệm trên cả local và remote Blackwell server.

---

## Danh sách công cụ

### 1. `evaluate_all_huggingface_models.py`
Công cụ chạy sweep đánh giá toàn bộ 72 cấu hình checkpoint của 11 Hugging Face Dataset repo tại ngưỡng NMS IoU = 0.50 (hoặc bất kỳ ngưỡng nào được cấu hình).
* **Tính năng**: 
  * Tự động tạo Fixed Dataset Split (Seeds 42, 43, 44) nếu chưa tồn tại.
  * Tự động tải các checkpoint từ Hugging Face về cache của server.
  * Đánh giá song song trên cả hai tập `val` và `test`.
  * Ghi kết quả thô thành file JSON `runs/eval_nms_05_results.json`.
* **Cách chạy**:
  ```bash
  python3 utils/evaluate_all_huggingface_models.py
  ```

### 2. `get_model_stats.py`
Công cụ trích xuất chính xác cấu trúc tham số (Parameters) và độ phức tạp tính toán (GFLOPs) của từng cấu hình mạng.
* **Tính năng**:
  * Chạy dummy forward pass trên CUDA của server để Ultralytics tính toán chính xác số FLOPs tại kích thước ảnh `512x512`.
  * Lưu kết quả vào file JSON `runs/model_stats.json`.
* **Cách chạy**:
  ```bash
  python3 utils/get_model_stats.py
  ```

### 3. `update_reports.py`
Công cụ đọc dữ liệu JSON kết quả thực nghiệm và tự động định dạng thành các bảng Markdown so sánh chi tiết, chèn trực tiếp vào báo cáo.
* **Tính năng**:
  * Tự động tính toán giá trị trung bình và độ lệch chuẩn (`mean ± std`) đối với các cấu hình chạy 3 seeds.
  * Tự động cập nhật bảng so sánh FPN-Only tại NMS = 0.50 vào báo cáo `docs/reports/investigate_pooling.md`.
  * Tự động chèn bảng so sánh lớn gồm 28 cấu hình YOLO (cả NMS=0.70 và NMS=0.50) vào báo cáo `docs/reports/report_yolo.md`.
  * Tự động tra cứu Params/GFLOPs tương ứng cho từng dòng cấu hình.
* **Cách chạy**:
  ```bash
  python3 utils/update_reports.py
  ```

### 4. `fetch_results.py`
Công cụ kết nối tới Marimo Server từ máy local để lấy trực tiếp các file kết quả hoặc log chạy ngầm trên server Blackwell mà không cần thao tác thủ công.
* **Cách chạy**:
  ```bash
  python3 utils/fetch_results.py
  ```

### 5. `diagnose_test_fails.py`
Công cụ phân tích chẩn đoán chi tiết các ca dự đoán sai (Failure Cases) trên tập Test của mô hình FPN-Only tốt nhất (A2_200: GER), phân loại lỗi theo kích thước vật thể (Tiny/Small/Medium/Large).
* **Cách chạy**:
  ```bash
  python3 utils/diagnose_test_fails.py
  ```

### 6. `marimo_client.py` và `marimo_run.py`
`marimo_client.py` là lớp dùng chung cho kết nối Marimo từ local: đọc credential theo thứ tự environment → config local, đặt timeout, retry qua các session stale và parse SSE. `marimo_run.py` chỉ còn là compatibility CLI mỏng.

Ưu tiên dùng environment để không ghi token vào command history:

```bash
export MARIMO_URL="https://..."
export MARIMO_TOKEN="..."
python3 utils/marimo_run.py -c "print('ready')"
```

Nếu cần giữ cấu hình local cho máy cá nhân, file `utils/marimo_config.json` đã bị ignore và được ghi permission `0600`:

```bash
python3 utils/marimo_run.py --set-config <SERVER_URL> <TOKEN>
```

Không tạo thêm `.jcode_*.py` để giữ logic kết nối, launch hoặc monitor. Nếu cần thao tác mới, thêm vào `utils.marimo_client` hoặc `utils.marimo_ops`, rồi thêm regression test.

### 7. `marimo_ops.py`
Các thao tác deterministic cho runner chạy trên Marimo. Đây là helper opt-in
cho runner mới hoặc runner đang được chọn, không phải yêu cầu migrate hàng loạt
các runner cũ.

```bash
# Kiểm tra remote checkout trước khi chạy
python3 -m utils.marimo_ops preflight \
  --repo /marimo/yolo_code \
  --expected-sha "$EXPECTED_SHA" \
  --python /marimo/mmdet-venv/bin/python \
  --epochs 100 --patience 0 \
  --upload-required --hf-repo-id "$HF_REPO_ID"

# Chạy detached, tạo train.pid, train.log và state.json
python3 -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/<experiment> \
  -- /marimo/mmdet-venv/bin/python train_all_<experiment>.py \
  --epochs 100 --patience 0 --upload

# Kiểm tra process, tiến độ, artifact và trạng thái upload
python3 -m utils.marimo_ops status \
  --run-dir /marimo/yolo_code/runs/<experiment>

# Kiểm tra artifact contract trước khi báo run hoàn tất
python3 -m utils.marimo_ops artifacts \
  --run-dir /marimo/yolo_code/runs/<experiment>
```

`status` cố ý phân biệt `process_alive` với `observed_status`. Ví dụ,
`not_running_unverified` nghĩa là PID đã chết nhưng chưa đủ bằng chứng để kết
luận run hoàn tất. Xem policy đầy đủ tại `.agents/workflows/marimo-train.md`.
