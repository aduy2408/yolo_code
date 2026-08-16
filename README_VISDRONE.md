# VisDrone2019: Tiny Object Detection with Customized YOLO (P2/P3/P4 GAP + FTAL)

Tài liệu này tổng hợp toàn bộ thông tin về mục tiêu, kiến trúc, kịch bản chuẩn bị dữ liệu, **thống kê chi tiết tập dữ liệu**, và hướng dẫn chạy thực nghiệm cho bài toán phát hiện vật thể nhỏ mật độ cao sử dụng họ mô hình YOLOv8 tùy chỉnh trên tập dữ liệu **VisDrone2019-DET**.

---

## 1. Tổng Quan Dự Án & Mục Tiêu

Đối với ảnh chụp từ thiết bị bay không người lái (UAV/drone), các đối tượng mục tiêu (như người đi bộ, ô tô, xe máy) thường có kích thước rất nhỏ và xuất hiện với mật độ cực kỳ dày đặc. Mô hình YOLOv8 tiêu chuẩn gặp nhiều khó khăn trong việc phát hiện các vật thể này do giảm cỡ mẫu không gian quá nhanh ở các tầng sâu.

Dự án này tích hợp và thử nghiệm:
- **Kiến trúc mạng (Architecture):** Sử dụng nhánh P2 (stride 4), P3 (stride 8) và P4 (stride 16) trong plain neck để giữ lại đặc trưng không gian hạt mịn và phát hiện vật thể kích thước lớn. Áp dụng cơ chế **GAP ChannelAttention** độc lập trên cả 3 nhánh P2, P3 và P4 trước khi đưa vào đầu dự đoán Detect.
- **Hàm Loss (FTAL - Factorized TAL):** Điều chỉnh trọng tâm gán nhãn mục tiêu phân loại sau TAL (Task Alignment Learning) để tối ưu hóa việc học của các hộp giới hạn cực nhỏ.

---

## 2. Thống Kê Chi Tiết Tập Dữ Liệu VisDrone2019-DET

Dữ liệu VisDrone2019 được phân tách thành 3 tập chính: **Train** (Huấn luyện), **Val** (Kiểm định), và **Test** (Thử nghiệm - tập test-dev).

### 2.1. Thống kê tổng quan các Split

| Chỉ số | Train Split | Val Split | Test Split |
| :--- | :---: | :---: | :---: |
| **Số lượng hình ảnh** | 6,471 | 548 | 1,610 |
| **Tổng số Bounding Box** | 353,550 | 40,169 | 77,547 |
| **Số BBox hợp lệ (Valid)** | 343,205 (97.1%) | 38,759 (96.5%) | 75,102 (96.8%) |
| **Số BBox bị bỏ qua (Ignored)** | 10,345 (2.9%) | 1,410 (3.5%) | 2,445 (3.2%) |
| **Kích thước trung bình BBox** | $38.6 \times 37.4$ px | $31.1 \times 31.4$ px | $30.7 \times 32.4$ px |

---

### 2.2. Phân Phối Kích Thước Vật Thể (Valid BBoxes)

Phân loại kích thước dựa trên diện tích hộp giới hạn (Area = Width $\times$ Height):
* **Tiny (Cực nhỏ):** $\text{Area} < 256\text{ px}^2$ (Tương đương kích thước dưới $16 \times 16$ px)
* **Small (Nhỏ):** $256 \le \text{Area} < 1024\text{ px}^2$ (Tương đương kích thước từ $16 \times 16$ px đến $32 \times 32$ px)
* **Medium (Vừa):** $1024 \le \text{Area} < 9216\text{ px}^2$ (Tương đương kích thước từ $32 \times 32$ px đến $96 \times 96$ px)
* **Large (Lớn):** $\text{Area} \ge 9216\text{ px}^2$ (Tương đương kích thước trên $96 \times 96$ px)

| Kích thước | Train (343,205) | Val (38,759) | Test (75,102) |
| :--- | :---: | :---: | :---: |
| **Tiny** ($\text{Area} < 256$) | 89,209 (**26.0%**) | 11,950 (**30.8%**) | 27,341 (**36.4%**) |
| **Small** ($256 \le \text{Area} < 1024$) | 118,316 (**34.5%**) | 14,625 (**37.7%**) | 23,475 (**31.3%**) |
| **Medium** ($1024 \le \text{Area} < 9216$) | 116,696 (**34.0%**) | 11,116 (**28.7%**) | 21,876 (**29.1%**) |
| **Large** ($\text{Area} \ge 9216$) | 18,984 (**5.5%**) | 1,068 (**2.8%**) | 2,410 (**3.2%**) |

> [!NOTE]
> Nhóm vật thể siêu nhỏ và nhỏ (Tiny + Small) chiếm tới **60.5%** ở tập Train, **68.5%** ở tập Val và **67.7%** ở tập Test. Tuy nhiên, vẫn có khoảng **3-5%** là vật thể kích thước lớn (Large). Do đó, cấu trúc mạng của chúng ta tích hợp cả 3 nhánh P2, P3 và P4 để bao phủ toàn diện dải tỉ lệ kích thước.

---

### 2.3. Phân Phối Nhãn Lớp Đối Tượng (Class Distribution)

Tỷ lệ phần trăm và số lượng của từng lớp đối tượng hợp lệ (1-10) trên mỗi tập dữ liệu:

| Lớp | Tên lớp | Tập Train | Tập Val | Tập Test |
| :--- | :--- | :---: | :---: | :---: |
| **Class 1** | `pedestrian` (Người đi bộ) | 79,337 (23.1%) | 8,844 (22.8%) | 21,006 (28.0%) |
| **Class 2** | `people` (Người/đám đông) | 27,059 (7.9%) | 5,125 (13.2%) | 6,376 (8.5%) |
| **Class 3** | `bicycle` (Xe đạp) | 10,480 (3.1%) | 1,287 (3.3%) | 1,302 (1.7%) |
| **Class 4** | `car` (Ô tô con) | 144,867 (42.2%) | 14,064 (36.3%) | 28,074 (37.4%) |
| **Class 5** | `van` (Xe bán tải/tải nhỏ) | 24,956 (7.3%) | 1,975 (5.1%) | 5,771 (7.7%) |
| **Class 6** | `truck` (Xe tải) | 12,875 (3.8%) | 750 (1.9%) | 2,659 (3.5%) |
| **Class 7** | `tricycle` (Xe ba bánh) | 4,812 (1.4%) | 1,045 (2.7%) | 530 (0.7%) |
| **Class 8** | `awning-tricycle` (Xe ba bánh mái che) | 3,246 (0.9%) | 532 (1.4%) | 599 (0.8%) |
| **Class 9** | `bus` (Xe buýt) | 5,926 (1.7%) | 251 (0.6%) | 2,940 (3.9%) |
| **Class 10**| `motor` (Xe máy) | 29,647 (8.6%) | 4,886 (12.6%) | 5845 (7.8%) |

---

## 3. Cấu Trúc Mã Nguồn VisDrone

```text
.
├── train_visdrone_scripts/              # Thư mục chứa toàn bộ script chạy VisDrone
│   └── train_all_visdrone_yolov8n_p2_gap_ftal.py # Script chuẩn bị dữ liệu & huấn luyện tự động
├── models_related/
│   ├── ultralytics/                     # Mã nguồn core YOLO tùy biến (FTAL, ChannelAttention)
│   └── models_config/yolov8/visdrone/
│       └── yolov8n_p2p3p4_visdrone_plain_gap.yaml  # File cấu hình cấu trúc mạng P2/P3/P4 plain neck với GAP
```

---

## 4. Hướng Dẫn Chạy Thực Nghiệm (Quick Start)

> [!IMPORTANT]
> Kích hoạt môi trường conda trước khi chạy:
> ```bash
> source /home/duylearch/miniconda3/bin/activate ml2
> ```

### Bước 1: Khởi động quá trình Huấn luyện & Đánh giá tự động
Chạy lệnh dưới đây từ thư mục gốc của repository:
```bash
python train_visdrone_scripts/train_all_visdrone_yolov8n_p2_gap_ftal.py \
    --data-root /mnt/data/varroa/VisDrone2019 \
    --output-dir /mnt/data/varroa/VisDrone2019/yolo_format \
    --epochs 100 \
    --imgsz 640 \
    --batch-size 8 \
    --device cuda
```

### Bước 2: Kiểm tra cấu hình Model YAML (Smoke Check)
```bash
PYTHONPATH=models_related/ultralytics python -c "
from ultralytics import YOLO
model = YOLO('models_related/models_config/yolov8/visdrone/yolov8n_p2p3p4_visdrone_plain_gap.yaml')
print(model.model)
"
```
