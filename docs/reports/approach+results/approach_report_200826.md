# Báo cáo Thí nghiệm: Tối ưu hóa Attention và FTAL Normalization trên LEVIR-Ship (20/08/2026)

Tài liệu này tổng hợp thiết kế phương pháp, cấu trúc mạng nơ-ron và kết quả thực nghiệm cho nhóm thí nghiệm tối ưu hóa Cơ chế chú ý nén khóa-trị (KVCA), vị trí Global Average Pooling (GAP) và chuẩn hóa mục tiêu FTAL.

---

## 1. Phương pháp & Cấu trúc mạng (Methodology)

### A. Chuẩn hóa mục tiêu FTAL (FTAL Denominator Control)
- **Vấn đề**: Các thực nghiệm FTAL (Factorized TAL) trước đây thay đổi phân phối giá trị mục tiêu (target) nhưng giữ nguyên khối lượng chuẩn hóa (normalization mass) của TAL cũ. Điều này có thể phóng đại gradient một cách không chủ ý.
- **Giải pháp**: Thiết lập biến môi trường `FTAL_NORM_MODE` để kiểm soát mẫu số (divisor) trong BCE classification loss:
  - `oldnorm` (mặc định): Sử dụng tổng target cũ của TAL làm divisor.
  - `newnorm`: Sử dụng tổng target mới của FTAL làm divisor để matched chính xác khối lượng chuẩn hóa.

### B. Cơ chế Attention Zero-Initialization
- **Vấn đề**: Các nhánh attention trong KVCA và KVCompressedTransformerEncoder không được khởi tạo dạng identity mapping ($x + 0$), gây ra hình phạt tối ưu hóa (optimization penalty) ngay từ các epoch đầu tiên.
- **Giải pháp**: Zero-initialize trọng số (`weight`) và bias của lớp BatchNorm cuối cùng (`proj_bn`) trong nhánh attention của `KVCompressedAttention` và `KVCompressedTransformerEncoder` để ép block hoạt động như một hàm đồng nhất (exact identity block) lúc bắt đầu huấn luyện.

### C. Vị trí tích hợp GAP (ChannelAttention) kết hợp KVCA Block
Chúng tôi khảo sát vị trí tối ưu của Global Average Pooling (GAP) dưới dạng `ChannelAttention` đối với khối chú ý `KVCompressedAttention` (KVCA) ở cuối nhánh FPN P2 (Stride-4):
- **Variant GAP After**: Đặt GAP ngay sau KVCA Block:
  `P2 output (C2f) -> KVCompressedAttention -> ChannelAttention -> Detect`
- **Variant GAP Before**: Đặt GAP ngay trước KVCA Block:
  `P2 output (C2f) -> ChannelAttention -> KVCompressedAttention -> Detect`

### D. YOLOv8 P2 Deep Supervision + GAP + FTAL (oldnorm)
- **Phương pháp**: Kết hợp cả giám sát auxiliary trực tiếp trên Backbone P2 (`p2_deep_sup_gain=1.0`), cơ chế tự chú ý kênh toàn cục `ChannelAttention` (GAP) tại ngõ ra của FPN P2, và cơ chế gán nhãn nâng cao FTAL (`factorized_tal_target=True`).
- **Mục tiêu**: Khảo sát sự tương tác tối ưu giữa cải thiện dòng gradient sớm ở backbone và lọc đặc trưng kênh trước Detect head dưới sự hướng dẫn của FTAL.

---

## 2. Kết quả Thực nghiệm (Experimental Results)

Tất cả các mô hình được huấn luyện đồng nhất trong **100 epochs** với `seed=42` trên tập dữ liệu LEVIR-Ship, đánh giá trên tập Test sử dụng explicit NMS IoU threshold **`0.5`**.

| Model / Variant | Normalization | test/mAP50 | test/mAP50-95 | So với Baseline (mAP50) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P2 Deep Supervision + GAP + FTAL** | **`oldnorm`** | **0.8030** | **0.2992** | **+5.00%** | Hoàn thành |
| **KVCA Block + GAP After** | **`oldnorm`** | 0.7844 | 0.2909 | **+3.14%** | Hoàn thành |
| **KVCA Block + GAP After** | **`newnorm`** | 0.7617 | 0.2799 | **+0.87%** | Hoàn thành |
| **KVCA Encoder** | **`oldnorm`** | 0.7582 | 0.2832 | **+0.52%** | Hoàn thành |
| **KVCA Block + GAP Before** | **`oldnorm`** | 0.7532 | 0.2835 | **+0.02%** | Hoàn thành |
| **Plain Control (Baseline)** | - | 0.7530 | 0.2741 | Baseline | Hoàn thành |
| **KVCA Block + GAP Before** | **`newnorm`** | 0.7215 | 0.2588 | **-3.15%** | Hoàn thành |
| **KVCA Backbone + GAP** | **`oldnorm`** | *Đang chạy* | *Đang chạy* | - | Đang chạy |

---

## 3. Phân tích & Đánh giá (Analysis)

1. **Sức mạnh kết hợp của Deep Supervision, GAP và FTAL (D)**:
   - Bản chạy `P2 Deep Supervision + GAP + FTAL (oldnorm)` đạt kết quả xuất sắc nhất sweeps (**0.8030 Test mAP50**), tăng **+5.00%** so với baseline và cao hơn cả Deep Supervision nguyên bản đơn lẻ (78.53%).
   - Điều này khẳng định sự tương tác tích cực (magic interaction): Deep supervision định hình đặc trưng sớm tốt ở backbone, GAP giúp lọc kênh tối ưu ở FPN đầu ra, và FTAL cải thiện nhãn gán phân loại.

2. **Hiệu quả của việc đặt GAP sau KVCA (GAP After)**:
   - Cấu hình `KVCA Block + GAP After + oldnorm` đạt hiệu năng vượt trội (**0.7844 Test mAP50**), tăng **+3.14%** so với baseline.
   - Việc đặt GAP *sau* khối tự chú ý (KVCA) hoạt động như một bộ lọc kênh tối ưu, giúp hiệu chỉnh lại độ quan trọng của các đặc trưng không gian đã được tinh lọc bởi KVCA trước khi đưa vào Detect head.

3. **Ảnh hưởng cực kỳ lớn của FTAL Normalization**:
   - Ở tất cả các cấu hình, `oldnorm` luôn cho hiệu năng tốt hơn đáng kể so với `newnorm` (ví dụ: `GAP After` giảm từ **78.44%** xuống **76.17%**, `GAP Before` giảm từ **75.32%** xuống **72.15%**).
   - *Giải thích*: Khi dùng `newnorm`, tổng các điểm mục tiêu sau FTAL nhỏ hơn rất nhiều so với TAL gốc. Việc chia cho một mẫu số nhỏ làm thang đo loss (gradient scale) bị đẩy lên rất cao một cách giả tạo, dẫn đến mất ổn định trong tối ưu hóa hội tụ. Việc giữ lại normalization mass cũ (`oldnorm`) hoạt động như một cơ chế ổn định hóa gradient (gradient stabilizer) hiệu quả.

4. **So sánh Encoder vs Block**:
   - Khối Attention đơn lẻ (`KVCA Block`) khi được bổ trợ bởi GAP sau nó cho kết quả vượt trội hơn so với Transformer Encoder đầy đủ (`KVCA Encoder` chỉ đạt **75.82%**). Điều này chỉ ra rằng lớp FFN (Feed-Forward Network) trong Transformer Encoder có thể quá nặng hoặc dư thừa thông tin đối với các đặc trưng cục bộ sớm ở stride-4.

---

## 4. Local-Basis Downsample P1→P2

Thí nghiệm seed 42 dùng cùng split LEVIR-Ship, 100 epochs, 512px và đánh giá test với NMS IoU **0.5**. Module giữ nguyên nhánh `Conv → BN → SiLU` pretrained, bổ sung residual Haar-like local-basis path với `gamma=0` khi khởi tạo.

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Plain Control | 0.8049 | 0.6223 | 0.7530 | 0.1353 | 0.2741 |
| **LBD fixed Haar** | **0.8400** | **0.7618** | **0.8149** | **0.1541** | **0.3156** |
| LBD adaptive Haar | 0.7621 | 0.6767 | 0.7255 | 0.0801 | 0.2529 |

`LBD fixed` tăng **+0.0619 AP50** và **+0.0415 mAP50-95** so với Plain Control. `LBD adaptive` giảm tương ứng **-0.0275 AP50** và **-0.0212 mAP50-95**. Vì mới có một seed, đây là kết quả định hướng; fixed Haar được giữ làm variant có tín hiệu, adaptive bị loại khỏi ưu tiên tiếp theo.

Artifact nguồn: [HF dataset repo](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-local-basis), [fixed metrics](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-local-basis/blob/main/runs/lbd_fixed/seed_42/evaluation_metrics.json), [adaptive metrics](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-local-basis/blob/main/runs/lbd_adaptive/seed_42/evaluation_metrics.json).

## 5. Tổng hợp các ablation LBD trên hai Marimo server

Các run dưới đây được đánh giá trên **test split 788 ảnh**, với `imgsz=512`, 100 epochs và explicit NMS IoU **`0.5`**. `mAP50-95` là fitness được ghi trong `evaluation_metrics.json`. Các checkpoint `best.pt` và `last.pt` của những run hoàn tất đã được upload lên Hugging Face dataset repo `duyle2408/levir-yolov8n-p2-local-basis`.

| Server | Variant | Seed | Test mAP50 | Test mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Marimo 1 | LBD fixed | 43 | 0.7958 | **0.3040** | 0.8325 | 0.7496 |
| Marimo 1 | LBD fixed | 44 | **0.8106** | 0.2998 | 0.8428 | 0.7703 |
| Marimo 1 | LBD fixed, no GAP/FTAL | 42 | 0.7595 | 0.2923 | 0.7835 | 0.7279 |
| Marimo 1 | LBD adaptive, no GAP/FTAL | 42 | 0.7677 | 0.2878 | 0.8053 | 0.7069 |
| Marimo 1 | Conv48 consistent | 42 | 0.7695 | 0.2843 | **0.8248** | 0.6899 |
| Marimo 1 | LBD48 fixed consistent | 42 | 0.7455 | 0.2711 | 0.7643 | 0.6850 |
| Marimo 1 | LBD48 adaptive consistent | 42 | 0.7690 | 0.2781 | 0.8052 | 0.7128 |
| Marimo 2 | Conv48 | 42 | 0.7010 | 0.2668 | 0.7552 | 0.6383 |
| Marimo 2 | LBD48 fixed | 42 | 0.7927 | 0.2891 | 0.8082 | 0.7388 |
| Marimo 2 | LBD48 adaptive | 42 | 0.7913 | 0.3019 | 0.8366 | 0.7208 |

### Nhận xét

- Kết quả tốt nhất theo **test mAP50** là `LBD fixed`, seed 44: **0.8106**.
- Kết quả tốt nhất theo **test mAP50-95** là `LBD fixed`, seed 43: **0.3040**.
- Trong nhóm consistent giữ nguyên 48 channels qua P2 backbone, neck, GAP và Detect, `Conv48 consistent` đạt mAP50 cao nhất (**0.7695**); hai biến thể LBD consistent chưa vượt được control.
- Ở nhóm expanded nhưng chưa giữ channel consistent, `LBD48 fixed` và `LBD48 adaptive` lần lượt đạt **0.7927** và **0.7913 mAP50**, cao hơn `Conv48` (**0.7010**).
- Các run consistent seed 42 và các run expanded seed 42 là kết quả một seed; chưa dùng để kết luận độ ổn định giữa seed.

Artifact tổng hợp:

- Marimo 1: `runs/levir_yolov8n_p2_lbd_expanded_consistent/summary_runs.csv`, `summary_aggregate.csv`.
- Marimo 1: `runs/levir_yolov8n_p2_local_basis/summary_runs.csv` và `runs/levir_yolov8n_p2_local_basis_nogap/summary_runs.csv`.
- Marimo 2: `runs/levir_yolov8n_p2_lbd_expanded/summary_runs.csv`, `summary_aggregate.csv`.
