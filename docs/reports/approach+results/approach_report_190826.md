# Báo cáo Thí nghiệm: Quét Biểu diễn P2 trên LEVIR-Ship (19/08/2026)

Tài liệu này tổng hợp thiết kế phương pháp, cấu trúc mạng nơ-ron và kết quả đánh giá thực nghiệm cho 5 variants thuộc sweep thiết kế biểu diễn sớm trên tập dữ liệu LEVIR-Ship.

---

## 1. Phương pháp & Cấu trúc mạng (Methodology)

Thí nghiệm được thực hiện trên cấu trúc YOLOv8n-P2 rút gọn (chỉ top-down FPN, Detect trực tiếp nhận P2 ở stride-4). Mục tiêu nhằm giải quyết câu hỏi: **"Tại sao biểu diễn đặc trưng P2 sớm lại chưa tối ưu cho việc phát hiện tàu nhỏ, và làm sao để cải thiện nó hiệu quả?"**

### Variant A: Plain Control (Baseline)
- **Cấu trúc**: YOLOv8n-P2 top-down standard. P2 chỉ nhận tín hiệu gradient gián tiếp truyền ngược từ đầu ra Detect (sau các cơ chế gán nhãn TAL).

### Variant B: Backbone P2 Deep Supervision
- **Phương pháp**: Đặt một auxiliary head giám sát trực tiếp tại P2 của backbone (Layer 2).
- **Cấu trúc**: Nhánh aux sử dụng Conv 3x3 -> GN -> SiLU -> Conv 1x1 để dự đoán heatmap tâm tàu (soft Gaussian targets ở stride 4). Nhánh aux này chỉ hoạt động trong quá trình huấn luyện và được loại bỏ hoàn toàn lúc inference.
- **Mục tiêu**: Giải quyết vấn đề dòng gradient phản hồi bị suy giảm/gián tiếp.

### Variant C: Canonical Raw-Crop Teacher
- **Phương pháp**: Distill thông tin đặc trưng của tàu từ ảnh gốc.
- **Cấu trúc**: Trong quá trình train, cắt trực tiếp các crop tàu từ ảnh gốc (zoom 1.5x, resize 32x32). Sử dụng một tiny raw-crop encoder huấn luyện bằng bài toán phân biệt tàu/nền biển (để tránh sập cấu trúc representation). Đồng thời lấy vector đặc trưng P2 tại đúng center cell của tàu đi qua projector, ép biểu diễn trùng với embedding của raw-crop teacher thông qua Cosine Similarity.
- **Mục tiêu**: Distill đặc trưng cục bộ lý tưởng từ ảnh thô độ phân giải cao vào mạng đặc trưng của detector.

### Variant D0 & D1: Raw Sidecar Supervised
- **Cấu trúc**: Một nhánh CNN phụ độc lập (Sidecar stem) nhận trực tiếp ảnh gốc `img0`, giảm độ phân giải xuống stride 4 ($R8$ - 8 channels). Nhánh chính FPN P2 được giảm kênh xuống 24 channels ($F24$). Hai nhánh được nối concat làm đầu vào cho Detect `[F24, R8] = 32 channels`.
- **D1 (Supervised)**: Nhánh Sidecar $R8$ được giám sát độc lập bằng auxiliary heatmap loss ( soft Gaussian targets ) trước khi concat.
- **D0 (Control)**: Nhánh Sidecar không nhận giám sát phụ (aux loss gain = 0.0) để cô lập tác động của cấu trúc vs giám sát.

---

## 2. Kết quả Thực nghiệm (Experimental Results)

Tất cả các mô hình được huấn luyện đồng nhất trong **100 epochs** với `seed=42`, đánh giá trên tập Test sử dụng explicit NMS IoU threshold **`0.5`**.

| Model / Variant | test/mAP50 | test/mAP50-95 | test/mAP75 | Precision | Recall | So với Baseline (mAP50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B (Backbone P2 Deep Supervision)** | **0.7853** | **0.2986** | **0.1159** | **0.8378** | **0.7422** | **+3.23%** |
| **C (Canonical Teacher)** | 0.7721 | 0.2835 | 0.0958 | 0.7948 | 0.7256 | **+1.91%** |
| **D0 (Raw Sidecar Control)** | 0.7654 | 0.2747 | 0.0985 | 0.7757 | 0.7254 | **+1.24%** |
| **A (Plain Control - Baseline)** | 0.7530 | 0.2741 | 0.1026 | 0.7838 | 0.7188 | Baseline |
| **D1 (Raw Sidecar Supervised)** | 0.7287 | 0.2622 | 0.0917 | 0.7604 | 0.6796 | **-2.43%** |

---

## 3. Phân tích & Đánh giá (Analysis)

1. **Sự vượt trội của Backbone Deep Supervision (B)**: 
   - Đạt mAP50 cao nhất (**78.53%**), tăng **+3.23%** so với baseline. Điều này xác thực giả thuyết dòng gradient truyền ngược từ Detect của YOLO bị loãng/suy giảm rất nhiều trước khi chạm tới các lớp sớm ở Stride-4. Việc giám sát trực tiếp bằng heatmap tâm tàu giúp định hình lại không gian đặc trưng của P2 cực kỳ tốt mà không làm tăng chi phí tính toán lúc inference.
2. **Hiệu quả của Canonical Teacher (C)**:
   - Tăng **+1.91%** mAP50. Việc học theo teacher giúp mạng thu nhận thêm tri thức về cấu trúc tàu thô.
3. **Sự thất bại của Raw Sidecar Supervised (D1)**:
   - D1 cho kết quả kém hơn cả baseline và D0. Điều này chỉ ra rằng việc ghép kênh biểu diễn thô từ sidecar stem ($R8$) và ép auxiliary loss lên nó trước khi concat trực tiếp tạo ra xung đột nghiêm trọng (feature conflict) ở đầu vào của Detect, làm nhiễu các đặc trưng ngữ cảnh/phát hiện vốn đã được học tốt ở nhánh FPN.
