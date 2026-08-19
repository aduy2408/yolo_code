# Báo cáo Kết quả Thực nghiệm Raw-Cue (#3, #4, #5, #6a, #6b)

Tất cả các thực nghiệm dưới đây được huấn luyện trên dataset **LEVIR-Ship** (seed 42, 100 epochs) và được đánh giá với giao thức bắt buộc **NMS IoU = 0.50**. Kết quả được lưu trữ và kiểm chứng trên Hugging Face.

## Bảng so sánh kết quả thực nghiệm

| # | Thực nghiệm (Variant) | Đặc trưng kiến trúc | VAL mAP50 | VAL mAP50-95 (mAP75) | TEST mAP50 | TEST mAP50-95 (mAP75) | HF Repository |
|---|---|---|:---:|:---:|:---:|:---:|---|
| - | **Plain P2 Baseline** | Baseline P2 tiêu chuẩn (không inject raw-cue) | 0.7865 | 0.2950 | 0.7513 | 0.1349 | `levir-ship-yolo-p2` |
| **#5** | **GT-Guided Cue Preservation** | Cân bằng bằng Auxiliary Head $D_{cue}(F)$ dự đoán GT cue (chỉ lúc train) | 0.7779 | 0.2955 | 0.7125 | 0.2630 | `levir-gt-cue-preservation` |
| **#3** | **Dedicated Cue Slots** | Concat cứng $F$, $B$, và các raw cue (Cb, Cr, E, H) pooled. Không học mixing. | **0.7876** | 0.3009 | 0.7253 | 0.2663 | `levir-raw-cue-archs` |
| **#4** | **Detached Residual Fusion** | Stop-gradient residual $sg(C - D(F))$ bù vào Backbone $B$. | 0.7811 | **0.3028** | **0.7432** | **0.2677** | `levir-raw-cue-archs` |
| **#6a** | **Split-Channel Control** | Chia đôi channel của $F$ thành Semantics (24ch) và Detail (8ch) | 0.7803 | 0.2953 | 0.7232 | 0.2641 | `levir-raw-cue-archs` |
| **#6b** | **GT Channel Specialization** | Giống #6a nhưng nhánh Detail (8ch) nhận GT cue supervision | 0.7663 | 0.2931 | 0.7070 | 0.2571 | `levir-raw-cue-archs` |

## Nhận xét quan trọng từ kết quả

1. **Direct Injection / Dedicated Cue Slots (#3)**:
   * Có mAP50 ở VAL tăng nhẹ so với baseline (`0.7876` vs `0.7865`), tuy nhiên khi chuyển sang TEST set, mAP50 bị sụt giảm đáng kể (`0.7252` vs `0.7513`). Điều này tiếp tục củng cố giả thuyết rằng việc đưa trực tiếp các raw cues thô (không có cơ chế fusion mềm hoặc learned selection) vào Detect Head làm suy yếu khả năng tổng quát hóa trên tập ẩn (out-of-distribution/test split).

2. **Detached Residual Fusion (#4)**:
   * Đây là cấu trúc đạt kết quả tốt nhất trong nhóm raw-cue trên tập **TEST** (`TEST mAP50 = 0.7432`, `TEST mAP50-95 = 0.2677`). 
   * Việc chỉ dùng raw-cue như một nguồn giám sát gián tiếp (phần dư $C - D(F)$ được stop-graded để Detect Head không thể tác động ngược làm suy giảm D) giúp giữ lại tính chất biểu diễn tự nhiên của P2 mà không bị nhiễu do thông tin pixel thô lấn át.

3. **Ablation Split-Channel (#6a vs #6b)**:
   * So sánh giữa #6a (chỉ split channel đơn thuần) và #6b (split + áp dụng GT cue supervision lên nhánh detail):
     * **#6b** cho kết quả kém hơn hẳn **#6a** ở cả VAL (`0.7663` vs `0.7802`) lẫn TEST (`0.7070` vs `0.7232`).
     * Điều này chỉ ra rằng việc áp dụng trực tiếp auxiliary loss dựa trên pixel (như độ lệch màu và gradient cục bộ của bounding box) lên một tập con các channel đặc trưng trung gian có xu hướng hạn chế không gian biểu diễn (representation space) của nhánh chi tiết thay vì giúp nó biểu diễn tốt hơn cho việc detect.
