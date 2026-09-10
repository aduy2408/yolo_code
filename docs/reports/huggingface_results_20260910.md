# Hugging Face results: DMM variants

> Bản này chỉ giữ lại các biến thể **DMM** theo yêu cầu. Các kết quả khác đã được loại khỏi phần hiển thị của báo cáo này; dữ liệu cũ vẫn còn trong lịch sử Git của file.

## Bảng kết quả DMM

| Repository | Run / variant | Seed | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test AP75 | Test P | Test R | Metrics source |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| duyle2408/levir-yolov8n-p2-dmm-lite-no-aug-no-mosaic-seed42-9ad2f90 | r1_dmm_lite | 42 | 0.8428 | 0.3389 | 0.8112 | 0.3085 | 0.1277 | 0.8172 | 0.7836 | [runs/r1_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-lite-no-aug-no-mosaic-seed42-9ad2f90/blob/main/runs/r1_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-lite-oacp-no-mosaic-seed44-3844e43 | r1_dmm_lite | 44 | 0.8430 | 0.3328 | 0.8059 | 0.3148 | 0.1441 | 0.8435 | 0.7511 | [runs/r1_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-lite-oacp-no-mosaic-seed44-3844e43/blob/main/runs/r1_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5 | r1_dmm_lite | - | 0.8367 | 0.3299 | 0.8209 | 0.3143 | 0.1366 | 0.8266 | 0.7773 | [runs/r1_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5/blob/main/runs/r1_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5 | r2_kvca_dmm_lite | - | 0.8484 | 0.3321 | 0.8193 | 0.3079 | 0.1102 | 0.8383 | 0.7730 | [runs/r2_kvca_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5/blob/main/runs/r2_kvca_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5 | r3_kvca_dmm_gated | - | 0.8360 | 0.3322 | 0.7992 | 0.3053 | 0.1173 | 0.8018 | 0.7672 | [runs/r3_kvca_dmm_gated/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5/blob/main/runs/r3_kvca_dmm_gated/evaluation_metrics.json) |

## Các biến thể đã loại khỏi báo cáo

<!--
Đã loại các nhóm không phải DMM: qmax/no-mosaic, nondeterministic server2, toàn bộ nhóm w1, yolov9t-p2-no-p5, baseline_cea/baseline_lea, w1_api*, dbss_oacp và gap_*_oacp.
-->

- Không hiển thị trong bảng kết quả hiện tại.
- Các artifact gốc vẫn có thể khôi phục từ commit trước: `d86bc68`.
