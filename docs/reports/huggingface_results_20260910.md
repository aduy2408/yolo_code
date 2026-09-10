# Tổng hợp kết quả các repository Hugging Face

> Ngày tổng hợp: 2026-09-10. Nguồn là các file `evaluation_metrics.json` trong đúng các repository được cung cấp.
> Các metric được giữ nguyên theo artifact. `mAP50-95` là trường `metrics/mAP50-95(B)`, `AP75` là AP tại IoU 0.75. Các giá trị được làm tròn 4 chữ số.

## Quy ước

- `Val` và `Test` không được trộn giữa các thí nghiệm.
- NMS IoU là `0.50` khi artifact có ghi rõ hoặc theo giao thức của các artifact YOLO này.
- Seed split và seed train được ghi theo tên repository/run khi có thể suy ra. Không tự suy diễn seed bị thiếu.
- Link ở cột cuối trỏ thẳng tới artifact metric trên Hugging Face.

## 1. LEVIR-Ship: các repository một run

| Repository | Run / variant | Seed | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test AP75 | Test P | Test R | Metrics source |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| duyle2408/levir-yolov8n-p2-baseline-no-aug-no-mosaic-seed42-d1f8206 | baseline_p2 | 42 | 0.8440 | 0.3395 | 0.8077 | 0.3084 | 0.1278 | 0.8236 | 0.7658 | [runs/baseline_p2/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-baseline-no-aug-no-mosaic-seed42-d1f8206/blob/main/runs/baseline_p2/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-lite-no-aug-no-mosaic-seed42-9ad2f90 | r1_dmm_lite | 42 | 0.8428 | 0.3389 | 0.8112 | 0.3085 | 0.1277 | 0.8172 | 0.7836 | [runs/r1_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-lite-no-aug-no-mosaic-seed42-9ad2f90/blob/main/runs/r1_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-lite-oacp-no-mosaic-seed44-3844e43 | r1_dmm_lite | 44 | 0.8430 | 0.3328 | 0.8059 | 0.3148 | 0.1441 | 0.8435 | 0.7511 | [runs/r1_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-lite-oacp-no-mosaic-seed44-3844e43/blob/main/runs/r1_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed43 | baseline_oacp | 43 | 0.7676 | 0.2895 | 0.8057 | 0.2862 | 0.0885 | 0.8381 | 0.7474 | [runs/baseline_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed43/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed44 | baseline_oacp | 44 | 0.7150 | 0.2625 | 0.6392 | 0.2290 | 0.0779 | 0.6988 | 0.6900 | [runs/baseline_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed44/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed43 | baseline_oacp | 43 | 0.8298 | 0.3162 | 0.7923 | 0.2931 | 0.1082 | 0.7972 | 0.7514 | [runs/baseline_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed43/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed44 | baseline_oacp | 44 | 0.8144 | 0.3190 | 0.7880 | 0.2944 | 0.0939 | 0.8097 | 0.7460 | [runs/baseline_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed44/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-ftal-1dfe88c-seed42 | w1_api_oacp_ftal | 42 | 0.8014 | 0.3056 | 0.7548 | 0.2795 | 0.1063 | 0.7967 | 0.7037 | [runs/w1_api_oacp_ftal/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-ftal-1dfe88c-seed42/blob/main/runs/w1_api_oacp_ftal/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-ftal-no-mosaic-seed42-62ff114 | baseline_p2 | 42 | 0.8352 | 0.3344 | 0.8184 | 0.3209 | 0.1467 | 0.8304 | 0.7658 | [runs/baseline_p2/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-ftal-no-mosaic-seed42-62ff114/blob/main/runs/baseline_p2/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3 | yolov8n_p2_oacp_no_mosaic | - | 0.8434 | 0.3375 | 0.8307 | 0.3263 | 0.1576 | 0.8275 | 0.7902 | [runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3/blob/main/runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed43 | yolov8n_p2_oacp_no_mosaic | 43 | 0.8354 | 0.3350 | 0.8182 | 0.3190 | 0.1333 | 0.8517 | 0.7675 | [runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed43/blob/main/runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed44 | yolov8n_p2_oacp_no_mosaic | 44 | 0.8086 | 0.3154 | 0.7921 | 0.3018 | 0.1195 | 0.8094 | 0.7688 | [runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed44/blob/main/runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-no-mosaic-nondeterministic-seed44-1ac1835-server2 | yolov8n_p2_oacp_no_mosaic | 44 | 0.8086 | 0.3154 | 0.7921 | 0.3018 | 0.1195 | 0.8094 | 0.7688 | [runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-nondeterministic-seed44-1ac1835-server2/blob/main/runs/yolov8n_p2_oacp_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-r1-reg-qmax-no-mosaic-seed43 | yolov8n_p2_oacp_r1_qmax_no_mosaic | 43 | 0.8503 | 0.3305 | 0.8090 | 0.3162 | 0.1398 | 0.8194 | 0.7757 | [runs/yolov8n_p2_oacp_r1_qmax_no_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-r1-reg-qmax-no-mosaic-seed43/blob/main/runs/yolov8n_p2_oacp_r1_qmax_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-yolov9t-no-oacp-no-mosaic-seed42 | yolov9t_no_oacp_no_mosaic_seed42 | 42 | 0.8112 | 0.3038 | 0.7796 | 0.2896 | 0.0928 | 0.8021 | 0.7452 | [runs/yolov9t_no_oacp_no_mosaic_seed42/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov9t-no-oacp-no-mosaic-seed42/blob/main/runs/yolov9t_no_oacp_no_mosaic_seed42/evaluation_metrics.json) |
| duyle2408/levir-yolov9t-oacp-no-mosaic-seed42 | yolov9t_oacp_no_mosaic_seed42 | 42 | 0.8266 | 0.3229 | 0.8079 | 0.3108 | 0.1165 | 0.8313 | 0.7647 | [runs/yolov9t_oacp_no_mosaic_seed42/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov9t-oacp-no-mosaic-seed42/blob/main/runs/yolov9t_oacp_no_mosaic_seed42/evaluation_metrics.json) |
| duyle2408/levir-yolov9t-oacp-official-e38ce07-retry1-seed42 | yolov9t_oacp_seed42 | 42 | 0.8287 | 0.3127 | 0.7556 | 0.2664 | 0.0891 | 0.7837 | 0.7572 | [runs/yolov9t_oacp_seed42/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov9t-oacp-official-e38ce07-retry1-seed42/blob/main/runs/yolov9t_oacp_seed42/evaluation_metrics.json) |
| duyle2408/levir-yolov9t-p2-no-p5-oacp-seed42 | yolov9t_p2_no_p5_oacp/seed_42 | 42 | 0.7144 | 0.2843 | 0.6529 | 0.2487 | 0.0986 | 0.7231 | 0.6192 | [runs/yolov9t_p2_no_p5_oacp/seed_42/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov9t-p2-no-p5-oacp-seed42/blob/main/runs/yolov9t_p2_no_p5_oacp/seed_42/evaluation_metrics.json) |

## 2. LEVIR-Ship: repository nhiều variant / ablation

| Repository | Run / variant | Seed | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test AP75 | Test P | Test R | Metrics source |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| duyle2408/levir-oacp-double-compare-9a77f69 | oacp_double_approx_mosaic | - | 0.7840 | 0.3039 | 0.7656 | 0.2827 | 0.0890 | 0.8204 | 0.7256 | [runs/oacp_double_approx_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-oacp-double-compare-9a77f69/blob/main/runs/oacp_double_approx_mosaic/evaluation_metrics.json) |
| duyle2408/levir-oacp-double-compare-9a77f69 | oacp_double_approx_no_mosaic | - | 0.8320 | 0.3226 | 0.8005 | 0.2936 | 0.1167 | 0.8042 | 0.7514 | [runs/oacp_double_approx_no_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-oacp-double-compare-9a77f69/blob/main/runs/oacp_double_approx_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-oacp-double-compare-9a77f69 | oacp_standard_mosaic | - | 0.7899 | 0.2964 | 0.7428 | 0.2633 | 0.0784 | 0.7716 | 0.7213 | [runs/oacp_standard_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-oacp-double-compare-9a77f69/blob/main/runs/oacp_standard_mosaic/evaluation_metrics.json) |
| duyle2408/levir-oacp-double-compare-9a77f69 | oacp_standard_no_mosaic | - | 0.7459 | 0.2730 | 0.6577 | 0.2222 | 0.0787 | 0.7231 | 0.6537 | [runs/oacp_standard_no_mosaic/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-oacp-double-compare-9a77f69/blob/main/runs/oacp_standard_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42 | baseline_cea | 42 | 0.8070 | 0.3103 | 0.7469 | 0.2713 | 0.0865 | 0.8105 | 0.6825 | [runs/baseline_cea/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42/blob/main/runs/baseline_cea/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42 | baseline_oacp | 42 | 0.8404 | 0.3273 | 0.8004 | 0.3045 | 0.1063 | 0.7939 | 0.7543 | [runs/baseline_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | baseline_cea | 42 | 0.8070 | 0.3103 | 0.7469 | 0.2713 | 0.0865 | 0.8105 | 0.6825 | [runs/baseline_cea/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/baseline_cea/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | baseline_lea | 42 | 0.7966 | 0.3158 | 0.7730 | 0.2774 | 0.0814 | 0.8128 | 0.7241 | [runs/baseline_lea/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/baseline_lea/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | baseline_oacp | 42 | 0.8404 | 0.3273 | 0.8004 | 0.3045 | 0.1063 | 0.7939 | 0.7543 | [runs/baseline_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | w1_api | 42 | 0.8158 | 0.3217 | 0.7884 | 0.2871 | 0.1013 | 0.8160 | 0.7586 | [runs/w1_api/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/w1_api/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | w1_api_cea | 42 | 0.8192 | 0.3240 | 0.7755 | 0.2923 | 0.1133 | 0.7826 | 0.7399 | [runs/w1_api_cea/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/w1_api_cea/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | w1_api_lea | 42 | 0.7779 | 0.3093 | 0.7772 | 0.2773 | 0.0798 | 0.8168 | 0.7672 | [runs/w1_api_lea/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/w1_api_lea/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | w1_api_oacp | 42 | 0.8217 | 0.3262 | 0.7686 | 0.2757 | 0.1011 | 0.7905 | 0.7356 | [runs/w1_api_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/w1_api_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | w1_cea | 42 | 0.8021 | 0.3102 | 0.7483 | 0.2813 | 0.1199 | 0.7974 | 0.6968 | [runs/w1_cea/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/w1_cea/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | w1_lea | 42 | 0.2631 | 0.0700 | 0.2009 | 0.0532 | 0.0039 | 0.3180 | 0.3563 | [runs/w1_lea/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/w1_lea/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | w1_oacp | 42 | 0.8258 | 0.3185 | 0.7774 | 0.2853 | 0.0892 | 0.7707 | 0.7629 | [runs/w1_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/w1_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5 | r1_dmm_lite | - | 0.8367 | 0.3299 | 0.8209 | 0.3143 | 0.1366 | 0.8266 | 0.7773 | [runs/r1_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5/blob/main/runs/r1_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5 | r2_kvca_dmm_lite | - | 0.8484 | 0.3321 | 0.8193 | 0.3079 | 0.1102 | 0.8383 | 0.7730 | [runs/r2_kvca_dmm_lite/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5/blob/main/runs/r2_kvca_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5 | r3_kvca_dmm_gated | - | 0.8360 | 0.3322 | 0.7992 | 0.3053 | 0.1173 | 0.8018 | 0.7672 | [runs/r3_kvca_dmm_gated/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5/blob/main/runs/r3_kvca_dmm_gated/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R1_sparse_mild | - | 0.8261 | 0.3249 | 0.8188 | 0.3103 | 0.1204 | 0.8454 | 0.7687 | [runs/R1_sparse_mild/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R1_sparse_mild/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R2_frequent_mild | - | 0.8332 | 0.3297 | 0.8242 | 0.3164 | 0.1184 | 0.8203 | 0.7859 | [runs/R2_frequent_mild/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R2_frequent_mild/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R3_frequent_current | - | 0.7489 | 0.2745 | 0.7116 | 0.2554 | 0.0836 | 0.7586 | 0.6501 | [runs/R3_frequent_current/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R3_frequent_current/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R4_strong | - | 0.8413 | 0.3275 | 0.8236 | 0.3209 | 0.1520 | 0.8161 | 0.7830 | [runs/R4_strong/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R4_strong/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R5_very_strong_stress | - | 0.8470 | 0.3369 | 0.8147 | 0.3115 | 0.1228 | 0.7959 | 0.7902 | [runs/R5_very_strong_stress/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R5_very_strong_stress/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R6_narrow_protection | - | 0.8420 | 0.3322 | 0.7968 | 0.3019 | 0.1102 | 0.8091 | 0.7673 | [runs/R6_narrow_protection/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R6_narrow_protection/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R7_wide_protection | - | 0.8428 | 0.3319 | 0.8050 | 0.3112 | 0.1259 | 0.8218 | 0.7615 | [runs/R7_wide_protection/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R7_wide_protection/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43 | dbss_oacp | - | 0.8317 | 0.3173 | 0.7971 | 0.2874 | 0.0996 | 0.8084 | 0.7601 | [runs/dbss_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43/blob/main/runs/dbss_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43 | gap_ftal_oacp | - | 0.8320 | 0.3146 | 0.7886 | 0.2939 | 0.0902 | 0.8103 | 0.7601 | [runs/gap_ftal_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43/blob/main/runs/gap_ftal_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43 | gap_oacp | - | 0.8195 | 0.3191 | 0.8041 | 0.3005 | 0.1152 | 0.8432 | 0.7457 | [runs/gap_oacp/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43/blob/main/runs/gap_oacp/evaluation_metrics.json) |

### Best Test mAP50 theo repository nhiều variant

| Repository | Variant tốt nhất theo Test mAP50 | Test mAP50 | Test mAP50-95 | Link |
|---|---|---:|---:|---|
| duyle2408/levir-oacp-double-compare-9a77f69 | oacp_double_approx_no_mosaic | 0.8005 | 0.2936 | [metrics](https://huggingface.co/datasets/duyle2408/levir-oacp-double-compare-9a77f69/blob/main/runs/oacp_double_approx_no_mosaic/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42 | baseline_oacp | 0.8004 | 0.3045 | [metrics](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42 | baseline_oacp | 0.8004 | 0.3045 | [metrics](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42/blob/main/runs/baseline_oacp/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5 | r1_dmm_lite | 0.8209 | 0.3143 | [metrics](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5/blob/main/runs/r1_dmm_lite/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758 | R2_frequent_mild | 0.8242 | 0.3164 | [metrics](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758/blob/main/runs/R2_frequent_mild/evaluation_metrics.json) |
| duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43 | gap_oacp | 0.8041 | 0.3005 | [metrics](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43/blob/main/runs/gap_oacp/evaluation_metrics.json) |

## 3. TinyPerson: fixed split 42, seed sweep

| Repository | Run / variant | Seed | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test AP75 | Test P | Test R | Metrics source |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed42 | yolov8n_p2p3p4_plain_oacp/seed_42 | 42 | 0.5769 | 0.2131 | 0.5203 | 0.1907 | 0.0916 | 0.5857 | 0.5103 | [runs/yolov8n_p2p3p4_plain_oacp/seed_42/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed42/blob/main/runs/yolov8n_p2p3p4_plain_oacp/seed_42/evaluation_metrics.json) |
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed43 | yolov8n_p2p3p4_plain_oacp/seed_43 | 43 | 0.5686 | 0.2056 | 0.5240 | 0.1915 | 0.0929 | 0.5928 | 0.5078 | [runs/yolov8n_p2p3p4_plain_oacp/seed_43/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed43/blob/main/runs/yolov8n_p2p3p4_plain_oacp/seed_43/evaluation_metrics.json) |
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed44 | yolov8n_p2p3p4_plain_oacp/seed_44 | 44 | 0.5681 | 0.2081 | 0.5335 | 0.1964 | 0.0934 | 0.5989 | 0.5161 | [runs/yolov8n_p2p3p4_plain_oacp/seed_44/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed44/blob/main/runs/yolov8n_p2p3p4_plain_oacp/seed_44/evaluation_metrics.json) |

## 4. TinyPerson: các repository khác

| Repository | Run / variant | Seed | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test AP75 | Test P | Test R | Metrics source |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc | yolov8n_p2p3p4_plain_oacp/seed_42 | 42 | 0.5329 | 0.1889 | 0.4943 | 0.1766 | 0.0806 | 0.5713 | 0.4888 | [runs/yolov8n_p2p3p4_plain_oacp/seed_42/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc/blob/main/runs/yolov8n_p2p3p4_plain_oacp/seed_42/evaluation_metrics.json) |
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc | yolov8n_p2p3p4_plain_oacp/seed_43 | 43 | 0.5386 | 0.1893 | 0.5046 | 0.1816 | 0.0844 | 0.5878 | 0.4929 | [runs/yolov8n_p2p3p4_plain_oacp/seed_43/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc/blob/main/runs/yolov8n_p2p3p4_plain_oacp/seed_43/evaluation_metrics.json) |
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc | yolov8n_p2p3p4_plain_oacp/seed_44 | 44 | 0.5478 | 0.1922 | 0.4998 | 0.1800 | 0.0837 | 0.5818 | 0.4865 | [runs/yolov8n_p2p3p4_plain_oacp/seed_44/evaluation_metrics.json](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc/blob/main/runs/yolov8n_p2p3p4_plain_oacp/seed_44/evaluation_metrics.json) |

## 5. Repository không có `evaluation_metrics.json`

| Repository | Artifact hiện có | Ghi chú |
|---|---|---|
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-9dda1cb-seed42 | `_trống_` | Không có metric định lượng để đưa vào bảng. |
| duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-25dcadc | `_trống_` | Không có metric định lượng để đưa vào bảng. |

## 6. Tóm tắt coverage

- Repository entries đã yêu cầu: **31**; số repository unique: **30**.
- Repository có metric: **28**.
- Repository không có `evaluation_metrics.json`: **2**.
- Tổng số run/variant có metric: **53**.
- LEVIR-Ship: **18** single-run rows và **29** multi-variant rows.
- TinyPerson: **6** rows.

## 7. Danh sách source repository

- [duyle2408/levir-oacp-double-compare-9a77f69](https://huggingface.co/datasets/duyle2408/levir-oacp-double-compare-9a77f69)
- [duyle2408/levir-yolov8n-p2-baseline-no-aug-no-mosaic-seed42-d1f8206](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-baseline-no-aug-no-mosaic-seed42-d1f8206)
- [duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-0867aa5-seed42)
- [duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-context-augmentations-807e763-seed42)
- [duyle2408/levir-yolov8n-p2-dmm-lite-no-aug-no-mosaic-seed42-9ad2f90](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-lite-no-aug-no-mosaic-seed42-9ad2f90)
- [duyle2408/levir-yolov8n-p2-dmm-lite-oacp-no-mosaic-seed44-3844e43](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-lite-oacp-no-mosaic-seed44-3844e43)
- [duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-dmm-oacp-no-mosaic-6a5dbd5)
- [duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758)
- [duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed43](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed43)
- [duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed44](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-b1c490b-seed44)
- [duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-combinations-c70fd43)
- [duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed43](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed43)
- [duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed44](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed44)
- [duyle2408/levir-yolov8n-p2-oacp-ftal-1dfe88c-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-ftal-1dfe88c-seed42)
- [duyle2408/levir-yolov8n-p2-oacp-ftal-no-mosaic-seed42-62ff114](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-ftal-no-mosaic-seed42-62ff114)
- [duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3)
- [duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed43](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed43)
- [duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed44](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-2d308a3-seed44)
- [duyle2408/levir-yolov8n-p2-oacp-no-mosaic-nondeterministic-seed44-1ac1835-server2](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-no-mosaic-nondeterministic-seed44-1ac1835-server2)
- [duyle2408/levir-yolov8n-p2-oacp-r1-reg-qmax-no-mosaic-seed43](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-oacp-r1-reg-qmax-no-mosaic-seed43)
- [duyle2408/levir-yolov9t-no-oacp-no-mosaic-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov9t-no-oacp-no-mosaic-seed42)
- [duyle2408/levir-yolov9t-oacp-no-mosaic-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov9t-oacp-no-mosaic-seed42)
- [duyle2408/levir-yolov9t-oacp-official-e38ce07-retry1-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov9t-oacp-official-e38ce07-retry1-seed42)
- [duyle2408/levir-yolov9t-p2-no-p5-oacp-seed42](https://huggingface.co/datasets/duyle2408/levir-yolov9t-p2-no-p5-oacp-seed42)
- [duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed42](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed42)
- [duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed43](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed43)
- [duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed44](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed44)
- [duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-9dda1cb-seed42](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-9dda1cb-seed42)
- [duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-25dcadc](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-25dcadc)
- [duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-servernew-25dcadc)
