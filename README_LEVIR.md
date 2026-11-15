# LEVIR Ship: Tiny Object Detection with Customized YOLO

Tài liệu này tổng hợp toàn bộ thông tin về mục tiêu, kiến trúc, các hướng tiếp cận và hướng dẫn chạy thực nghiệm cho bài toán phát hiện tàu biển siêu nhỏ (tiny ships) sử dụng họ mô hình YOLOv8/YOLOv10 tùy chỉnh trên tập dữ liệu **LEVIR-Ship**.

---

## 1. Tổng Quan Dự Án & Mục Tiêu
Phát hiện vật thể cực nhỏ (tiny objects) luôn là thách thức đối với các phiên bản YOLO chuẩn do quá trình downsampling (stride lớn) làm mất mát đặc trưng không gian. Dự án này nghiên cứu và cải tiến các thành phần:
- **Kiến trúc mạng (Architecture):** Thêm nhánh P2 (stride 4), GCTS local routing, CFR.
- **Hàm Loss (Loss Functions):** Non-uniform DFL, Pair-competitive loss, Box Consensus loss.
- **Xử lý đặc trưng nền (Feature Filtering):** DBSS (triệt nhiễu nền biển), HIT (vận chuyển đặc trưng bất khả quy).

---

## 2. Cấu Trúc Mã Nguồn (Organized Structure)

Sau khi sắp xếp lại, cấu trúc thư mục chính như sau:

```text
.
├── train_levir_scripts/      # Thư mục chứa toàn bộ script chạy train, test và chẩn đoán
│   ├── train_all_*.py        # Huấn luyện tự động qua các hạt giống (seeds)
│   ├── test_*.py             # Các file smoke test kiểm tra nhanh tính đúng đắn
│   └── analyze_*.py          # Các script đo đạc phương sai (variance) và chẩn đoán lỗi
├── docs/
│   └── reports/              # Thư mục chứa các tài liệu phân tích và báo cáo
│       ├── approach_report.md  # Báo cáo lý thuyết chi tiết cơ chế của từng method
│       ├── report_yolo.md      # Kết quả thực nghiệm và so sánh tổng quan
│       └── report_levir_dbss_generalization.md # Đánh giá khả năng tổng quát hóa của DBSS
├── models_related/
│   ├── ultralytics/          # Mã nguồn core YOLO được tùy biến (nn/tasks.py, utils/loss.py,...)
│   └── models_config/        # Các file YAML cấu hình kiến trúc model cho từng thực nghiệm
└── README_LEVIR.md            # Tài liệu tổng quan và hướng dẫn nhanh
```

---

## 3. Các Phương Pháp Thực Nghiệm Chính

Chi tiết kỹ thuật sâu hơn được trình bày cụ thể trong [approach_report.md](file:///mnt/data/varroa/yolo_related/docs/reports/approach_report.md):

| Phương Pháp | Mô Tả Cơ Chế | File Can Thiệp Chính |
| :--- | :--- | :--- |
| **YOLO-P2 Baseline** | Thêm nhánh P2 (stride 4) tăng mật độ anchor dự đoán vật thể nhỏ | `nn/tasks.py`, `models_config/` |
| **Non-uniform DFL** | Thay đổi phân phối bin DFL, tập trung bin dày ở khoảng cách nhỏ | `utils/loss.py` |
| **Pair-competitive DFL** | Thêm loss cạnh tranh giữa các bin lân cận để tăng độ sắc nét | `utils/loss.py` |
| **CFR (Conflict-Guided)** | Tái cấu trúc đặc trưng tại vị trí xung đột giữa phân loại và định vị | `nn/modules/block.py` |
| **Box Consensus Loss** | Phạt sự lệch pha (variance) của các dự đoán quanh Ground Truth | `utils/loss.py` |
| **DBSS** | Triệt tiêu các tín hiệu nền lặp lại (sóng, bờ biển) bằng Ridge projection | `nn/modules/block.py` |
| **HIT** | Đo độ khó pixel bằng quan hệ không gian/channel rồi vận chuyển residual | `nn/modules/block.py` |

---

## 4. Trạng Thái CACS và NegativeCanvas

Bộ đánh giá checkpoint CACS cho **OACP-R2 + NegativeCanvas R2/R4** đã hoàn tất và được xác minh trên Hugging Face:

- **HF repository:** `duyle2408/cacs-checkpoint-eval-20261018-runs`
- **Coverage:** `24/24` checkpoint evaluations hoàn tất
- **Ma trận:** YOLOv8/YOLOv9/YOLO11 × LEVIR-Ship/TinyPerson × NC-R2/NC-R4 × seed 42/44
- **Completion markers:** `24/24` `cacs_evaluation_complete.json`
- **Evaluation artifacts:** `24/24` `evaluation_metrics.json`
- **Provenance:** split seed `42`, workers `8`, NMS IoU `0.50`; Copy-Paste được áp dụng trước OACP-R2 trong pipeline CACS

Các bảng kết quả và protocol chi tiết nằm trong:

- [`docs/reports/augmentation_report.md`](docs/reports/augmentation_report.md)
- [`docs/reports/complete_main_report.md`](docs/reports/complete_main_report.md)
- [`docs/reports/yolov9_yolo11_seed44_size_evaluation.md`](docs/reports/yolov9_yolo11_seed44_size_evaluation.md)

### Bảng Copy-Paste cuối — TinyPerson

Protocol: split seed `42`, NMS IoU `0.50`. `Native AP50` và `Native mAP50-95` là metric trên test corner-window của detector. `Merged AP50` và `Merged AP50-Small` là metric sau khi đưa prediction về ảnh gốc, gộp corner-window bằng NMS rồi chạy TinyBenchmark. Các giá trị là mean ± sample standard deviation khi đủ hai seed.

| Model | Variant | Seeds | Native AP50 | Native mAP50-95 | Merged AP50 | Native AP50-Small | Merged AP50-Small |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8 | CP1 | 43,44 | 0.49403 ± 0.00569 | 0.17504 ± 0.00287 | **0.51611 ± 0.00488** | 0.51450 ± 0.00990 | 0.66430 ± 0.00368 |
| YOLOv8 | NC-R2 | 43,44 | 0.50918 ± 0.00610 | 0.18176 ± 0.00525 | **0.53212 ± 0.01289** | — | 0.67292 ± 0.00636 |
| YOLOv8 | NC-R4 | 43,44 | 0.50947 ± 0.01687 | 0.18308 ± 0.00992 | **0.51568 ± 0.00277** | 0.50970 | 0.66113 ± 0.00131 |
| YOLO11 | CP1 | 42,44 | **0.49504 ± 0.00093** | **0.17608 ± 0.00118** | **0.51456 ± 0.00092** | — | **0.66056 ± 0.00069** |
| YOLO11 | NC-R2 | 43,44 | 0.50734 ± 0.00835 | 0.18117 ± 0.00480 | **0.53132 ± 0.00935** | 0.51378 ± 0.00932 | 0.67338 ± 0.00784 |
| YOLO11 | NC-R4 | 43,44 | 0.50575 ± 0.02040 | 0.18018 ± 0.00956 | **0.52808 ± 0.02354** | 0.51837 ± 0.01068 | 0.67489 ± 0.01472 |

`*` YOLOv8 NC-R4 seed44 đã được backfill merged AP50 bằng evaluator merged-only với stable vectorized NMS: `Merged AP50=0.51372`, `Merged mAP50-95=0.18369`, `Merged AP50-Small=0.66206`. Aggregate dùng seed43/44; `Native AP50-Small` vẫn chỉ có seed43 vì seed44 artifact không có field này.
### Backfill parity canonical — LEVIR-Ship và TinyPerson

Các bảng Copy-Paste cũ phía trên giữ provenance lịch sử. Bảng dưới đây là kết quả backfill theo đúng `baseline_no_mosaic_parity`, không trộn với `augmentation_matrix_current`.

Protocol chung: split seed `42`, NMS IoU `0.50`. TinyPerson dùng merged corner-window evaluator; LEVIR-Ship dùng native `val/test`, `imgsz=512`. Artifact:

- TinyPerson: `duyle2408/augmentation-merged-ap50-backfill-runs/baseline_no_mosaic_parity/`
- LEVIR-Ship: `duyle2408/augmentation-levir-parity-eval-runs/baseline_no_mosaic_parity/`

#### TinyPerson merged

| Model | Variant | Seed | Merged AP50 | Merged mAP50-95 | Merged AP50-Small |
| :--- | :--- | ---: | ---: | ---: | ---: |
| YOLOv8 | CP1 | 43,44 | 0.51600 ± 0.00480 | 0.18499 ± 0.00051 | 0.66413 ± 0.00371 |
| YOLOv8 | NC-R2 | 43 | 0.54712 | 0.20035 | 0.69069 |
| YOLOv8 | NC-R2 | 44 | 0.54136 | 0.19716 | 0.67772 |
| YOLOv8 | NC-R4 | 43 | 0.54455 | 0.19807 | 0.68116 |
| YOLOv8 | NC-R4 | 44 | 0.55188 | 0.20322 | 0.69233 |
| YOLO11 | NC-R2 | 43 | 0.55074 | 0.19954 | 0.69282 |
| YOLO11 | NC-R2 | 44 | 0.53788 | 0.19522 | 0.67883 |
| YOLO11 | NC-R4 | 43 | 0.54123 | 0.19877 | 0.68323 |
| YOLO11 | NC-R4 | 44 | 0.54480 | 0.19771 | 0.68550 |

#### LEVIR-Ship native

| Model | Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test AP50-small |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8 | NC-R2 | 43 | 0.60745 | 0.19814 | 0.55932 | 0.17361 | 0.50995 |
| YOLOv8 | NC-R2 | 44 | 0.82352 | 0.33112 | 0.80796 | 0.30350 | 0.85032 |
| YOLOv8 | NC-R4 | 43 | 0.80584 | 0.31144 | 0.78774 | 0.27978 | 0.82400 |
| YOLOv8 | NC-R4 | 44 | 0.79565 | 0.30638 | 0.78798 | 0.29457 | 0.81112 |
| YOLO11 | NC-R2 | 43 | 0.82436 | 0.32321 | 0.79315 | 0.29509 | 0.79905 |
| YOLO11 | NC-R2 | 44 | 0.82443 | 0.32663 | 0.81806 | 0.29785 | 0.80447 |
| YOLO11 | NC-R4 | 43 | 0.82349 | 0.32427 | 0.79119 | 0.29376 | 0.79283 |
| YOLO11 | NC-R4 | 44 | 0.82663 | 0.33128 | 0.79327 | 0.30321 | 0.78570 |

Canonical parity rows use `duyle2408/yolov8-cp-nc-seed43-44-runs`. Its seed43 NC-R2 result remains `Test AP50 ≈ 0.5593`, matching both the artifact and Colab rerun. The separate `duyle2408/levirship-tinyperson-yolov8-yolo11-copy-paste-nc43-44-runs` rows use `translate=0.0`, `scale=0.0`, so they are not baseline parity and are excluded.

### Kết quả evaluation SOTA trên Marimo

Các kết quả dưới đây dùng split seed `42`, training seed `42`, NMS IoU `0.50`. TinyPerson merged dùng protocol ghép các corner-window về ảnh gốc trước khi chạy TinyBenchmark evaluator.

**TinyPerson merged test:**

| Model | `test_merged/AP50` | `test_merged/mAP50-95` | `test_merged/AP50-Small` | `test_merged/mAP50-95-Small` |
| :--- | ---: | ---: | ---: | ---: |
| STW-YOLO | 0.5892 | 0.2241 | **0.7004** | 0.2903 |
| TPH-YOLOv5 | 0.5474 | 0.2041 | 0.6370 | 0.2565 |

**Native AP50-Small theo split:**

| Model | Dataset | `val/AP50-Small` | `test/AP50-Small` |
| :--- | :--- | ---: | ---: |
| STW-YOLO | LEVIR-Ship | 0.8340 | 0.8407 |
| STW-YOLO | Varroa | 0.8978 | 0.8818 |
| STW-YOLO | TinyPerson | 0.5982 | 0.5418 |

TPH-YOLOv5 đã được evaluate theo cùng protocol merged original-image, với corner predictions được dịch về ảnh gốc, NMS IoU `0.50`, sau đó chạy TinyBenchmark evaluator. Artifact merged nằm tại `tph_test_merged_predictions.json` và `merged_summary.json` trên run Marimo tương ứng. Protocol và script nằm trong `evaluate_test/run_tph_merged_only.py` và các report trong `docs/reports/`.

### STW-YOLO standard seed43 backfill

Protocol: fixed split seed `42`, training seed `43`, native validation/test
evaluation, NMS IoU `0.50`. These are split-qualified native metrics; they are
not TinyPerson merged-corner metrics.

| Dataset | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | HF artifact |
| :--- | ---: | ---: | ---: | ---: | :--- |
| LEVIR-Ship | 0.8295 | 0.3385 | 0.8205 | 0.3208 | `duyle2408/stw-yolo-standard-seed43-backfill-runs/runs/levirship/seed_43` |
| TinyPerson | 0.6241 | 0.2394 | 0.5792 | 0.2204 | `duyle2408/stw-yolo-standard-seed43-backfill-runs/runs/tinyperson/seed_43` |

The required checkpoint, manifest, native metrics, bucket metrics, raw
predictions/ground truth, and upload markers were verified for both prefixes.
The AP50-Small values below come from the separate native size-bucket protocol;
they are not inferred from the native AP50.

#### STW seed43 native size-bucket evaluation

The bucket protocol is separate from the native Ultralytics AP50/mAP rows above:
TinyBenchmark area ranges, IoU `0.50:0.05:0.75`, maxDets `200`, evaluated on
native validation/test images. These are not TinyPerson merged-corner metrics.

| Dataset | Split | AP50 | AP50-Tiny1 | AP50-Tiny2 | AP50-Tiny3 | AP50-Small | AP50-Medium | mAP50-75 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEVIR-Ship | Val | 0.7373 | — | 0.2969 | 0.7084 | 0.8298 | 0.8020 | 0.4907 |
| LEVIR-Ship | Test | 0.7188 | 0.0000 | 0.4061 | 0.7284 | 0.8187 | 0.6773 | 0.4699 |
| TinyPerson | Val | 0.4945 | 0.3451 | 0.5282 | 0.5504 | 0.5418 | 0.6454 | 0.3184 |
| TinyPerson | Test | 0.4620 | 0.2507 | 0.4753 | 0.5170 | 0.5434 | 0.5441 | 0.2906 |

`—` means the bucket had no valid ground-truth instances in that split. The
raw predictions, ground truth, manifest, metrics, and upload marker are
verified under the HF prefixes in the table above.

#### TPH-YOLOv5 native size-bucket backfill — both training seeds

Protocol: native validation/test images, TinyBenchmark area buckets, IoU
`0.50:0.05:0.75`, maxDets `200`. These metrics are separate from the
Ultralytics native AP50/mAP rows and from TinyPerson merged-corner metrics.

| Dataset | Seed | Split | AP50 | AP50-Tiny1 | AP50-Tiny2 | AP50-Tiny3 | AP50-Small | AP50-Medium | mAP50-75 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEVIR-Ship | 42 | Val | 0.2452 | — | 0.2391 | 0.3049 | 0.3178 | 0.6399 | 0.1490 |
| LEVIR-Ship | 42 | Test | 0.6719 | 0.0085 | 0.3566 | 0.7220 | 0.7180 | 0.6906 | 0.3921 |
| LEVIR-Ship | 43 | Val | 0.7524 | — | 0.3783 | 0.7508 | 0.8323 | 0.9026 | 0.4726 |
| LEVIR-Ship | 43 | Test | 0.7403 | 0.0000 | 0.4538 | 0.7814 | 0.8006 | 0.7180 | 0.4424 |
| TinyPerson | 42 | Val | 0.2689 | 0.2162 | 0.3600 | 0.3303 | 0.3602 | 0.3688 | 0.1565 |
| TinyPerson | 42 | Test | 0.2787 | 0.1718 | 0.3174 | 0.3307 | 0.3625 | 0.3605 | 0.1697 |
| TinyPerson | 43 | Val | 0.5786 | 0.2759 | 0.5653 | 0.6976 | 0.7381 | 0.6817 | 0.3314 |
| TinyPerson | 43 | Test | 0.4992 | 0.2747 | 0.5058 | 0.5655 | 0.5785 | 0.6141 | 0.2960 |

Artifacts and upload markers are verified in
`duyle2408/tph-yolov5-bucket-eval-runs` under
`runs/{levirship,tinyperson}/seed_{42,43}`.

## 4.1 Kết quả augmentation seed 43/44 — bảng cuối

Protocol cố định: split seed `42`, training seeds `43/44`, NMS IoU `0.50`, LEVIR `imgsz=512` không Mosaic, TinyPerson `imgsz=640` có Mosaic. Các giá trị là **mean ± sample std** trên hai seed. Bỏ qua Varroa theo phạm vi audit.
Bảng hợp nhất đầy đủ `AP50`, `AP50-Small` và `mAP50-95`, với protocol riêng cho native/merged size metrics, nằm trong [`docs/reports/oacp_r2_seed43_44_results.md`](docs/reports/oacp_r2_seed43_44_results.md).


| Model | Variant | Dataset | Val AP50 | Val AP50-Small | Val mAP50-95 | Test AP50 | Test AP50-Small | Test mAP50-95 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8 | OACP R2 | LEVIR-Ship | 0.7380 ± 0.0015 | 0.7790 ± 0.0280 | 0.2721 ± 0.0124 | 0.7131 ± 0.0174 | 0.7880 ± 0.0098 | 0.2603 ± 0.0016 |
| YOLO11 | OACP R2 | LEVIR-Ship | 0.7285 ± 0.0313 | 0.7696 ± 0.0250 | 0.2718 ± 0.0122 | 0.6775 ± 0.0127 | 0.7407 ± 0.0372 | 0.2485 ± 0.0052 |
| YOLOv8 | OACP R2 | TinyPerson | 0.5194 ± 0.0090 | 0.5432 ± 0.0111 | 0.1860 ± 0.0004 | 0.4934 ± 0.0010 | 0.6537 ± 0.0042* | 0.1724 ± 0.0013 |
| YOLO11 | OACP R2 | TinyPerson | 0.5165 ± 0.0124 | 0.4937 ± 0.0129 | 0.1839 ± 0.0017 | 0.4996 ± 0.0019 | 0.6631 ± 0.0086* | 0.1763 ± 0.0013 |
| YOLOv8 | NegativeCanvas R2 | LEVIR-Ship | 0.7154 ± 0.1529 | 0.6684 ± 0.2192 | 0.2646 ± 0.0941 | 0.6836 ± 0.1759 | 0.6789 ± 0.2393 | 0.2385 ± 0.0920 |
| YOLOv8 | NegativeCanvas R4 | LEVIR-Ship | 0.8007 ± 0.0072 | 0.8343 ± 0.0107 | 0.3086 ± 0.0032 | 0.7879 ± 0.0002 | 0.8344 ± 0.0055 | 0.2873 ± 0.0101 |
| YOLO11 | NegativeCanvas R2 | LEVIR-Ship | 0.7544 ± 0.0992 | 0.7858 ± 0.0757 | 0.2879 ± 0.0552 | 0.7465 ± 0.1010 | 0.8124 ± 0.0721 | 0.2633 ± 0.0487 |
| YOLO11 | NegativeCanvas R4 | LEVIR-Ship | 0.7565 ± 0.0992 | 0.8063 ± 0.0323 | 0.2922 ± 0.0554 | 0.7204 ± 0.1031 | 0.8074 ± 0.0010 | 0.2666 ± 0.0516 |
| YOLOv8 | NegativeCanvas R2 | TinyPerson | 0.5584 ± 0.0074 | 0.5483 ± 0.0120 | 0.2008 ± 0.0027 | 0.5190 ± 0.0078 | 0.6841 ± 0.0095* | 0.1878 ± 0.0032 |
| YOLOv8 | NegativeCanvas R4 | TinyPerson | 0.5477 ± 0.0052 | 0.5218 ± 0.0017 | 0.2034 ± 0.0031 | 0.5195 ± 0.0026 | 0.6868 ± 0.0075* | 0.1886 ± 0.0021 |
| YOLO11 | NegativeCanvas R2 | TinyPerson | 0.5247 ± 0.0193 | 0.5174 ± 0.0082 | 0.1929 ± 0.0063 | 0.5073 ± 0.0084 | 0.6734 ± 0.0078* | 0.1812 ± 0.0048 |
| YOLO11 | NegativeCanvas R4 | TinyPerson | 0.5279 ± 0.0170 | 0.4810 ± 0.0083 | 0.1935 ± 0.0078 | 0.5057 ± 0.0204 | 0.6749 ± 0.0147* | 0.1802 ± 0.0096 |

`*` TinyPerson `Test AP50-Small` là `test_merged/AP50-Small` theo evaluator corner-window chính thức; không trộn với native `test_size/AP50-Small`. OACP LEVIR size metrics được backfill tại `duyle2408/oacp-r2-seed43-44-size-eval-runs`; OACP TinyPerson native size metrics tại `duyle2408/oacp-r2-seed43-44-tiny-size-eval-runs`. NegativeCanvas dùng các artifact size-evaluation canonical đã audit. **Lưu ý:** các dòng YOLO11 NegativeCanvas trong bảng cũ đã bị trộn seed43 `augmentation_matrix_current` với seed44 `baseline_no_mosaic_parity`; bảng corrected parity nằm trong `docs/reports/seed43_44_yolov8_yolo11_augmentation_audit.md` và thay thế các dòng YOLO11 cũ.

---


## 5. Hướng Dẫn Chạy Thực Nghiệm (Quick Start)

> [!IMPORTANT]
> Trước khi chạy bất kỳ script nào, hãy kích hoạt môi trường conda thích hợp:
> ```bash
> conda activate ml2
> # Hoặc dùng source kích hoạt trực tiếp:
> source /home/duylearch/miniconda3/bin/activate ml2
> ```

### Chạy Huấn Luyện (Matrix Sweep)
Các script huấn luyện nằm trong thư mục `train_levir_scripts/`. Hãy chạy chúng từ thư mục gốc của repository:

```bash
# Huấn luyện tổ hợp P2 + NUDFL + PC + CFR
python train_levir_scripts/train_all_levir_yolov8n_p2_nudfl_pc_cfr.py --data-root <LEVIR_DATA_ROOT> --device cuda

# Huấn luyện mô hình P2 Box Consensus
python train_levir_scripts/train_all_levir_yolov8n_p2_consensus.py --data-root <LEVIR_DATA_ROOT> --device cuda
```

### CI/CD / MLOps

Workflow `.github/workflows/mlops.yml` được kích hoạt thủ công bằng
`workflow_dispatch`, với các operation:
`train`, `evaluate`, `export`, `evaluate_jetson`, `export_jetson`, và
`jetson_pipeline`.

Job Marimo tự clone repository vào `marimo_repo`, checkout đúng SHA của run,
và tự gọi `misc.prepare_levir_ship.prepare` khi `dataset_yaml` để trống.
Default smoke training dùng `epochs=1`, `batch_size=8`, `workers=4`; full run
có thể thay đổi epoch và các input liên quan.

Training tạo và upload checkpoint, metrics, `manifest.json`, và
`upload_complete.json` lên Hugging Face. Quality gate bắt buộc
`evaluation_metrics.json`, `manifest.json`, và `nms_iou=0.5`; có thể bật thêm
ngưỡng `min_test_map50` và `min_test_map50_95`.

Marimo Environment cần `MARIMO_URL`, `MARIMO_TOKEN`, và `HF_TOKEN`.
Jetson job tải checkpoint từ Hugging Face, chuyển qua SSH, export ONNX/TensorRT
và evaluate bằng Ultralytics. Hướng dẫn input, secrets, artifact và cách
dispatch nằm trong [`docs/ci_cd_workflow.md`](docs/ci_cd_workflow.md).

### Chạy Smoke Test Kiểm Tra
Luôn khuyến khích chạy smoke test trước khi khởi động tiến trình train dài hạn:
```bash
python train_levir_scripts/test_levir_yolov8n_p2_nudfl_pc_cfr.py
```

### Phân Tích Phương Sai (Variance Diagnostics)
Để đo đạc độ ổn định của các dự đoán hộp giới hạn:
```bash
python train_levir_scripts/analyze_p2_box_field_variance.py --checkpoint <PATH_TO_PT>
```

---

## 5. Hướng Dẫn Phát Triển (Developer Guide)

### Bản Clone Ultralytics Nội Bộ (Local Ultralytics Clone)
Các khối tùy chỉnh (custom blocks), loss tùy chỉnh và tác vụ mạng (tasks) được nhận diện trực tiếp bằng cơ chế nội bộ của Ultralytics thông qua bản clone nội bộ nằm tại:
```text
models_related/ultralytics
```
Các script trong `train_levir_scripts/` tự động thiết lập import ưu tiên bản local clone này. Để kiểm tra thủ công xem bạn có đang sử dụng đúng bản clone nội bộ này không, hãy chạy:
```bash
PYTHONPATH=models_related/ultralytics python -c "import ultralytics; print(ultralytics.__file__)"
```
Kết quả kỳ vọng phải là đường dẫn nội bộ: `<repo>/models_related/ultralytics/ultralytics/__init__.py`. Nếu kết quả chứa `site-packages/`, các class tùy chỉnh sẽ lỗi `KeyError`.

### Các Khối Tùy Chỉnh Hiện Tại (Current Custom Blocks)
Các block sau đã được đăng ký trong bản local clone:
- `CFR` (Conflict-Guided Fine Reconstruction)
- `DBSSBlock` (Dynamic Background Subspace Suppression)
- `HITBlock` (Hardness-Induced Transport)

Nơi đăng ký cụ thể:
- `models_related/ultralytics/ultralytics/nn/modules/block.py` (Khai báo class và thêm vào `__all__`)
- `models_related/ultralytics/ultralytics/nn/modules/__init__.py` (Import và thêm vào `__all__`)
- `models_related/ultralytics/ultralytics/nn/tasks.py` (Import và thêm vào `base_modules` trong `parse_model()`)

*Chú ý:* File `models_related/custom_blocks/custom_blocks.py` chỉ là file tham chiếu có thể đọc trực tiếp; bộ giải mã YAML thực tế của Ultralytics sẽ gọi các class được đăng ký trong thư mục clone nói trên.

---

### Hướng Dẫn Từng Bước (Step-by-Step Guides)

#### 1. Đổi một Block trong YAML
Nên chọn phương pháp đổi tên block trước (thay vì chèn thêm) để giữ nguyên chỉ số index của các layer phía sau.
Ví dụ trong file config YAML:
- Dòng gốc: `[-1, 3, C2f, [128, True]]`
- Dòng thay thế: `[-1, 3, CFR, [128, True]]`

*Định dạng layer:* `[from, repeats, module, args]`
Đối với các block nằm trong `base_modules` của `parse_model()`, Ultralytics sẽ tự động truyền kích thước kênh của layer trước làm tham số `c1` đầu tiên của constructor. Bạn chỉ cần viết các tham số còn lại (ví dụ `c2`, `shortcut`) vào phần `args` của YAML.

#### 2. Thêm một Class Block Mới
Để thêm một block mới tên là `MyBlock`:
1. Thêm định nghĩa class vào `models_related/ultralytics/ultralytics/nn/modules/block.py`.
2. Thêm `"MyBlock"` vào danh sách `__all__` của `block.py`.
3. Import `MyBlock` và thêm vào `__all__` trong `models_related/ultralytics/ultralytics/nn/modules/__init__.py`.
4. Import `MyBlock` trong `models_related/ultralytics/ultralytics/nn/tasks.py`.
5. Đăng ký trong `parse_model()` của `tasks.py`:
   - Thêm vào danh sách `base_modules` nếu hàm `__init__` nhận đầu vào dạng `(c1, c2, ...)`.
   - Thêm vào danh sách `repeat_modules` nếu block cần tham số lặp lại `n`, ví dụ `C2f(c1, c2, n, ...)`.
6. Gọi tên `MyBlock` trong file YAML cấu hình thực nghiệm của bạn.

#### 3. Chèn một Layer Mới
*Lưu ý:* Việc chèn thêm một dòng module mới vào YAML sẽ làm dịch chuyển toàn bộ index của các layer phía sau. Hãy nhớ cập nhật lại các tham số tham chiếu `from` ở hạ nguồn (ví dụ: `[-1, 9]` hoặc `[15, 18, 21]`).

---

### Kiểm Tra Sau Khi Chỉnh Sửa (Testing & Verification)

1. **Kiểm tra import:**
   ```bash
   PYTHONPATH=models_related/ultralytics python -c "from ultralytics.nn.tasks import parse_model; print('Import OK')"
   ```
2. **Khởi tạo thử mô hình từ YAML:**
   ```bash
   PYTHONPATH=models_related/ultralytics python -c "from ultralytics import YOLO; m=YOLO('models_related/models_config/yolov8/levir/yolov8n_p2_levir_nudfl_pc_cfr.yaml'); print(m.model.model[2])"
   ```
3. **Chạy dummy forward pass:**
   ```bash
   PYTHONPATH=models_related/ultralytics python -c "import torch; from ultralytics import YOLO; m=YOLO('models_related/models_config/yolov8/levir/yolov8n_p2_levir_nudfl_pc_cfr.yaml'); y=m.model(torch.randn(1, 3, 640, 640)); print(type(y))"
   ```

---

## Kết quả DETR trên Varroa

Nguồn artifact: [`duyle2408/detr_eval_20261006`](https://huggingface.co/duyle2408/detr_eval_20261006), file [`evaluation/evaluation_metrics.json`](https://huggingface.co/duyle2408/detr_eval_20261006/raw/main/evaluation/evaluation_metrics.json). Có hai training seed: `42` và `43`. `AP50` và `mAP50:95` là mean ± sample standard deviation tính trực tiếp từ per-seed artifacts.

`AP50-small` lấy từ bảng Varroa lịch sử. Artifact HF hiện không lưu `AP50-small` theo từng seed, chỉ lưu COCO `AP-small` (IoU `0.50:0.95`), nên std của `AP50-small` chưa được xác minh và không được suy diễn từ `AP-small`.

| Model | Val AP50 | Val mAP50:95 | Val AP50-small | Test AP50 | Test mAP50:95 | Test AP50-small |
|---|---:|---:|---:|---:|---:|---:|
| Deformable DETR R50 | 75.70 ± 5.09 | 23.15 ± 1.34 | 73.25 (std chưa xác minh) | 73.85 ± 1.63 | 23.50 ± 1.98 | 68.77 (std chưa xác minh) |
| RT-DETR R18 | 83.32 ± 0.26 | 23.34 ± 0.54 | 82.60 (std chưa xác minh) | 77.05 ± 1.25 | 22.11 ± 0.48 | 73.93 (std chưa xác minh) |

Các giá trị `AP50-small` trên là metric tại IoU `0.50`, area `small`; không thay thế bằng COCO `AP-small`. NMS IoU của protocol là `0.50`.
