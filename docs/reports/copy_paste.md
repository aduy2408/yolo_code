# Copy-Paste Runs — YOLOv8 và YOLO11

Cập nhật: 2026-11-06

## Quy ước đọc metrics

- Tất cả metric là fraction, không phải percentage.
- `AP50` và `mAP50-95` trong bảng native là metric trên split `test` của detector.
- `AP50-Small` có hai protocol khác nhau:
  - `native test AP50-Small`: TinyBenchmark area bucket trên ảnh test native; IoU `0.50:0.05:0.75`, `maxDets=200`.
  - `test_merged/AP50-Small`: TinyPerson official merged corner-window evaluator trên ảnh gốc.
- Không thay thế native metric bằng merged metric.
- NMS IoU: `0.50`.
- TinyPerson split seed: `42`.

## 1. Kết luận coverage

| Family | Detector | Dataset/protocol | Coverage đã xác minh |
|---|---|---|---|
| Historical CP1/CP2 post-hoc | YOLOv8 canonical P3/P4/P5 | LEVIR-Ship | CP1 và CP2, seeds `42,43,44` |
| Current augmentation matrix CP1 | YOLOv8 canonical P3/P4/P5 | TinyPerson/LEVIR/Varroa | CP1 seeds `42,43,44` |
| Current model-specific CP1/CP2 | YOLO11 canonical P3/P4/P5 | TinyPerson | CP1/CP2 seeds `43,44` |
| Negative Canvas R2/R4 | YOLOv8 | LEVIR-Ship/TinyPerson | seeds `43,44` |
| Negative Canvas R2/R4 | YOLO11 | LEVIR-Ship/TinyPerson | seeds `43,44`; seed43 parity and seed44 historical artifacts are kept provenance-separated |

**Correction quan trọng:** YOLO11 CP1 và CP2 đều đã chạy đủ hai seed `43,44`. YOLO11 NC-R2 và NC-R4 cũng có hai seed `43,44`; chỉ không được gộp seed43 parity với seed44 historical thành một aggregate không ghi protocol.

**Correction quan trọng:** YOLOv8 CP2 không bị thiếu trong family historical. CP1/CP2 YOLOv8 đã có đủ seeds `42,43,44`; artifact family này khác với repo `augmentation-copy-paste-runs` của matrix mới.

## 2. Historical YOLOv8 CP1/CP2 — LEVIR-Ship

Canonical YOLOv8 P3/P4/P5, post-hoc Copy-Paste family, split seed `42`. Bảng này là aggregate trên seeds `42,43,44`.

| Run | Seeds | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | AP50-Small |
|---|---|---:|---:|---:|---:|---:|
| CP1 — one single-object paste | 42,43,44 | 0.8262 | 0.3264 | 0.7974 ± 0.0144 | 0.3109 ± 0.0079 | — |
| CP2 — two single-object pastes | 42,43,44 | 0.8233 | 0.3322 | 0.8046 ± 0.0174 | 0.3167 ± 0.0082 | — |

`AP50-Small` không có trong post-hoc artifact này. Không suy luận từ native test AP50.

Source: `duyle2408/levir-copy-paste-posthoc-test-runs`, provenance checkpoint `duyle2408/stw-yolo-runs`; report: `docs/reports/augmentation_report.md`, sections 2.2 and 4.3.
### 2.1 Historical YOLOv8 CP1/CP2 — TinyPerson, no Mosaic

Đây chính là **YOLOv8 CP2** mà bảng trước đã bỏ sót. Artifact nằm trong `duyle2408/tinyperson-copy-paste` và là family historical no-Mosaic; không phải current seed43/44 Mosaic matrix ở mục 3.

| Run | Test AP50 | Test mAP50-95 | Test AP50-Small |
|---|---:|---:|---:|
| CP1 | 0.4636 | 0.1669 | — |
| CP2 | **0.4653** | **0.1666** | — |

Source: `docs/reports/augmentation_report.md`, section 6.4, historical TinyPerson CP0–CP3 table. Artifact này không có split-qualified AP50-Small và không có cặp seed43/44, nên không được gộp vào mean/std TinyPerson seed43/44 ở mục 11.


## 3. YOLOv8 current CP1 matrix — TinyPerson seed 43/44

Đây là canonical YOLOv8 P3/P4/P5, Mosaic on, `imgsz=640`. Đây là family khác với historical post-hoc CP1/CP2 ở mục 2.

| Run | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Native test AP50-Small | Merged test AP50-Small |
|---|---:|---:|---:|---:|---:|---:|---:|
| CP1 | 43 | 0.5069 | 0.1823 | 0.4900 | 0.1730 | 0.5075 | 0.6617 |
| CP1 | 44 | 0.5061 | 0.1842 | 0.4980 | 0.1771 | 0.5215 | 0.6669 |
| CP2 | 43/44 | — | — | — | — | — | — |

`CP2` trong bảng này không có artifact verified tại prefix current matrix; không được dùng dòng này để phủ nhận CP2 historical ở mục 2.

## 4. YOLO11 CP1/CP2 — TinyPerson

### 4.1 Seed 43 — run đã được kiểm tra trong session hiện tại

| Run | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Native test AP50-Small | Merged test AP50-Small |
|---|---:|---:|---:|---:|---:|---:|---:|
| CP1 | 43 | 0.52898 | 0.19902 | 0.52086 | 0.18835 | — | 0.69343 |
| CP2 | 43 | 0.55595 | 0.20339 | 0.52177 | 0.18833 | 0.53857 | — |

CP2 có `test_merged/available=0.0`; native bucket backfill đã upload riêng.

### 4.2 Seed 44 — rerun matched to the image/paper CP1 protocol

This rerun supersedes the earlier seed-44 size-evaluation CP1 row that reported merged `AP50-Small=0.69746`. The earlier row belongs to a different artifact/protocol family and is not used in the final CP1 table. This matched run uses the image-family configuration: `cp1_single1`, Mosaic on, `p=0.5`, YOLO11n canonical P3/P4/P5, `imgsz=640`, train seed `44`, split seed `42`, NMS IoU `0.50`.

| Run | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Merged test AP50 | Merged test mAP50-95 | Merged test AP50-Small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CP1 | 44 | 0.4960 | 0.1769 | 0.49569 | 0.17691 | 0.51327 | 0.18719 | 0.66007 |

Source: `duyle2408/model-augmentation-seed42-cp1-runs/yolov11/tinyperson/copy_paste/cp1_single1/mosaic/seed_44`; merged metrics verified on Hugging Face.

### 4.3 Superseded YOLO11 CP1/CP2 rows

The previous seed-43/44 CP1/CP2 rows below are retained only as provenance. They must not be combined with the matched image-family CP1 rows above. In particular, the earlier CP1 values around `0.693–0.697` for merged `AP50-Small` are excluded from the final table.

| Run | Seed | Test AP50 | Test mAP50-95 | Merged test AP50-Small | Status |
|---|---:|---:|---:|---:|---|
| CP1 | 43 | 0.52086 | 0.18835 | 0.69343 | Superseded; different artifact family |
| CP2 | 43 | 0.52177 | 0.18833 | — | Different artifact family |
| CP1 | 44 | 0.52266 | 0.18803 | 0.69746 | Superseded; different artifact family |
| CP2 | 44 | 0.51585 | 0.18819 | 0.68831 | Different artifact family |

## 5. Negative Canvas R2/R4 — TinyPerson seed 43/44

Các dòng seed 43 là artifact model-specific trong `levirship-tinyperson-yolov8-yolo11-copy-paste-nc43-44-runs`, protocol `augmentation_matrix_current`. Seed 44 lấy từ artifact size-evaluation riêng. Hai protocol được giữ tách biệt.

| Model | Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Native test AP50-Small | Merged test AP50-Small |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8 | NC-R2 | 43 | 0.5189 | 0.1816 | 0.5049 | 0.1780 | 0.5199 | 0.6684 |
| YOLOv8 | NC-R2 | 44 | 0.5532 | 0.1989 | 0.5135 | 0.1855 | — | 0.6774 |
| YOLOv8 | NC-R4 | 43 | 0.5139 | 0.1866 | 0.4975 | 0.1761 | 0.5097 | 0.6602 |
| YOLOv8 | NC-R4 | 44 | 0.5440 | 0.2056 | 0.5214 | 0.1901 | — | 0.6920 |
| YOLO11 | NC-R2 | 43 | 0.5111 | 0.1885 | 0.5014 | 0.1778 | 0.5072 | 0.6678 |
| YOLO11 | NC-R2 | 44 | 0.5384 | 0.1974 | 0.5132 | 0.1846 | 0.5204 | 0.6789 |
| YOLO11 | NC-R4 | 43 | 0.5159 | 0.1880 | 0.4913 | 0.1734 | 0.5108 | 0.6645 |
| YOLO11 | NC-R4 | 44 | 0.5399 | 0.1990 | 0.5202 | 0.1869 | 0.5259 | 0.6853 |

## 6. Negative Canvas R2/R4 — LEVIR-Ship seed 43/44

LEVIR không có merged TinyPerson metric. `AP50-Small` dưới đây là native test bucket nếu artifact backfill có field; các core metric lấy từ model-specific audit.

| Model | Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8 | NC-R2 | 43 | 0.7188 | 0.2572 | 0.6758 | 0.2419 |
| YOLOv8 | NC-R2 | 44 | 0.7486 | 0.2773 | 0.6752 | 0.2481 |
| YOLOv8 | NC-R4 | 43 | 0.7149 | 0.2536 | 0.7049 | 0.2511 |
| YOLOv8 | NC-R4 | 44 | 0.7427 | 0.2635 | 0.6816 | 0.2419 |
| YOLO11 | NC-R2 | 43 | 0.6842 | 0.2488 | 0.6751 | 0.2289 |
| YOLO11 | NC-R4 | 43 | 0.6863 | 0.2530 | 0.6475 | 0.2300 |

YOLO11 seed 44 có artifact riêng trong `duyle2408/yolov9-yolo11-seed44-size-eval-runs`:

| Model | Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Val AP50-Small | Test AP50-Small |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11 | NC-R2 | 44 | 0.82450 | 0.32696 | 0.81800 | 0.29779 | 0.83931 | 0.86341 |
| YOLO11 | NC-R4 | 44 | 0.82659 | 0.33135 | 0.79330 | 0.30308 | 0.82916 | 0.80667 |

## 7. Corrected YOLO11 parity rerun

Seed 43 YOLO11 NC-R2/R4 được rerun với `baseline_no_mosaic_parity` (`scale=0.5`, `translate=0.1`) để khớp seed-42 workflow. Các giá trị dưới đây là mean ± sample SD của seed 43/44 parity artifacts, không được trộn với bảng historical ở mục 5.

| Model | Dataset | Variant | Val AP50 | Val AP50-Small | Val mAP50-95 | Test AP50 | Test AP50-Small | Test mAP50-95 | Merged Test AP50-Small |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11 | LEVIR-Ship | NC-R2 | 0.823769 ± 0.001031 | 0.822394 ± 0.023921 | 0.325032 ± 0.002732 | 0.805516 ± 0.017653 | 0.831259 ± 0.045471 | 0.296434 ± 0.001918 | — |
| YOLO11 | LEVIR-Ship | NC-R4 | 0.825049 ± 0.002182 | 0.821418 ± 0.010948 | 0.327918 ± 0.004850 | 0.792255 ± 0.001478 | 0.799825 ± 0.009683 | 0.298528 ± 0.006432 | — |
| YOLO11 | TinyPerson | NC-R2 | 0.542617 ± 0.005962 | 0.510842 ± 0.001126 | 0.198432 ± 0.001516 | 0.518101 ± 0.006867 | 0.527061 ± 0.009468 | 0.186621 ± 0.002914 | 0.685662 ± 0.009521 |
| YOLO11 | TinyPerson | NC-R4 | 0.545606 ± 0.008045 | 0.487605 ± 0.017529 | 0.199907 ± 0.001275 | 0.519385 ± 0.001113 | 0.524395 ± 0.002154 | 0.187623 ± 0.000963 | 0.684272 ± 0.001449 |

## 8. Historical YOLOv8 NC-R2/R4 — seed 42 reference

Đây là matched single-seed canonical YOLOv8 LEVIR experiment, không phải seed 43/44 model-specific matrix.

| Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test AP50-Small |
|---|---:|---:|---:|---:|---:|---:|
| NC-R2 | 42 | 0.8440 | 0.3322 | 0.8207 | 0.3133 | — |
| NC-R4 | 42 | 0.8305 | 0.3352 | 0.8200 | 0.3158 | — |

## 9. Artifact sources

- Historical YOLOv8 CP1/CP2: `duyle2408/levir-copy-paste-posthoc-test-runs`; checkpoint provenance `duyle2408/stw-yolo-runs`.
- Current YOLOv8 augmentation matrix: `duyle2408/augmentation-copy-paste-runs`.
- YOLOv8/YOLO11 NC seed 43/44: `duyle2408/levirship-tinyperson-yolov8-yolo11-copy-paste-nc43-44-runs`.
- YOLO11 seed 44 CP/NC size evaluation: `duyle2408/yolov9-yolo11-seed44-size-eval-runs`.
- YOLO11 seed 43 CP2 bucket backfill: `duyle2408/augmentation-size-metrics-runs`.
- Supporting reports:
  - `docs/reports/augmentation_report.md`
  - `docs/reports/seed43_44_yolov8_yolo11_augmentation_audit.md`
  - `docs/reports/yolov9_yolo11_seed44_size_evaluation.md`

## 10. Interpretation rule

Không được tạo một mean/std chung từ các bảng trên nếu chưa giữ nguyên:

1. detector YAML/model family;
2. dataset và image/window protocol;
3. Mosaic/no-Mosaic;
4. training seed set;
5. `augmentation_matrix_current` so với `baseline_no_mosaic_parity`;
6. native AP50-Small so với merged AP50-Small.

Các số trong file này được gom vào một file duy nhất để tra cứu, nhưng các provenance family khác nhau vẫn được giữ tách biệt.
## 11. Final TinyPerson table — protocol-separated

The final table does **not** combine incompatible artifact families. The matched image-family YOLO11 CP1 result is reported separately from the existing seed43/44 matrix rows. `AP50-Small` is always the merged TinyPerson metric when the column says `Merged`.

### 11.1 Matched image-family CP1

| Model | Variant | Seeds | Test AP50 | Test mAP50-95 | Merged test AP50 | Merged test mAP50-95 | Merged test AP50-Small |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO11 | CP1 | 42,44 | 0.49504 ± 0.00093 | 0.17608 ± 0.00118 | 0.51456 ± 0.00092 | 0.18524 ± 0.00138 | **0.66056 ± 0.00069** |

Seed42 is the image-table artifact; seed44 is the matched rerun verified in `duyle2408/model-augmentation-seed42-cp1-runs`. Values are mean ± sample standard deviation.

### 11.2 Existing seed43/44 matrix family

These rows remain useful for the NC comparison and for CP2 provenance, but are **not** merged with the image-family CP1 row above.

| Model | Variant | Seeds | Native test AP50 | Native test mAP50-95 | Merged test AP50 | Native test AP50-Small | Merged test AP50-Small |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8 | CP1 | 43,44 | 0.49403 ± 0.00569 | 0.17504 ± 0.00287 | **0.51611 ± 0.00488** | 0.51450 ± 0.00990 | 0.66430 ± 0.00368 |
| YOLOv8 | NC-R2 | 43,44 | 0.50918 ± 0.00610 | 0.18176 ± 0.00525 | **0.53212 ± 0.01289** | — | 0.67292 ± 0.00636 |
| YOLOv8 | NC-R4 | 43,44 | 0.50947 ± 0.01687 | 0.18308 ± 0.00992 | **0.51568 ± 0.00277** | 0.50970 | 0.66113 ± 0.00131 |
| YOLO11 | **CP1*** | 42,44 | **0.49504 ± 0.00093** | **0.17608 ± 0.00118** | **0.51456 ± 0.00092** | — | **0.66056 ± 0.00069** |
| YOLO11 | NC-R2 | 43,44 | 0.50734 ± 0.00835 | 0.18117 ± 0.00480 | **0.53132 ± 0.00935** | 0.51378 ± 0.00932 | 0.67338 ± 0.00784 |
| YOLO11 | NC-R4 | 43,44 | 0.50575 ± 0.02040 | 0.18018 ± 0.00956 | **0.52808 ± 0.02354** | 0.51837 ± 0.01068 | 0.67489 ± 0.01472 |

`*` YOLO11 CP1 is the matched image-family protocol (seeds 42,44). YOLOv8 CP1 merged AP50 was backfilled with merged-only stable vectorized NMS and verified on `duyle2408/augmentation-merged-ap50-backfill-runs`. YOLOv8 NC-R4 seed44 was then backfilled with the same protocol: merged AP50 `0.513719`, merged mAP50-95 `0.183692`, merged AP50-Small `0.662059`; the row now aggregates seeds 43/44. Native AP50-Small remains seed43-only because the seed44 artifact has no native size-bucket field.

Std is sample standard deviation (`ddof=1`).

Riêng YOLOv8 CP1/CP2 historical LEVIR post-hoc family có aggregate riêng trên seeds `42,43,44`:

| Model | Variant | Test AP50 | Test mAP50-95 | AP50-Small |
|---|---|---:|---:|---:|
| YOLOv8 | CP1 | 0.7974 ± 0.0144 | 0.3109 ± 0.0079 | 0.79089 ± 0.02442 |
| YOLOv8 | CP2 | 0.8046 ± 0.0174 | 0.3167 ± 0.0082 | — |
## 12. Corrected `baseline_no_mosaic_parity` backfill

These rows are the canonical parity re-evaluations. TinyPerson uses the official merged corner-window evaluator; LEVIR-Ship uses native split evaluation. All runs use split seed `42`, NMS IoU `0.50`, and the local pinned evaluation workflow. Results were uploaded to `duyle2408/augmentation-merged-ap50-backfill-runs` (TinyPerson) and `duyle2408/augmentation-levir-parity-eval-runs` (LEVIR-Ship).

### 12.1 TinyPerson merged metrics

| Model | Variant | Seed | Merged AP50 | Merged mAP50-95 | Merged AP50-Small |
|---|---|---:|---:|---:|---:|
| YOLOv8 | CP1 | 43,44 | 0.51600 ± 0.00480 | 0.18499 ± 0.00051 | 0.66413 ± 0.00371 |
| YOLOv8 | NC-R2 | 43 | 0.54712 | 0.20035 | 0.69069 |
| YOLOv8 | NC-R2 | 44 | 0.54136 | 0.19716 | 0.67772 |
| YOLOv8 | NC-R4 | 43 | 0.54455 | 0.19807 | 0.68116 |
| YOLOv8 | NC-R4 | 44 | 0.55188 | 0.20322 | 0.69233 |
| YOLO11 | NC-R2 | 43 | 0.55074 | 0.19954 | 0.69282 |
| YOLO11 | NC-R2 | 44 | 0.53788 | 0.19522 | 0.67883 |
| YOLO11 | NC-R4 | 43 | 0.54123 | 0.19877 | 0.68323 |
| YOLO11 | NC-R4 | 44 | 0.54480 | 0.19771 | 0.68550 |

### 12.2 LEVIR-Ship native metrics

| Model | Variant | Seed | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test AP50-small |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8 | NC-R2 | 43 | 0.81507 | 0.31812 | 0.80631 | 0.30100 | 0.83403 |
| YOLOv8 | NC-R2 | 44 | 0.82352 | 0.33112 | 0.80796 | 0.30350 | 0.85032 |
| YOLOv8 | NC-R4 | 43 | 0.80584 | 0.31144 | 0.78774 | 0.27978 | 0.82400 |
| YOLOv8 | NC-R4 | 44 | 0.79565 | 0.30638 | 0.78798 | 0.29457 | 0.81112 |
| YOLO11 | NC-R2 | 43 | 0.82436 | 0.32321 | 0.79315 | 0.29509 | 0.79905 |
| YOLO11 | NC-R2 | 44 | 0.82443 | 0.32663 | 0.81806 | 0.29785 | 0.80447 |
| YOLO11 | NC-R4 | 43 | 0.82349 | 0.32427 | 0.79119 | 0.29376 | 0.79283 |
| YOLO11 | NC-R4 | 44 | 0.82663 | 0.33128 | 0.79327 | 0.30321 | 0.78570 |
| YOLO11 | CP1 | 43 | — | — | 0.82387 | 0.31485 | 0.80738 |
| YOLO11 | CP1 | 44 | — | — | 0.81290 | 0.30554 | 0.80454 |

YOLO11 CP1 LEVIR is filled from the available seed43 parity artifact and seed44 CP1 artifact. Seed44 native `AP50-small=0.80454` was backfilled with the local size-bucket evaluator and uploaded to `duyle2408/yolov9-yolo11-cp-nc-seed44-runs`. YOLOv8 CP1 native `AP50-small` was backfilled for seeds 43 and 44 from `duyle2408/stw-yolo-runs`, yielding `0.79089 ± 0.02442`. YOLOv8 NC-R2 seed43 was retrained on the seed44 commit and backfilled with native `AP50-small=0.83403`; its two-seed aggregate is `AP50=0.80714 ± 0.00116`, `mAP50-95=0.30225 ± 0.00177`, `AP50-small=0.84217 ± 0.01152`.

The canonical YOLOv8 NC-R2 parity rows now use `duyle2408/yolov8-nc-r2-seed43-levir-parity-commit-a82-runs` for seed43 and the existing seed44 parity artifact. The former seed43 `0.55932` artifact is retained only as superseded provenance because it used a different source commit.
