# Tổng hợp Approach 15–22 — Tiny Object Detection trên LEVIR-Ship

Phần này tiếp tục từ Approach 1–14. Chỉ giữ lại **mục tiêu, cách làm, kết quả và kết luận** của từng hướng.

---

# 15. Scale-Tempered TAL Targets

## Mục tiêu

GAP cho thấy nhiều candidate có box khá ổn nhưng confidence vẫn thấp. Hướng này thử **tăng positive classification target của object nhỏ** để classifier mạnh dạn hơn.

## Cách làm

Với TAL target gốc `q`:

\[
q' = q + \lambda\left(q^{\tau(s)}-q\right), \qquad s=\sqrt{wh}.
\]

Với object nhỏ, `tau < 1` nên target được tăng lên. Effect giảm dần khi object lớn hơn, chủ yếu áp dụng cho P2. Có warm-up để không thay target quá mạnh từ đầu train.

Các variant chính:

| Variant | `tau_min` | `lambda` |
|---|---:|---:|
| mild | `0.75` | `0.5` |
| medium | `0.50` | `0.5` |
| strong | `0.50` | `1.0` |

## Kết quả

Seed 42:

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| GAP | `0.8266` | `0.7701` | **`0.8162`** | **`0.1305`** | **`0.3106`** |
| mild | `0.8283` | `0.7744` | `0.8084` | `0.1213` | `0.3053` |
| medium | `0.8235` | `0.7615` | `0.7898` | `0.1111` | `0.2891` |
| strong | `0.8292` | **`0.7950`** | `0.8154` | `0.1144` | `0.3027` |

Strong variant tăng recall nhưng AP75/mAP giảm.

## Kết luận

**CLOSED.** Chỉ tăng target magnitude không đủ. Kết quả này dẫn tới FTAL, nơi target ceiling và ranking giữa positive được xử lý riêng.

---

# 16. Factorized TAL — FTAL

## Mục tiêu

Scale tempering tăng tất cả positive theo cách tương đối giống nhau. FTAL thử tách hai việc:

1. mức target cao nhất của một GT;
2. ranking giữa các positive của GT đó.

Mục tiêu là tăng supervision cho candidate tốt nhất nhưng không nâng tất cả candidate yếu lên cùng lúc.

## Cách làm

Với positive TAL scores `q_i`:

\[
r_i=\frac{q_i}{q_{max}}.
\]

Gọi `u_max` là IoU tốt nhất trong các candidate của GT. Target mới:

\[
q_i^{new}=u_{max}^{\tau}r_i^{\kappa}.
\]

Sau đó blend với TAL target cũ:

\[
q_i^{final}=q_i+\lambda(q_i^{new}-q_i).
\]

Config chính:

- `tau = 0.75`;
- `kappa = 1.5`;
- `lambda = 0.5`;
- chỉ áp dụng cho object nhỏ, khoảng `sqrt(wh) < 32 px`;
- P2-only + warm-up.

## Kết quả seed 42

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| GAP | `0.8266` | `0.7701` | `0.8162` | `0.1305` | `0.3106` |
| ceiling-only | `0.8310` | `0.7787` | `0.8189` | `0.1280` | `0.3157` |
| separation-only | **`0.8527`** | `0.7572` | `0.8204` | `0.1216` | `0.3145` |
| **FTAL k=1.5** | `0.8393` | **`0.7882`** | **`0.8283`** | **`0.1388`** | **`0.3161`** |
| FTAL k=2.0 | `0.8321` | `0.7773` | `0.8208` | `0.1188` | `0.3091` |

`k=1.5` là variant tốt nhất trong nhóm.

## Multi-seed FTAL k=1.5

| Seed | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 42 | `0.8393` | `0.7882` | `0.8283` | `0.1388` | `0.3161` |
| 43 | `0.8488` | `0.7759` | `0.8277` | `0.1291` | `0.3195` |
| 44 | `0.8157` | `0.7759` | `0.8147` | `0.1402` | `0.3180` |
| Mean | `0.8346 ± 0.0170` | `0.7800 ± 0.0071` | `0.8235 ± 0.0076` | `0.1360 ± 0.0060` | **`0.3179 ± 0.0017`** |

## Control quan trọng

FTAL không tự động tốt trên Plain P2:

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Plain P2 | `0.8049` | `0.6223` | `0.7213` | **`0.1353`** | **`0.2815`** |
| Plain P2 + FTAL | `0.7937` | **`0.7342`** | **`0.7621`** | `0.0985` | `0.2701` |

FTAL tăng recall/AP50 nhưng làm AP75/mAP giảm nếu không có GAP.

`lambda=1` cũng không tốt hơn tổng thể: AP50 `0.7956`, AP75 `0.1421`, mAP `0.3107`. Các mode mass-preserving, geometry-ranking và agreement-gated cũng không vượt `k=1.5`.

## Kết luận

**KEEP AS REFERENCE.** GAP + FTAL k=1.5 là một trong những reference mạnh nhất của project. FTAL nên được xem là một interaction với GAP, không phải TAL replacement dùng ở đâu cũng tốt.

---

# 17. GGCF — Geometry-Guided Candidate Field

## Mục tiêu

Thử cho candidate biết rõ geometry của box hiện tại thay vì bắt head tự suy ra hoàn toàn từ feature.

## Cách làm

1. Chọn các candidate P2 có score cao.
2. Lấy local patch `7×7` quanh candidate.
3. Thêm bốn channel khoảng cách tương đối tới `left/right/top/bottom` của coarse box.
4. Một encoder nhỏ dự đoán residual cho box và classification score.

Ba variant:

- `G1`: cùng architecture nhưng geometry channels = 0;
- `G2`: dùng geometry thật;
- `G3`: dùng refined box của GGCF trong TAL assignment.

## Kết quả

Seed 42:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| GAP | `0.8162` | `0.1305` | `0.3106` |
| GAP+FTAL | **`0.8283`** | **`0.1388`** | **`0.3161`** |
| G1 zero geometry | `0.7867` | `0.1165` | `0.2962` |
| G2 GGCF | `0.8115` | `0.1063` | `0.3015` |
| G3 refined assignment | `0.7747` | `0.1114` | `0.2815` |

G2 tốt hơn G1, nên geometry cue có ích, nhưng vẫn dưới GAP+FTAL. G3 còn tệ hơn.

## Kết luận

**CLOSED.** Geometry field có signal nhưng không đủ để thành direction chính. Đặc biệt không nên đưa refined box chưa ổn định vào assignment quá sớm.

---

# 18. CFR — Conflict-Guided Fine Reconstruction

## Mục tiêu

Thay vì fuse P1 vào inference, CFR dùng P1 như **auxiliary target trong train** để ép P2 giữ lại detail. Inference không thêm cost.

## Cách làm

```text
P2
 ↓
Conv1×1
 ↓
Upsample ×2
 ↓
DWConv3×3
 ↓
Conv1×1
 ├→ reconstruct P1
 └→ reconstruct P1 local detail
```

Target được detach:

- `P1`;
- `P1 - AvgPool3(P1)`.

Các vùng positive có conflict cao được weight mạnh hơn, đại khái:

\[
W=M(1+\eta C).
\]

Detect vẫn nhận P2 gốc; decoder chỉ tồn tại trong train.

## Kết quả

Các run đầu bị contamination do `P2OffsetRegression` vô tình bật, nên không dùng để kết luận.

Clean seed-42 rerun, offset tắt:

| Split | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Val | `0.7177` | `0.6576` | `0.7128` | `0.1284` | `0.2806` |
| Test | `0.6986` | `0.6264` | `0.6646` | `0.0941` | `0.2477` |

Kết quả thấp hơn rõ so với GAP/GAP+FTAL.

## Kết luận

**CLOSED AS MAIN.** Ý tưởng training-only reconstruction vẫn có thể dùng như auxiliary loss, nhưng reconstruct P1/detail theo cách này không giúp detector đủ nhiều.

---

# 19. CRC và Candidate Verifier

## Mục tiêu

Nhiều candidate có geometry khá tốt nhưng confidence thấp. Hướng này thử thêm một branch nhỏ chỉ sửa classification score, không thay regression.

## CRC — Contextual Ring Classification

Lấy center feature `F` và ring context quanh nó:

```text
F ──────────────┐
                ├→ concat → 1×1 fusion → residual vào cls path
RingPoolR5(F) ──┘
```

Regression vẫn dùng feature cũ.

## Kết quả CRC

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| GAP+FTAL | **`0.8393`** | **`0.7882`** | **`0.8283`** | `0.1388` | **`0.3161`** |
| GAP+FTAL+CRC | `0.8263` | `0.7721` | `0.8100` | **`0.1420`** | `0.3133` |

CRC tăng AP75 rất nhẹ (`+0.0032`) nhưng giảm AP50, recall và mAP.

## Candidate verifier variants

Plain P2 còn thử:

- `a1_box_fovea`;
- `a3_semantic_structural`;
- `a4_raw_adapted`.

Trong run này `verifier_alpha=0`, nên output riêng của verifier không được cộng vào inference score. Vì vậy ba variant ra cùng metric và không nên coi đây là comparison công bằng giữa ba cơ chế.

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Plain P2 | `0.8049` | `0.6223` | `0.7213` | **`0.1353`** | **`0.2815`** |
| a1 / a3 / a4 | `0.7838` | `0.7188` | `0.7530` | `0.1026` | `0.2741` |

## Kết luận

**CRC CLOSED; verifier DEPRIORITIZED.** Local context có thể chứa thông tin, nhưng dense trainable verifier hiện tại không tạo gain tổng thể.

---

# 20. Ring / Local Context

## Mục tiêu

Kiểm tra xem tiny object có nên được biểu diễn bằng center feature cùng context ngay xung quanh thay vì center feature đơn lẻ hay không.

## Cách làm

Các cue đã xem xét gồm:

- fixed ring pooling;
- local average;
- inner/outer region;
- center + ring;
- inner minus ring;
- box-conditioned inner/ring.

Primitive chính `RingPoolR5` lấy trung bình annulus khoảng `1 < distance <= 5` quanh mỗi P2 cell, channel-wise.

Một lesson quan trọng là không nên mặc định dùng `F - Ring`; giữ `[F, Ring]` riêng thường hợp lý hơn vì center và context đều có thể chứa signal.

## Kết quả / Kết luận

Các probe cho thấy ring context có signal, nhưng khi đưa thành dense CRC branch thì detector không tốt hơn. Ngoài ra ring vẫn được lấy từ **cùng P2 feature**, nên nó không phải một nguồn thông tin độc lập.

**CLOSED AS MAIN DIRECTION.** Không tiếp tục thêm ring radius/gate/attention trên cùng P2.

---

# 21. Border / Partial-View Failure

## Mục tiêu

Một số miss xuất hiện gần biên ảnh. Cần biết lỗi do:

- padding / vị trí gần biên;
- regression;
- hay object bị cắt mất một phần.

## Kết quả

Recall của small object:

| Nhóm | Recall |
|---|---:|
| Center | `68.35%` |
| Gần biên nhưng vẫn full visible | `72.00%` |
| Touching / bị cắt bởi biên | `50.00%` |

Với nhiều touching misses, model vẫn có candidate box tốt, ví dụ IoU:

```text
0.700, 0.709, 0.718, 0.790, 0.847
```

nhưng confidence thấp.

## Kết luận

**Không cần border-specific architecture.** Vấn đề chính là **partial observation**, không phải chỉ vì object nằm gần mép ảnh. Vì vậy hướng này được chuyển sang augmentation ở Approach 22.

---

# 22. Targeted Partial-View Clipping

## Mục tiêu

Train model quen với tiny object bị crop mất một phần ở biên ảnh.

## Cách làm

Chỉ dùng trong train:

- probability tối đa `0.25`;
- target object area `< 400 px²`;
- giữ khoảng `55–85%` object visible;
- dịch ảnh bằng integer pixel để tránh interpolation blur;
- vùng ảnh mới lộ ra được fill bằng negative sea image khi có thể.

Cùng ablation còn có `small_weight`: tăng classification weight cho object nhỏ.

## Kết quả

Validation:

| Variant | AP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| P2 + WIoU | `0.7946` | `0.3277` | `0.8396` | `0.7201` |
| small_weight | `0.7507` | `0.2923` | `0.7616` | `0.6960` |
| partial_clip | **`0.8205`** | **`0.3317`** | **`0.8447`** | **`0.7670`** |
| small_weight + clip | `0.7449` | `0.3028` | `0.7656` | `0.6702` |

Test:

| Variant | AP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| P2 + WIoU | **`0.7797`** | **`0.2966`** | **`0.8099`** | `0.7112` |
| small_weight | `0.7264` | `0.2697` | `0.7017` | `0.6796` |
| partial_clip | `0.7714` | `0.2959` | `0.7890` | **`0.7284`** |
| small_weight + clip | `0.7245` | `0.2805` | `0.7426` | `0.6466` |

Partial clip tăng test recall `0.7112 → 0.7284`, còn AP50/mAP gần như giữ nguyên.

Placement của augmentation cũng quan trọng:

| Variant | AP50 | AP75 | Recall |
|---|---:|---:|---:|
| P1-DRR + alternate partial clip | `0.7616` | `0.1145` | `0.7356` |
| P1-DRR + old post-Mosaic clip | `0.7122` | `0.0955` | `0.6471` |

## Kết luận

**KEEP AS AUXILIARY AUGMENTATION.** Partial clipping có ích cho recall và robustness. `small_weight` và post-Mosaic clipping thì bỏ.

---

# Tóm tắt Approach 15–22

| # | Approach | Kết luận |
|---:|---|---|
| 15 | Scale-Tempered TAL | **CLOSED; precursor to FTAL** |
| 16 | FTAL | **KEEP AS REFERENCE** |
| 17 | GGCF | **CLOSED** |
| 18 | CFR | **AUXILIARY ONLY / CLOSED AS MAIN** |
| 19 | CRC / verifier | **CLOSED / DEPRIORITIZED** |
| 20 | Ring / local context | **CLOSED AS MAIN** |
| 21 | Border / partial-view analysis | **NO STANDALONE MODULE** |
| 22 | Targeted Partial Clip | **KEEP AS AUXILIARY AUGMENTATION** |

## Nguồn chính

- `docs/reports/approach+results/report_factorized_tal.md`
- `docs/reports/approach+results/report_ggcf.md`
- `docs/reports/report_yolo.md`
- `docs/investigate_gap_feature_miss/report_p2_detail_hf_plain_gap.md`
- `docs/investigate_arch/verifier_plain_p2.md`
- `docs/reports/detection_results.md`
- `docs/reports/approach+results/report_ablation_results.md`
- `docs/reports/investigate_pooling.md`
