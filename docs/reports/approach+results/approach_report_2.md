# Tổng hợp Approach 7–14 — Tiny Object Detection trên LEVIR-Ship

Phần này tiếp tục từ Approach 1–6. Chỉ giữ lại **mục tiêu, cách làm, kết quả và kết luận** của từng hướng.

---

# 7. KVCA — Key/Value Compressed Attention

## Mục tiêu

P2 có resolution cao (`128×128` ở input 512), nên full self-attention khá tốn. KVCA giữ Query ở full resolution nhưng nén Key/Value để thử dùng global context với chi phí thấp hơn.

## Cách làm

```text
P2
 ├→ Q full resolution
 └→ K,V spatially compressed
          ↓
     Multi-head attention
          ↓
       residual
          ↓
        Detect
```

Các ablation chính:

- shared KVCA cho cả cls/reg;
- classification-only KVCA;
- `SR=8`, `SR=4`;
- transformer encoder sâu hơn;
- Channel-KVCA.

## Kết quả

Seed 42:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| shared KVCA SR8 | **`0.7857`** | `0.1029` | **`0.2988`** |
| cls-only KVCA | `0.7641` | `0.0831` | `0.2705` |
| KVCA SR4 | `0.7633` | **`0.1107`** | `0.2863` |
| Channel-KVCA | `0.7948` | `0.1080` | `0.2913` |

3-seed comparison:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| shared KVCA | `0.7979 ± 0.0109` | `0.1045 ± 0.0130` | `0.2987 ± 0.0056` |
| shared CBAM | **`0.8076 ± 0.0102`** | **`0.1115 ± 0.0058`** | **`0.3089 ± 0.0054`** |

`results_hf.md` xác nhận các result seed-42 ở trên và bổ sung Val AP50, ví dụ shared KVCA `0.8339`, cls-only `0.8027`, SR4 `0.8286`, Channel-KVCA `0.8356`.

## Kết luận

**CLOSED.** KVCA phức tạp hơn nhưng không tốt hơn CBAM/GAP. Shared placement tốt hơn cls-only, nhưng không đáng tiếp tục tăng attention complexity.

---

# 8. Local-vs-Global Attention

## Mục tiêu

Sau KVCA, thử xem vấn đề có phải do global attention nhìn quá nhiều background hay không. Vì vậy giới hạn attention vào local neighborhood.

## Cách làm

- `patch_kvca_r0`: mỗi query chỉ nhìn KV group của chính nó.
- `patch_kvca_r1`: query nhìn `3×3` KV groups xung quanh.
- `NATTEN k3`: full-resolution local attention.
- PAN-P3 attention: dùng attention ở cả P2 và P3.

## Kết quả

Seed 42:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| Patch-KVCA r0 | `0.7689` | `0.0902` | `0.2804` |
| Patch-KVCA r1 | **`0.7847`** | `0.0966` | **`0.2852`** |
| NATTEN k3 | `0.7594` | **`0.1032`** | `0.2783` |

PAN-P3 3 seeds:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| PAN-P3 KVCA | `0.7677 ± 0.0193` | **`0.1488 ± 0.0125`** | `0.3112 ± 0.0126` |
| PAN-P3 CBAM | **`0.7893 ± 0.0078`** | `0.1459 ± 0.0047` | **`0.3135 ± 0.0052`** |

HF aggregate cho các row này khớp với test values trên và bổ sung Val AP50.

## Kết luận

**CLOSED.** `r1 > r0`, tức context hơi rộng hơn có ích, nhưng local attention vẫn không tạo win rõ. Không tiếp tục tìm “attention radius tối ưu”.

---

# 9. P1 Detail Fusion / P1-GER / P1-DRR

## Mục tiêu

P1 có stride 2 nên giữ nhiều detail hơn P2. Hướng này thử lấy thông tin từ P1 để cứu các object nhỏ đã yếu ở P2.

## Cách làm

### Plain P1 fusion

Downsample P1 về kích thước P2 rồi fuse trực tiếp.

### P1-GER

Thêm gate để chỉ đưa P1 vào P2 ở một số vị trí thay vì cộng toàn bộ.

### P1-DRR

Lấy local detail từ P1:

\[
L=ReLU(P1_{down}-AvgPool_{3×3}(P1_{down})).
\]

Sau đó project detail và dùng P2 semantics + detail strength để tạo gate. Output:

\[
P2'=P2+G\odot R.
\]

Có thêm restraint loss để hạn chế gate mở trên background.

## Kết quả

Kết quả từ report P1 cũ:

| Variant | AP50 | AP75 | Recall |
|---|---:|---:|---:|
| FPN-only baseline | `0.7553` | `0.0923` | `0.6940` |
| Plain P1 fusion 200e | `0.7480` | `0.0846` | **`0.7486`** |
| P1-GER 200e | **`0.7875`** | `0.0916` | `0.7476` |
| Regression-only P1 detail | `0.7751` | **`0.1078`** | `0.7213` |

HF aggregate ghi một evaluation history khác:

| Variant | Val AP50 | Test AP50 | Test AP75 | mAP50-95 |
|---|---:|---:|---:|---:|
| topdown baseline | `0.7514` | `0.7213` | `0.1366` | `0.2865` |
| P1 fusion 200e | `0.7910` | `0.7419` | `0.1185` | `0.2803` |
| P1-GER 200e | **`0.8046`** | **`0.7632`** | `0.1237` | **`0.2937`** |
| P1-GER 500e | `0.7911` | `0.7096` | `0.1109` | `0.2630` |
| P1-DRR + partial clip | `0.7793` | `0.7195` | **`0.1516`** | `0.2929` |
| P1-DRR + CBAM | `0.7387` | `0.7096` | `0.1319` | `0.2833` |

Hai bảng được giữ riêng vì khác evaluation history.

## Kết luận

**CLOSED AS MAIN DIRECTION.** P1 có thông tin hữu ích và có thể tăng recall/AP50, nhưng cũng mang nhiều sea texture. Gain không đủ ổn định để justify thêm P1 branch làm architecture chính.

---

# 10. High-Frequency / Edge / Detail Preservation

## Mục tiêu

Kiểm tra giả thuyết model mất high-frequency/detail của tiny object.

## Cách làm

### Random HF Attenuation

Blur ảnh để lấy low-frequency, sau đó giảm ngẫu nhiên phần high-frequency trong train.

### Masked P2 Detail Reconstruction

Tính local detail:

\[
D=X-AvgPool(X)
\]

rồi mask một phần detail trong train và dùng auxiliary decoder để reconstruct lại.

## Kết quả 3 seeds từ HF

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| `hf_plain` | `0.6997 ± 0.0949` | `0.1228 ± 0.0270` | `0.2701 ± 0.0411` |
| `hf_gap` | `0.7712 ± 0.0099` | **`0.1553 ± 0.0066`** | **`0.3103 ± 0.0052`** |
| `detail_plain` | `0.7373 ± 0.0199` | `0.1253 ± 0.0040` | `0.2864 ± 0.0099` |
| `detail_gap` | **`0.7775 ± 0.0173`** | `0.1402 ± 0.0079` | `0.3072 ± 0.0138` |

`local_detail_repc2f` seed 42: AP50 `0.7532`, AP75 `0.1304`, mAP `0.2889`.

## Kết luận

**AUXILIARY ONLY.** Detail/HF có thể hữu ích khi đi cùng GAP, nhưng không có bằng chứng rằng chỉ cần tăng edge/high-frequency là đủ. Không theo hướng fixed edge enhancer làm architecture chính.

---

# 11. DBSS — Dynamic Background Subspace Suppression

## Mục tiêu

Biển/background lặp lại khá nhiều. DBSS thử model background thành một subspace, sau đó lấy phần feature khó được background giải thích làm signal foreground.

## Cách làm

1. Embed P2 feature.
2. Chọn một số background basis từ feature map.
3. Dùng ridge projection để reconstruct background.
4. Lấy residual `R = X - X_bg`.
5. Dùng residual để tạo correction cho feature.

Có thêm các variant P2-aware và routed.

## Kết quả

Explicit re-evaluation:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| YOLOv8n P2 baseline | **`0.8146 ± 0.0350`** | `0.1288 ± 0.0249` | **`0.3151 ± 0.0280`** |
| YOLOv8n + DBSS | `0.8118 ± 0.0368` | **`0.1305 ± 0.0226`** | `0.3105 ± 0.0241` |
| DBSS P2-aware | `0.7960 ± 0.0358` | `0.1238 ± 0.0149` | `0.3017 ± 0.0283` |
| DBSS P2-routed | `0.7699 ± 0.0558` | `0.1000 ± 0.0353` | `0.2893 ± 0.0329` |

HF aggregate ghi evaluation khác:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| YOLOv8n DBSS | `0.7631 ± 0.0252` | `0.1418 ± 0.0066` | `0.3032 ± 0.0110` |
| DBSS P2-aware | `0.7521 ± 0.0353` | `0.1417 ± 0.0209` | `0.2946 ± 0.0253` |
| DBSS P2-routed | `0.7002 ± 0.0624` | `0.1217 ± 0.0213` | `0.2740 ± 0.0331` |
| DBSS pre-P2 | `0.7296 ± 0.0165` | `0.1301 ± 0.0198` | `0.2859 ± 0.0075` |

Basis-count seed 42: k4 `0.2687`, k12 `0.2894`, k16 `0.2730`, k20 `0.3009` mAP50-95.

DBSS còn tốn hơn `100 MFLOPs` cho dynamic basis/projection.

## Kết luận

**CLOSED.** Có signal nhưng không thắng P2 baseline đủ rõ để justify complexity của basis selection + ridge solve.

---

# 12. HIT — Hardness-Induced Transport

## Mục tiêu

Thay vì model background như DBSS, HIT tìm các vị trí feature “khó reconstruct” cả theo spatial lẫn channel rồi transport signal đó sang vị trí khác.

## Cách làm

```text
X
├→ spatial reconstruction → residual Rs
└→ channel reconstruction → residual Rc
            ↓
      hardness map
            ↓
       top-q locations
            ↓
       predict offset
            ↓
     Gaussian transport
            ↓
       residual fuse
```

## Kết quả

Explicit evaluation:

| Model | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| YOLOv8n + HIT | `0.7955 ± 0.0311` | `0.1073 ± 0.0163` | `0.2968 ± 0.0169` |
| YOLOv8n + DBSS | **`0.8118 ± 0.0368`** | **`0.1305 ± 0.0226`** | **`0.3105 ± 0.0241`** |
| YOLOv5n + HIT | `0.6650 ± 0.0596` | `0.0900 ± 0.0217` | `0.2420 ± 0.0349` |
| YOLOv10n + HIT | `0.7013 ± 0.0501` | `0.1584 ± 0.0213` | `0.2905 ± 0.0228` |

HF aggregate cũng cho cùng xu hướng nhưng với evaluation values khác: YOLOv8n HIT mAP `0.2903`, YOLOv5n HIT `0.2442`, YOLOv10n HIT `0.2787`.

## Kết luận

**CLOSED.** Cơ chế reconstruct + select + offset + transport quá phức tạp so với gain thực tế.

---

# 13. Regression-Specific Adapters

## Mục tiêu

Thử cho regression một feature path riêng thay vì thay toàn bộ detector.

## Các variant

### `reg_local`

Một local residual block chỉ đi vào box tower.

### P1 regression-only

P1 detail chỉ được đưa vào regression, classification vẫn dùng P2 sạch.

### HV-Decoupled Regression

Tách regression thành một tower cho left/right và một tower cho top/bottom.

## Kết quả

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| Plain P2 | `0.7213` | **`0.1353`** | `0.2815` |
| `reg_local` | **`0.7411`** | `0.1323` | **`0.2937`** |
| `hv_decoupled` | `0.7214` | `0.1116` | `0.2699` |

HF row cho P1 regression-only: AP50 `0.7437`, AP75 `0.1463`, mAP `0.2914`. Một report cũ ghi AP50 `0.7751`, AP75 `0.1078`; giữ hai source riêng.

P1 regression-only còn làm compute tăng mạnh trong implementation đó, khoảng `6.34 → 14.73 GFLOPs`.

## Kết luận

**CLOSED / DEPRIORITIZED.** `reg_local` có gain mAP nhẹ nhưng AP75 không tăng. Không có bằng chứng regression capacity là bottleneck chính.

---

# 14. Classification / Detect Head Redesign

## Mục tiêu

Nhiều candidate có box khá tốt nhưng confidence thấp, nên thử xem classification head có quá yếu hay cls/reg bị tách quá sớm hay không.

## Các variant

1. `cls_context_mid_cbam`: thêm local context vào classification tower.
2. Shared-head: share một hoặc toàn bộ feature blocks giữa cls/reg.
3. Tăng classification width từ 32 lên 64 bằng DWConv hoặc full Conv.

## Kết quả

Classification context:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| plain P2 | `0.7213` | **`0.1353`** | `0.2815` |
| cls context | **`0.7633`** | `0.1284` | **`0.2946`** |

Shared-head trên GAP+FTAL:

| Head | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| decoupled | **`0.8283`** | **`0.1388`** | **`0.3161`** |
| share1 | `0.7420` | `0.0985` | `0.2730` |
| fully shared | `0.7846` | `0.0948` | `0.2863` |

Tăng cls width:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| default | **`0.8283`** | **`0.1388`** | **`0.3161`** |
| cls64 DW | `0.7945` | `0.1136` | `0.2922` |
| cls64 fullConv | `0.7924` | `0.1092` | `0.2918` |

HF aggregate khớp với các test values trên và bổ sung Val AP50.

## Kết luận

**CLOSED.** Làm classifier lớn hơn hoặc share cls/reg nhiều hơn đều không giúp. Giữ standard decoupled Detect head.

---

# Tóm tắt Approach 7–14

| # | Approach | Kết luận |
|---:|---|---|
| 7 | KVCA | **CLOSED** |
| 8 | Local/global attention | **CLOSED** |
| 9 | P1 detail fusion / rescue | **CLOSED AS MAIN** |
| 10 | HF / edge / detail preservation | **AUXILIARY ONLY** |
| 11 | DBSS | **CLOSED** |
| 12 | HIT | **CLOSED** |
| 13 | Regression adapters | **CLOSED / DEPRIORITIZED** |
| 14 | Classification / Detect redesign | **CLOSED; KEEP DECOUPLED HEAD** |

## Nguồn chính

- `docs/reports/approach+results/results_hf.md`
- `docs/reports/approach+results/report_yolov8n_p2_attention.md`
- `docs/reports/investigate_pooling.md`
- `docs/investigate_gap_feature_miss/report_p2_detail_hf_plain_gap.md`
- `docs/reports/report_yolo.md`
- `docs/reports/approach+results/report_factorized_tal.md`
