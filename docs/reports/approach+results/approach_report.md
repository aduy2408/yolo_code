# Tổng hợp Approach 1–6 — Tiny Object Detection trên LEVIR-Ship

File này ghi lại các hướng đã train trong giai đoạn đầu. Mục tiêu là giữ lại **ý tưởng, cách làm, kết quả và quyết định**, không ghi lại toàn bộ probe/diagnostic phụ.

Protocol giữa các nhóm thí nghiệm không phải lúc nào cũng giống hệt nhau. Khi có hai bộ số từ hai nguồn khác nhau, report giữ riêng thay vì trộn chúng.

---

# 1. P2-first và P2-only

## Mục tiêu

LEVIR-Ship có phần lớn object nhỏ, vì vậy P3/P4/P5 có resolution khá thấp cho bài toán này. Hướng đầu tiên là thêm **P2 stride 4** để detect object nhỏ ở feature map có độ phân giải cao hơn.

Sau đó dùng thêm bản **P2-only**, tức chỉ Detect trên P2. Mục đích của P2-only là tập trung toàn bộ head vào scale nhỏ và tạo một baseline gọn để thử các module P2 sau này.

## Cách làm

Full P2:

```text
Backbone + FPN/PAN
        ↓
Detect([P2, P3, P4, P5])
```

P2-only:

```text
P5 → P4 → P3 → P2 → Detect([P2])
```

## Kết quả

Explicit NMS IoU = 0.5:

| Model | Seeds | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 42/43/44 | `0.7171 ± 0.0372` | `0.0774 ± 0.0065` | `0.2466 ± 0.0143` |
| YOLOv8n + P2 | 42/43/44 | **`0.8146 ± 0.0350`** | **`0.1288 ± 0.0249`** | **`0.3151 ± 0.0280`** |

Kết quả lấy trực tiếp từ `results_hf.md` cho full P2:

| Config | Seeds | Val AP50 | Test AP50 | Test AP75 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `levir-ship-yolo-p2/default` | 42/43/44 | `0.7865 ± 0.0193` | `0.7513 ± 0.0071` | `0.1349 ± 0.0081` | `0.2950 ± 0.0063` |

Hai bảng trên đến từ hai lịch sử evaluation khác nhau nên giữ riêng.

P2-only seed 42:

| Variant | Val AP50 | Test AP50 | Test AP75 | Test mAP50-95 |
|---|---:|---:|---:|---:|
| `plain_p2_only` | `0.7565` | `0.7213` | `0.1353` | `0.2815` |

P2-only khoảng `1.61M` params và `5.58 GFLOPs`.

## Kết luận

**KEEP.** P2 là nền tảng chính cho toàn bộ project. Full P2 mạnh hơn, còn P2-only được giữ làm baseline gọn để nghiên cứu representation ở P2.

---

# 2. GCTS — Grid-Cell Target Selection

## Mục tiêu

Khi P2 được downsample về P3, một vùng `2×2` trên P2 bị gộp thành một cell P3. Với object nhỏ, vị trí object trong bốn sub-cell này có thể quan trọng. GCTS thử học cách chọn/routing sub-cell phù hợp thay vì gộp chúng như bình thường.

## Cách làm

### GCTS v1

Dùng `pixel_unshuffle(2)` để tách mỗi block `2×2` P2 thành bốn candidate. Một selector dự đoán trọng số cho bốn candidate rồi fuse lại trước khi đưa xuống P3.

Có hai kiểu target cho selector:

- one-hot;
- bilinear.

### GCTS v2

Không thay trực tiếp downsampling nữa mà dùng P2 để hỗ trợ P3 head. Selector chọn thông tin P2 cho classification và tạo expected local coordinate cho regression.

## Kết quả

GCTS v1:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| `bilinear_w01` | `0.7380` | **`0.1414`** | `0.2917` |
| `bilinear_w02` | `0.7388` | `0.1300` | `0.2825` |
| `onehot_w01` | **`0.7541`** | `0.1355` | **`0.2951`** |
| `onehot_w02` | `0.7451` | `0.1365` | `0.2927` |

GCTS v2 seed 42:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| `v2_e05` | `0.7205` | **`0.1386`** | `0.2788` |
| `v2_e05_nogate` | `0.7276` | `0.1324` | **`0.2952`** |
| `v2_e10` | **`0.7428`** | `0.1349` | `0.2874` |

YOLOv8n follow-up:

- P2 baseline mAP50-95: `0.3151 ± 0.0280`.
- `gcts_backbone_p2_p3`: `0.2963 ± 0.0285`.

## Kết luận

**CLOSED.** Routing theo sub-cell có ảnh hưởng tới metric nhưng không tạo gain ổn định. Không tiếp tục tăng độ phức tạp của selector.

---

# 3. Redesign phần localization: Offset Regression, NUDFL và Pair-Competitive DFL

## Mục tiêu

P2 đã tăng resolution nhưng AP75 vẫn thấp. Nhóm này thử thay đổi trực tiếp cách regression box hoạt động.

Ba hướng chính:

1. cho bốn cạnh box lấy feature ở vị trí hơi khác nhau;
2. tăng độ phân giải của DFL ở các khoảng cách nhỏ;
3. ép hai bin đúng của DFL thắng các bin cạnh bên.

## 3.1. P2 Offset Regression

Head dự đoán offset riêng cho `left/top/right/bottom`, sau đó dùng `grid_sample` lấy feature riêng cho từng cạnh.

Kết quả seed 42:

| AP50 | AP75 | mAP50-95 |
|---:|---:|---:|
| `0.7636` | `0.0933` | `0.2806` |

Không tốt hơn P2 baseline.

## 3.2. Non-Uniform DFL

Thay vì bin đều `[0,1,...,15]`, dùng nhiều bin hơn gần 0 vì sai số nhỏ quan trọng hơn với tiny object.

Codebook chính:

```text
[0.00, 0.35, 0.70, 1.05, 1.40, 1.80, 2.30, 2.90,
 3.60, 4.50, 5.60, 6.90, 8.40, 10.20, 12.40, 15.00]
```

Ví dụ trên YOLOv8n P3:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| P3 baseline | `0.7063` | `0.0671` | `0.2432` |
| P3 NUDFL | `0.7014` | `0.0716` | `0.2387` |

AP75 tăng rất ít nhưng AP50/mAP giảm.

## 3.3. Pair-Competitive DFL

Ngoài DFL target bình thường, thêm loss để cặp bin chứa target phải có score cao hơn hai bin ngoài cặp đó.

Ý chính:

\[
L_{PC}=\text{softplus}(m+s_{cmp}-s_{target}).
\]

NUDFL + PC-DFL sau đó được train chung với CFR. Clean rerun seed 42, không dùng offset regression:

| AP50 | AP75 | mAP50-95 |
|---:|---:|---:|
| `0.6646` | `0.0941` | `0.2477` |

## Kết luận

**CLOSED / DEPRIORITIZED.** Thay regression representation có thể đổi AP75 nhưng không cho gain tổng thể ổn định. Các thí nghiệm sau chuyển trọng tâm sang classification và feature representation.

---

# 4. P2 Spatial / Channel Attention

## Mục tiêu

Thử xem P2 đã có đủ thông tin nhưng cần reweight trước Detect hay không. Hai loại chính là:

- spatial attention: chọn vị trí;
- channel attention: chọn/reweight channel.

Có cả shared attention cho cls + reg và classification-only attention.

## Cách làm

CBAM được dùng làm baseline attention: channel attention trước, sau đó spatial attention. Ngoài ra còn tách riêng channel-only và spatial-only để xem phần nào có ích.

## Kết quả

Seed 42 P2-only:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| `plain_p2_only` | `0.7213` | **`0.1353`** | `0.2815` |
| `cls_context_mid_cbam` | **`0.7633`** | `0.1284` | **`0.2946`** |

Shared CBAM 3 seeds đạt mAP50-95 `0.3089 ± 0.0054`.

Trong ablation channel-vs-spatial:

- channel-only AP75 khoảng `0.1305`;
- spatial-only AP75 khoảng `0.0928`.

Spatial attention làm kết quả xấu rõ hơn, trong khi channel attention an toàn hơn.

## Kết luận

**Spatial/CBAM direction CLOSED, channel side được giữ lại.** Kết quả này dẫn tới GAP Channel Attention ở Approach 5.

---

# 5. GAP Channel Attention

## Mục tiêu

Bỏ spatial attention và chỉ dùng global average của từng channel để reweight P2. Ý tưởng là giữ nguyên spatial layout của P2 nhưng cho model điều chỉnh channel nào quan trọng hơn.

## Cách làm

```text
P2
 ├→ Global Average Pooling
 │      ↓
 │    Conv 1×1
 │      ↓
 │    Sigmoid
 └──── multiply → P2' → Detect
```

So sánh ba descriptor:

- GAP: global average;
- GMP: global max;
- GAP + GMP.

## Kết quả

| Variant | Seeds | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|
| GAP | 42 | `0.8162` | **`0.1305`** | **`0.3106`** |
| GAP | 43 | `0.7913` | `0.1060` | `0.2893` |
| GAP | 44 | `0.8057` | `0.1093` | `0.3061` |
| GAP mean | 42/43/44 | `0.8044` | **`0.1153`** | **`0.3020`** |
| GMP mean | 42/43/44 | **`0.8048`** | `0.0984` | `0.2912` |
| GAP+GMP | 42 | `0.7891` | `0.1028` | `0.2862` |

GAP ổn định hơn GMP, đặc biệt ở AP75/mAP.

## Kết luận

**KEEP AS REFERENCE.** GAP là một trong những baseline P2 mạnh nhất và sau đó kết hợp tốt với FTAL.

---

# 6. Feature Amplitude Calibration / Perturbation

## Mục tiêu

Sau GAP, đặt câu hỏi đơn giản hơn: có phải chỉ cần điều chỉnh **amplitude** của P2 là đủ, không cần học channel semantics hay không?

## Các variant

### A1 — Global scalar

Một scalar học được cho toàn bộ P2:

\[
Y=\alpha X.
\]

### A2 — Dynamic amplitude calibrator

Từ một số global statistics của P2, MLP dự đoán một scalar riêng cho mỗi ảnh.

### A3 — Train-time amplitude perturbation

Trong train, nhân P2 với random scalar `U(0.7,1.3)`. Inference không thay đổi feature.

### A4 — Calibrator + perturbation

Kết hợp A2 và A3.

## Kết quả

Trong matched experiment, control mAP50-95 khoảng `0.2741`:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| global scalar | **`0.8178`** | `0.0880` | **`0.2818`** |
| amplitude calibrator | — | — | terminated |
| amplitude perturbation | `0.8000` | **`0.0912`** | `0.2801` |
| calibrator + perturbation | `0.7960` | `0.0890` | `0.2763` |

Một follow-up `MatchedChannelPerturbation` đạt AP50 `0.7982`, AP75 `0.1051`, mAP `0.2985`, nhưng run chỉ continuation 41 epochs nên chỉ xem như evidence phụ.

## Kết luận

**CLOSED.** Điều chỉnh amplitude có thể giúp nhẹ nhưng không đủ mạnh để trở thành direction chính.

---

# Tóm tắt Approach 1–6

| # | Approach | Kết luận |
|---:|---|---|
| 1 | P2-first / P2-only | **KEEP** |
| 2 | GCTS | **CLOSED** |
| 3 | Offset / NUDFL / PC-DFL | **CLOSED / DEPRIORITIZED** |
| 4 | Spatial / Channel Attention | **Spatial closed, channel retained** |
| 5 | GAP Channel Attention | **KEEP AS REFERENCE** |
| 6 | Amplitude calibration / perturbation | **CLOSED** |

## Nguồn chính

- `docs/reports/approach+results/results_hf.md`
- `docs/reports/report_yolo.md`
- `docs/reports/approach+results/report_yolov8n_p2_attention.md`
- `docs/reports/approach+results/report_channel.md`
