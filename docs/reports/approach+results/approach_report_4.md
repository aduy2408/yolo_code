# Tổng hợp Approach 22–27 — Tiny Object Detection trên LEVIR-Ship

Phần này tiếp tục từ Approach 22. Approach 22 chỉ được nhắc ngắn để nối mạch; phần chính là Approach 23–27.

---

# 22. Targeted Partial-View Clipping — Bridge

Approach 22 đã được ghi đầy đủ ở report 3. Kết luận giữ nguyên: partial clipping giúp tăng recall cho object bị cắt một phần ở biên ảnh, nhưng không phải architecture chính.

**Status: KEEP AS AUXILIARY AUGMENTATION.**

---

# 23. Raw Color / Multi-Cue Evidence Fusion

## Mục tiêu

Thử đưa thêm thông tin trực tiếp từ ảnh gốc vào P2, vì có thể P2 đã mất một số low-level cue hữu ích cho tiny object.

## Cách làm

Từ RGB tạo một cue bank gồm:

- 4 color cues: `Cb`, `Cr`, `R-G`, `B-(R+G)/2`;
- 2 Sobel edge cues;
- 2 frequency cues: high-pass và DoG;
- 1 local variance cue.

Tổng cộng 9 cues, sau đó downsample về stride 4 để align với P2.

Ba variant chính:

### Color slots

Giữ final P2 32 channels nhưng chia budget:

```text
semantic P2      24 ch
low-level B       4 ch
raw color         4 ch
----------------------
final            32 ch
```

### Color formation

Dùng low-level feature + 4 color cues để học một evidence branch 8 channels, sau đó concat với 24 semantic channels.

### Multi-cue formation

Giống color formation nhưng dùng đủ 9 cues gồm color, edge, frequency và local variance.

## Kết quả

Lưu ý: runner của nhóm này ghi nhầm field `map75` bằng mAP50-95, nên không báo AP75 ở bảng dưới.

| Variant | Precision | Recall | AP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Plain P2 | **`0.8049`** | `0.6223` | `0.7213` | **`0.2815`** |
| Color slots | `0.7740` | `0.6437` | `0.6997` | `0.2664` |
| Color formation | `0.7538` | **`0.6552`** | `0.7182` | `0.2725` |
| Multi-cue formation | `0.7910` | `0.6394` | **`0.7246`** | `0.2692` |

Multi-cue tăng AP50 rất nhẹ nhưng mAP vẫn thấp hơn Plain P2. Color formation tăng recall nhưng cũng không thắng tổng thể.

## Kết luận

**CLOSED.** Chỉ expose thêm color/edge/frequency cue trực tiếp không đủ. Nếu dùng raw image sau này thì nên để branch đó tự học representation riêng trước khi fuse.

---

# 24. DCFB — Dual Channel Formation Backbone

## Mục tiêu

Thử giữ hai kiểu representation song song trong backbone:

- `M`: stream bình thường, có channel mixing;
- `I`: stream chủ yếu dùng depthwise conv để hạn chế channel mixing.

Ý tưởng là một stream học semantic bình thường, stream còn lại giữ local/channel-specific evidence lâu hơn.

## Cách làm

Khoảng `67%` channels cho mixed stream và `33%` cho isolated stream.

### Progressive DCFB

Hai stream tồn tại qua nhiều backbone stage và được cross-condition sau mỗi stage bằng projection, product/difference và gate.

### Late concat

Hai stream xử lý riêng, chỉ concat ở cuối khi cần collapse về một tensor cho FPN.

## Kết quả

Seed 42:

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Plain P2 | **`0.8049`** | `0.6223` | `0.7213` | **`0.1353`** | **`0.2815`** |
| DCFB late concat | `0.7366` | `0.6264` | `0.6961` | `0.0620` | `0.2433` |
| DCFB progressive | `0.7764` | **`0.7069`** | **`0.7400`** | `0.0835` | `0.2664` |

Progressive tốt hơn late concat, đặc biệt ở recall/AP50, nhưng AP75 và mAP vẫn thấp hơn Plain P2 rõ rệt.

## Kết luận

**CLOSED.** Tách channel processing không tạo ra một nguồn thông tin thật sự độc lập vì cả hai stream vẫn bắt đầu từ cùng feature. Lesson chính: **channel isolation không đồng nghĩa với independent representation**.

---

# 25. Native Cross-Reconstruction / FFM

## Mục tiêu

Thử dùng hai feature đã có sẵn ở P2 làm hai representation:

- `A`: feature trước C2f;
- `B`: feature sau C2f.

Sau đó fuse chúng theo concat hoặc FFM-style cross reconstruction.

## Cách làm

### Native concat

```text
A pre-C2f ─┐
            ├→ concat → Conv1×1 → new P2
B post-C2f ─┘
```

### Native FFM

Dùng global pooled vector từ A và B để tạo một rank-1 reconstructed feature, sau đó fuse:

```text
[A, B, reconstructed B] → Conv1×1
```

## Kết quả

Seed 42:

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Plain P2 | **`0.8049`** | `0.6223` | `0.7213` | **`0.1353`** | **`0.2815`** |
| Native concat | `0.7895` | `0.6710` | **`0.7280`** | `0.0867` | `0.2655` |
| Native FFM | `0.7376` | **`0.6954`** | `0.7050` | `0.0995` | `0.2500` |

Concat tăng AP50 nhẹ nhưng giảm AP75/mAP. FFM còn tệ hơn.

## Kết luận

**CLOSED.** `B = C2f(A)`, nên A và B không phải hai representation độc lập mà chỉ là ancestor/descendant trong cùng một stream. Fuse hai tensor ở hai depth khác nhau không tự động tạo complementary information.

---

# 26. Independent Representation Formation

## Mục tiêu

Đây **không phải một model đã train**, mà là principle rút ra sau các failure ở raw cues, DCFB và Native Cross.

Ý chính:

> Nếu muốn hai branch bổ sung thông tin cho nhau, mỗi branch nên tự hình thành representation riêng từ input trước khi fuse.

## Principle

Thay vì:

```text
feature X → transform thành Y → fuse X,Y
```

hoặc:

```text
same feature
├→ normal Conv
└→ depthwise Conv
```

thì ưu tiên:

```text
raw image
├→ branch A → representation A
└→ branch B → representation B

A + B → fuse → Detect
```

Ba rule chính:

1. Branch phụ nên bắt đầu gần raw input.
2. Hai branch phải có inductive bias khác nhau.
3. Mỗi branch nên form representation trước rồi mới fuse; tránh cross-talk quá sớm.

## Kết luận

**RETAIN AS DESIGN PRINCIPLE.** Đây không phải detector result, mà là guideline cho các architecture sau.

---

# 27. Local Contrast Basis

## Mục tiêu

Đây là architecture đầu tiên thử đúng principle ở Approach 26: từ raw RGB tạo một branch semantic bình thường và một branch local-relative riêng.

Ý tưởng của relative branch là tiny ship có thể khó phân biệt bằng absolute RGB, nhưng vẫn khác local neighborhood xung quanh.

## Cách làm

### Semantic branch

```text
RGB → Conv s2 → Conv s2 → C2f → M
```

### Local-relative branch

Dùng hai scale `k=9` và `k=17`:

\[
R_k=X-AvgPool_k(X).
\]

Mỗi scale giữ signed RGB residual và magnitude, sau đó đi qua cùng một encoder để tạo `R_s` và `R_l`.

Ba variant chính:

### `contrast_basis`

\[
[R_s,R_l,R_s\odot R_l,|R_s-R_l|] \rightarrow C2f.
\]

### `contrast_no_cross`

Không dùng product/difference, chỉ lặp hai representation để giữ cùng channel budget:

\[
[R_s,R_l,R_s,R_l] \rightarrow C2f.
\]

### `raw_control`

Dùng một auxiliary branch cùng capacity nhưng input là raw RGB + magnitude thay vì local contrast.

Cuối cùng fuse semantic `M` và relative representation `R` bằng C2f:

\[
F=C2f([M,R]).
\]

## Kết quả main family

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Plain P2 | `0.8049` | `0.6223` | **`0.7213`** | **`0.1353`** | **`0.2815`** |
| raw control | `0.7547` | **`0.6542`** | `0.7099` | `0.0844` | `0.2592` |
| contrast no-cross | **`0.8055`** | `0.6366` | `0.7202` | `0.1200` | **`0.2769`** |
| contrast basis | `0.7626` | `0.6365` | `0.7087` | `0.1001` | `0.2655` |

`contrast_no_cross` là variant tốt nhất trong family, nhưng vẫn dưới Plain P2 ở AP75 và mAP. Product/difference giữa hai scale làm kết quả xấu hơn.

## FFM fusion follow-up

Đổi fusion sang FFM-style làm kết quả giảm mạnh:

| Precision | Recall | AP50 | AP75 | mAP50-95 |
|---:|---:|---:|---:|---:|
| `0.6804` | `0.5733` | `0.6172` | `0.0529` | `0.2101` |

## Single-17 follow-up

Thử bỏ scale 9 và chỉ dùng local contrast `17×17`:

| Variant | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| `single17` | `0.6706` | `0.0754` | `0.2296` |
| `single17_no_form` | `0.5574` | `0.0520` | `0.1908` |
| `single_raw_form` | `0.6597` | `0.0787` | `0.2407` |

Formation block có ích so với no-form, nhưng toàn bộ family vẫn dưới baseline.

## Kết luận

**CLOSED AS SPECIFIC ARCHITECTURE.** Local contrast 9/17, explicit cross-scale interaction, FFM fusion và single17 đều không thắng Plain P2.

Tuy nhiên principle ở Approach 26 vẫn được giữ: nếu thử branch thứ hai sau này, nên form representation độc lập từ raw image trước khi fuse. Failure ở đây chỉ cho thấy **local mean subtraction chưa phải representation phù hợp**.

---

# Tóm tắt Approach 22–27

| # | Approach | Kết luận |
|---:|---|---|
| 22 | Partial-view clipping | **AUXILIARY** |
| 23 | Raw color / multi-cue fusion | **CLOSED** |
| 24 | DCFB | **CLOSED** |
| 25 | Native Cross / FFM | **CLOSED** |
| 26 | Independent representation formation | **RETAIN PRINCIPLE** |
| 27 | Local Contrast Basis | **CLOSED IMPLEMENTATION; RETAIN PRINCIPLE** |

## Nguồn chính

- `docs/reports/approach+results/approach_report_3.md`
- `docs/reports/approach+results/results_hf.md`
- `train_levir_scripts/train_levir_raw_cue.py`
- `train_levir_scripts/train_all_levir_yolov8n_p2_dcfb.py`
- `train_levir_scripts/train_all_levir_yolov8n_p2_native_cross.py`
- `train_levir_scripts/train_all_levir_yolov8n_p2_contrast_basis.py`
- `models_related/ultralytics/ultralytics/nn/modules/raw_cue_fusion.py`
- `models_related/ultralytics/ultralytics/nn/modules/local_contrast.py`
