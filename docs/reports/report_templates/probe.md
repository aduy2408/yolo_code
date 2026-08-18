

```text
B = backbone P2, Layer 2
F = fused P2, Layer 18
R = representation mới cần probe
```

`R` có thể là:

```text
CbCr
Opponent color
edge
frequency
raw RGB
B
[B, CbCr]
fixed interaction
...
```

Protocol chính thức như sau.

1. Model và candidate source

YOLO checkpoint frozen hoàn toàn.

Chỉ dùng raw P2 candidates trước NMS:

```text
image
 ↓
YOLO frozen
 ↓
raw P2 decoded boxes
```

Không dùng NMS, không dùng final detections.

Mỗi P2 cell `i` có:

[
b_i = \text{decoded box}
]

và với GT `g`:

[
u_i = IoU(b_i,g).
]

Chỉ xét GT có ít nhất hai candidate:

[
u_i \ge 0.5.
]

Đây là candidate-quality probe, không phải object/background probe.

---

2. Feature extraction

Với candidate `i` tại P2 cell `(y_i,x_i)`:

[
F_i = F[:,y_i,x_i]
]

[
B_i = B[:,y_i,x_i].
]

Nếu representation từ ảnh gốc như CbCr:

```text
image
 ↓ fixed transform
CbCr
 ↓ deterministic pooling to stride 4
C
```

thì:

[
C_i=C[:,y_i,x_i].
]

Không crop per candidate nếu không cần.

Không random Conv.

Không train auxiliary architecture.

Probe chỉ được dùng:

* representation đã train trong checkpoint;
* fixed deterministic transforms;
* hoặc algebraic fixed interactions được định nghĩa trước.

---

3. Split

Cực kỳ cố định:

```text
TRAIN   không dùng
VAL     fit preprocessing + probe weights
TEST    evaluation only
```

Tất cả thứ sau chỉ được fit trên VAL:

```text
mean/std
PCA nếu có
linear probe
MLP diagnostic nếu có
```

TEST tuyệt đối không tham gia.

---

4. Standardization

Mỗi base component được standardize riêng.

Ví dụ:

[
F,;B,;C.
]

Trên VAL:

[
\mu_F,\sigma_F,\quad
\mu_B,\sigma_B,\quad
\mu_C,\sigma_C.
]

Sau đó:

[
\tilde F = \frac{F-\mu_F}{\sigma_F+\epsilon}
]

và tương tự B/C.

TEST dùng đúng statistics từ VAL.

Quan trọng: nếu probe `[F,B,C]`, không fit scaler mới trên concatenation. Standardize từng component rồi mới concat:

[
x_i=[\tilde F_i,\tilde B_i,\tilde C_i].
]

Như vậy comparison giữa các probe sạch hơn.

---

5. Pair construction

Với mỗi GT, chỉ lấy:

[
u_i\ge0.5.
]

Tạo pair `(i,j)` nếu:

[
|u_i-u_j|\ge0.05.
]

Nếu:

[
u_i>u_j
]

thì candidate `i` phải rank cao hơn.

Không mix candidates từ hai GT khác nhau.

Không dùng negative background ở quality probe.

---

6. Probe training

Official probe là Bradley–Terry linear ranking.

Feature của candidate:

[
x_i.
]

Pair feature:

[
d_{ij}=x_i-x_j.
]

Train:

[
P(i>j)=\sigma(w^\top d_{ij})
]

với BCE.

Sau khi train xong, candidate score là:

[
s_i=w^\top x_i.
]

Không cần bias vì pair difference triệt bias, nhưng có bias cũng không ảnh hưởng ranking.

Đây là probe chính thức.

Không dùng object/background BCE để kết luận quality representation nữa.

---

7. Mandatory representations

Mọi experiment phải có ít nhất:

```text
P0 = F
P1 = R
P2 = [F,R]
```

Nếu R phụ thuộc B, ví dụ chroma + original P2:

```text
P0 = F
P1 = B
P2 = [F,B]
P3 = R
P4 = [F,R]
```

Nếu câu hỏi là `B + C`:

```text
P0 = F
P1 = [F,B]
P2 = [F,C]
P3 = [F,B,C]
```

Không được chỉ report standalone R.

Metric quyết định luôn là:

[
[F,R] \text{ vs } F.
]

Vì cái mình cần là information **complementary**, không phải representation tự nó mạnh.

---

8. Mandatory metrics

Tao sẽ cố định đúng 6 metric.

Thứ nhất, Overall PairAcc:

[
PairAcc=
P(s_i>s_j\mid u_i>u_j+0.05).
]

Đây là metric chính.

Thứ hai, PairAcc theo difficulty:

```text
ΔIoU 0.05–0.10
ΔIoU 0.10–0.20
ΔIoU >0.20
```

Đặc biệt quan trọng bucket `0.05–0.10`.

Thứ ba, Best-IoU Rank.

Với mỗi GT:

```text
candidate có IoU cao nhất thật
↓
đứng rank bao nhiêu theo probe score?
```

Lower tốt hơn.

Thứ tư, Spearman:

[
\rho(s_i,u_i)
]

trong từng GT, rồi average GT-wise.

Thứ năm, Regret:

[
Regret_g=
\max_i u_i-u_{\arg\max s_i}.
]

Lower tốt hơn.

Thứ sáu, GT-level win rate.

Với mỗi GT:

[
\Delta PA_g =
PA_g([F,R])-PA_g(F).
]

Report:

```text
% GT improved
% GT degraded
median ΔPairAcc
mean ΔPairAcc
```

Cái này tránh chuyện một vài GT có nhiều candidate pair kéo global number.

---

9. Rescue / Damage

Nếu `[F,R]` là candidate architecture proposal thì bắt buộc thêm:

[
Rescue =
P(FR\ correct\mid F\ wrong)
]

và

[
Damage =
P(FR\ wrong\mid F\ correct).
]

Ta muốn:

[
Rescue > Damage.
]

Không chỉ nhìn net PairAcc.

Ví dụ:

```text
F PairAcc      .681
[F,R]          .696

Rescue         10.2%
Damage          5.8%
```

đẹp.

Nếu:

```text
Rescue 18%
Damage 17%
```

thì representation đang reshuffle nhiều hơn là sửa lỗi.

---

10. Reporting format cố định

Mọi probe từ giờ report đúng table này:

| Representation | PairAcc | Δ.05-.10 | Δ.10-.20 | Δ>.20 | BestRank | Spearman | Regret |
| -------------- | ------: | -------: | -------: | ----: | -------: | -------: | -----: |
| F              |         |          |          |       |          |          |        |
| R              |         |          |          |       |          |          |        |
| [F,R]          |         |          |          |       |          |          |        |

Sau đó một bảng GT-level:

| Representation | GT improved | GT degraded | Median ΔPA | Rescue | Damage |
| -------------- | ----------: | ----------: | ---------: | -----: | -----: |
| [F,R] vs F     |             |             |            |        |        |

Thế là đủ.

---

11. Decision gate

Từ giờ tao sẽ dùng gate này để tránh mỗi lần lại đổi tiêu chuẩn.

`FAIL`:

```text
ΔPairAcc < +0.005
hoặc
GT improved ≈ GT degraded
hoặc
Rescue <= Damage
```

Không build.

`WEAK SIGNAL`:

```text
ΔPairAcc +0.005 → +0.015
các metric khác cùng chiều
```

Có information, nhưng chưa đủ để justify architecture phức tạp.

Có thể build nếu mechanism cực cheap hoặc có lý do architectural mạnh.

`PASS`:

```text
ΔPairAcc >= +0.015
hard-pair bucket cũng tăng
BestRank giảm
Regret giảm
GT improved > degraded rõ
Rescue > Damage rõ
```

Build.

`STRONG PASS`:

```text
ΔPairAcc >= +0.02
và gain xuất hiện nhất quán ở hard pairs
```

Build ngay, không cần probe thêm.

---

12. Một distinction cực quan trọng

Từ giờ nên có hai loại probe và **không bao giờ trộn conclusion của chúng**.

`Quality Probe` là protocol ở trên:

```text
IoU >= .5
same-GT pairs
Bradley–Terry
```

Dùng để quyết định architecture cho candidate quality / AP75 / ranking.

Còn nếu muốn hỏi:

> feature có phân biệt ship/background không?

thì đó là `Detection Separability Probe` riêng:

```text
positive vs hard negative
AP
AUC
```

Nó chỉ là diagnostic phụ.

Không bao giờ dùng AP/AUC của separability probe để nói “architecture đáng build” nữa. Local contrast đã chứng minh tại sao.

---

Vậy ví dụ chroma hiện tại sẽ được ghi rất đơn giản:

```text
Probe ID: QP-CbCr-v1

Baseline:
F = Layer18 fused P2

R:
C = fixed CbCr
    C0 = stride4 average
    C3 = local 3×3 average
    C5 = local 5×5 average

Representations:
F
C
[F,C]

Candidates:
raw P2 pre-NMS
IoU >= .5

Pairs:
same GT
ΔIoU >= .05

Training:
VAL Bradley-Terry linear
VAL standardization

Evaluation:
TEST only

Decision:
based on PairAcc + hard-pair + GT-level rescue/damage
```

Còn nếu probe `B + C`:

```text
Probe ID: QP-B-CbCr-v1

F
[F,B]
[F,C]
[F,B,C]
```

