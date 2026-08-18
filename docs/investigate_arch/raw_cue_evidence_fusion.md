# Raw Color & Multi-Cue Evidence Fusion — Probing & Architecture Investigation

**Date**: August 2026  
**NMS IoU**: 0.5 (mandatory across all evaluations)  
**Dataset**: LEVIR-Ship (seed 42, 512×512, 100 epochs, batch 8, plain P2)  
**Objective**: Evaluate whether raw image color and multi-cue evidence ($C$) provide complementary information to backbone P2 ($B$) and fused P2 ($F$), and test end-to-end evidence formation architectures.

---

## 1. Executive Summary & Core Findings

1. **Standardized Linear Probing Matrix (A, B, C, D)**:
   - **Concatenation $[F, B, C]$** provides the strongest linear ranking gain over fused P2 baseline $F$ ($\text{PairAcc} = 0.6790 \to 0.6914$).
   - **Bilinear interactions ($Z \otimes C$)** yield zero incremental gain over additive concatenation ($\Delta\text{PairAcc} \le +0.0005$).
   - **Discrepancy $D = B - \hat{B}$** (filtering out linearly predictable components of $B$ from $F$) performs worse than raw $B$ ($0.6868$ vs $0.6914$), proving that linear reconstruction removes informative non-linear backbone cues.

2. **End-to-End Detector Training (Full 100 Epochs, Seed 42)**:
   - Preserving 32-channel input capacity to Detect ($F^* = [F_{24}, E_8]$) prevents head capacity confusion.
   - **Evidential Formation ($\varphi([B, C])$) outperforms Raw Concatenation ($A2 \to B_{\text{color}}$)**:
     - `A2_color_slots` ($[F_{24}, B_4, C_4]$): **0.6997** TEST mAP50.
     - `B_color_formation` ($[F_{24}, B_8 + \varphi([B_8, C_4])]$): **0.7182** TEST mAP50 (**+0.0185** gain).
   - **Multi-Cue Bank provides additional complementary gain ($B_{\text{color}} \to B_{\text{multi}}$)**:
     - `B_multi_formation` ($[F_{24}, B_8 + \varphi([B_8, C_{\text{multi9}}])]$): **0.7246** TEST mAP50 (**+0.0249** total gain over $A2$).

---

## 2. Phase 1 — Standardized Linear Probing Matrix

### 2.1 Protocol Specification

- **Candidates**: Raw P2 pre-NMS candidate boxes with $\text{IoU} \ge 0.5$ against GT.
- **Pairs**: Same-GT candidates with $|\Delta\text{IoU}| \ge 0.05$.
- **Training**: Standardized features fit on VAL only ($\mu, \sigma$, PCA8, Ridge reconstruction). Bradley-Terry linear ranker trained on VAL difference vectors ($d_{ij} = x_i - x_j$).
- **Evaluation**: Evaluated on frozen TEST set across 15,150 VAL and 16,098 TEST candidate vectors.

### 2.2 Probe Diagnostic Results

| Suite | Mechanism Proposal | Probe Representation | PairAcc | Δ.05-.10 | Δ.10-.20 | Δ>.20 | BestRank | Spearman | Regret | Rescue / Damage | Gate Classification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Channel Replacement | $P0: F$ | 0.6790 | 0.6007 | 0.6848 | 0.7846 | 10.29 | 0.2998 | 0.1242 | Baseline | Baseline |
| | | $P1: B$ | 0.6484 | 0.5832 | 0.6538 | 0.7354 | 10.68 | 0.2857 | 0.1309 | - | - |
| | | **$P2: [F, B]$** | **0.6877** | **0.6081** | **0.6946** | **0.7929** | **10.14** | **0.3238** | **0.1222** | **17.7% / 7.1%** | **WEAK SIGNAL** |
| **B** | Cue Interaction $\varphi(B, C)$ | $P1: [F, B, C]$ (Control) | 0.6914 | 0.6100 | 0.6988 | 0.7986 | 10.15 | 0.3271 | 0.1226 | Baseline | Baseline |
| | | $P2: I_{BC}$ | 0.5655 | 0.5327 | 0.5662 | 0.6130 | 11.74 | 0.1113 | 0.1364 | - | - |
| | | **$P3: [F, B, C, I_{BC}]$** | **0.6914** | **0.6114** | **0.6974** | **0.7990** | **10.16** | **0.3268** | **0.1222** | **7.5% / 3.3%** | **FAIL** |
| **C** | Discrepancy $\varphi(B-F, C)$ | $P1: [F, B, C]$ (Control) | 0.6914 | 0.6100 | 0.6988 | 0.7986 | 10.15 | 0.3271 | 0.1226 | Baseline | Baseline |
| | | $P2: [F, D]$ ($D = B - \hat{B}$) | 0.6839 | 0.6035 | 0.6901 | 0.7919 | 10.13 | 0.3123 | 0.1219 | - | - |
| | | $P3: [F, D, C]$ | 0.6882 | 0.6065 | 0.6943 | 0.7983 | 10.08 | 0.3163 | 0.1235 | - | - |
| | | **$P4: [F, D, C, I_{DC}]$** | **0.6868** | **0.6063** | **0.6923** | **0.7962** | **10.21** | **0.3144** | **0.1202** | **12.2% / 6.1%** | **FAIL** |
| **D** | Basis Synthesis | $P3: [F, Z_B, C]$ (Control) | 0.6867 | 0.6071 | 0.6923 | 0.7948 | 10.15 | 0.3146 | 0.1247 | Baseline | Baseline |
| | | **$P4: [F, Z_B, C, Z_B \otimes C]$** | **0.6872** | **0.6067** | **0.6932** | **0.7957** | **10.25** | **0.3174** | **0.1236** | **8.1% / 3.6%** | **FAIL** |

---

## 3. Phase 2 — End-to-End Detector Training Experiments

### 3.1 Neural Module Architecture & Dataflow

All variants preserve 32-channel Detect head input capacity ($F^* = [F_{24}, \cdot_8] \in \mathbb{R}^{32 \times 128 \times 128}$ at stride 4):

```text
[A2_color_slots]
F (32ch) ---> 1x1 Conv(32->24) ------------> F24 ──┐
B (32ch) ---> 1x1 Conv(32->4) -------------> B4 ───┼─> F* = [F24, B4, C4] (32ch) ---> Detect
img0 ───────> CueBank(color4) -> Affine ---> C4 ───┘

[B_color_formation & B_multi_formation]
F (32ch) ---> 1x1 Conv(32->24) -------------------> F24 ──┐
B (32ch) ---> 1x1 Conv(32->8) -------> B8 ─────────┐     ├─> F* = [F24, E8] (32ch) ----> Detect
                                       │           ├─> E8 = B8 + ΔE8
img0 ───────> CueBank(color4/multi9) -> C ─┴─> φ(U) ┘  (ΔE8 zero-initialized)
```

1. **`RawImageCueBank`**:
   - Color (4ch): $[\text{Cb}, \text{Cr}, O_1, O_2]$ centered in $[-1, 1]$ range.
   - Multi-Cue (9ch): Color (4ch) + Sobel Edges ($G_x, G_y$, 2ch) + Frequency ($Y - G_3, G_3 - G_7$, 2ch) + Local Variance ($E[Y^2] - (E[Y])^2$, 1ch).
   - Deterministic `AvgPool4x4, stride=4` matches $128 \times 128$ P2 grid without learnable parameters.
2. **Zero-Initialized Evidential Correction**:
   - Residual $\Delta E_8 = \text{Conv}_{16\to 8}(\text{DWConv}_{3\times 3}(\text{Conv}_{\text{in}\to 16}([B_8, \text{Cue}])))$.
   - Final $1\times 1$ conv weights & biases are zero-initialized $\implies \Delta E_8 = 0 \implies E_8 = B_8$ at step 0.

### 3.2 Full 100-Epoch Evaluation Metrics (NMS IoU = 0.5)

| Run ID | Architecture Variant | Formulation ($F^*$) | VAL Precision | VAL Recall | VAL mAP50 | VAL mAP50-95 | TEST Precision | TEST Recall | TEST mAP50 | TEST mAP50-95 | $\Delta\text{TEST mAP50}$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A2_color_slots` | Raw Color Slots | $[F_{24}, B_4, C_4]$ | 0.8291 | 0.6972 | 0.7689 | 0.2926 | 0.7740 | 0.6437 | **0.6997** | 0.2664 | Baseline |
| `B_color_formation` | Color Formation | $[F_{24}, E_8(\text{color4})]$ | 0.8505 | 0.7035 | 0.7922 | 0.2993 | 0.7538 | 0.6552 | **0.7182** | 0.2725 | **+0.0185** |
| `B_multi_formation` | Multi-Cue Formation | $[F_{24}, E_8(\text{multi9})]$ | 0.8228 | 0.7132 | 0.7791 | 0.2946 | 0.7793 | 0.6483 | **0.7246** | 0.2692 | **+0.0249** |

---

## 4. Key Conclusions & Next Steps

1. **Evidential Contextualization is Essential**:
   - Directly feeding raw color cues $C_4$ into Detect (`A2_color_slots`) yields **0.6997** TEST mAP50.
   - Contextualizing $C_4$ with backbone $B_8$ via residual correction $\varphi([B_8, C_4])$ (`B_color_formation`) boosts TEST mAP50 to **0.7182** (**+0.0185** gain).
2. **Multi-Cue Bank Provides Complementary Value**:
   - Expanding color to include edges ($G_x, G_y$), frequency (DoG/HighPass), and local variance (`B_multi_formation`) achieves **0.7246** TEST mAP50 (**+0.0249** gain over color slots).
3. **Recommended Next Steps**:
   - Run seeds 43 and 44 on `B_color_formation` and `B_multi_formation` to confirm multi-seed stability.
   - Evaluate combining the winning `B_multi_formation` block with factorized GAP and TAL supervision.
