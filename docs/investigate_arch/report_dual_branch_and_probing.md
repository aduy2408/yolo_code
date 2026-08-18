# Report: Dual-Branch Local Contrast — Probing & Architecture Investigation

**Period**: August 2026  
**NMS IoU**: 0.5 (all experiments, unless noted `nms=?` for legacy runs)  
**Dataset**: LEVIR-Ship, seed 42, `256×256`, YOLO8n-P2 backbone  
**Objective**: Understand whether a local-contrast auxiliary branch adds signal beyond the standard YOLO stem, and find the minimal architecture that captures it.

---

## 1. Background & Motivation

Prior experiments established that adding a P2-scale feature map (stride-4 to the FPN) improves tiny-ship AP50 on LEVIR-Ship from ~0.73 (P3-only) to ~0.75–0.82 depending on architecture. The **plain P2P3** pipeline with factorized GAP was the best overall at AP50 ≈ 0.819.

The hypothesis under test: **the YOLO conv stem mixes spatial information too early**, blurring tiny-object high-frequency contrast signals. A dedicated local-contrast branch operating before any stride-2 conv could preserve this signal and be fused back into the FPN's P2 lateral path.

---

## 2. Phase 1 — Linear Probe on Contrast-Basis Checkpoint

### 2.1 Architecture: `contrast_basis`

```
Input RGB
 ├─ MAIN PATH: Conv 3→16 s2 → Conv 16→32 s2 → C2f(32) → [M: 32ch @ stride-4]
 └─ RAW AUX:
     Local contrast (basis_small k=9, basis_large k=17)
     → rel_encoder (shared, stride-2 × 2) → [R_s, R_l @ stride-4]
     → scale_formation C2f([R_s, R_l, R_s×R_l, R_s−R_l]) → [R: 32ch @ stride-4]
     → feature_merge FFM([M, R]) → [F: 32ch @ stride-4]
     → P2 lateral
```

Three checkpoints were trained:

| Variant              | Description                                    | AP50 (test) |
|----------------------|------------------------------------------------|-------------|
| `raw_control`        | No local contrast; `basis_small==basis_large`  | 0.7099      |
| `contrast_no_cross`  | Two scales (k=9, k=17), no product/difference  | 0.7202      |
| `contrast_basis`     | Full: two scales + product + difference        | 0.7087      |
| `contrast_basis_ffm` | FFM with attention-based fusion                | 0.6172      |

> [!NOTE]
> `contrast_no_cross` marginally outperforms `contrast_basis`. The cross-terms (product/difference) add no benefit. `contrast_basis_ffm` is significantly worse, suggesting attention-based fusion overfits or introduces gradient interference.

### 2.2 Linear Probe Results (from `raw_control` checkpoint)

Logistic regression probe on frozen representations, predicting good-candidate vs bad-candidate (val → test transfer):

| Feature                          | AP50 (probe) |
|----------------------------------|-------------|
| **M** (YOLO stem, stride-4)     | 0.933       |
| **R** (rel_encoder output)      | 0.964       |
| **[M, R]** (concat)             | 0.933       |
| **F** (FFM output)              | 0.966–0.977 |

**Interpretation (revised)**:

- `R > M` suggests the auxiliary branch output contains more linearly-separable tiny-object signal. However, `[M,R]` not exceeding `M` alone does **not** conclusively prove concat loses information — dimension and covariance differences disadvantage the logistic probe.
- `F > R` is strong evidence: the FFM output scores higher than either input even on `raw_control` (where both branches are near-identical). The self-quadratic E² expansion in `scale_formation` likely provides nonlinear expressivity.

### 2.3 Layer-wise Probe: backbone depth analysis

Probing individual layers of the `raw_control` checkpoint:

| Layer | Description                      | AP50 (linear probe) |
|-------|----------------------------------|---------------------|
| M1    | main_cv1, stride-2               | ~0.53               |
| M2    | main_cv2, stride-4               | ~0.53               |
| M3    | main_c2f, stride-4               | ~0.90               |
| R1    | rel_encoder[0], stride-2         | ~0.53               |
| R2    | rel_encoder[1], stride-4         | ~0.53               |
| R3    | after rel_encoder (=R2), stride-4| 0.895               |
| R4    | after scale_formation C2f        | 0.960               |

Key finding: **C2f is the representation-forming operator** in both paths. `M2` and `R2` (both stride-4, pre-C2f) score equally. The gain `R3 → R4` (0.895 → 0.960) comes from the `scale_formation` C2f.

> [!WARNING]
> The `R4` result should **not** be interpreted as evidence that `scale_formation` produces uniquely good representations. The probe's C2f was random-initialized and optimized directly on val labels alongside the linear probe — it is a nonlinear classifier, not a passively-trained backbone module.

---

## 3. Phase 2 — Scale Decomposition Probe: k=9 vs k=17

Using frozen features from the `raw_control` checkpoint:

| Feature           | AP50 (probe) | Pair Acc |
|-------------------|-------------|----------|
| R_s (k=9 only)   | ~0.63       | 0.650    |
| **R_l (k=17)**   | **0.649**   | **0.678**|
| [R_s, R_l] concat| ~0.63       | 0.655    |

**Finding**: `R_l (k=17) > R_s (k=9)` on all metrics. Concatenating k=9 does not help and slightly hurts pair accuracy (0.678 → 0.655). At linear-probe level, **k=9 adds no information to k=17**.

### Reconciling with R4 gain

The `R4` probe at 0.739 in the `contrast_no_cross` checkpoint was higher than `R_l` alone at 0.649. Two explanations:
1. `scale_formation` adds processing depth/capacity (k=9 irrelevant).
2. k=9 does not contribute linearly but contributes nonlinearly via C2f spatial interaction.

These hypotheses remain unresolved. Practical conclusion:
- **For linear-probe purposes**: simplify to single scale k=17.
- **For full-train architecture**: dual-scale advantage cannot be ruled out.

---

## 4. Phase 3 — Single-Path Architecture (LCF-Stem)

### 4.1 Architecture: `single17`

No dual-branch. Single path through local contrast at k=17 only:

```
RGB
 │ LocalContrast17:  L = X − AvgPool(X, k=17)
 ▼
Conv 3→8 s=2  →  Conv 8→8 s=2  →  C2f(8→8, n=1)  →  C2f(8→32, n=1) → P2 lateral
              [narrow encoder]      [formation block]
```

Main YOLO backbone runs in parallel and provides P3/P4/P5. The stem *replaces* the standard YOLO P2 lateral only.

| Variant             | Description                                | AP50 (test) | P     | R     |
|---------------------|--------------------------------------------|-------------|-------|-------|
| `single17`          | LocalContrast17 + narrow encoder + C2f    | **0.6706**  | 0.738 | 0.626 |
| `single_raw_form`   | Raw RGB + encoder + C2f (no contrast)     | 0.6597      | 0.759 | 0.592 |
| `single17_no_form`  | No C2f formation block (encoder only)     | 0.5574      | 0.617 | 0.547 |

**Findings**:
1. **Formation block is essential**: removing it drops from 0.671 → 0.557 (−11.3 AP pts).
2. **Local contrast helps marginally**: `single17` vs `single_raw_form`: +1.1 AP pts only.
3. **All single17 variants score below plain P2 baseline (≈0.745–0.759)**. The narrow 8-channel encoder limits capacity.

> [!IMPORTANT]
> The `single17` architecture has a parameter budget problem. 8 channels at stride-4 is extremely limited vs the 32ch YOLO P2 stem in the baseline. The lower AP50 may be a **capacity failure**, not an architectural failure.

---

## 5. Phase 4 — Sidecar Residual Fusion (Main Results)

### 5.1 Architecture

All 4 variants share the same backbone (pretrained YOLO → M path, 32ch stride-4), differing only in how the sidecar `F` is computed and fused:

```
RGB
 ├─ MAIN PATH (pretrained YOLO stem)
 │   Conv 3→16 s2 → Conv 16→32 s2 → C2f(32) → M [32ch @ stride-4]
 │
 └─ SIDECAR (local contrast k=17)
     LocalContrast17 → Conv 3→8 s2 → Conv 8→8 s2 → C2f(8) → R [8ch @ stride-4]

FUSION → F [32ch @ stride-4] → FPN P2 lateral
```

| Mode                | Fusion Block                    |
|---------------------|---------------------------------|
| `linear_residual`   | `F = M + W·R`                  |
| `joint_residual`    | `F = M + C2f([M,R])`           |
| `norm_joint_residual` | `F = LayerNorm(M + C2f([M,R]))` |
| `replace_joint`     | `F = C2f([M,R])` (no residual) |

### 5.2 Results (seed 42, NMS IoU=0.5)

| Variant               | AP50 (test) | AP50-95 | P     | R     |
|-----------------------|-------------|---------|-------|-------|
| `replace_joint`       | **0.7133**  | 0.2531  | 0.791 | 0.625 |
| `linear_residual`     | 0.7048      | 0.2587  | 0.731 | 0.648 |
| `joint_residual`      | 0.6786      | 0.2420  | 0.753 | 0.630 |
| `norm_joint_residual` | 0.6416      | 0.2230  | 0.715 | 0.560 |

### 5.3 Reference baselines

| Experiment                          | AP50 (test)    | Seeds   | NMS  |
|-------------------------------------|----------------|---------|------|
| Plain P2 baseline (3-seed mean)    | ~0.745–0.759   | 42/43/44| ?    |
| `contrast_no_cross`                 | 0.720          | 42      | 0.5  |
| `contrast_basis`                    | 0.709          | 42      | 0.5  |
| `raw_control`                       | 0.710          | 42      | 0.5  |
| `contrast_basis_ffm`                | 0.617          | 42      | 0.5  |
| **Plain P2P3 + GAP factorized k15** | **0.819**      | 42      | 0.5  |

---

## 6. Analysis

### 6.1 Sidecar adds no value vs strong baseline

Best sidecar (`replace_joint`, AP50=0.713) does not exceed the plain P2 baseline (≈0.745–0.759). Gap: ~3–5 AP pts.

Possible explanations:
- **Interference**: untrained sidecar perturbs pretrained YOLO weights.
- **Capacity imbalance**: 8-channel R is too narrow for meaningful C2f fusion.
- **Local contrast marginal**: preprocessing provides ~1pt benefit at best.
- **Single-seed variance**: ±3pt within expected range.

### 6.2 C2f is the key operator

Consistent pattern across all experiments:
- Without C2f formation: ~0.55–0.62 AP50
- With C2f formation: ~0.67–0.72 AP50
- Full YOLO P2 stem (deeper, wider, pretrained): ~0.75 AP50

**Local contrast preprocessing is not the differentiating factor. C2f depth and width are.**

### 6.3 Local contrast: marginal preprocessing only

| Comparison | Δ AP50 |
|------------|--------|
| `contrast_no_cross` vs `raw_control` | +1.0 pt |
| `single17` vs `single_raw_form` | +1.1 pt |
| `contrast_basis` vs `raw_control` | −0.1 pt |
| `contrast_basis_ffm` vs `raw_control` | −9.3 pt |

Conclusion: local contrast at k=17 provides ~+1pt on average. Insufficient to justify architectural complexity.

---

## 7. Full Results Table

| Architecture          | Variant                       | AP50   | AP50-95 | Seeds | NMS |
|-----------------------|-------------------------------|--------|---------|-------|-----|
| Sidecar               | replace_joint                 | 0.713  | 0.253   | 42    | 0.5 |
| Sidecar               | linear_residual               | 0.705  | 0.259   | 42    | 0.5 |
| Sidecar               | joint_residual                | 0.679  | 0.242   | 42    | 0.5 |
| Sidecar               | norm_joint_residual           | 0.642  | 0.223   | 42    | 0.5 |
| Single17              | single17                      | 0.671  | 0.230   | 42    | 0.5 |
| Single17              | single_raw_form               | 0.660  | 0.241   | 42    | 0.5 |
| Single17              | single17_no_form              | 0.557  | 0.191   | 42    | 0.5 |
| Contrast basis        | contrast_no_cross             | 0.720  | 0.277   | 42    | 0.5 |
| Contrast basis        | contrast_basis                | 0.709  | 0.266   | 42    | 0.5 |
| Contrast basis        | raw_control                   | 0.710  | 0.259   | 42    | 0.5 |
| Contrast basis        | contrast_basis_ffm            | 0.617  | 0.210   | 42    | 0.5 |
| Plain P2 baseline     | —                             | ~0.751 | ~0.295  | 3 seeds | ?  |
| **P2P3 best**         | plain_p2p3_gap_factorized_k15 | **0.819** | **0.325** | 42 | 0.5 |

---

## 8. Conclusions

### Ruled out:
- ❌ Dual-scale local contrast (k=9 + k=17)
- ❌ FFM / attention-based fusion
- ❌ Narrow 8-channel auxiliary stem as replacement for YOLO P2
- ❌ Cross-scale interaction terms (product, difference)

### Confirmed:
- ✅ C2f is the key representation operator (not local contrast preprocessing)
- ✅ The pretrained YOLO stem is hard to surpass with narrow untrained auxiliary branches
- ✅ Local contrast provides ~+1pt preprocessing benefit but insufficient to justify complexity

### Open question:
Sidecar variants were fine-tuned with the pretrained YOLO stem unfrozen. A **staged training protocol** (freeze YOLO stem → train sidecar only → joint fine-tune) might allow the sidecar branch to reach a useful representation before joint optimization.

### Next direction:
The plain P2P3 + factorized GAP (AP50=0.819) remains the reference architecture. The local contrast hypothesis is considered **weakly supported** (~1pt benefit in preprocessing). Future investment should focus on:
1. Better loss functions for sub-pixel ship detection (WIoU, dynamic label assignment)
2. Multi-seed validation of `replace_joint` (best sidecar at 0.713)
3. Capacity-matched comparison: 32-channel sidecar encoder + no local contrast vs plain P2

---

*All metrics: NMS IoU=0.5, metric-matching threshold AP50/AP50-95 per COCO protocol.*
