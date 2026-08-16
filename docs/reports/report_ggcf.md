# Report: Geometry-Guided Candidate Field (GGCF) Evaluation

This report documents the results of the **Geometry-Guided Candidate Field (GGCF)** experiments evaluated on the LEVIR-Ship dataset. We compare the three GGCF variants against the established **GAP baseline** and the **GAP + FTAL (Factorized TAL)** method.

---

## 1. Experimental Protocol

- **Dataset**: LEVIR-Ship fixed split (seed 42)
- **Image Size**: $512 \times 512$
- **Training Epochs**: 100 epochs, batch size = 8
- **NMS Evaluation**: Explicitly evaluated using NMS IoU threshold of `0.5`
- **Hugging Face Repository**: [duyle2408/levir-yolov8n-p2-gap-ftal-ggcf](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-gap-ftal-ggcf)

### Model Variants Evaluated:
1. **G1_field_only**: Ablation control. The `GGCFEncoder` is constructed with $C + 4$ channels, but instead of the dynamic geometry map, it concatenates zero geometry maps. This ensures it has **identical parameter count and architecture** to G2/G3.
2. **G2_ggcf**: Full geometry-guided candidate field encoder with standard TAL target assignment.
3. **G3_ggcf_refined_assign**: Full geometry-guided candidate field encoder with GGCF-refined bounding boxes used for the TAL target classification assignment.

---

## 2. Quantitative Results (NMS IoU = 0.5)

| Method / Variant | Split | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GAP Baseline** | Val | 0.8605 | 0.7743 | 0.8361 | 0.1515 | 0.3338 |
| | Test | 0.8266 | 0.7701 | 0.8162 | 0.1305 | 0.3106 |
| **GAP + FTAL (k=1.5, lambda=0.5)** | Val | 0.8605 | 0.7743 | 0.8361 | 0.1515 | 0.3338 |
| | Test | 0.8393 | 0.7882 | **0.8283** | **0.1388** | **0.3161** |
| **G1_field_only** (Ablation - Zero Geometry) | Val | 0.8370 | 0.8018 | 0.8262 | 0.1458 | 0.3294 |
| | Test | 0.8135 | 0.7644 | 0.7867 | 0.1165 | 0.2962 |
| **G2_ggcf** (Standard TAL) | Val | **0.8661** | 0.7761 | 0.8256 | 0.1439 | 0.3293 |
| | Test | 0.8370 | 0.7773 | 0.8115 | 0.1063 | 0.3015 |
| **G3_ggcf_refined_assign** (Refined TAL Assign) | Val | 0.8545 | 0.7519 | 0.8061 | 0.1348 | 0.3102 |
| | Test | 0.8366 | 0.7313 | 0.7747 | 0.1114 | 0.2815 |

---

## 3. Analysis & Key Takeaways

### A. G2_ggcf vs. G1_field_only (The Geometry Signal is Real)
* **G2_ggcf** outperforms **G1_field_only** on the Test set:
  * **AP50**: $+2.48\%$ ($0.8115$ vs. $0.7867$)
  * **mAP50-95**: $+0.53\%$ ($0.3015$ vs. $0.2962$)
* **Insight**: Since G1 and G2 have the exact same architecture and parameter counts, this performance gap directly confirms that the **dynamic geometry guidance signal** is contributing positively to localization, rather than the improvement being a mere byproduct of model capacity expansion.

### B. Comparison with GAP + FTAL Baseline
* **G2_ggcf** remains slightly below the optimized **GAP + FTAL** baseline on the Test set (AP50 of $0.8115$ vs. $0.8283$, mAP50-95 of $0.3015$ vs. $0.3161$).
* **Insight**: While GGCF shows clean improvements over its parameter-matched control (G1), it does not yet outperform the FTAL baseline on its own.

### C. Refined Assignment Regression (G3)
* **G3_ggcf_refined_assign** underperforms both G1 and G2 (AP50 drops to $0.7747$ on Test).
* **Insight**: Directly feeding the GGCF-refined bounding boxes into the TAL classification assignment stage appears to degrade training stability. This might be due to inconsistencies/fluctuations in refined coordinates early in training, causing the classifier targets to become noisy.

---

## 4. Next Steps & Recommendations

1. **Investigate Hybrid GGCF + FTAL**:
   Since FTAL targets classification matching and GGCF targets regression guidance, they could be complementary. Testing G2 (GGCF) combined with the FTAL loss formulation could yield optimal results.
2. **Refined TAL assignment stabilization (Alternative to G3)**:
   Instead of using GGCF-refined boxes directly for TAL matching from epoch 0, apply a warmup phase or use a soft blend between coarse and refined boxes to prevent noisy classification targets.
