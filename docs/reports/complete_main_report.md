# Complete Main Report: Augmentation Matrix

**Updated:** 2026-10-03  
**Scope:** YOLO augmentation matrix only: OACP, Mosaic, and Copy-Paste.  
**Metric policy:** preserve split-qualified metrics exactly as reported. Use `--` when a metric is absent from the source artifact. Do not relabel unsplit metrics as validation or test metrics.

## 1. Report status

This report consolidates the 12 requested augmentation variants. Existing metrics are copied from the current augmentation report and related uploaded artifacts. Missing variants are labeled **Missing run**. Existing runs with metrics that are not split-qualified remain **Existing, unsplit metrics only** and are candidates for later checkpoint evaluation rather than automatic retraining.

The final matrix coverage is:

| Method | Variants | LEVIR missing runs | TinyPerson missing runs | Varroa missing runs | Total missing |
|---|---:|---:|---:|---:|---:|
| OACP | 3 | 0 | 0 | 3 | **3** |
| Mosaic | 5 | 0 | 1 | 5 | **6** |
| Copy-Paste | 4 | 0 | 0 | 4 | **4** |
| **Total** | **12** | **0** | **1** | **12** | **13 runs** |

**Evaluation backlog:** 13 runs are missing entirely and require training or another valid source checkpoint. Additional `--` cells may only require checkpoint re-evaluation if the corresponding checkpoint exists. They must not be treated as missing training runs automatically.

---

## 2. OACP: 3 variants

The current adaptive-OACP family contains LEVIR and TinyPerson results. The report does not provide the remaining split-qualified fields for those rows, so they remain `--` rather than being filled from another experiment family.

| Variant | Dataset | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Spacing-adaptive | LEVIR | -- | -- | -- | -- | 0.8079 | 0.0977 | 0.3065 | -- | Existing |
| Spacing-adaptive | TinyPerson | -- | -- | 0.2134 | -- | 0.5252 | -- | 0.1923 | -- | Existing |
| Spacing-adaptive | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| Load-adaptive | LEVIR | -- | -- | -- | -- | **0.8204** | **0.1240** | **0.3109** | -- | Existing |
| Load-adaptive | TinyPerson | -- | -- | 0.2085 | -- | **0.5341** | -- | **0.1955** | -- | Existing |
| Load-adaptive | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| Mass-adaptive | LEVIR | -- | -- | -- | -- | 0.7731 | 0.1017 | 0.2881 | -- | Existing |
| Mass-adaptive | TinyPerson | -- | -- | 0.2113 | -- | 0.5310 | -- | 0.1940 | -- | Existing |
| Mass-adaptive | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |

**Protocol note:** this is the historical adaptive-OACP family. LEVIR uses P2/P3/P4 + Mosaic and TinyPerson uses adaptive OACP + Mosaic. Do not mix validation values from the separate R2 no-Mosaic audits into these rows.

---

## 3. Mosaic: 5 variants

LEVIR M0/M2-M5 have split-qualified validation/test AP50 and mAP50-95 in the current source report. TinyPerson has Standard/M2/M3/M4; M5 is not available in the requested matrix. AP75 and AP50-S are left as `--` where the family artifact does not expose them.

| Variant | Dataset | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Standard | LEVIR | 0.8044 | -- | 0.3118 | -- | 0.7653 | -- | 0.2876 | -- | Existing |
| Standard | TinyPerson | **0.5245** | -- | **0.1889** | -- | **0.4948** | -- | **0.1750** | -- | Existing |
| Standard | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| M2 Cluster-preserving | LEVIR | 0.7863 | -- | 0.2735 | -- | 0.7410 | -- | 0.2634 | -- | Existing |
| M2 Cluster-preserving | TinyPerson | 0.4689 | -- | 0.1660 | -- | 0.4458 | -- | 0.1575 | -- | Existing |
| M2 Cluster-preserving | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| M3 Post-scale-constrained | LEVIR | 0.7577 | -- | 0.2735 | -- | 0.7278 | -- | 0.2546 | -- | Existing |
| M3 Post-scale-constrained | TinyPerson | 0.5038 | -- | 0.1759 | -- | 0.4818 | -- | 0.1736 | -- | Existing |
| M3 Post-scale-constrained | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| M4 Adaptive-geometry | LEVIR | 0.7887 | -- | 0.2819 | -- | 0.7765 | -- | 0.2767 | -- | Existing |
| M4 Adaptive-geometry | TinyPerson | 0.5006 | -- | 0.1750 | -- | 0.4874 | -- | 0.1740 | -- | Existing |
| M4 Adaptive-geometry | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| M5 Hard-negative | LEVIR | **0.8180** | -- | **0.3079** | -- | **0.7883** | -- | **0.2921** | -- | Existing |
| M5 Hard-negative | TinyPerson | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| M5 Hard-negative | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |

TinyPerson Standard Mosaic values `0.5245/0.1889` validation and `0.4948/0.1750` test are from the matched canonical baseline artifact. The older controlled policy artifact's Test AP75 value is not inserted here because it belongs to a different subfamily.

---

## 4. Copy-Paste: 4 variants

LEVIR CP1/CP3 use corrected post-hoc split-qualified results. NegativeCanvas R1/R4 have explicit validation/test artifacts. TinyPerson CP1 and CP3 have results in the source report, but only as unsplit metrics, so they are not relabeled as validation or test values here.

| Variant | Dataset | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CP1 Single-object | LEVIR | 0.8262 | -- | 0.3264 | -- | 0.7974 | -- | 0.3109 | -- | Existing |
| CP1 Single-object | TinyPerson | -- | -- | -- | -- | -- | -- | -- | -- | Existing, **unsplit metrics only** |
| CP1 Single-object | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| CP3 Clustered | LEVIR | 0.8281 | -- | 0.3324 | -- | 0.7985 | -- | 0.3046 | -- | Existing |
| CP3 Clustered | TinyPerson | -- | -- | -- | -- | -- | -- | -- | -- | Existing, **unsplit metrics only** |
| CP3 Clustered | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| NegativeCanvas R1 | LEVIR | 0.8213 | -- | 0.3248 | -- | **0.8222** | -- | 0.3156 | -- | Existing |
| NegativeCanvas R1 | TinyPerson | **0.5287** | -- | **0.1892** | -- | **0.5037** | -- | **0.1817** | **0.6772** | Existing |
| NegativeCanvas R1 | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |
| NegativeCanvas R4 | LEVIR | **0.8305** | -- | **0.3352** | -- | 0.8200 | -- | **0.3158** | -- | Existing |
| NegativeCanvas R4 | TinyPerson | 0.4990 | -- | 0.1781 | -- | 0.4946 | -- | 0.1756 | **0.6646** | Existing |
| NegativeCanvas R4 | Varroa | -- | -- | -- | -- | -- | -- | -- | -- | **Missing run** |

TinyPerson NegativeCanvas `AP50-S` values are merged original-image TinyBenchmark metrics (`test_merged/AP50-Small`), not standard corner-window AP50 values.

TinyPerson unsplit source metrics retained for later evaluation:

| Variant | Dataset | Unsplit AP50 | Unsplit mAP50-95 | Split-qualified fields |
|---|---|---:|---:|---|
| CP1 Single-object | TinyPerson | 0.4636 | 0.1669 | -- |
| CP3 Clustered | TinyPerson | 0.4881 | 0.1751 | -- |

These values remain explicitly unsplit and must not be copied into `val/*` or `test/*` columns.

---

## 5. Evaluation backlog

### 5.1 Missing training runs

The following 13 matrix entries have no valid run represented in the current source report and require a new training run or an explicitly identified equivalent checkpoint:

- OACP: Spacing-adaptive / Varroa; Load-adaptive / Varroa; Mass-adaptive / Varroa.
- Mosaic: Standard / Varroa; M2 / Varroa; M3 / Varroa; M4 / Varroa; M5 / TinyPerson; M5 / Varroa.
- Copy-Paste: CP1 / Varroa; CP3 / Varroa; NegativeCanvas R1 / Varroa; NegativeCanvas R4 / Varroa.

### 5.2 Existing checkpoints with missing metrics

Rows containing `--` are not automatically missing runs. Before training, check whether the existing run has a checkpoint and manifest. If it does, schedule read-only evaluation to populate only the missing fields, preserving:

- dataset and exact split protocol;
- detector YAML and commit;
- training seed and split seed;
- `val/*` versus `test/*` labels;
- TinyPerson `test/*` versus `test_merged/*` protocol;
- NMS IoU and image-size settings from the original manifest.

### 5.3 Evaluation output contract

For every evaluated row, write an artifact containing at least:

```text
val/AP50
val/mAP50-95
test/AP50
test/mAP50-95
```

Add AP75 and AP50-S only when the evaluator emits them under an explicit protocol label. Do not infer missing metrics from plots, aggregate a different run, or relabel unsplit metrics.

---

## 6. Provenance and source reports

- [`docs/reports/augmentation_report.md`](augmentation_report.md)
- [`docs/reports/all_datasets_baseline_validation_test.md`](all_datasets_baseline_validation_test.md)
- [`docs/reports/baseline_validation_test_recovered.md`](baseline_validation_test_recovered.md)
- [`docs/reports/report_yolo.md`](report_yolo.md)
- [`docs/reports/huggingface_results_20260912.md`](huggingface_results_20260912.md)

The source augmentation report records the final 38/38 uploaded augmentation matrix separately from this 12-variant planning matrix. The two counts are intentionally not conflated: **38/38** is the completed uploaded matrix, while **13 missing runs** refers to the requested 12-variant cross-dataset coverage described in this report.
