# Complete Main Report: Augmentation Matrix

**Updated:** 2026-10-03  
**Scope:** YOLO augmentation matrix only: OACP, Mosaic, and Copy-Paste.  
**Metric policy:** preserve split-qualified metrics exactly as reported. Use `--` when a metric is absent from the source artifact. Do not relabel unsplit metrics as validation or test metrics.

## 1. Report status

This report consolidates the 12 requested augmentation variants from the completed YOLO augmentation matrix. All 38 augmentation jobs are complete and upload-verified. A row or metric marked as pending below means that this report has not yet mapped or evaluated that metric field for the corresponding completed run. It does **not** mean that the training job is missing.

The final matrix coverage is:

| Method | Variants | LEVIR metric gaps in this report | TinyPerson metric gaps in this report | Varroa metric gaps in this report | Total metric gaps |
|---|---:|---:|---:|---:|---:|
| OACP | 3 | 0 | 0 | 0 | **0** |
| Mosaic | 5 | 0 | 1 | 0 | **1** |
| Copy-Paste | 4 | 0 | 2 | 0 | **2** |
| **Total** | **12** | **0** | **3** | **0** | **3 rows** |

**Evaluation backlog:** all 38 jobs are already complete. The remaining 3 rows summarized here are report-level split-metric gaps, not missing training jobs. A read-only checkpoint backfill verified 20 selected prefixes and populated the Varroa rows plus the missing core fields for the selected LEVIR/TinyPerson rows.

---

## 2. OACP: 3 variants

The adaptive-OACP family now includes verified split-qualified backfill results for the selected LEVIR, Varroa, and TinyPerson checkpoints. Existing source metrics are preserved where already reported; only missing core fields are filled from the backfill artifact.

| Variant | Dataset | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Spacing-adaptive | LEVIR | 0.6494 | -- | 0.2189 | -- | 0.8079 | 0.0977 | 0.3065 | -- | Existing + backfill |
| Spacing-adaptive | TinyPerson | -- | -- | 0.2134 | -- | 0.5252 | -- | 0.1923 | -- | Existing |
| Spacing-adaptive | Varroa | 0.9199 | -- | 0.3372 | -- | 0.9092 | -- | 0.3424 | -- | Backfill verified |
| Load-adaptive | LEVIR | 0.6669 | -- | 0.2253 | -- | **0.8204** | **0.1240** | **0.3109** | -- | Existing + backfill |
| Load-adaptive | TinyPerson | 0.5342 | -- | 0.2085 | -- | **0.5341** | -- | **0.1955** | -- | Existing + backfill |
| Load-adaptive | Varroa | 0.9342 | -- | 0.3314 | -- | 0.9063 | -- | 0.3354 | -- | Backfill verified |
| Mass-adaptive | LEVIR | 0.5685 | -- | 0.1796 | -- | 0.7731 | 0.1017 | 0.2881 | -- | Existing + backfill |
| Mass-adaptive | TinyPerson | -- | -- | 0.2113 | -- | 0.5310 | -- | 0.1940 | -- | Existing |
| Mass-adaptive | Varroa | 0.9187 | -- | 0.3343 | -- | 0.9096 | -- | 0.3305 | -- | Backfill verified |

**Protocol note:** this is the historical adaptive-OACP family. LEVIR uses P2/P3/P4 + Mosaic and TinyPerson uses adaptive OACP + Mosaic. Do not mix validation values from the separate R2 no-Mosaic audits into these rows.

---

## 3. Mosaic: 5 variants

LEVIR M0/M2-M5 have split-qualified validation/test AP50 and mAP50-95 in the current source report. The Varroa Mosaic rows below are now populated by the verified checkpoint backfill. TinyPerson M5 remains outside this backfill pass. AP75 and AP50-S are left as `--` where the family artifact does not expose them.

| Variant | Dataset | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Standard | LEVIR | 0.8044 | -- | 0.3118 | -- | 0.7653 | -- | 0.2876 | -- | Existing |
| Standard | TinyPerson | **0.5245** | -- | **0.1889** | -- | **0.4948** | -- | **0.1750** | -- | Existing |
| Standard | Varroa | 0.9262 | -- | 0.3283 | -- | 0.9018 | -- | 0.3276 | -- | Backfill verified |
| M2 Cluster-preserving | LEVIR | 0.7863 | -- | 0.2735 | -- | 0.7410 | -- | 0.2634 | -- | Existing |
| M2 Cluster-preserving | TinyPerson | 0.4689 | -- | 0.1660 | -- | 0.4458 | -- | 0.1575 | -- | Existing |
| M2 Cluster-preserving | Varroa | 0.8158 | -- | 0.2744 | -- | 0.7995 | -- | 0.2584 | -- | Backfill verified |
| M3 Post-scale-constrained | LEVIR | 0.7577 | -- | 0.2735 | -- | 0.7278 | -- | 0.2546 | -- | Existing |
| M3 Post-scale-constrained | TinyPerson | 0.5038 | -- | 0.1759 | -- | 0.4818 | -- | 0.1736 | -- | Existing |
| M3 Post-scale-constrained | Varroa | 0.9129 | -- | 0.3320 | -- | 0.9047 | -- | 0.3202 | -- | Backfill verified |
| M4 Adaptive-geometry | LEVIR | 0.7887 | -- | 0.2819 | -- | 0.7765 | -- | 0.2767 | -- | Existing |
| M4 Adaptive-geometry | TinyPerson | 0.5006 | -- | 0.1750 | -- | 0.4874 | -- | 0.1740 | -- | Existing |
| M4 Adaptive-geometry | Varroa | 0.8762 | -- | 0.2969 | -- | 0.8500 | -- | 0.2891 | -- | Backfill verified |
| M5 Hard-negative | LEVIR | **0.8180** | -- | **0.3079** | -- | **0.7883** | -- | **0.2921** | -- | Existing |
| M5 Hard-negative | TinyPerson | -- | -- | -- | -- | -- | -- | -- | -- | Existing run, split metrics not in this backfill |
| M5 Hard-negative | Varroa | 0.9335 | -- | 0.3326 | -- | 0.9098 | -- | 0.3349 | -- | Backfill verified |

TinyPerson Standard Mosaic values `0.5245/0.1889` validation and `0.4948/0.1750` test are from the matched canonical baseline artifact. The older controlled policy artifact's Test AP75 value is not inserted here because it belongs to a different subfamily.

---

## 4. Copy-Paste: 4 variants

LEVIR CP1/CP3 use corrected post-hoc split-qualified results. NegativeCanvas R1/R4 have explicit validation/test artifacts. TinyPerson CP1 and CP3 have results in the source report, but only as unsplit metrics, so they are not relabeled as validation or test values here.

| Variant | Dataset | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-S | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-S | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CP1 Single-object | LEVIR | 0.8262 | -- | 0.3264 | -- | 0.7974 | -- | 0.3109 | -- | Existing |
| CP1 Single-object | TinyPerson | -- | -- | -- | -- | -- | -- | -- | -- | Existing, **unsplit metrics only** |
| CP1 Single-object | Varroa | 0.8640 | -- | 0.3160 | -- | 0.8579 | -- | 0.3150 | -- | Backfill verified |
| CP3 Clustered | LEVIR | 0.8281 | -- | 0.3324 | -- | 0.7985 | -- | 0.3046 | -- | Existing |
| CP3 Clustered | TinyPerson | -- | -- | -- | -- | -- | -- | -- | -- | Existing, **unsplit metrics only** |
| CP3 Clustered | Varroa | 0.8571 | -- | 0.3127 | -- | 0.8639 | -- | 0.3201 | -- | Backfill verified |
| NegativeCanvas R1 | LEVIR | 0.8213 | -- | 0.3248 | -- | **0.8222** | -- | 0.3156 | -- | Existing |
| NegativeCanvas R1 | TinyPerson | **0.5287** | -- | **0.1892** | -- | **0.5037** | -- | **0.1817** | **0.6772** | Existing |
| NegativeCanvas R1 | Varroa | 0.8688 | -- | 0.3175 | -- | 0.8521 | -- | 0.3064 | -- | Backfill verified |
| NegativeCanvas R4 | LEVIR | **0.8305** | -- | **0.3352** | -- | 0.8200 | -- | **0.3158** | -- | Existing |
| NegativeCanvas R4 | TinyPerson | 0.4990 | -- | 0.1781 | -- | 0.4946 | -- | 0.1756 | **0.6646** | Existing |
| NegativeCanvas R4 | Varroa | 0.8816 | -- | 0.3166 | -- | 0.8453 | -- | 0.3053 | -- | Backfill verified |

TinyPerson NegativeCanvas `AP50-S` values are merged original-image TinyBenchmark metrics (`test_merged/AP50-Small`), not standard corner-window AP50 values.

TinyPerson unsplit source metrics retained for later evaluation:

| Variant | Dataset | Unsplit AP50 | Unsplit mAP50-95 | Split-qualified fields |
|---|---|---:|---:|---|
| CP1 Single-object | TinyPerson | 0.4636 | 0.1669 | -- |
| CP3 Clustered | TinyPerson | 0.4881 | 0.1751 | -- |

These values remain explicitly unsplit and must not be copied into `val/*` or `test/*` columns.

---

## 5. Evaluation backlog

### 5.1 Completed runs with metric mapping gaps

The following 3 report rows still need metric mapping or checkpoint evaluation in this selected view. The underlying augmentation jobs are already included in the completed 38-run matrix:

- Mosaic: M5 / TinyPerson.
- Copy-Paste: CP1 / TinyPerson; CP3 / TinyPerson.

### 5.2 Existing checkpoints with missing metrics

Rows containing `--` are not automatically metric gaps in this report. Before training, check whether the existing run has a checkpoint and manifest. If it does, schedule read-only evaluation to populate only the missing fields, preserving:

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

The source augmentation report records the final **38/38 completed and upload-verified** augmentation matrix. This 12-variant view is a reporting slice over that completed matrix. The **3 pending rows** above are metric extraction/evaluation gaps in this report, not missing training jobs.

## 7. Verified checkpoint backfill

The read-only evaluation completed on the supplied Marimo endpoint without retraining. The task-specific output repository is `duyle2408/augmentation-evaluation-backfill-runs`. It contains **20/20** `evaluation_backfill_complete.json` markers and **20/20** `evaluation_metrics.json` files for the selected prefixes. Every verified metrics artifact contains `val/AP50`, `val/mAP50-95`, `test/AP50`, and `test/mAP50-95`.

The evaluator used commit `87c9924069d774fdd2f1212773303070bf1d7551`, split seed `42`, NMS IoU `0.5`, image size `640`, and the following test protocols: LEVIR-Ship standard held-out test split, Varroa standard held-out test split, and TinyPerson standard corner-window test plus merged evaluator. The generic `utils.marimo_ops artifacts` training contract is not used as the completion gate for this evaluation because checkpoint evaluation intentionally produces metrics, manifests, and completion markers rather than training weights or `results.csv`.
