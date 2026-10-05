# Complete Main Report: Augmentation Matrix

**Updated:** 2026-10-05
**Scope:** YOLO augmentation matrix only: OACP, Mosaic, and Copy-Paste.  
**Metric policy:** preserve split-qualified metrics exactly as reported. Use `--` when a metric is absent from the source artifact. Do not relabel unsplit metrics as validation or test metrics.

## 1. Report status

This report consolidates the 12 requested augmentation variants from the completed YOLO augmentation matrix. All 38 augmentation jobs are complete and upload-verified. The all-run read-only evaluation now also produced split-qualified core metrics and size-bucket metrics for all 38 checkpoints. A row or metric marked as `--` below means that this report slice has not copied that field into its table, not that the underlying run, checkpoint, or all-run evaluation is missing.

The final matrix coverage is:

| Method | Variants | LEVIR metric gaps in this report | TinyPerson metric gaps in this report | Varroa metric gaps in this report | Total metric gaps |
|---|---:|---:|---:|---:|---:|
| OACP | 3 | 0 | 0 | 0 | **0** |
| Mosaic | 5 | 0 | 1 | 0 | **1** |
| Copy-Paste | 4 | 0 | 2 | 0 | **2** |
| **Total** | **12** | **0** | **3** | **0** | **3 rows** |

**Evaluation backlog:** there is no remaining training or checkpoint-evaluation backlog for the 38-run matrix. The three rows summarized with `--` below are report-slice mapping gaps. The complete all-run artifact set contains the required core and size-bucket metrics for every run, with TinyPerson merged metrics under the explicit `test_merged/*` protocol where applicable.

### 1.2 Consolidated AP75 and AP50-Small table

This is the top-level size-metric table for the OACP, Mosaic, and Copy-Paste
tables below. Values come from the completed size evaluator. For TinyPerson,
the final small-object column is the merged original-image metric; for LEVIR
and Varroa it is the native `test_size/AP50-Small` metric. `--` means that the
exact requested variant/protocol is not present in the 38-run size queue.

| Method | Variant | Dataset | Mosaic | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Val AP75 | Val AP50-Small | Test AP75 | Test AP50-Small |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OACP | Spacing-adaptive | LEVIR | Off | 0.7343 | 0.2611 | 0.6978 | 0.2512 | 0.0789 | 0.6900 | 0.0554 | 0.6126 |
| OACP | Spacing-adaptive | TinyPerson | On | 0.5168 | 0.1848 | 0.5009 | 0.1767 | 0.0907 | 0.5214 | 0.0788 | 0.6527 |
| OACP | Spacing-adaptive | Varroa | On | 0.9199 | 0.3372 | 0.9092 | 0.3424 | 0.1069 | 0.8891 | 0.1174 | 0.8632 |
| OACP | Load-adaptive | LEVIR | Off | 0.7337 | 0.2691 | 0.7087 | 0.2529 | 0.0658 | 0.6966 | 0.0676 | 0.6416 |
| OACP | Load-adaptive | TinyPerson | On | 0.5342 | 0.1885 | 0.4941 | 0.1738 | 0.0864 | 0.5690 | 0.0777 | 0.6462 |
| OACP | Load-adaptive | Varroa | On | 0.9342 | 0.3314 | 0.9063 | 0.3354 | 0.0927 | 0.8844 | 0.0977 | 0.8422 |
| OACP | Mass-adaptive | LEVIR | Off | 0.7168 | 0.2651 | 0.6621 | 0.2372 | 0.0459 | 0.5923 | 0.0306 | 0.5532 |
| OACP | Mass-adaptive | TinyPerson | On | 0.5062 | 0.1800 | 0.4956 | 0.1726 | 0.0929 | 0.5151 | 0.0794 | 0.6541 |
| OACP | Mass-adaptive | Varroa | On | 0.9187 | 0.3343 | 0.9096 | 0.3305 | 0.0981 | 0.8530 | 0.1012 | 0.8709 |
| Mosaic | Standard | LEVIR | On | -- | -- | -- | -- | -- | -- | -- | -- |
| Mosaic | Standard | TinyPerson | On | 0.5228 | 0.1893 | 0.4993 | 0.1743 | 0.0991 | 0.5077 | 0.0771 | 0.6584 |
| Mosaic | Standard | Varroa | On | 0.9262 | 0.3283 | 0.9018 | 0.3276 | 0.0789 | 0.8858 | 0.0870 | 0.8596 |
| Mosaic | M2 Cluster-preserving | LEVIR | On | 0.6812 | 0.2221 | 0.6307 | 0.2035 | 0.0542 | 0.6085 | 0.0557 | 0.6051 |
| Mosaic | M2 Cluster-preserving | TinyPerson | On | 0.4931 | 0.1686 | 0.4655 | 0.1621 | 0.0727 | 0.5307 | 0.0711 | 0.6087 |
| Mosaic | M2 Cluster-preserving | Varroa | On | 0.8158 | 0.2744 | 0.7995 | 0.2584 | 0.0956 | 0.8188 | 0.0628 | 0.8254 |
| Mosaic | M3 Post-scale-constrained | LEVIR | On | 0.6473 | 0.1966 | 0.5887 | 0.1790 | 0.0386 | 0.6951 | 0.0368 | 0.6579 |
| Mosaic | M3 Post-scale-constrained | TinyPerson | On | 0.4931 | 0.1738 | 0.4803 | 0.1677 | 0.0815 | 0.5215 | 0.0713 | 0.6399 |
| Mosaic | M3 Post-scale-constrained | Varroa | On | 0.9129 | 0.3320 | 0.9047 | 0.3202 | 0.0901 | 0.8538 | 0.0896 | 0.8482 |
| Mosaic | M4 Adaptive-geometry | LEVIR | On | 0.6198 | 0.1982 | 0.6043 | 0.1904 | 0.0402 | 0.6698 | 0.0497 | 0.6340 |
| Mosaic | M4 Adaptive-geometry | TinyPerson | On | 0.5065 | 0.1851 | 0.4876 | 0.1724 | 0.0880 | 0.5565 | 0.0816 | 0.6567 |
| Mosaic | M4 Adaptive-geometry | Varroa | On | 0.8762 | 0.2969 | 0.8500 | 0.2891 | 0.0849 | 0.9134 | 0.0807 | 0.8868 |
| Mosaic | M5 Hard-negative | LEVIR | On | 0.6912 | 0.2314 | 0.6706 | 0.2090 | 0.0759 | 0.7004 | 0.0641 | 0.6416 |
| Mosaic | M5 Hard-negative | TinyPerson | On | 0.4981 | 0.1772 | 0.4708 | 0.1643 | 0.0826 | 0.4916 | 0.0728 | 0.6568 |
| Mosaic | M5 Hard-negative | Varroa | On | 0.9335 | 0.3326 | 0.9098 | 0.3349 | 0.0889 | 0.8727 | 0.1029 | 0.8466 |
| Copy-Paste | CP1 Single-object | LEVIR | Off | 0.7466 | 0.2786 | 0.6936 | 0.2495 | 0.0695 | 0.4818 | 0.0654 | 0.4980 |
| Copy-Paste | CP1 Single-object | TinyPerson | Off | 0.4351 | 0.1500 | 0.4363 | 0.1514 | 0.0624 | 0.4598 | 0.0650 | 0.6005 |
| Copy-Paste | CP1 Single-object | Varroa | On | 0.8640 | 0.3160 | 0.8579 | 0.3150 | 0.1028 | 0.7869 | 0.1080 | 0.7612 |
| Copy-Paste | CP3 Clustered | LEVIR | Off | 0.7057 | 0.2622 | 0.6790 | 0.2451 | 0.0565 | 0.6150 | 0.0500 | 0.5499 |
| Copy-Paste | CP3 Clustered | TinyPerson | Off | 0.4459 | 0.1585 | 0.4478 | 0.1558 | 0.0736 | 0.5238 | 0.0635 | 0.6101 |
| Copy-Paste | CP3 Clustered | Varroa | On | 0.8571 | 0.3127 | 0.8639 | 0.3201 | 0.1121 | 0.7650 | 0.1019 | 0.7599 |
| Copy-Paste | NegativeCanvas R1 | LEVIR | Off | 0.6907 | 0.2506 | 0.6562 | 0.2316 | 0.0676 | 0.6892 | 0.0539 | 0.6126 |
| Copy-Paste | NegativeCanvas R1 | TinyPerson | Off | 0.4324 | 0.1513 | 0.4227 | 0.1442 | 0.0719 | 0.4083 | 0.0592 | 0.5863 |
| Copy-Paste | NegativeCanvas R1 | Varroa | On | 0.8688 | 0.3175 | 0.8521 | 0.3064 | 0.1021 | 0.7761 | 0.1040 | 0.7385 |
| Copy-Paste | NegativeCanvas R4 | LEVIR | Off | 0.7173 | 0.2599 | 0.6708 | 0.2313 | 0.0635 | 0.6195 | 0.0581 | 0.6267 |
| Copy-Paste | NegativeCanvas R4 | TinyPerson | Off | 0.4378 | 0.1537 | 0.4393 | 0.1497 | 0.0631 | 0.4255 | 0.0611 | 0.6160 |
| Copy-Paste | NegativeCanvas R4 | Varroa | On | 0.8816 | 0.3166 | 0.8453 | 0.3053 | 0.0874 | 0.7882 | 0.0832 | 0.7272 |

The LEVIR M2-M5 rows above are provenance-matched historical checkpoint evaluations, not values imputed from the 38-run size queue. Source checkpoints came from `duyle2408/mosaic-only-yolo-runs`; the read-only backfill used the native LEVIR split, split seed `42`, image size `640`, workers `8`, NMS IoU `0.50`, and uploaded verified artifacts to `duyle2408/augmentation-checkpoint-eval-20261004-runs` under `historical_mosaic/levir/<variant>/seed_42`. The LEVIR Standard row remains `--` because no matching Standard checkpoint was evaluated in this backfill.

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

### 5.1 Report-slice mapping gaps

The following 3 report rows still need table mapping in this selected view. The underlying augmentation jobs and their all-run evaluation artifacts are already complete:

- Mosaic: M5 / TinyPerson.
- Copy-Paste: CP1 / TinyPerson; CP3 / TinyPerson.

### 5.2 Existing checkpoints and unmapped table fields

Rows containing `--` are not automatically metric gaps in this report. The complete all-run evaluation has already populated the required core and bucket artifacts. If this table is expanded with an optional field, use the existing checkpoint and manifest and preserve:

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

The source augmentation report records the final **38/38 completed and upload-verified** augmentation matrix. This 12-variant view is a reporting slice over that completed matrix. The **3 rows** above are table-mapping gaps only, not missing training jobs or missing all-run evaluation artifacts.

## 7. Verified checkpoint backfill

The read-only evaluation completed on the supplied Marimo endpoint without retraining. The task-specific output repository is `duyle2408/augmentation-evaluation-backfill-runs`. It contains **20/20** `evaluation_backfill_complete.json` markers and **20/20** `evaluation_metrics.json` files for the selected prefixes. Every verified metrics artifact contains `val/AP50`, `val/mAP50-95`, `test/AP50`, and `test/mAP50-95`.

The evaluator used commit `87c9924069d774fdd2f1212773303070bf1d7551`, split seed `42`, NMS IoU `0.5`, image size `640`, and the following test protocols: LEVIR-Ship standard held-out test split, Varroa standard held-out test split, and TinyPerson standard corner-window test plus merged evaluator. The generic `utils.marimo_ops artifacts` training contract is not used as the completion gate for this evaluation because checkpoint evaluation intentionally produces metrics, manifests, and completion markers rather than training weights or `results.csv`.

## 8. Complete all-run size and bucket evaluation

The final read-only evaluator processed every augmentation checkpoint without retraining and uploaded the results to the task-specific repository `duyle2408/augmentation-size-metrics-runs`.

| Artifact or check | Verified count | Result |
|---|---:|---|
| Local `size_metrics_complete.json` markers | 38/38 | Complete |
| Local `size_metrics_manifest.json` files | 38/38 | Complete |
| Local `evaluation_metrics.json` files | 38/38 | Complete |
| Required split-qualified core fields | 38/38 | `val/AP50`, `val/mAP50-95`, `test/AP50`, `test/mAP50-95` present |
| `val_size/*` and `test_size/*` bucket fields | 38/38 | Complete |
| TinyPerson `test_merged/*` artifacts | 19 prefixes | Present where the merged protocol applies |
| HF completion markers | 38/38 | Upload verified |
| HF manifests | 38/38 | Upload verified |
| HF evaluation metrics | 38/38 | Upload verified |
| Required remote triplet paths | 114/114 | No missing paths |

The evaluator recorded finite numeric values for the required core and bucket
metrics, explicit split labels, and protocol metadata. TinyPerson standard
corner-window `test/*` metrics remain distinct from original-image merged
`test_merged/*` metrics. No validation metric was relabeled as test, and no
unsplit metric was promoted into a split-qualified field.

The upload-only phase exited with return code `0` through `python -m
utils.marimo_ops launch`. The generic `utils.marimo_ops artifacts` command still
defaults to training artifacts such as `weights/best.pt`, `weights/last.pt`, and
`results.csv`, so it is not the completion gate for this evaluator-only run.
The evaluator-specific marker, manifest, metric, protocol, and HF remote-path
checks are the authoritative completion evidence here.

### 8.1 AP75 and AP50-Small values from the uploaded size evaluator

The earlier 12-row tables were a compact reporting slice and therefore did not
show the newly backfilled size-evaluator fields. The values below are read from
the uploaded `evaluation_metrics.json` artifacts. They are explicitly labeled
as native size-bucket metrics. For TinyPerson, the `test_merged/*` columns use
the original-image merged protocol. `--` means that the merged protocol does
not apply to that dataset, not that the native `test_size/*` metric is absent.

| Dataset | Method | Variant | Mosaic | `val_size/AP75` | `val_size/AP50-Small` | `test_size/AP75` | `test_size/AP50-Small` | `test_merged/AP75` | `test_merged/AP50-Small` |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| LEVIR | Copy-Paste | CP1 | Off | 0.0695 | 0.4818 | 0.0654 | 0.4980 | -- | -- |
| LEVIR | Copy-Paste | CP3 | Off | 0.0565 | 0.6150 | 0.0500 | 0.5499 | -- | -- |
| LEVIR | Copy-Paste | NegativeCanvas R1 | Off | 0.0676 | 0.6892 | 0.0539 | 0.6126 | -- | -- |
| LEVIR | Copy-Paste | NegativeCanvas R4 | Off | 0.0635 | 0.6195 | 0.0581 | 0.6267 | -- | -- |
| TinyPerson | Copy-Paste | CP1 | On | 0.0879 | 0.5495 | 0.0815 | 0.5196 | 0.0882 | 0.6658 |
| TinyPerson | Copy-Paste | CP1 | Off | 0.0624 | 0.4598 | 0.0650 | 0.4543 | 0.0710 | 0.6005 |
| TinyPerson | Copy-Paste | CP3 | On | 0.0870 | 0.5339 | 0.0771 | 0.5127 | 0.0816 | 0.6571 |
| TinyPerson | Copy-Paste | CP3 | Off | 0.0736 | 0.5238 | 0.0635 | 0.4593 | 0.0668 | 0.6101 |
| TinyPerson | Copy-Paste | NegativeCanvas R1 | On | 0.0890 | 0.5071 | 0.0759 | 0.5074 | 0.0845 | 0.6542 |
| TinyPerson | Copy-Paste | NegativeCanvas R1 | Off | 0.0719 | 0.4083 | 0.0592 | 0.4422 | 0.0612 | 0.5863 |
| TinyPerson | Copy-Paste | NegativeCanvas R4 | On | 0.0923 | 0.5248 | 0.0814 | 0.5106 | 0.0899 | 0.6580 |
| TinyPerson | Copy-Paste | NegativeCanvas R4 | Off | 0.0631 | 0.4255 | 0.0611 | 0.4662 | 0.0669 | 0.6160 |
| Varroa | Copy-Paste | CP1 | On | 0.1028 | 0.7869 | 0.1080 | 0.7612 | -- | -- |
| Varroa | Copy-Paste | CP3 | On | 0.1121 | 0.7650 | 0.1019 | 0.7599 | -- | -- |
| Varroa | Copy-Paste | NegativeCanvas R1 | On | 0.1021 | 0.7761 | 0.1040 | 0.7385 | -- | -- |
| Varroa | Copy-Paste | NegativeCanvas R4 | On | 0.0874 | 0.7882 | 0.0832 | 0.7272 | -- | -- |
| TinyPerson | Mosaic | M2 Cluster-preserving | On | 0.0727 | 0.5307 | 0.0711 | 0.4582 | 0.0789 | 0.6087 |
| TinyPerson | Mosaic | M3 Post-scale-constrained | On | 0.0815 | 0.5215 | 0.0713 | 0.4818 | 0.0800 | 0.6399 |
| TinyPerson | Mosaic | M4 Adaptive-geometry | On | 0.0880 | 0.5565 | 0.0816 | 0.5173 | 0.0878 | 0.6567 |
| TinyPerson | Mosaic | M5 Hard-negative | On | 0.0826 | 0.4916 | 0.0728 | 0.4907 | 0.0822 | 0.6568 |
| TinyPerson | Mosaic | Standard | On | 0.0991 | 0.5077 | 0.0771 | 0.4999 | 0.0844 | 0.6584 |
| Varroa | Mosaic | M2 Cluster-preserving | On | 0.0956 | 0.8188 | 0.0628 | 0.8254 | -- | -- |
| Varroa | Mosaic | M3 Post-scale-constrained | On | 0.0901 | 0.8538 | 0.0896 | 0.8482 | -- | -- |
| Varroa | Mosaic | M4 Adaptive-geometry | On | 0.0849 | 0.9134 | 0.0807 | 0.8868 | -- | -- |
| Varroa | Mosaic | M5 Hard-negative | On | 0.0889 | 0.8727 | 0.1029 | 0.8466 | -- | -- |
| Varroa | Mosaic | Standard | On | 0.0789 | 0.8858 | 0.0870 | 0.8596 | -- | -- |
| LEVIR | OACP | Load-adaptive | Off | 0.0658 | 0.6966 | 0.0676 | 0.6416 | -- | -- |
| LEVIR | OACP | Mass-adaptive | Off | 0.0459 | 0.5923 | 0.0306 | 0.5532 | -- | -- |
| LEVIR | OACP | Spacing-adaptive | Off | 0.0789 | 0.6900 | 0.0554 | 0.6126 | -- | -- |
| TinyPerson | OACP | Load-adaptive | On | 0.0864 | 0.5690 | 0.0777 | 0.5046 | 0.0832 | 0.6462 |
| TinyPerson | OACP | Load-adaptive | Off | 0.0635 | 0.4689 | 0.0550 | 0.4393 | 0.0606 | 0.5732 |
| TinyPerson | OACP | Mass-adaptive | On | 0.0929 | 0.5151 | 0.0794 | 0.5047 | 0.0854 | 0.6541 |
| TinyPerson | OACP | Mass-adaptive | Off | 0.0696 | 0.4372 | 0.0570 | 0.4303 | 0.0612 | 0.5787 |
| TinyPerson | OACP | Spacing-adaptive | On | 0.0907 | 0.5214 | 0.0788 | 0.5084 | 0.0846 | 0.6527 |
| TinyPerson | OACP | Spacing-adaptive | Off | 0.0631 | 0.4379 | 0.0579 | 0.4515 | 0.0612 | 0.5845 |
| Varroa | OACP | Load-adaptive | On | 0.0927 | 0.8844 | 0.0977 | 0.8422 | -- | -- |
| Varroa | OACP | Mass-adaptive | On | 0.0981 | 0.8530 | 0.1012 | 0.8709 | -- | -- |
| Varroa | OACP | Spacing-adaptive | On | 0.1069 | 0.8891 | 0.1174 | 0.8632 | -- | -- |

These values are the backfill the report previously omitted. The underlying
artifacts already contained them, so no retraining was required and no metric
was inferred from another run.

**Important provenance clarification for LEVIR Copy-Paste:** the four LEVIR
Copy-Paste rows above are the canonical augmentation-matrix checkpoints for
training seeds 43 and 44. The historical Copy-Paste report contains seed-42
core metrics such as CP3 `mAP50-95=0.3234` and NegativeCanvas R1/R4 test
`mAP50-95=0.3156/0.3158`, but those historical LEVIR Copy-Paste checkpoints
were not included in this native size-bucket backfill. Therefore there is no
verified seed-42 LEVIR Copy-Paste `AP50-Small` value in this table. The
`0.7879` LEVIR `test_size/AP50-Small` value cited in the augmentation report is
the YOLOv8 no-Mosaic MuSGD baseline, not a Copy-Paste result. Both the
baseline evaluator and this augmentation evaluator use the same native
`size_bucket_evaluator` protocol; the low Copy-Paste values are not caused by
substituting TinyPerson's merged metric.

**Matrix/config audit (2026-10-05):** the current runner does not define a
72-job matrix. Its default job specification is 12 OACP + 10 Mosaic + 16
Copy-Paste = 38 jobs per training seed; the seed-43/44 sweep therefore has
76 prefixes. For LEVIR Copy-Paste, the runner explicitly uses the canonical
YOLOv8 P3/P4/P5 YAML, `mosaic=0`, `close_mosaic=0`, LEVIR image size 512,
100 epochs, patience 0, workers 8, optimizer `auto`, AMP enabled, NMS IoU
0.50, and the fixed split seed 42. The LEVIR split is regenerated by
`misc.prepare_levir_ship.prepare` into `levir_ship_augmentation_matrix_split_42`
with counts 2320/788/788. The historical CP table uses checkpoints from
`duyle2408/stw-yolo-runs` at commit `32f5899c...` and calls its evaluation
split “LEVIR post-hoc”; the current 43/44 matrix uses commit
`53686d0364de3c720672c2fc05a3e0b6f4a5f`. Thus the original seed-42 core table
and the current seed-43/44 native AP50-Small table are not a like-for-like
re-evaluation, even though both describe canonical P3/P4/P5 and split seed 42.


The follow-up multi-seed augmentation sweep is complete and upload-verified.
The sweep used the pinned commit
`53686d0364de3c720426672c2fc05a3e0b6f4a5f`, fixed split seed `42`, training
seeds `43` and `44`, `100` epochs, patience `0`, workers `8`, NMS IoU `0.50`,
and the three task-specific repositories listed in Section 1.1. Completion was
counted only when an evaluation artifact contained all four split-qualified
fields:

```text
val/AP50
val/mAP50-95
test/AP50
test/mAP50-95
```

The final acceptance check through the live Marimo kernels and the Hugging Face
listing interface reported **76/76 local-complete**, **76/76 HF-verified**, and
**0 pending**. The checked artifacts all contained the required metric fields
and the corresponding `upload_complete.json` marker.

### 9.1 Recovery-slot accounting

The recovery servers covered the missing queue ranges as follows. Counts below
are local split-qualified evaluation artifacts and upload markers in each
isolated run directory.

| Slot | Queue range | Final state | Local complete | HF markers | Completion note |
|---|---:|---|---:|---:|---|
| A | 20-22 | exited 0 | 3/3 | 3/3 | Completed normally |
| B | 51-55 | exited `-15` | 3/5 | 3/5 | Stopped after queue job 4 began; duplicate 54-55 owned by extra1 |
| C | 62-66 | exited `-15` | 3/5 | 3/5 | Stopped after queue job 4 began; duplicate 65-66 owned by extra2 |
| D | 76 | exited 0 | 1/1 | 1/1 | Completed normally |
| extra1 | 54-55 | exited 0 | 2/2 | 2/2 | Completed duplicate-owned shard |
| extra2 | 65-66 | exited 0 | 2/2 | 2/2 | Completed duplicate-owned shard |

The B and C queues were not stopped prematurely. Their logs were checked for
the duplicate boundary before the authorized stop, and their durable
`state.json` records were verified after termination. The protected legacy
endpoint for the original 23-28 slot remained HTTP 410 and was not relaunched.
It did not reduce the final unique sweep coverage because the replacement and
recovery shards supplied the missing queue indices.

### 9.2 Final acceptance evidence

- All tracked recovery run directories have terminal state records.
- Every inspected `evaluation_metrics.json` contains the four required
  validation/test fields.
- Every completed recovery artifact has `upload_complete.json`.
- The live Hugging Face tree contains `76` seed 43/44 completion markers across
  the three task-specific repositories.
- No pending queue item remains, and no additional monitor is required.

### 9.3 Complete size-metric acceptance for the seed 43/44 sweep

The seed 43/44 size-metric evaluator completed and remote acceptance was verified
against the task-specific Hugging Face **dataset** repository
`duyle2408/augmentation-seed43-44-size-metrics-runs`. The evaluator used commit
`53686d0364de3c720426672c2fc05a3e0b6f4a5f`, split seed `42`, image size `640`,
batch size `8`, and workers `8`. The corrected LEVIR root was
`/marimo/LevirShip/LevirShipData`.

The accepted coverage is:

| Metric/artifact family | Coverage |
|---|---:|
| `val/AP50`, `val/mAP50-95`, `test/AP50`, `test/mAP50-95` | **76/76** |
| `val_size/*` fields | **76/76** |
| `test_size/*` fields | **76/76** |
| `val_size/AP50-Small` | **76/76** |
| `test_size/AP50-Small` | **76/76** |
| `val_size/AP75` | **76/76** |
| `test_size/AP75` | **76/76** |
| `size_metrics_complete.json` | **76/76** |
| `size_metrics_manifest.json` | **76/76** |
| `test_merged/AP50-Small` | **38/38 TinyPerson prefixes** |
| `test_merged/AP75` | **38/38 TinyPerson prefixes** |

Remote acceptance checked `76` completion markers, `76` evaluation metric
paths, and `76` manifests. Every marker directory had all three required
siblings, and all remote completion markers reported `verified: true`. The
repository is a Hugging Face **dataset** repository because the evaluator's
upload implementation uses `repo_type="dataset"`.

The initial evaluator launch used `/marimo/LevirShip` and failed closed because
that path did not contain the expected `All Images` and `All Annotations`
directories. The corrected relaunch used `/marimo/LevirShip/LevirShipData` and
produced the accepted artifact set. The earlier 38-run size-evaluator sections
in this report remain separate from this full 76-prefix seed 43/44 acceptance.

### 9.4 Seed 43/44 per-prefix metrics

The following table contains the actual values from the 76 remote
`evaluation_metrics.json` artifacts. Values are rounded to four decimals. `--`
means the metric is not applicable to that dataset/protocol, notably
`test_merged/*` outside TinyPerson.

| Prefix | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | val/AP75 | val/AP50-Small | test/AP75 | test/AP50-Small | test_merged/AP50-Small | test_merged/AP75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `augmentation-copy-paste-runs/levir/copy_paste/cp1_single1/no_mosaic/seed_43` | 0.7357 | 0.2666 | 0.6972 | 0.2501 | 0.0612 | 0.6657 | 0.0565 | 0.6372 | -- | -- |
| `augmentation-copy-paste-runs/levir/copy_paste/cp1_single1/no_mosaic/seed_44` | 0.7152 | 0.2662 | 0.6519 | 0.2387 | 0.0790 | 0.5868 | 0.0682 | 0.5874 | -- | -- |
| `augmentation-copy-paste-runs/levir/copy_paste/cp3_cluster1/no_mosaic/seed_43` | 0.7174 | 0.2643 | 0.6905 | 0.2416 | 0.0298 | 0.4517 | 0.0268 | 0.4676 | -- | -- |
| `augmentation-copy-paste-runs/levir/copy_paste/cp3_cluster1/no_mosaic/seed_44` | 0.6863 | 0.2639 | 0.6562 | 0.2428 | 0.0777 | 0.5971 | 0.0590 | 0.5670 | -- | -- |
| `augmentation-copy-paste-runs/levir/copy_paste/negative_canvas_r1/no_mosaic/seed_43` | 0.7053 | 0.2646 | 0.7006 | 0.2494 | 0.0818 | 0.7164 | 0.0739 | 0.6573 | -- | -- |
| `augmentation-copy-paste-runs/levir/copy_paste/negative_canvas_r1/no_mosaic/seed_44` | 0.7406 | 0.2703 | 0.7162 | 0.2596 | 0.0515 | 0.6253 | 0.0372 | 0.6929 | -- | -- |
| `augmentation-copy-paste-runs/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_43` | 0.7149 | 0.2536 | 0.7049 | 0.2511 | 0.0956 | 0.6651 | 0.0673 | 0.6177 | -- | -- |
| `augmentation-copy-paste-runs/levir/copy_paste/negative_canvas_r4/no_mosaic/seed_44` | 0.7427 | 0.2635 | 0.6816 | 0.2419 | 0.0736 | 0.6813 | 0.0600 | 0.6769 | -- | -- |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp1_single1/mosaic/seed_43` | 0.5069 | 0.1823 | 0.4900 | 0.1730 | 0.0886 | 0.5083 | 0.0773 | 0.5075 | 0.6617 | 0.0857 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp1_single1/mosaic/seed_44` | 0.5061 | 0.1842 | 0.4980 | 0.1771 | 0.0912 | 0.5411 | 0.0791 | 0.5215 | 0.6669 | 0.0851 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp1_single1/no_mosaic/seed_43` | 0.4154 | 0.1443 | 0.4411 | 0.1529 | 0.0707 | 0.4172 | 0.0631 | 0.4674 | 0.6111 | 0.0715 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp1_single1/no_mosaic/seed_44` | 0.4249 | 0.1500 | 0.4176 | 0.1441 | 0.0722 | 0.3918 | 0.0641 | 0.4396 | 0.6036 | 0.0692 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp3_cluster1/mosaic/seed_43` | 0.5224 | 0.1866 | 0.4988 | 0.1760 | 0.0946 | 0.5328 | 0.0778 | 0.5021 | 0.6667 | 0.0852 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp3_cluster1/mosaic/seed_44` | 0.5400 | 0.1898 | 0.4881 | 0.1729 | 0.0932 | 0.5390 | 0.0779 | 0.5103 | 0.6555 | 0.0829 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp3_cluster1/no_mosaic/seed_43` | 0.4561 | 0.1594 | 0.4360 | 0.1516 | 0.0788 | 0.4750 | 0.0664 | 0.4650 | 0.6064 | 0.0719 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/cp3_cluster1/no_mosaic/seed_44` | 0.4490 | 0.1608 | 0.4427 | 0.1553 | 0.0812 | 0.4467 | 0.0629 | 0.4631 | 0.6107 | 0.0702 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r1/mosaic/seed_43` | 0.5223 | 0.1855 | 0.4896 | 0.1752 | 0.0900 | 0.5390 | 0.0792 | 0.4973 | 0.6557 | 0.0876 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r1/mosaic/seed_44` | 0.5203 | 0.1862 | 0.4948 | 0.1734 | 0.0844 | 0.5212 | 0.0752 | 0.5041 | 0.6536 | 0.0813 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r1/no_mosaic/seed_43` | 0.4334 | 0.1480 | 0.4219 | 0.1428 | 0.0681 | 0.4834 | 0.0592 | 0.4523 | 0.5763 | 0.0657 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r1/no_mosaic/seed_44` | 0.4282 | 0.1494 | 0.4153 | 0.1409 | 0.0700 | 0.3845 | 0.0542 | 0.4378 | 0.5976 | 0.0605 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_43` | 0.5139 | 0.1866 | 0.4975 | 0.1761 | 0.0945 | 0.5191 | 0.0764 | 0.5097 | 0.6602 | 0.0824 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r4/mosaic/seed_44` | 0.5026 | 0.1807 | 0.4910 | 0.1735 | 0.0840 | 0.5390 | 0.0801 | 0.4994 | 0.6621 | 0.0868 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r4/no_mosaic/seed_43` | 0.4383 | 0.1485 | 0.4267 | 0.1445 | 0.0698 | 0.4292 | 0.0595 | 0.4539 | 0.6034 | 0.0647 |
| `augmentation-copy-paste-runs/tinyperson/copy_paste/negative_canvas_r4/no_mosaic/seed_44` | 0.4272 | 0.1480 | 0.4310 | 0.1456 | 0.0689 | 0.4174 | 0.0591 | 0.4619 | 0.6033 | 0.0658 |
| `augmentation-copy-paste-runs/varroa/copy_paste/cp1_single1/mosaic/seed_43` | 0.8566 | 0.3171 | 0.8582 | 0.3011 | 0.1036 | 0.7750 | 0.0886 | 0.7719 | -- | -- |
| `augmentation-copy-paste-runs/varroa/copy_paste/cp1_single1/mosaic/seed_44` | 0.8594 | 0.3165 | 0.8589 | 0.3193 | 0.0938 | 0.7698 | 0.1124 | 0.7483 | -- | -- |
| `augmentation-copy-paste-runs/varroa/copy_paste/cp3_cluster1/mosaic/seed_43` | 0.8539 | 0.3131 | 0.8604 | 0.3117 | 0.1065 | 0.7819 | 0.1006 | 0.7663 | -- | -- |
| `augmentation-copy-paste-runs/varroa/copy_paste/cp3_cluster1/mosaic/seed_44` | 0.8748 | 0.3174 | 0.8582 | 0.3153 | 0.0927 | 0.7664 | 0.0909 | 0.7629 | -- | -- |
| `augmentation-copy-paste-runs/varroa/copy_paste/negative_canvas_r1/mosaic/seed_43` | 0.8654 | 0.3114 | 0.8688 | 0.3149 | 0.0855 | 0.7558 | 0.1005 | 0.7552 | -- | -- |
| `augmentation-copy-paste-runs/varroa/copy_paste/negative_canvas_r1/mosaic/seed_44` | 0.8670 | 0.3206 | 0.8453 | 0.3102 | 0.0998 | 0.7608 | 0.0966 | 0.7612 | -- | -- |
| `augmentation-copy-paste-runs/varroa/copy_paste/negative_canvas_r4/mosaic/seed_43` | 0.8580 | 0.3170 | 0.8559 | 0.3153 | 0.0985 | 0.7704 | 0.0865 | 0.7466 | -- | -- |
| `augmentation-copy-paste-runs/varroa/copy_paste/negative_canvas_r4/mosaic/seed_44` | 0.8847 | 0.3215 | 0.8574 | 0.3101 | 0.1006 | 0.7703 | 0.0897 | 0.7704 | -- | -- |
| `augmentation-mosaic-runs/tinyperson/mosaic/M2_cluster_preserving/mosaic/seed_43` | 0.4845 | 0.1725 | 0.4506 | 0.1575 | 0.0777 | 0.4807 | 0.0679 | 0.4281 | 0.5830 | 0.0782 |
| `augmentation-mosaic-runs/tinyperson/mosaic/M2_cluster_preserving/mosaic/seed_44` | 0.5005 | 0.1751 | 0.4552 | 0.1610 | 0.0759 | 0.5092 | 0.0725 | 0.4493 | 0.5938 | 0.0827 |
| `augmentation-mosaic-runs/tinyperson/mosaic/M3_post_scale_constrained/mosaic/seed_43` | 0.4901 | 0.1768 | 0.4820 | 0.1664 | 0.0839 | 0.4870 | 0.0695 | 0.4924 | 0.6353 | 0.0766 |
| `augmentation-mosaic-runs/tinyperson/mosaic/M3_post_scale_constrained/mosaic/seed_44` | 0.4959 | 0.1770 | 0.4718 | 0.1635 | 0.0846 | 0.5115 | 0.0655 | 0.4796 | 0.6305 | 0.0718 |
| `augmentation-mosaic-runs/tinyperson/mosaic/M4_adaptive_geometry/mosaic/seed_43` | 0.5005 | 0.1816 | 0.4823 | 0.1708 | 0.0874 | 0.5391 | 0.0791 | 0.5163 | 0.6408 | 0.0875 |
| `augmentation-mosaic-runs/tinyperson/mosaic/M4_adaptive_geometry/mosaic/seed_44` | 0.5196 | 0.1827 | 0.4842 | 0.1705 | 0.0782 | 0.5419 | 0.0757 | 0.5069 | 0.6288 | 0.0826 |
| `augmentation-mosaic-runs/tinyperson/mosaic/M5_hard_negative/mosaic/seed_43` | 0.5076 | 0.1817 | 0.4764 | 0.1675 | 0.0860 | 0.4898 | 0.0750 | 0.4916 | 0.6730 | 0.0835 |
| `augmentation-mosaic-runs/tinyperson/mosaic/M5_hard_negative/mosaic/seed_44` | 0.5038 | 0.1800 | 0.4891 | 0.1695 | 0.0845 | 0.4620 | 0.0704 | 0.4832 | 0.6668 | 0.0814 |
| `augmentation-mosaic-runs/tinyperson/mosaic/standard/mosaic/seed_43` | 0.5196 | 0.1859 | 0.4981 | 0.1753 | 0.0866 | 0.4966 | 0.0747 | 0.5079 | 0.6626 | 0.0833 |
| `augmentation-mosaic-runs/tinyperson/mosaic/standard/mosaic/seed_44` | 0.5071 | 0.1893 | 0.4967 | 0.1733 | 0.0890 | 0.4973 | 0.0725 | 0.4893 | 0.6517 | 0.0820 |
| `augmentation-mosaic-runs/varroa/mosaic/M2_cluster_preserving/mosaic/seed_43` | 0.8140 | 0.2593 | 0.7818 | 0.2598 | 0.0589 | 0.7920 | 0.0596 | 0.7698 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/M2_cluster_preserving/mosaic/seed_44` | 0.7724 | 0.2429 | 0.7545 | 0.2376 | 0.0680 | 0.7860 | 0.0672 | 0.7949 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/M3_post_scale_constrained/mosaic/seed_43` | 0.9177 | 0.3324 | 0.9122 | 0.3353 | 0.1154 | 0.8963 | 0.1093 | 0.8812 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/M3_post_scale_constrained/mosaic/seed_44` | 0.9236 | 0.3348 | 0.8864 | 0.3249 | 0.0999 | 0.8831 | 0.0966 | 0.8620 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/M4_adaptive_geometry/mosaic/seed_43` | 0.8421 | 0.2693 | 0.8048 | 0.2557 | 0.0573 | 0.8791 | 0.0523 | 0.8768 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/M4_adaptive_geometry/mosaic/seed_44` | 0.8293 | 0.2680 | 0.8076 | 0.2596 | 0.0582 | 0.7290 | 0.0626 | 0.7281 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/M5_hard_negative/mosaic/seed_43` | 0.9175 | 0.3366 | 0.9028 | 0.3325 | 0.0906 | 0.8267 | 0.1041 | 0.8074 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/M5_hard_negative/mosaic/seed_44` | 0.9274 | 0.3405 | 0.9172 | 0.3395 | 0.0913 | 0.8501 | 0.1044 | 0.8522 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/standard/mosaic/seed_43` | 0.9114 | 0.3328 | 0.9078 | 0.3252 | 0.0993 | 0.8328 | 0.0932 | 0.8350 | -- | -- |
| `augmentation-mosaic-runs/varroa/mosaic/standard/mosaic/seed_44` | 0.9276 | 0.3331 | 0.9121 | 0.3361 | 0.0879 | 0.8965 | 0.1089 | 0.8826 | -- | -- |
| `augmentation-oacp-runs/levir/oacp/load_adaptive/no_mosaic/seed_43` | 0.7311 | 0.2668 | 0.7145 | 0.2592 | 0.0640 | 0.6213 | 0.0436 | 0.5815 | -- | -- |
| `augmentation-oacp-runs/levir/oacp/load_adaptive/no_mosaic/seed_44` | 0.7173 | 0.2721 | 0.6935 | 0.2476 | 0.0528 | 0.5857 | 0.0521 | 0.5070 | -- | -- |
| `augmentation-oacp-runs/levir/oacp/mass_adaptive/no_mosaic/seed_43` | 0.7431 | 0.2707 | 0.6955 | 0.2484 | 0.0601 | 0.6766 | 0.0623 | 0.6186 | -- | -- |
| `augmentation-oacp-runs/levir/oacp/mass_adaptive/no_mosaic/seed_44` | 0.7040 | 0.2592 | 0.6879 | 0.2407 | 0.0509 | 0.5691 | 0.0275 | 0.5528 | -- | -- |
| `augmentation-oacp-runs/levir/oacp/spacing_adaptive/no_mosaic/seed_43` | 0.7627 | 0.2828 | 0.7174 | 0.2568 | 0.0676 | 0.6364 | 0.0559 | 0.5768 | -- | -- |
| `augmentation-oacp-runs/levir/oacp/spacing_adaptive/no_mosaic/seed_44` | 0.7077 | 0.2674 | 0.7004 | 0.2540 | 0.0525 | 0.5487 | 0.0628 | 0.5499 | -- | -- |
| `augmentation-oacp-runs/tinyperson/oacp/load_adaptive/mosaic/seed_43` | 0.5250 | 0.1853 | 0.4914 | 0.1712 | 0.0895 | 0.5374 | 0.0730 | 0.5149 | 0.6536 | 0.0789 |
| `augmentation-oacp-runs/tinyperson/oacp/load_adaptive/mosaic/seed_44` | 0.5172 | 0.1883 | 0.4902 | 0.1710 | 0.0867 | 0.5391 | 0.0784 | 0.5080 | 0.6594 | 0.0827 |
| `augmentation-oacp-runs/tinyperson/oacp/load_adaptive/no_mosaic/seed_43` | 0.4318 | 0.1472 | 0.4216 | 0.1413 | 0.0615 | 0.4061 | 0.0557 | 0.4460 | 0.6009 | 0.0592 |
| `augmentation-oacp-runs/tinyperson/oacp/load_adaptive/no_mosaic/seed_44` | 0.4485 | 0.1503 | 0.4238 | 0.1423 | 0.0731 | 0.4386 | 0.0596 | 0.4706 | 0.5939 | 0.0638 |
| `augmentation-oacp-runs/tinyperson/oacp/mass_adaptive/mosaic/seed_43` | 0.5185 | 0.1867 | 0.4913 | 0.1731 | 0.0865 | 0.5273 | 0.0769 | 0.5126 | 0.6586 | 0.0845 |
| `augmentation-oacp-runs/tinyperson/oacp/mass_adaptive/mosaic/seed_44` | 0.5135 | 0.1813 | 0.4970 | 0.1758 | 0.0812 | 0.4997 | 0.0813 | 0.5067 | 0.6569 | 0.0872 |
| `augmentation-oacp-runs/tinyperson/oacp/mass_adaptive/no_mosaic/seed_43` | 0.4432 | 0.1489 | 0.4316 | 0.1453 | 0.0660 | 0.4681 | 0.0594 | 0.4497 | 0.6011 | 0.0668 |
| `augmentation-oacp-runs/tinyperson/oacp/mass_adaptive/no_mosaic/seed_44` | 0.4347 | 0.1469 | 0.4360 | 0.1485 | 0.0621 | 0.4023 | 0.0613 | 0.4686 | 0.6094 | 0.0652 |
| `augmentation-oacp-runs/tinyperson/oacp/spacing_adaptive/mosaic/seed_43` | 0.5255 | 0.1878 | 0.4946 | 0.1738 | 0.0889 | 0.5026 | 0.0751 | 0.5071 | 0.6492 | 0.0841 |
| `augmentation-oacp-runs/tinyperson/oacp/spacing_adaptive/mosaic/seed_44` | 0.5079 | 0.1863 | 0.4985 | 0.1767 | 0.0974 | 0.5297 | 0.0811 | 0.5100 | 0.6586 | 0.0860 |
| `augmentation-oacp-runs/tinyperson/oacp/spacing_adaptive/no_mosaic/seed_43` | 0.4207 | 0.1444 | 0.4203 | 0.1396 | 0.0652 | 0.4162 | 0.0530 | 0.4223 | 0.5867 | 0.0582 |
| `augmentation-oacp-runs/tinyperson/oacp/spacing_adaptive/no_mosaic/seed_44` | 0.4319 | 0.1493 | 0.4227 | 0.1434 | 0.0683 | 0.4397 | 0.0590 | 0.4445 | 0.5935 | 0.0654 |
| `augmentation-oacp-runs/varroa/oacp/load_adaptive/mosaic/seed_43` | 0.9305 | 0.3350 | 0.8985 | 0.3277 | 0.0880 | 0.9024 | 0.0816 | 0.8749 | -- | -- |
| `augmentation-oacp-runs/varroa/oacp/load_adaptive/mosaic/seed_44` | 0.9307 | 0.3398 | 0.9124 | 0.3396 | 0.1083 | 0.8792 | 0.1125 | 0.8569 | -- | -- |
| `augmentation-oacp-runs/varroa/oacp/mass_adaptive/mosaic/seed_43` | 0.9197 | 0.3255 | 0.9107 | 0.3330 | 0.0977 | 0.8595 | 0.0937 | 0.8510 | -- | -- |
| `augmentation-oacp-runs/varroa/oacp/mass_adaptive/mosaic/seed_44` | 0.8961 | 0.3358 | 0.9037 | 0.3254 | 0.1101 | 0.8576 | 0.0872 | 0.8584 | -- | -- |
| `augmentation-oacp-runs/varroa/oacp/spacing_adaptive/mosaic/seed_43` | 0.9047 | 0.3293 | 0.9089 | 0.3306 | 0.0994 | 0.8362 | 0.0968 | 0.8057 | -- | -- |
| `augmentation-oacp-runs/varroa/oacp/spacing_adaptive/mosaic/seed_44` | 0.9199 | 0.3317 | 0.9058 | 0.3378 | 0.0960 | 0.8715 | 0.1181 | 0.8468 | -- | -- |