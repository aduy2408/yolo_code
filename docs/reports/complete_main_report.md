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

## 9. Seed 43/44 augmentation sweep completion

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
