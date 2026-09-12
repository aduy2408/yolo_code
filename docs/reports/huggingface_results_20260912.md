# Hugging Face experiment results compilation

**Compiled:** 2026-09-12 22:51 ICT
**Scope:** the repositories listed in the accompanying request.
**Primary metric:** mAP50-95(B), with mAP50(B) shown for readability. Values are reported as fractions, not percentages.

## Important interpretation notes

1. The Mosaic policy matrix and the YOLOv8n resize-policy matrix contain both LEVIR-Ship and TinyPerson runs. They are separated below by the `dataset` field in each manifest.
2. The LEVIR mosaic-policy matrix is the cleanest controlled comparison: same upstream-style YOLOv8 YAML, split seed 42, training seed 42, 100 epochs, workers 8, and only the mosaic policy changed.
3. Several older adaptive-mosaic and YOLOv9t repositories upload `evaluation_metrics.json` with explicit test metrics. The copy-paste repositories generally upload only `experiment_manifest.json` plus `results.csv`; their reported manifest metrics are therefore labeled **reported**, not silently relabeled as test metrics.
4. `mAP75` is only shown where the artifact contains it. A blank value means the repository did not upload that field.
5. The final repository in the user's list, `duyle2408/levir-yolov8n-p2p3p4-oacp-mass-adaptive-20260910`, returned HTTP 401 from the public dataset API during collection and is marked unavailable rather than guessed.

## 1. LEVIR-Ship: mosaic policy matrix

| Policy | Method/config | Test mAP50 | Test mAP50-95 | Test AP75 | Val mAP50 | Val mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| M0 standard | Standard Mosaic, no OACP | 0.7235 | 0.2453 | 0.0710 | 0.7987 | 0.2859 |
| M1 visibility | Visibility-aware Mosaic | **0.7351** | **0.2580** | 0.0685 | 0.7944 | **0.2944** |
| M2 occupancy | Occupancy-matched Mosaic | 0.7324 | 0.2454 | 0.0653 | 0.7639 | 0.2736 |
| M3 context | Context-contrast Mosaic | 0.7235 | 0.2474 | 0.0675 | 0.7790 | 0.2862 |

**Finding:** M1 visibility is the best policy in this controlled matrix. It improves test mAP50-95 by about **5.2% relative** over M0, while M2 and M3 do not improve the primary test metric. All four use the canonical upstream-style P3/P4/P5 YOLOv8 YAML, seed 42, split seed 42, 100 epochs, and workers 8.

Repository: [levir-ship-mosaic-policy-matrix](https://huggingface.co/datasets/duyle2408/levir-ship-mosaic-policy-matrix)

## 2. LEVIR-Ship: adaptive Mosaic and P2/P3/P4 experiments

| Repository / variant | Model and method | Mosaic/OACP | Test mAP50 | Test mAP50-95 | Test AP75 | Val mAP50 | Val mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| `levirship-yolov8n-oacp-spacing-adaptive-022e5d9` | YOLOv8n P2/P3/P4, OACP spacing-adaptive | Mosaic | 0.8079 | 0.3065 | 0.0977 | 0.8374 | 0.3293 |
| `levirship-yolov8n-oacp-load-adaptive-022e5d9` | YOLOv8n P2/P3/P4, OACP load-adaptive | Mosaic | **0.8204** | **0.3109** | **0.1240** | 0.8365 | 0.3264 |
| `levirship-yolov8n-oacp-mass-adaptive-mosaic` | YOLOv8n P2/P3/P4, OACP mass-adaptive | Mosaic | 0.7731 | 0.2881 | 0.1017 | 0.8293 | 0.3166 |
| `levir-yolov8n-p2p3p4-oacp-spacing-adaptive-022e5d9` | YOLOv8n P2/P3/P4, OACP spacing-adaptive | Mosaic | 0.8079 | 0.3065 | 0.0977 | 0.8374 | 0.3293 |
| `levir-yolov8n-p2p3p4-oacp-load-adaptive-022e5d9` | YOLOv8n P2/P3/P4, OACP load-adaptive | Mosaic | **0.8204** | **0.3109** | **0.1240** | 0.8365 | 0.3264 |
| `levir-yolov8n-p2p3p4-oacp-density-nomosaic-20260911` | YOLOv8n P2/P3/P4, OACP density allocation | No Mosaic | 0.7960 | 0.3054 | 0.1186 | 0.8148 | 0.3114 |
| `levir-yolov8n-p2p3p4-oacp-budget-nomosaic-20260911` | YOLOv8n P2/P3/P4, OACP budget allocation | No Mosaic | 0.7960 | 0.3054 | 0.1186 | 0.8148 | 0.3114 |
| `levir-yolov9t-p2p3p4-oacp-budget-20260911` | YOLOv9t P2/P3/P4, OACP budget | Mosaic | 0.7156 | 0.2575 | 0.0936 | 0.7643 | 0.2870 |
| `levir-yolov9t-p2p3p4-oacp-density-20260911` | YOLOv9t P2/P3/P4, OACP density | Mosaic | 0.7156 | 0.2575 | 0.0936 | 0.7643 | 0.2870 |
| `levir-yolov9t-p2p3p4-oacp-budget-nomosaic-20260911` | YOLOv9t P2/P3/P4, OACP budget | No Mosaic | 0.7265 | 0.2720 | 0.1159 | 0.7647 | 0.2954 |
| `levir-yolov9t-p2p3p4-oacp-density-nomosaic-20260911` | YOLOv9t P2/P3/P4, OACP density | No Mosaic | 0.7265 | 0.2720 | 0.1159 | 0.7647 | 0.2954 |

**Finding:** among these LEVIR runs, **load-adaptive OACP Mosaic** is strongest at test mAP50 and mAP50-95. The no-Mosaic density/budget runs are competitive, but the tested Mosaic budget-density run is materially weaker. The duplicate-looking `levirship-*` and `levir-*` repositories contain matching metrics and should be treated as naming/upload duplicates unless their manifests are intentionally different.

## 3. LEVIR-Ship: Copy-Paste ablation

### Copy-Paste without Mosaic

| Variant | Description | Reported mAP50 | Reported mAP50-95 | Reported precision | Reported recall |
|---|---|---:|---:|---:|---:|
| CP0 | No Copy-Paste control | 0.8181 | 0.3202 | 0.8512 | 0.7625 |
| CP1 | One single-object copy | 0.8008 | 0.3174 | 0.8454 | 0.7337 |
| CP2 | Two single-object copies | **0.8135** | 0.3180 | 0.8401 | **0.7670** |
| CP3 | One clustered copy, cluster expansion 3.0 | 0.8047 | **0.3234** | **0.8708** | 0.7322 |

**Finding:** CP3 gives the highest reported mAP50-95, but CP0 is essentially tied and has higher mAP50. Copy-Paste is not a clear improvement in this four-way LEVIR ablation. These are manifest-reported metrics because the repositories do not provide the same explicit test-evaluation artifact as the Mosaic matrix.

### Copy-Paste plus Mosaic

| Repository | Method | Reported mAP50 | Reported mAP50-95 |
|---|---|---:|---:|
| `levir-ship-copy-paste-mosaic` | YOLOv8n, CP3 clustered Copy-Paste + Mosaic | 0.7689 | 0.2848 |

**Mosaic status is explicit for this row:** `mosaic=1.0`, `close_mosaic=10`. The non-mosaic Copy-Paste ablation above is the separate `levir-ship-copy-paste` repository, whose manifest records `mosaic=0.0`.

## 4. TinyPerson: baselines and adaptive OACP Mosaic

| Repository / variant | Model and method | Mosaic/OACP | Test mAP50 | Test mAP50-95 | Val mAP50 | Val mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| `tinyperson-yolo-baselines-no-mosaic` / YOLOv8 | YOLOv8 baseline, no Mosaic | No Mosaic, no OACP | 0.4953 | 0.1752 | 0.5067 | 0.1796 |
| `tinyperson-yolo-baselines-no-mosaic` / YOLOv9 | YOLOv9 baseline, no Mosaic | No Mosaic, no OACP | **0.5052** | **0.1793** | 0.5000 | 0.1777 |
| `tinyperson-yolov8n-oacp-spacing-adaptive-022e5d9` | YOLOv8n OACP, spacing-adaptive | Mosaic | 0.5252 | 0.1923 | 0.5773 | 0.2134 |
| `tinyperson-yolov8n-oacp-load-adaptive-022e5d9` | YOLOv8n OACP, load-adaptive | Mosaic | **0.5341** | **0.1955** | 0.5716 | 0.2085 |
| `tinyperson-yolov8n-oacp-mass-adaptive-022e5d9` | YOLOv8n OACP, mass-adaptive | Mosaic | 0.5310 | 0.1940 | **0.5805** | 0.2113 |
| `tinyperson-yolov9t-p2p3p4-oacp-budget-density-20260911` / density | YOLOv9t P2/P3/P4 OACP, density | Mosaic | 0.4716 | 0.1708 | 0.5115 | 0.1831 |
| `tinyperson-yolov9t-p2p3p4-oacp-budget-density-20260911` / budget | YOLOv9t P2/P3/P4 OACP, budget | Mosaic | 0.4744 | 0.1720 | **0.5181** | 0.1823 |

**Finding:** TinyPerson behaves differently from LEVIR-Ship. Adaptive OACP Mosaic improves over the YOLOv8 no-Mosaic baseline by roughly **8.8% relative in test mAP50-95** for the load-adaptive run. YOLOv9t P2/P3/P4 budget-density runs are below the YOLOv8 baseline in the reported test metrics, so the larger/newer model is not automatically better for this setup.

## 5. TinyPerson: Copy-Paste and P2/P3/P4 CP3

### Copy-Paste ablation without Mosaic

| Variant | Description | Reported mAP50 | Reported mAP50-95 | Reported precision | Reported recall |
|---|---|---:|---:|---:|---:|
| CP0 | No Copy-Paste control | 0.4643 | 0.1590 | **0.6032** | 0.4491 |
| CP1 | One single-object copy | 0.4636 | **0.1669** | 0.5897 | 0.4663 |
| CP2 | Two single-object copies | 0.4653 | 0.1666 | 0.5949 | 0.4521 |
| CP3 | One clustered copy | **0.4881** | **0.1751** | 0.5958 | **0.4815** |

### CP3 / OACP / Mosaic combinations

| Repository | Configuration | Reported mAP50 | Reported mAP50-95 |
|---|---|---:|---:|
| `tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp` | YOLOv8n P2/P3/P4, CP3 clustered, Mosaic, no OACP | 0.5529 | 0.1996 |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic` | YOLOv8n P2/P3/P4, OACP mass, CP3 clustered, no Mosaic | 0.5226 | 0.1863 |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic` | YOLOv8n P2/P3/P4, OACP mass, CP3 clustered, Mosaic | **0.5537** | **0.2061** |
| `tinyperson-copy-paste-oacp-mass-cp3-mosaic` | YOLOv8n, OACP mass, CP3 clustered, Mosaic | 0.4990 | 0.1803 |
| `tinyperson-copy-paste-mosaic` | YOLOv8n, CP3 clustered, Mosaic | 0.5056 | 0.1851 |

**Mosaic status is explicit for every row in this table:** the repositories ending in `-mosaic` use `mosaic=1.0` with `close_mosaic=10`; the repository ending in `-no-mosaic` uses `mosaic=0.0`; the older `tinyperson-copy-paste-oacp-mass-cp3-mosaic` manifest also records `mosaic=1.0`.

**Finding:** the strongest TinyPerson Copy-Paste result in the requested set is **P2/P3/P4 + OACP mass + CP3 + Mosaic**. The P2/P3/P4 detector is important here: the similarly named non-P2/P3/P4 CP3 Mosaic run is much lower. This is evidence that the detector resolution/configuration is at least as important as the augmentation choice.

## 6. Resize-policy matrix

| Dataset / policy | Method | Test mAP50 | Test mAP50-95 | Test AP75 | Val mAP50 | Val mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| LEVIR M0 | Standard Mosaic/resize | 0.7652 | 0.2841 | 0.1208 | 0.8064 | 0.3099 |
| LEVIR M1 | Visibility policy | **0.7905** | **0.2904** | 0.1147 | **0.8066** | 0.3031 |
| LEVIR M2 | Occupancy policy | 0.7841 | 0.2865 | 0.1018 | 0.7917 | 0.2935 |
| TinyPerson M0 | Standard policy | 0.4844 | 0.1719 | 0.0764 | 0.5077 | 0.1779 |

**Finding:** the same broad policy ranking appears on LEVIR: visibility is best on test mAP50 and mAP50-95. The TinyPerson M0 run is not directly comparable to the LEVIR rows because it is a different dataset and only the standard policy is present in this repository.

## 7. Exact repository coverage and augmentation-status audit

The requested list contains 28 repository names. The tables above consolidate duplicate-looking repositories only when their uploaded metrics match, but the exact names are retained here:

- `tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp`: CP3 clustered, `mosaic=1.0`, no OACP.
- `levir-ship-mosaic-policy-matrix`: M0 standard, M1 visibility, M2 occupancy, M3 context; all `mosaic=1.0`, no OACP.
- `yolov8n-mosaic-resize-policy-matrix-seed42`: LEVIR M0/M1/M2 plus TinyPerson M0. The manifests explicitly record the dataset and policy.
- `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic`: CP3 clustered + mass OACP, `mosaic=1.0`.
- `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic`: CP3 clustered + mass OACP, `mosaic=0.0`.
- `tinyperson-copy-paste-oacp-mass-cp3-mosaic`: CP3 clustered + mass OACP, `mosaic=1.0`.
- `tinyperson-copy-paste-mosaic`: CP3 clustered, `mosaic=1.0`.
- `levir-ship-copy-paste-mosaic`: CP3 clustered, `mosaic=1.0`.
- `tinyperson-copy-paste`: CP0/CP1/CP2/CP3 ablation, all `mosaic=0.0`.
- `levir-ship-copy-paste`: CP0/CP1/CP2/CP3 ablation, all `mosaic=0.0`.
- `tinyperson-yolo-baselines-no-mosaic`: YOLOv8 and YOLOv9 baselines, `mosaic=0.0`.
- `levir-yolov9t-p2p3p4-oacp-density-nomosaic-20260911`: YOLOv9t P2/P3/P4 density, `mosaic=0.0`.
- `tinyperson-yolov9t-p2p3p4-oacp-budget-density-20260911`: YOLOv9t P2/P3/P4 budget and density variants, `mosaic=1.0`.
- `levir-yolov9t-p2p3p4-oacp-budget-nomosaic-20260911`: YOLOv9t P2/P3/P4 budget, `mosaic=0.0`.
- `levir-yolov8n-p2p3p4-oacp-density-nomosaic-20260911`: YOLOv8n P2/P3/P4 density, `mosaic=0.0`.
- `levir-yolov8n-p2p3p4-oacp-budget-nomosaic-20260911`: YOLOv8n P2/P3/P4 budget, `mosaic=0.0`.
- `levirship-yolov8n-oacp-spacing-adaptive-mosaic`, `levirship-yolov8n-oacp-load-adaptive-mosaic`, `levirship-yolov8n-oacp-mass-adaptive-mosaic`: YOLOv8n P2/P3/P4 adaptive OACP Mosaic runs, all `mosaic=1.0`.
- `tinyperson-yolov8n-oacp-spacing-adaptive-022e5d9`, `tinyperson-yolov8n-oacp-load-adaptive-022e5d9`, `tinyperson-yolov8n-oacp-mass-adaptive-022e5d9`: TinyPerson YOLOv8n adaptive OACP Mosaic runs, all `mosaic=1.0`.
- `levir-yolov8n-p2p3p4-oacp-spacing-adaptive-022e5d9`, `levir-yolov8n-p2p3p4-oacp-load-adaptive-022e5d9`: LEVIR YOLOv8n P2/P3/P4 adaptive OACP Mosaic runs, both `mosaic=1.0`.
- `levir-yolov9t-p2p3p4-oacp-density-20260911`, `levir-yolov9t-p2p3p4-oacp-budget-20260911`: YOLOv9t P2/P3/P4 density/budget Mosaic runs, `mosaic=1.0`.
- `levir-yolov8n-p2p3p4-oacp-load-adaptive-022e5d9`: LEVIR YOLOv8n P2/P3/P4 load-adaptive Mosaic, `mosaic=1.0`.
- `tinyperson-yolov8n-oacp-mass-adaptive-022e5d9`: TinyPerson YOLOv8n mass-adaptive Mosaic, `mosaic=1.0`.
- `levir-yolov8n-p2p3p4-oacp-budget-density-20260910`: LEVIR YOLOv8n P2/P3/P4 budget-density Mosaic, `mosaic=1.0`.
- `levir-yolov8n-p2p3p4-oacp-mass-adaptive-20260910`: public API returned HTTP 401, so no metric or augmentation status is asserted.

This audit resolves the earlier ambiguity: **CP3 means the Copy-Paste variant, not Mosaic status. Mosaic status must be read from the manifest and is now stated explicitly for every CP3 row.**


## 8. Overall ranking by dataset

### LEVIR-Ship

1. **YOLOv8n P2/P3/P4 + OACP load-adaptive Mosaic:** test mAP50-95 **0.3109**.
2. **YOLOv8n P2/P3/P4 + OACP spacing-adaptive Mosaic:** 0.3065.
3. **YOLOv8n P2/P3/P4 + OACP density/budget no Mosaic:** 0.3054.
4. **Canonical P3/P4/P5 visibility-aware Mosaic:** 0.2580.
5. **Canonical P3/P4/P5 standard Mosaic:** 0.2453.

The P2/P3/P4 detector family is the dominant factor in the current LEVIR results. Within the canonical P3/P4/P5 Mosaic policy matrix, visibility-aware Mosaic is the best policy.

### TinyPerson

1. **YOLOv8n P2/P3/P4 + OACP mass + CP3 + Mosaic:** reported mAP50-95 **0.2061**.
2. **YOLOv8n P2/P3/P4 + CP3 + Mosaic, no OACP:** 0.1996.
3. **YOLOv8n OACP load-adaptive Mosaic:** test mAP50-95 0.1955.
4. **YOLOv8n OACP mass-adaptive Mosaic:** 0.1940.
5. **YOLOv8n baseline no Mosaic:** 0.1752.
6. **YOLOv9t P2/P3/P4 OACP budget/density Mosaic:** 0.1708 to 0.1720.

## Recommended next comparison

For a publication-quality conclusion, use the following matched matrix next:

- Keep `models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml` or the explicitly selected P2/P3/P4 YAML fixed.
- Keep split seed 42, workers 8, epochs, image size, NMS, and training protocol fixed.
- Compare only: no augmentation, standard Mosaic, visibility Mosaic, OACP load-adaptive Mosaic, CP3 Mosaic, and OACP mass + CP3 Mosaic.
- Run at least seeds 42, 43, and 44 for the final candidates. The single-seed rows above are useful for direction, not uncertainty estimation.
- Do not combine canonical P3/P4/P5 Mosaic policy results with P2/P3/P4 detector results as if they were one architecture ablation.

## Artifact links

- [TinyPerson YOLOv8n P2/P3/P4 CP3 Mosaic, no OACP](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp)
- [LEVIR-Ship Mosaic policy matrix](https://huggingface.co/datasets/duyle2408/levir-ship-mosaic-policy-matrix)
- [YOLOv8n Mosaic resize policy matrix](https://huggingface.co/datasets/duyle2408/yolov8n-mosaic-resize-policy-matrix-seed42)
- [TinyPerson P2/P3/P4 OACP mass CP3 Mosaic](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic)
- [TinyPerson P2/P3/P4 OACP mass CP3 no Mosaic](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic)
- [LEVIR-Ship Copy-Paste](https://huggingface.co/datasets/duyle2408/levir-ship-copy-paste)
- [TinyPerson Copy-Paste](https://huggingface.co/datasets/duyle2408/tinyperson-copy-paste)
- [TinyPerson YOLO baselines no Mosaic](https://huggingface.co/datasets/duyle2408/tinyperson-yolo-baselines-no-mosaic)
