# Augmentation Report: OACP, Copy-Paste, and Mosaic Runs

**Updated:** 2026-09-16
**Scope:** the Hugging Face repositories and run names supplied in the request, filtered to experiments involving **OACP**, **Copy-Paste**, **Mosaic**, or a direct no-augmentation/no-Mosaic control.
**Primary metric:** mAP50-95(B). Values are fractions, not percentages.

## 1. Executive summary

The filtered results do not represent one clean ablation table. They contain several experiment families with different detectors, datasets, split/training seeds, schedules, and metric artifacts. The most reliable conclusions are therefore **within matched families**, not from a single global ranking.

**Live-run status:** the currently running matched adaptive Mosaic + current-OACP experiment is not treated as completed evidence here. Its validation/test metrics and remote upload acceptance are not yet available. The numeric adaptive-OACP rows below come from previously uploaded artifacts identified in the compiled results reports; they are not a status claim about the still-running job.

Main observations:

1. **OACP is most consistently useful when paired with a small-object detector and a data-aware policy.** On LEVIR-Ship, YOLOv8n P2/P3/P4 with load-adaptive OACP + Mosaic reaches test mAP50-95 **0.3109**, ahead of spacing-adaptive OACP + Mosaic (**0.3065**) and the no-Mosaic density/budget variants (**0.3054**).
2. **The detector configuration is a major confounder.** P2/P3/P4 runs should not be compared directly with the canonical P3/P4/P5 Mosaic policy matrix. The strongest TinyPerson combination, P2/P3/P4 + mass-adaptive OACP + clustered Copy-Paste + Mosaic, reports **0.2061**, but that gain includes both architecture and augmentation changes.
3. **Visibility-aware Mosaic is the best isolated Mosaic policy in the controlled canonical YOLOv8 matrix.** On LEVIR-Ship it reaches test mAP50-95 **0.2580**, versus **0.2453** for standard Mosaic, a relative improvement of about **5.2%**.
4. **Copy-Paste alone is not uniformly beneficial.** In the four-way no-Mosaic LEVIR ablation, CP3 has the highest reported mAP50-95 (**0.3234**) but is essentially tied with the no-Copy-Paste control (**0.3202**) within the limitations of the reported artifact. On TinyPerson, CP3 is more favorable (**0.1751** versus **0.1590** for CP0).
5. **The historical OACP results require provenance caution.** The older aggressive sweep used accidental double-OACP. The corrected single-pass sweep selected R4 by validation mAP50, while R2 had the highest test mAP50 and R7 the highest test AP75. These are screening results, not a final multi-seed claim.
6. **Mosaic is not automatically helpful.** Its effect depends on policy, object visibility, detector scale, and dataset. No-Mosaic OACP can be competitive on LEVIR, while adaptive OACP + Mosaic is clearly stronger for the reported TinyPerson YOLOv8n runs than the no-Mosaic baseline.

## 2. Method definitions

### 2.1 OACP

OACP (Object-Aware Context Perturbation) modifies **far context around eligible tiny objects** while protecting the object and a local expanded neighborhood. The implementation:

- identifies eligible small-object boxes;
- builds a protected mask around each object, normally using `protected_expand=3.0`;
- computes a distance-based soft far-context mask;
- downsamples and upsamples the image with a random resolution scale, normally `0.65-0.85`;
- blends the degraded image into the far-context mask with random strength, normally `0.20-0.40`;
- leaves the ground-truth object region and protected local context unchanged.

The base configuration uses `p=0.20`, but the experiment families include frequent/mild settings (`p=0.40`), budget and density allocation, and adaptive variants. The implementation supports these variants:

| Variant | Method | Main control signal |
|---|---|---|
| `current` | fixed-protection OACP | fixed protected region and random far-context perturbation |
| `budget` | budget-limited OACP | random perturbation budget over available background |
| `density` | density-targeted OACP | target density/coverage of the perturbation |
| `mass_adaptive` | mass-adaptive OACP | target perturbed image mass, default range about `0.25-0.40` |
| `load_adaptive` | load-adaptive OACP | selects far-context perturbation area from scene/object load, saturating at a configured object count; preserves historical protection and does not change OACP strength, resolution scale, or application probability |
| `spacing_adaptive` | spacing-adaptive OACP | expands protection/context according to nearest-object spacing |

A critical implementation detail is the historical **double-OACP** path. The old pipeline applied OACP once in a wrapper and again in the normal transform pipeline. The corrected pipeline applies it once unless legacy mode is explicitly enabled. Results from the two paths must not be pooled as if they were the same method.

### 2.2 Copy-Paste

The project Copy-Paste transform uses raw training-image object crops and inserts them into the current canvas. The baseline is deliberately narrow:

- hard paste, native scale (`scale=1.0`), no padding, no blending;
- one or two copies;
- random or collision-aware placement;
- optional empty-target augmentation;
- optional natural clusters selected from a source crop.

The reported CP variants are:

| Variant | Method |
|---|---|
| CP0 | no Copy-Paste control |
| CP1 | one single-object copy |
| CP2 | two single-object copies |
| CP3 | one natural clustered copy; cluster crop expansion is normally `3.0` and requires at least two objects |

The implementation also contains adaptive Copy-Paste policies, including `load_adaptive` and `layout_adaptive`, but the main reported CP0-CP3 ablations are fixed-policy comparisons. CP3 therefore means **clustered Copy-Paste**, not Mosaic status.

### 2.3 Mosaic policies

All Mosaic rows should be interpreted together with the manifest's `mosaic` and `close_mosaic` fields. In the requested runs, Mosaic is generally enabled with `mosaic=1.0` and disabled for the final 10 epochs with `close_mosaic=10`.

The controlled LEVIR policy matrix compares:

| Policy | Method |
|---|---|
| Standard | ordinary four-image Mosaic |
| Visibility | scores candidate source crops by retained ground-truth box visibility and prefers candidates that preserve visible object evidence |
| Occupancy | selects candidates to match object occupancy/effective object count |
| Context | selects candidates using context-compatibility/contrast criteria |

The project helpers compute source crops, box visibility fractions, and a soft effective object count. The scene-compatible wrapper computes per-image descriptors such as relative object size, object spacing, and occupancy, then gates candidates whose descriptors drift outside the positive-scene support. These policy changes affect Mosaic selection, not detector architecture.

## 3. Filtered run inventory

### 3.1 Mosaic-only and Mosaic-policy runs

- `duyle2408/levir-ship-mosaic-policy-matrix`: standard, visibility, occupancy, and context Mosaic; canonical YOLOv8 P3/P4/P5; no OACP.
- `duyle2408/yolov8n-mosaic-resize-policy-matrix-seed42`: LEVIR standard/visibility/occupancy policies and a TinyPerson standard-policy control.
- `duyle2408/tinyperson-yolo11-oacp-mosaic-matrix`: YOLO11n legacy/single-pass OACP versus Mosaic/no-Mosaic matrix.
- `duyle2408/tinyperson-yolov9-yolov10-legacy-oacp-mosaic`: YOLOv9t and YOLOv10n with legacy double-OACP and standard Mosaic.
- `duyle2408/tinyperson-yolov9t-p2p3p4-oacp-budget-density-20260911`: YOLOv9t P2/P3/P4 budget and density OACP with Mosaic.
- `duyle2408/levir-yolov9t-p2p3p4-oacp-density-20260911` and `...oacp-budget-20260911`: YOLOv9t P2/P3/P4 OACP with Mosaic.
- `duyle2408/levirship-yolov8n-oacp-spacing-adaptive-mosaic`, `...load-adaptive-mosaic`, and `...mass-adaptive-mosaic`: YOLOv8n P2/P3/P4 adaptive OACP + Mosaic.

### 3.2 Copy-Paste runs

- `duyle2408/tinyperson-copy-paste`: TinyPerson CP0/CP1/CP2/CP3, no Mosaic.
- `duyle2408/levir-ship-copy-paste`: LEVIR-Ship CP0/CP1/CP2/CP3, no Mosaic.
- `duyle2408/tinyperson-copy-paste-mosaic`: TinyPerson CP3 + Mosaic.
- `duyle2408/levir-ship-copy-paste-mosaic`: LEVIR-Ship CP3 + Mosaic.
- `duyle2408/tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp`: P2/P3/P4 CP3 + Mosaic, no OACP.
- `duyle2408/tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic`: P2/P3/P4 mass-adaptive OACP + CP3, no Mosaic.
- `duyle2408/tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic`: P2/P3/P4 mass-adaptive OACP + CP3 + Mosaic.
- `duyle2408/tinyperson-copy-paste-oacp-mass-cp3-mosaic`: YOLOv8n mass-adaptive OACP + CP3 + Mosaic without the P2/P3/P4 detector.

### 3.3 OACP-only, no-Mosaic, seed, and parameter runs

The supplied list also contains the following OACP families and controls:

- Adaptive OACP: `tinyperson-yolov8n-oacp-spacing-adaptive-022e5d9`, `...load-adaptive-022e5d9`, `...mass-adaptive-022e5d9`; `levir-yolov8n-p2p3p4-oacp-spacing-adaptive-022e5d9`, `...load-adaptive-022e5d9`, `...mass-adaptive-20260910`.
- Budget/density: `tinyperson-yolov8n-p2p3p4-oacp-budget-density-20260910`, `levir-yolov8n-p2p3p4-oacp-budget-density-20260910`, `levir-yolov8n-p2p3p4-oacp-budget-20260911`, `levir-yolov8n-p2p3p4-oacp-density-nomosaic-20260911`, `levir-yolov8n-p2p3p4-oacp-budget-nomosaic-20260911`, `levir-yolov9t-p2p3p4-oacp-density-nomosaic-20260911`, and `levir-yolov9t-p2p3p4-oacp-budget-nomosaic-20260911`.
- No-Mosaic OACP controls and seed sweeps: `tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-25dcadc`, `...servernew-25dcadc`, `levir-yolov8n-p2-oacp-no-mosaic-2d308a3` plus seeds 43/44, `levir-yolov8n-p2-oacp-no-mosaic-nondeterministic-seed44-1ac1835-server2`, `levir-yolov8n-p2-oacp-ftal-no-mosaic-seed42-62ff114`, and `levir-yolov9t-oacp-no-mosaic-seed42`.
- Fixed-split and parameter comparisons: `tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed42/43/44`, `tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-9dda1cb-seed42`, `levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed43/44`, `levir-yolov8n-p2-oacp-ftal-1dfe88c-seed42`, `levir-yolov8n-p2-oacp-b1c490b-seed43/44`, `levir-yolov8n-p2-oacp-combinations-c70fd43`, `levir-yolov8n-p2-oacp-ap50-narrow-expand-20260910`, and `levir-oacp-r2-r6-seed-verify-b53f5ee-seed43/44`.
- Aggressive/corrected OACP sweeps: `levir-oacp-aggressive-singlepass-3fc7b92`, `levir-oacp-strong-ecaec0b`, `levir-oacp-double-compare-9a77f69`, `levir-oacp-r2-r6-seed-verify-b53f5ee-seed43/44`, and historical `levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758`.
- Baseline controls used for interpretation: `tinyperson-yolo-baselines-no-mosaic`, `tinyperson-yolov8n-baselines`, `levir-yolov8n-p2-baseline-no-aug-no-mosaic-seed42-d1f8206`, and the no-augmentation DMM controls listed in the previous Hugging Face compilation.
- Mixed OACP/model controls: `levir-yolov8n-p2p3p4-samc-oacp-current-seed42`, `levir-yolov8n-p2p3p4-samc-seed42`, `tinyperson-yolov8n-p2p3p4-samc-seed42`, `tinyperson-yolov8n-p2p3p4-oacp-current-seed42`, `tinyperson-yolov9t-cp3-mosaic-no-oacp-rerun`, and `tinyperson-yolov9t-no-oacp-no-mosaic-seed42`. These are retained as related controls, but are not pooled with the clean OACP/Copy-Paste/Mosaic ablations because they also change the detector, attention/module stack, or model family.

The repository list also includes broader baseline/model runs such as `...samc...`, DETR, and MMDetection repositories. They are not augmentation results and are excluded from the tables below unless they serve as an explicitly named control.

## 3.4 Newly uploaded augmentation results: 2026-09-14 to 2026-09-16

This subsection adds the relevant repositories from the latest HF list. The date shown is the repository creation/upload date returned by the Hugging Face dataset API. The uploaded manifests do not contain a separate wall-clock training start/end timestamp, so this is not presented as an exact training date. All rows below use split seed 42 and training seed 42 unless the row explicitly lists multiple seeds. Metrics are fractions, and every metric is labeled by split.

### Copy-Paste post-hoc test evaluation

Repository: [`duyle2408/levir-copy-paste-posthoc-test-runs`](https://huggingface.co/datasets/duyle2408/levir-copy-paste-posthoc-test-runs), uploaded **2026-09-16**. These are post-hoc evaluations of checkpoints in [`duyle2408/stw-yolo-runs`](https://huggingface.co/datasets/duyle2408/stw-yolo-runs), whose Copy-Paste run repository was created/uploaded **2026-09-14**. They use the LEVIR post-hoc split, NMS IoU 0.50, and a fixed split seed of 42. Values are mean across the available training seeds, with sample standard deviation in parentheses.

| Method | Seeds | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|
| CP0, no Copy-Paste control | 42, 43, 44 | 0.8297 | 0.3278 | 0.8203 (0.0094) | 0.3147 (0.0048) |
| CP1, one single-object copy | 42, 43, 44 | 0.8262 | 0.3264 | 0.7974 (0.0144) | 0.3109 (0.0079) |
| CP2, two single-object copies | 42, 43, 44 | 0.8233 | 0.3322 | 0.8046 (0.0174) | 0.3167 (0.0082) |
| CP3, one clustered copy | 42, 43 | 0.8281 | 0.3324 | 0.7985 (0.0108) | 0.3046 (0.0011) |
| Crowd mild | 42, 43 | 0.8132 | 0.3270 | 0.8012 (0.0110) | 0.2993 (0.0060) |
| Crowd moderate | 42, 43 | 0.8262 | 0.3276 | 0.8124 (0.0039) | 0.3085 (0.0024) |
| Negative Copy-Paste, offline | 42, 43 | 0.8333 | 0.3402 | 0.8113 (0.0081) | 0.3101 (0.0028) |
| Scale-matched Copy-Paste | 42, 43 | 0.8225 | 0.3302 | 0.8103 (0.0047) | 0.3125 (0.0015) |

**Interpretation:** the corrected post-hoc test table changes the earlier Copy-Paste interpretation. CP2 has the highest mean test mAP50-95 (**0.3167**) among the standard CP0-CP3 variants, while CP0 remains a strong control (**0.3147**). CP3 is lower in this post-hoc protocol (**0.3046**) and should not be described as the best Copy-Paste method without specifying the older artifact. The post-hoc artifact is the source of truth for these test values.

### Mosaic-only policy variants

Repository: [`duyle2408/mosaic-only-yolo-runs`](https://huggingface.co/datasets/duyle2408/mosaic-only-yolo-runs), uploaded **2026-09-15** and updated **2026-09-16**. These runs use the canonical upstream-style YOLOv8 P3/P4/P5 YAML, 100 epochs, workers 8, Mosaic enabled, and no OACP or Copy-Paste. Each row is a single seed-42 run.

| Dataset | Method | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---|---:|---:|---:|---:|
| LEVIR-Ship | M2 cluster-preserving Mosaic | 0.7863 | 0.2735 | 0.7410 | 0.2634 |
| LEVIR-Ship | M3 post-scale-constrained Mosaic | 0.7577 | 0.2735 | 0.7278 | 0.2546 |
| LEVIR-Ship | M4 adaptive-geometry Mosaic | 0.7887 | 0.2819 | 0.7765 | 0.2767 |
| LEVIR-Ship | M5 hard-negative Mosaic | **0.8180** | **0.3079** | **0.7883** | **0.2921** |
| TinyPerson | M2 cluster-preserving Mosaic | 0.4689 | 0.1660 | 0.4458 | 0.1575 |
| TinyPerson | M3 post-scale-constrained Mosaic | **0.5038** | **0.1759** | 0.4818 | 0.1736 |
| TinyPerson | M4 adaptive-geometry Mosaic | 0.5006 | 0.1750 | **0.4874** | **0.1740** |

**Interpretation:** M5 hard-negative Mosaic is strongest on LEVIR-Ship among this new set, while TinyPerson favors M3/M4 over M2. These are new policies and should be compared with the earlier standard/visibility/occupancy/context matrix only with the detector, schedule, and artifact protocol held constant.

### Post-Mosaic OACP variants

Repository: [`duyle2408/levir-post-mosaic-oacp-a63b4bd`](https://huggingface.co/datasets/duyle2408/levir-post-mosaic-oacp-a63b4bd), uploaded **2026-09-15**. These are LEVIR-Ship YOLOv8n/P2-family OACP experiments with explicit validation and test evaluation artifacts. The method column follows the manifest's OACP placement and variant fields.

| Method | Mosaic | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Fixed OACP, effect-adaptive R2 | Off | 0.8268 | 0.3323 | 0.8050 | 0.3085 |
| Fixed OACP, effect-fixed R2 | Off | **0.8576** | **0.3396** | 0.8089 | **0.3104** |
| Load-adaptive OACP, P2, post-Mosaic | On | 0.8254 | 0.3133 | 0.7848 | 0.2871 |
| Load-adaptive OACP, P2, no Mosaic | Off | 0.8198 | 0.3205 | 0.8026 | 0.3096 |
| Fixed OACP, R2, p=0.40 pre-transform | Off | 0.8266 | 0.3340 | **0.8296** | **0.3219** |
| Sparse-probability load-adaptive OACP, pre-transform | Off | 0.8424 | 0.3415 | 0.8063 | 0.3126 |
| Load-adaptive OACP, P2/P3/P4, post-Mosaic | On | 0.8298 | 0.3094 | 0.7843 | 0.2879 |
| Spatial-load-adaptive OACP, P2/P3/P4, post-Mosaic | On | 0.8018 | 0.3116 | 0.7469 | 0.2778 |

**Interpretation:** within this repository, the fixed R2 p=0.40 pre-transform run has the strongest test result (**test AP50 0.8296, test mAP50-95 0.3219**). The post-Mosaic load-adaptive variants are weaker than the no-Mosaic controls in this artifact, so the earlier conclusion that load-adaptive OACP + Mosaic is strongest must remain restricted to its matched experiment family and must not be generalized to this post-Mosaic implementation.

### Repository with no usable result

[`duyle2408/tinyperson-copy-paste-runs`](https://huggingface.co/datasets/duyle2408/tinyperson-copy-paste-runs) was created on **2026-09-16**, but currently contains only `.gitattributes` and no manifest, checkpoint metric, or evaluation artifact. It is therefore excluded from the result tables rather than treated as a failed or zero-valued Copy-Paste run.

### 3.5 Deduplication and baseline-reference audit

The latest list contains related artifacts that should be grouped, but not blindly pooled. The following consolidation is now used throughout this report:

| Consolidated family | Repositories/artifacts | Action | Reason |
|---|---|---|---|
| LEVIR Copy-Paste post-hoc family | `stw-yolo-runs` + `levir-copy-paste-posthoc-test-runs` | **Merge as one family**; use the post-hoc artifact for test metrics and retain the STW repository as checkpoint/config provenance | The post-hoc JSON explicitly evaluates the STW checkpoints on the fixed post-hoc test split. The older manifest-reported CP table in Section 5.4 is retained only as historical evidence and must not be averaged with the post-hoc table. |
| Mosaic policy results | `mosaic-only-yolo-runs` versus `levir-ship-mosaic-policy-matrix` | **Keep separate subfamilies** | Both use canonical YOLOv8-style detection, but the policy sets and uploaded artifacts differ. The newer M2-M5 policies are not duplicate rows of standard/visibility/occupancy/context. |
| Post-Mosaic OACP | `levir-post-mosaic-oacp-a63b4bd` versus earlier adaptive-OACP repositories | **Keep separate subfamilies** | Placement (`post_mosaic`), detector/configuration, and artifact protocol differ. Similar method names do not establish a matched duplicate. |
| Empty TinyPerson Copy-Paste repository | `tinyperson-copy-paste-runs` | **Exclude** | It contains no result artifact, so it cannot be merged with `tinyperson-copy-paste` or used as a zero-valued result. |

This prevents double counting. In particular, the same CP checkpoint is not counted once from `stw-yolo-runs` and again from the post-hoc repository, and the older manifest-only CP0-CP3 values are not mixed with the corrected post-hoc test values.

#### Baseline coverage by experiment family

| Experiment family | Baseline currently available | Reference used in this report | Remaining baseline gap |
|---|---|---|---|
| LEVIR canonical Mosaic M2-M5 | **Partial** | Standard Mosaic M0 in the controlled policy matrix, Section 5.1 | No same-repository no-Mosaic/no-augmentation control for M2-M5. Use M0 only as a Mosaic-policy reference, not as a no-augmentation baseline. |
| TinyPerson Mosaic M2-M4 | **Partial** | TinyPerson standard-policy control in the resize-policy matrix and M2-M4 rows above | No same-detector, same-artifact no-Mosaic/no-augmentation control in `mosaic-only-yolo-runs`. |
| LEVIR Copy-Paste CP0-CP3 post-hoc | **Present** | CP0 is the matched no-Copy-Paste, no-Mosaic control in the same post-hoc protocol | No gap for the CP0-CP3 comparison. Crowd/negative/scale variants still use CP0 as the nearest control and should not be interpreted as a complete factorial design. |
| LEVIR post-Mosaic OACP | **Partial** | No-Mosaic OACP rows in the same repository and earlier OACP controls | No pure no-augmentation P2/P3/P4 baseline in the same repository. The no-Mosaic OACP rows are augmentation controls, not no-augmentation controls. |
| LEVIR adaptive OACP + Mosaic | **Partial** | Earlier standard/no-Mosaic OACP family in Section 5.2 | No single matched canonical P2/P3/P4 no-OACP baseline across every adaptive variant. Detector and policy effects remain confounded. |
| TinyPerson CP3/OACP/Mosaic combinations | **Partial** | CP3 + Mosaic without OACP is available for the P2/P3/P4 group | Missing a same P2/P3/P4, no-OACP/no-Copy-Paste/no-Mosaic baseline. The YOLOv8 baseline elsewhere is not an architecture-matched substitute. |

For publication-quality comparisons, the missing controls should be added as explicit rows rather than inferred from another repository. The minimum next baseline set is: canonical detector with no augmentation, canonical detector with standard Mosaic, P2/P3/P4 detector with no augmentation, and P2/P3/P4 detector with standard Mosaic, each using the same dataset split, training seed set, workers, schedule, and evaluation protocol as its augmentation family.

## 4. Per-run settings audit

This section records the augmentation switches explicitly, rather than inferring them from a checkpoint name. `On` means `mosaic=1.0`; `Off` means `mosaic=0.0`. Unless a row says otherwise, Mosaic-enabled runs use `close_mosaic=10`, while the matched Copy-Paste-only protocol uses `close_mosaic=0`.

### 4.1 Copy-Paste settings shared by CP0-CP3 runs

The LEVIR-Ship and TinyPerson CP0-CP3 repositories use the same controlled settings:

| Setting | Value |
|---|---|
| Mosaic | `0.0` / disabled |
| `close_mosaic` | `0` |
| MixUp / CutMix | `0.0` / disabled |
| Copy-Paste probability | `p=0.5` |
| Scale / padding | `1.0` / `0.0` |
| Blend | hard paste |
| Placement | random |
| Max overlap / trials | `0.0` / `30` |
| Empty target / same source | allowed / allowed |

| Repository | CP0 | CP1 | CP2 | CP3 |
|---|---|---|---|---|
| `tinyperson-copy-paste` | disabled | one single object | two single objects | one cluster, `expand=3.0`, minimum 2 objects |
| `levir-ship-copy-paste` | disabled | one single object | two single objects | one cluster, `expand=3.0`, minimum 2 objects |

For the CP3 + Mosaic repositories, the Copy-Paste method remains the same clustered method, but Mosaic is enabled separately:

| Repository | Detector family | Copy-Paste | OACP | Mosaic |
|---|---|---|---|---|
| `tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp` | YOLOv8n P2/P3/P4 | CP3 cluster | none | On, `close_mosaic=10` |
| `tinyperson-copy-paste-mosaic` | YOLOv8n | CP3 cluster | none | On, `close_mosaic=10` |
| `levir-ship-copy-paste-mosaic` | YOLOv8n | CP3 cluster | none | On, `close_mosaic=10` |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic` | YOLOv8n P2/P3/P4 | CP3 cluster | mass-adaptive | Off, `mosaic=0.0` |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic` | YOLOv8n P2/P3/P4 | CP3 cluster | mass-adaptive | On, `close_mosaic=10` |
| `tinyperson-copy-paste-oacp-mass-cp3-mosaic` | YOLOv8n | CP3 cluster | mass-adaptive | On, `close_mosaic=10` |

The combined OACP + Copy-Paste repositories should not be compared with the plain CP3 repository as a Copy-Paste-only ablation. They change at least one additional factor.

### 4.2 OACP and Mosaic settings by named run family

| Run/repository family | Detector / dataset | OACP path and setting | Mosaic setting | Other important settings / status |
|---|---|---|---|---|
| `levir-ship-mosaic-policy-matrix` | canonical YOLOv8 P3/P4/P5, LEVIR-Ship | none | On, `close_mosaic=10`; standard, visibility, occupancy, or context policy | split seed 42, train seed 42, 100 epochs, workers 8; only Mosaic policy changes |
| `yolov8n-mosaic-resize-policy-matrix-seed42` | YOLOv8n policy matrix, LEVIR + TinyPerson | none | On; standard/visibility/occupancy policies for LEVIR, standard control for TinyPerson | manifests record dataset and policy; do not merge datasets |
| `levirship-yolov8n-oacp-spacing-adaptive-mosaic` | YOLOv8n P2/P3/P4, LEVIR-Ship | single-pass spacing-adaptive OACP | On, `close_mosaic=10` | adaptive context expansion based on object spacing |
| `levirship-yolov8n-oacp-load-adaptive-mosaic` | YOLOv8n P2/P3/P4, LEVIR-Ship | single-pass load-adaptive OACP | On, `close_mosaic=10` | adaptive budget based on scene/object load |
| `levirship-yolov8n-oacp-mass-adaptive-mosaic` | YOLOv8n P2/P3/P4, LEVIR-Ship | single-pass mass-adaptive OACP | On, `close_mosaic=10` | target perturbed image mass |
| `tinyperson-yolov8n-oacp-spacing/load/mass-adaptive-022e5d9` | YOLOv8n, TinyPerson | corresponding single-pass adaptive variant | On, `close_mosaic=10` | variant is encoded in each repository name |
| `levir-yolov8n-p2p3p4-oacp-spacing/load-adaptive-022e5d9` | YOLOv8n P2/P3/P4, LEVIR-Ship | corresponding single-pass adaptive variant | On, `close_mosaic=10` | variant is encoded in each repository name |
| `levir-yolov8n-p2p3p4-oacp-density/budget-nomosaic-20260911` | YOLOv8n P2/P3/P4, LEVIR-Ship | density or budget OACP | Off, `mosaic=0.0` | no-Mosaic comparison family |
| `levir-yolov9t-p2p3p4-oacp-density/budget-20260911` | YOLOv9t P2/P3/P4, LEVIR-Ship | density or budget OACP | On, `close_mosaic=10` | model-family change versus YOLOv8n |
| `levir-yolov9t-p2p3p4-oacp-density/budget-nomosaic-20260911` | YOLOv9t P2/P3/P4, LEVIR-Ship | density or budget OACP | Off, `mosaic=0.0` | matched no-Mosaic counterpart |
| `tinyperson-yolov9t-p2p3p4-oacp-budget-density-20260911` | YOLOv9t P2/P3/P4, TinyPerson | budget and density variants | On, `close_mosaic=10` | two variants in one repository |
| `tinyperson-yolo11-oacp-mosaic-matrix` | YOLO11n, TinyPerson | legacy double OACP or new single-pass R2 | On or Off | split/train seed 42, 100 epochs, imgsz 640, batch 8, workers 8; exact variants below |
| `tinyperson-yolov9-yolov10-legacy-oacp-mosaic` | YOLOv9t and YOLOv10n, TinyPerson | legacy double OACP | On, `close_mosaic=10` | split/train seed 42, 100 epochs, imgsz 640, batch 8, workers 8 |
| `levir-oacp-aggressive-singlepass-3fc7b92` | YOLOv8n P2, LEVIR-Ship | corrected single-pass R1-R7 sweep | Off, matching the no-Mosaic sweep | split/train seed 42; R4 selected by validation mAP50 |
| `levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758` | YOLOv8n P2, LEVIR-Ship | historical legacy double OACP | Off | nondeterministic historical screen; not pooled with corrected single-pass runs |
| `levir-yolov8n-p2p3p4-samc-oacp-current-seed42` | SAMC + YOLOv8n P2/P3/P4 | current OACP | read from manifest; not a clean OACP-only comparison | attention/module stack also changes |
| `tinyperson-yolov8n-p2p3p4-oacp-current-seed42` | YOLOv8n P2/P3/P4, TinyPerson | current OACP | read from manifest | current run status/metric acceptance is separate from completed artifacts |

`read from manifest` is deliberate. A repository name can identify an OACP variant but cannot prove `mosaic`, `close_mosaic`, workers, or exact OACP parameters. Those rows remain provenance-qualified until their manifest is available.

The queued OACP workflow is sequential and uses a separate Hugging Face repository for each model/variant. Verified LEVIR density and TinyPerson budget artifacts are reused rather than retrained. Only missing or untrusted artifacts should enter the queue again.

**TinyPerson resolution clarification:** the prepared corner windows are named `sw640_sh512`, meaning source crops are **640×512 (width×height)**. The detector training/evaluation setting is `imgsz=640`, so Ultralytics receives a **640×640 square model input** after its resize/letterbox pipeline. Therefore, `640×512` describes the dataset window geometry, not the tensor resolution used by the detector. The same distinction applies to the merged corner-window test protocol.

### 4.2a Exact detector YAML map

This is the part that must not be inferred from the checkpoint name. In particular, `yolov8n.pt` only identifies the pretrained weights; it does **not** tell us whether the detector head is the canonical P3/P4/P5 graph, a P2/P3/P4 graph, or a P2-only graph.

| Label used in this report | Exact YAML / source | Detection levels and important structure | Runs that belong here |
|---|---|---|---|
| **Canonical YOLOv8 baseline** | `models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml` | Upstream-style YOLOv8. The `Detect` layer consumes **P3, P4, P5** (`[15, 18, 21]`). There is no P2 detection output. | `levir-ship-mosaic-policy-matrix`, `mosaic-only-yolo-runs`, `misc/train_mosaic_policy_matrix.py`, and other rows explicitly described as canonical YOLOv8 P3/P4/P5. |
| **YOLOv8n P2/P3/P4, plain neck, LEVIR** | `models_related/models_config/yolov8/levir/yolov8n_p2p3p4_levir_plain.yaml` (the context-augmentation runner source is `misc/train_context_aug_matrix.py`) | Custom small-object detector. The `Detect` layer consumes **P2, P3, P4** (`[18, 15, 12]`), using a plain `RepC2f` neck. | LEVIR adaptive OACP rows whose run name/configuration says `p2p3p4`, including the spacing/load-adaptive Mosaic and density/budget families. |
| **YOLOv8n P2/P3/P4, plain neck, TinyPerson** | `models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3p4_plain.yaml` | Same output scale pattern, **P2, P3, P4**, but with the TinyPerson-specific YAML and `nc: 1`. | TinyPerson rows explicitly labeled `YOLOv8n P2/P3/P4`, including `tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp` and the P2/P3/P4 OACP + CP3 combinations. |
| **YOLOv8n four-scale historical P2 family** | `models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml` | The `Detect` layer consumes **P2, P3, P4, P5** (`[19, 22, 25, 28]`). The filename contains `p2`, but this is not a P2-only head and not the same graph as the P2/P3/P4 plain YAML. | `levir-yolov8n-p2-*` OACP/no-Mosaic rows driven by `misc/train_context_aug_matrix.py` and the dedicated P2 OACP runners. |
| **YOLOv8n P2-only historical family** | `models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml` or an explicitly named P2 attention/loss variant. | **P2 only** at the detection head. For the plain YAML, `Detect` consumes `[18]`. These are not the same model as canonical P3/P4/P5, P2/P3/P4, or the four-scale P2/P3/P4/P5 family. | P2-only historical FTAL/architecture experiments that explicitly select the `p2_fpn_only` YAML. Do not assign every `levir-yolov8n-p2-*` repository to this row. |
| **YOLOv9t / YOLOv10n / YOLO11n families** | The corresponding version-specific YAML, not a YOLOv8 YAML. | Model family and output graph differ from YOLOv8. The `P2/P3/P4` label still means the detector has those output levels, but it does not make the model equivalent to YOLOv8n P2/P3/P4. | `tinyperson-yolo11-oacp-mosaic-matrix`, YOLOv9t/YOLOv10n rows, and the YOLOv9t P2/P3/P4 rows. |

**How to read the tables:**

1. Rows labeled **canonical YOLOv8 baseline** are the only rows using the upstream P3/P4/P5 detector unless a table explicitly says otherwise.
2. Rows labeled **P2/P3/P4** add a high-resolution P2 output and therefore change the detector architecture. Any augmentation gain in those rows is a combined detector-plus-augmentation result.
3. Rows labeled **P2-only** are a separate group from both P2/P3/P4 and the historical four-scale P2/P3/P4/P5 family. The substring `p2` in a repository name is not enough to identify this group.
4. Where only a Hugging Face result table was available and the manifest did not expose the YAML filename, the report keeps the architecture label but marks the exact path as provenance-qualified rather than guessing it from the repository name.

### 4.3 Exact YOLO11 matrix variants

The four YOLO11 variants are not interchangeable:

| Variant | OACP path | OACP parameters | Mosaic |
|---|---|---|---|
| `legacy_default_oacp_mosaic` | legacy double-pass | `p=.20`, strength `.20-.40`, scale `.65-.85`, expand `3.0` | On, `close_mosaic=10` |
| `legacy_default_oacp_no_mosaic` | legacy double-pass | same as above | Off, `mosaic=0.0` |
| `r2_frequent_mild_oacp_mosaic` | corrected single-pass | `p=.40`, strength `.10-.25`, scale `.80-.95`, expand `3.0` | On, `close_mosaic=10` |
| `r2_frequent_mild_oacp_no_mosaic` | corrected single-pass | same as above | Off, `mosaic=0.0` |

All four use YOLO11n, TinyPerson, train seed 42, split seed 42, 100 epochs, patience 0, image size 640, batch size 8, workers 8, and NMS IoU 0.5.

## 5. Results grouped by experiment family

### 5.1 LEVIR-Ship: controlled Mosaic policy matrix

All four rows use the canonical upstream-style YOLOv8 P3/P4/P5 YAML, fixed split seed 42, training seed 42, 100 epochs, and workers 8. Only the Mosaic policy changes.

| Policy | Test mAP50 | Test mAP50-95 | Test AP75 | Val mAP50-95 |
|---|---:|---:|---:|---:|
| Standard Mosaic | 0.7235 | 0.2453 | 0.0710 | 0.2859 |
| Visibility-aware Mosaic | **0.7351** | **0.2580** | 0.0685 | **0.2944** |
| Occupancy-matched Mosaic | 0.7324 | 0.2454 | 0.0653 | 0.2736 |
| Context-contrast Mosaic | 0.7235 | 0.2474 | 0.0675 | 0.2862 |

**Interpretation:** visibility-aware selection is the only policy with a clear primary-metric improvement in this controlled matrix. Occupancy and context policies do not improve test mAP50-95 over standard Mosaic.

### 5.2 LEVIR-Ship: adaptive OACP and detector-scale effects

| Configuration | Mosaic | Test mAP50 | Test mAP50-95 | Test AP75 |
|---|---|---:|---:|---:|
| YOLOv8n P2/P3/P4, spacing-adaptive OACP | On | 0.8079 | 0.3065 | 0.0977 |
| YOLOv8n P2/P3/P4, load-adaptive OACP | On | **0.8204** | **0.3109** | **0.1240** |
| YOLOv8n P2/P3/P4, mass-adaptive OACP | On | 0.7731 | 0.2881 | 0.1017 |
| YOLOv8n P2/P3/P4, density/budget OACP | Off | 0.7960 | 0.3054 | 0.1186 |
| YOLOv9t P2/P3/P4, density/budget OACP | On | 0.7156 | 0.2575 | 0.0936 |
| YOLOv9t P2/P3/P4, density/budget OACP | Off | 0.7265 | 0.2720 | 0.1159 |

**Interpretation:** load-adaptive OACP + Mosaic is the strongest reported LEVIR adaptive-OACP row. The YOLOv9t rows are lower, demonstrating that a newer/larger detector is not automatically better in this small-object setting. The P2/P3/P4 advantage is a detector effect plus augmentation, not an isolated OACP effect.

### 5.3 TinyPerson: adaptive OACP

| Configuration | Test mAP50 | Test mAP50-95 | Validation mAP50-95 |
|---|---:|---:|---:|
| YOLOv8 baseline, no Mosaic/no OACP | 0.4953 | 0.1752 | 0.1796 |
| YOLOv9 baseline, no Mosaic/no OACP | 0.5052 | 0.1793 | 0.1777 |
| YOLOv8n, spacing-adaptive OACP + Mosaic | 0.5252 | 0.1923 | 0.2134 |
| YOLOv8n, load-adaptive OACP + Mosaic | **0.5341** | **0.1955** | 0.2085 |
| YOLOv8n, mass-adaptive OACP + Mosaic | 0.5310 | 0.1940 | **0.2113** |
| YOLOv9t P2/P3/P4, budget/density + Mosaic | 0.4716-0.4744 | 0.1708-0.1720 | 0.1823-0.1831 |

**Interpretation:** adaptive OACP + Mosaic improves over the YOLOv8 no-Mosaic baseline by about **8.8% relative in test mAP50-95** for load-adaptive OACP. The YOLOv9t P2/P3/P4 rows are below the YOLOv8 baseline in this reported setup.

### 5.4 Copy-Paste without Mosaic

Metrics below are manifest-reported because these Copy-Paste repositories do not expose the same explicit evaluation artifact as the Mosaic matrix.

#### LEVIR-Ship

| Variant | Description | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| CP0 | no Copy-Paste | 0.8181 | 0.3202 | 0.8512 | 0.7625 |
| CP1 | one single-object copy | 0.8008 | 0.3174 | 0.8454 | 0.7337 |
| CP2 | two single-object copies | 0.8135 | 0.3180 | 0.8401 | **0.7670** |
| CP3 | one clustered copy | 0.8047 | **0.3234** | **0.8708** | 0.7322 |

CP3 has the best reported mAP50-95, but the margin over CP0 is small. CP1 is worse than the control on the main metrics, so simply increasing synthetic object count is not a reliable improvement.

#### TinyPerson

| Variant | Description | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| CP0 | no Copy-Paste | 0.4643 | 0.1590 | **0.6032** | 0.4491 |
| CP1 | one single-object copy | 0.4636 | 0.1669 | 0.5897 | 0.4663 |
| CP2 | two single-object copies | 0.4653 | 0.1666 | 0.5949 | 0.4521 |
| CP3 | one clustered copy | **0.4881** | **0.1751** | 0.5958 | **0.4815** |

CP3 is more clearly favorable on TinyPerson, consistent with clustered object layouts being more representative than independent random copies.

### 5.5 Copy-Paste + OACP + Mosaic combinations

| Repository/run | Detector | Configuration | Reported mAP50 | Reported mAP50-95 |
|---|---|---|---:|---:|
| `tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp` | YOLOv8n P2/P3/P4 | CP3 + Mosaic, no OACP | 0.5529 | 0.1996 |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic` | YOLOv8n P2/P3/P4 | mass OACP + CP3, no Mosaic | 0.5226 | 0.1863 |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic` | YOLOv8n P2/P3/P4 | mass OACP + CP3 + Mosaic | **0.5537** | **0.2061** |
| `tinyperson-copy-paste-oacp-mass-cp3-mosaic` | YOLOv8n | mass OACP + CP3 + Mosaic | 0.4990 | 0.1803 |
| `tinyperson-copy-paste-mosaic` | YOLOv8n | CP3 + Mosaic | 0.5056 | 0.1851 |
| `levir-ship-copy-paste-mosaic` | YOLOv8n | CP3 + Mosaic | 0.7689 | 0.2848 |

**Interpretation:** in the matched P2/P3/P4 TinyPerson group, adding Mosaic to mass OACP + CP3 improves reported mAP50-95 from **0.1863** to **0.2061**. However, the comparison with the plain YOLOv8n combinations is confounded by detector architecture, so it should not be described as a pure augmentation gain.

## 6. Seed and provenance notes

- Split seed and training seed are separate parameters. The main matched comparisons use fixed `split_seed=42`; seed sweeps vary training seed only.
- The corrected OACP single-pass sweep `duyle2408/levir-oacp-aggressive-singlepass-3fc7b92` reports:

| Run | Validation mAP50 | Test mAP50 | Test mAP50-95 | Test AP75 |
|---|---:|---:|---:|---:|
| R1 sparse mild | 0.7969 | 0.7807 | 0.2954 | 0.1134 |
| R2 frequent mild | 0.8299 | **0.8297** | 0.3164 | 0.1251 |
| R3 frequent current | 0.8368 | 0.8013 | 0.3145 | 0.1449 |
| R4 strong | **0.8457** | 0.7951 | 0.3145 | 0.1232 |
| R5 very strong stress | 0.8273 | 0.8090 | 0.3007 | 0.1061 |
| R6 narrow protection | 0.8248 | 0.8262 | 0.3152 | 0.1324 |
| R7 wide protection | 0.8385 | 0.8141 | 0.3168 | **0.1472** |

R4 was selected from R1-R5 by validation mAP50. The historical repository `levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758` is a legacy double-OACP screen and should not be pooled with this corrected sweep.

The fixed-split seed groups in the supplied list are useful for stability checks, but they are not interchangeable with the single-seed policy matrices. In particular, nondeterministic runs and runs with different workers or detector YAMLs should remain separate groups.

## 7. Conclusions and recommended next matrix

### What the current evidence supports

- For canonical YOLOv8 P3/P4/P5 Mosaic, prefer **visibility-aware Mosaic** over standard, occupancy, or context policy based on the controlled LEVIR matrix.
- For P2/P3/P4 small-object detectors, **load-adaptive OACP + Mosaic** is the strongest adaptive OACP configuration currently reported on LEVIR, while **mass-adaptive OACP + CP3 + Mosaic** is the strongest reported TinyPerson combination.
- Clustered Copy-Paste (CP3) is promising when object co-occurrence or clustering matches the target distribution, but the corrected LEVIR post-hoc table favors CP2 on mean test mAP50-95. CP3 should therefore be treated as dataset/protocol-dependent, not as a universal default.
- Treat OACP, Copy-Paste, Mosaic, detector YAML, and seed as separate factors. The current repository list contains many useful results, but not all of them are matched factorial comparisons.

### What cannot yet be claimed

- No global claim that OACP always improves performance across datasets.
- No claim that P2/P3/P4 is an augmentation improvement over P3/P4/P5, because it changes the detector architecture.
- No universal claim that Copy-Paste improves LEVIR. In the corrected post-hoc family, CP2 is highest among CP0-CP3, while the older manifest-only table gives a different ranking and is retained only as historical evidence.
- No pooling of legacy double-OACP and corrected single-pass OACP results.
- No uncertainty estimate from the single-seed Mosaic and Copy-Paste matrices.
- No validation/test result or upload-acceptance claim for the currently running adaptive Mosaic + current-OACP job.

### Recommended publication-quality follow-up

Hold the following fixed: detector YAML, image size, epochs, workers, `split_seed=42`, evaluation/NMS protocol, and Copy-Paste configuration. Then run at least training seeds 42, 43, and 44 for:

1. no augmentation;
2. standard Mosaic;
3. visibility-aware Mosaic;
4. OACP load-adaptive + Mosaic;
5. CP3 + Mosaic;
6. OACP mass-adaptive + CP3 + Mosaic.

Run this matrix separately for the canonical P3/P4/P5 detector and the explicitly selected P2/P3/P4 detector. Report mean ± sample standard deviation and retain the exact manifest, commit, split seed, training seed, worker count, and augmentation policy for every run.

## 8. Source artifacts

- Existing compiled results: [`docs/reports/huggingface_results_20260912.md`](huggingface_results_20260912.md)
- Earlier compiled results: [`docs/reports/huggingface_results_20260910.md`](huggingface_results_20260910.md)
- OACP method/provenance report: [`oacp.md`](../../oacp.md)
- OACP implementation: [`project_ultralytics/context_augment.py`](../../project_ultralytics/context_augment.py)
- Copy-Paste implementation: [`project_ultralytics/copy_paste.py`](../../project_ultralytics/copy_paste.py)
- Mosaic policy helpers: [`project_ultralytics/mosaic_policy.py`](../../project_ultralytics/mosaic_policy.py)
- Scene-compatible Mosaic helpers: [`project_ultralytics/scene_compatible_mosaic.py`](../../project_ultralytics/scene_compatible_mosaic.py)

Metric provenance follows the existing compiled report: explicit evaluation artifacts are labeled as test/validation metrics, while Copy-Paste rows whose repositories upload only manifest metrics are labeled **reported** rather than silently relabeled as test metrics.
