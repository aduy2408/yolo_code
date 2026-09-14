# Augmentation Report: OACP, Copy-Paste, and Mosaic Runs

**Updated:** 2026-09-14
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
- Prefer **clustered Copy-Paste (CP3)** over isolated single-object copies when object co-occurrence or clustering is part of the target dataset distribution.
- Treat OACP, Copy-Paste, Mosaic, detector YAML, and seed as separate factors. The current repository list contains many useful results, but not all of them are matched factorial comparisons.

### What cannot yet be claimed

- No global claim that OACP always improves performance across datasets.
- No claim that P2/P3/P4 is an augmentation improvement over P3/P4/P5, because it changes the detector architecture.
- No claim that Copy-Paste improves LEVIR from the current manifest-only CP ablation; CP3 is only marginally ahead of CP0 on mAP50-95.
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
