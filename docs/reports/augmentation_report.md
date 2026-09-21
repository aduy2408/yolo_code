# Augmentation Report: OACP, Copy-Paste, and Mosaic Runs

**Updated:** 2026-09-21
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
7. **The new R2 no-Mosaic OACP audit does not overturn the earlier ranking.** In the matched YOLOv8n P2/P3/P4, split-seed-42 audit, load-adaptive and effect-adaptive OACP are close at test mAP50-95 (**0.3094** and **0.3102**), while curriculum and size-adaptive variants are lower (**0.2992** and **0.2977**). These are single-seed screening results, not a replacement for the earlier matched Mosaic family.
8. **The newly completed context-adaptive audit favors the C3 expand/protection policy in this single-seed screen.** C3 reaches test mAP50-95 **0.3145**, ahead of C1 strength (**0.3123**) and the existing R2 current audit (**0.3113**), while C2 scale is lower (**0.3051**). The detector, split seed, training seed, and no-Mosaic protocol are matched across these four context-audit rows.
9. **The completed negative-canvas ablation isolates four Copy-Paste hypotheses on the same canonical protocol.** R2 has the strongest validation result, while R1 has the strongest test mAP50-95 among R1-R4. The larger-donor policy in R3 reduces test performance, and weak post-resize blur in R4 recovers most of that loss without exceeding R1 on test mAP50-95.
10. **On TinyPerson, R1 is slightly above the available HF baseline on standard test metrics, while R4 is below it.** R1 improves test AP50 by **+0.0033** and test mAP50-95 by **+0.0030** relative to the seed-42 TinyPerson baseline. R4 changes those deltas to **-0.0058** and **-0.0031**. These comparisons are directional because the available baseline is the P2/P3/P4 TinyPerson model, while the negative-canvas runs use the canonical P3/P4/P5 YAML.

## 2. Consolidated result tables

The tables below are the compact handoff view requested for the report. They
keep controls, Copy-Paste, Mosaic, and OACP in separate families instead of
pooling unlike detectors or unlike augmentation protocols. Metrics are
fractions, and every row uses split-qualified names. Where an older source
artifact did not expose a required metric, the detailed section retains that
artifact-specific row rather than substituting another metric.

### 2.1 Control and optimizer audit

The first three adaptive controls use the same canonical YOLOv8 P3/P4/P5
detector, no Mosaic, seed 42, split seed 42, 100 epochs, patience 0, workers 8,
and final val/test evaluation with NMS IoU 0.50. Only the requested optimizer
changes between the MuSGD, SGD, and AdamW rows. The canonical no-augmentation
baseline is included as a separate upstream-baseline reference.

| Dataset / detector | Method | Optimizer | Mosaic | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---|---|---:|---:|---:|---:|
| LEVIR / canonical YOLOv8 P3/P4/P5 | No augmentation baseline reference | SGD | Off | 0.7524 | 0.2710 | 0.7152 | 0.2615 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | Adaptive CP control | MuSGD (`auto`) | Off | 0.8181 | 0.3202 | 0.8105 | 0.3013 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | Adaptive CP control | **SGD** | Off | 0.7738 | 0.2934 | 0.7614 | 0.2862 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | Adaptive CP control | **AdamW** | Off | 0.7730 | 0.2932 | 0.7298 | 0.2625 |

**Method note:** the adaptive CP control has `copy_paste_enabled=false`, so it
does not paste objects. The MuSGD row is the original `optimizer=auto` run;
the SGD and AdamW rows explicitly force their optimizer. The AdamW result is
verified in `duyle2408/levir-adaptive-copy-paste-adamw-runs`, and the SGD result
is verified in `duyle2408/levir-adaptive-copy-paste-sgd-runs`.

### 2.2 Copy-Paste

The first four rows are the conventional post-hoc CP0-CP3 family. The remaining
rows are the newer adaptive Copy-Paste matrix, all no-Mosaic LEVIR controls with
the same seed/split provenance and MuSGD optimizer.

| Method | Description | Optimizer / provenance | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| CP0 | No Copy-Paste control | Post-hoc CP family; optimizer not recorded | 0.8297 | 0.3278 | 0.8203 | 0.3147 |
| CP1 | One single-object hard paste | Post-hoc CP family; optimizer not recorded | 0.8262 | 0.3264 | 0.7974 | 0.3109 |
| CP2 | Two single-object hard pastes | Post-hoc CP family; optimizer not recorded | 0.8233 | 0.3322 | 0.8046 | 0.3167 |
| CP3 | One natural clustered hard paste | Post-hoc CP family; optimizer not recorded | 0.8281 | 0.3324 | 0.7985 | 0.3046 |
| Adaptive scale-conditioned | Budgeted positive paste conditioned on destination/source object scale; shrink-only factor max 1.0 | MuSGD | 0.8248 | 0.3076 | 0.7960 | 0.2856 |
| Adaptive scale-deficit | Paste budget driven by the positive object-count deficit, with scale conditioning | MuSGD | 0.8068 | 0.3242 | 0.7691 | 0.3035 |
| Adaptive cluster | Budgeted positive paste using natural object clusters | MuSGD | 0.8105 | 0.3205 | 0.7940 | 0.2972 |
| Online negative | Online hard-negative bank, BCE hardness, spatial candidate deduplication, collision-safe placement | MuSGD | 0.8116 | 0.3209 | 0.7986 | 0.2919 |
| Online negative, scale-matched | Online hard-negative bank plus source/destination scale matching | MuSGD | 0.8240 | 0.3258 | 0.7899 | 0.2999 |

For the adaptive matrix, positive variants use `copy_paste_p=1.0` as the
transform gate while their actual paste count is controlled by the adaptive
budget. Online negative variants use `negcp=0.30`; this is not the same
effective probability as the positive variants.

### 2.3 Negative-canvas Copy-Paste ablation

These four runs are a matched LEVIR-Ship study using the canonical YOLOv8
P3/P4/P5 detector, seed 42, split seed 42, 100 epochs, patience 0, workers 8,
MuSGD (`optimizer=auto`), Mosaic off, hard collision-aware placement, one paste
maximum, and target size at most 20 px. The probability is `negative_cp_p=0.30`
conditional on an **originally negative** training image. Each row has a clean
artifact contract and a verified upload in
`duyle2408/levir-negative-canvas-r1-r4-runs`.

| Run | Target policy | Donor policy | Rendering | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---|---|---:|---:|---:|---:|
| R1 | Empirical small-size distribution | Matched, 1.0 <= source/target < 1.5 | Resize only | 0.8213 | 0.3248 | **0.8222** | **0.3156** |
| R2 | Scale deficit, gamma 0.5, capped ratio 3 | Matched, 1.0 <= source/target < 1.5 | Resize only | **0.8440** | 0.3322 | 0.8207 | 0.3133 |
| R3 | Scale deficit, gamma 0.5, capped ratio 3 | Larger, 1.5 <= source/target <= 2.5 | Resize only | 0.8230 | 0.3291 | 0.7915 | 0.3003 |
| R4 | Scale deficit, gamma 0.5, capped ratio 3 | Larger, 1.5 <= source/target <= 2.5 | Resize + weak Gaussian blur, sigma 0.5 | 0.8305 | **0.3352** | 0.8200 | 0.3158 |

**Interpretation:** R1 tests whether negative images can serve as synthetic-positive
canvases. R2-R1 isolates deficit-aware target-scale sampling. R3-R2 isolates
larger-to-small donor geometry and is negative on the test split in this screen.
R4-R3 isolates fixed weak post-resize degradation and recovers the R3 test result
to approximately the R1 level, but does not establish a winner. These are
single-seed hypothesis checks, not a parameter sweep or a multi-seed claim.

Remote prefixes, all with verified `upload_complete.json`:

- `copy_paste/levir/negative_canvas_r1/seed_42`
- `copy_paste/levir/negative_canvas_r2/seed_42`
- `copy_paste/levir/negative_canvas_r3/seed_42`
- `copy_paste/levir/negative_canvas_r4/seed_42`

### 2.4 TinyPerson negative-canvas Copy-Paste

These results are sourced from the public Hugging Face dataset
`duyle2408/tinyperson-negative-canvas-r1-r4-runs`, not from the unavailable
Marimo server. The protocol uses the official TinyPerson `sw640/sh512`
corner-window dataset, source-image-grouped split seed 42, training seed 42,
100 epochs, patience 0, workers 8, NMS IoU 0.50, and the TinyBenchmark
merged-corner evaluator. Standard `test/*` metrics are corner-window detector
metrics. `test_merged/*` metrics are the merged original-image protocol and
must not be substituted for standard test metrics.

| Run | Target / donor / rendering | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | test_merged/AP50 | test_merged/mAP50-75 | test_merged/AP50-Small |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | Empirical + matched + resize only | 0.5287 | 0.1892 | **0.5037** | **0.1817** | **0.5276** | **0.3099** | **0.6772** |
| R4 | Deficit + larger + resize + weak blur | 0.4990 | 0.1781 | 0.4946 | 0.1756 | 0.5187 | 0.3032 | 0.6646 |
| TinyPerson baseline, seed 42 | Available HF `yolov8n_p2p3p4_plain` artifact | 0.5421 | 0.1925 | 0.5004 | 0.1787 | not reported | not reported | not reported |

The baseline row is from
`duyle2408/tinyperson-yolov8n-baselines/runs/yolov8n_p2p3p4_plain/seed_42`.
Its artifact reports standard validation/test metrics but
`test_merged/available=0`, so no merged AP50 or AP50-Small value is imputed.
The negative-canvas runs report the merged metrics from their verified HF
artifacts:

- R1: `test_merged/AP50-Small = 0.6771773148`.
- R4: `test_merged/AP50-Small = 0.6645901893`.

Relative to the available baseline, R1 is +0.0033 test AP50 and +0.0030 test
mAP50-95, while R4 is -0.0058 test AP50 and -0.0031 test mAP50-95. R1 also
exceeds R4 by 0.0126 merged AP50-Small. Do not treat these as a clean causal
architecture-matched ablation until a P3/P4/P5 TinyPerson no-negative-canvas
baseline is available.

HF sources:

- [R1 evaluation metrics](https://huggingface.co/datasets/duyle2408/tinyperson-negative-canvas-r1-r4-runs/blob/main/copy_paste/tinyperson/negative_canvas_r1/seed_42/evaluation_metrics.json)
- [R4 evaluation metrics](https://huggingface.co/datasets/duyle2408/tinyperson-negative-canvas-r1-r4-runs/blob/main/copy_paste/tinyperson/negative_canvas_r4/seed_42/evaluation_metrics.json)
- [TinyPerson baseline evaluation metrics](https://huggingface.co/datasets/duyle2408/tinyperson-yolov8n-baselines/blob/main/runs/yolov8n_p2p3p4_plain/seed_42/evaluation_metrics.json)

### 2.5 Mosaic

These rows are Mosaic-only or matched Mosaic-policy references with no OACP or
Copy-Paste in the stated configuration. Mosaic policy changes the source-tile
selection, not the detector architecture.

| Dataset / detector | Method | Mosaic protocol | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| LEVIR / canonical YOLOv8 P3/P4/P5 | No augmentation baseline | Off | 0.7524 | 0.2710 | 0.7152 | 0.2615 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | Standard Mosaic baseline | On, close_mosaic=10 | 0.8044 | 0.3118 | 0.7653 | 0.2876 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | M2 cluster-preserving Mosaic | On, close_mosaic=10 | 0.7863 | 0.2735 | 0.7410 | 0.2634 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | M3 post-scale-constrained Mosaic | On, close_mosaic=10 | 0.7577 | 0.2735 | 0.7278 | 0.2546 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | M4 adaptive-geometry Mosaic | On, close_mosaic=10 | 0.7887 | 0.2819 | 0.7765 | 0.2767 |
| LEVIR / canonical YOLOv8 P3/P4/P5 | M5 hard-negative Mosaic | On, close_mosaic=10 | 0.8180 | 0.3079 | 0.7883 | 0.2921 |

The older controlled standard/visibility/occupancy/context policy artifact has
test mAP50-95 values `0.2453`, `0.2580`, `0.2454`, and `0.2474`, respectively,
but does not expose all four split-qualified metrics in the same summary
artifact. Those rows remain in Section 6.1 without imputing missing AP50 values.

### 2.6 OACP

The OACP rows below are kept separate from Copy-Paste. They perturb far context
around eligible small objects while protecting the object and local context.

| Dataset / detector | Method | Mosaic | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| LEVIR / YOLOv8n P2/P3/P4 | Hardness-adaptive R2 | Off | 0.8332 | 0.3241 | 0.8016 | 0.3102 |
| LEVIR / YOLOv8n P2/P3/P4 | Curriculum R2 | Off | 0.8082 | 0.3250 | 0.7871 | 0.2992 |
| LEVIR / YOLOv8n P2/P3/P4 | Load-adaptive R2 | Off | 0.8337 | 0.3291 | 0.8189 | 0.3094 |
| LEVIR / YOLOv8n P2/P3/P4 | Size-adaptive R2 | Off | 0.8257 | 0.3288 | 0.7854 | 0.2977 |
| LEVIR / YOLOv8n P2/P3/P4 | Effect-adaptive R2 | Off | 0.8316 | 0.3298 | 0.8074 | 0.3102 |
| LEVIR / four-scale P2 family | Current OACP R2 audit baseline | Off | 0.8348 | 0.3262 | 0.8156 | 0.3113 |
| LEVIR / four-scale P2 family | C1 strength-adaptive | Off | 0.8280 | 0.3294 | 0.8034 | 0.3123 |
| LEVIR / four-scale P2 family | C2 scale-adaptive | Off | 0.8330 | 0.3251 | 0.7844 | 0.3051 |
| LEVIR / four-scale P2 family | C3 expand/protection-adaptive | Off | 0.8444 | 0.3366 | 0.8207 | 0.3145 |

The OACP detector rows are not pooled with the canonical YOLOv8 control or the
adaptive Copy-Paste matrix because the P2/P3/P4 and four-scale P2 families are
different detector architectures. Historical double-OACP rows also remain
separate from corrected single-pass OACP.

## 3. Method definitions

### 3.1 OACP

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

### 3.2 Copy-Paste

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

### 3.3 Negative-canvas Copy-Paste

`NegativeCanvasCopyPaste` is a separate transform and does not modify the older
`ScaleMatchedCopyPaste` or `AdaptiveCopyPaste` behavior. It first checks the
original dataset label to ensure the source image was negative, then applies
`negative_cp_p` conditional on that image. It samples one target size either
from the empirical small-object pool or from inverse-frequency scale bins,
selects a donor from one of two disjoint source/target size ranges, resizes with
`INTER_AREA`, optionally applies fixed weak blur, and performs one hard,
collision-safe paste. Donor feasibility and placement failures are reported in
the transform diagnostics rather than silently falling back to another policy.

| Run | Configured method |
|---|---|
| R1 | `negative_canvas`, empirical target, matched donor, no degradation |
| R2 | `negative_canvas`, deficit target, matched donor, no degradation |
| R3 | `negative_canvas`, deficit target, larger donor, no degradation |
| R4 | `negative_canvas`, deficit target, larger donor, `weak_blur` |

### 3.4 Mosaic policies

All Mosaic rows should be interpreted together with the manifest's `mosaic` and `close_mosaic` fields. In the requested runs, Mosaic is generally enabled with `mosaic=1.0` and disabled for the final 10 epochs with `close_mosaic=10`.

The controlled LEVIR policy matrix compares:

| Policy | Method |
|---|---|
| Standard | ordinary four-image Mosaic |
| Visibility | scores candidate source crops by retained ground-truth box visibility and prefers candidates that preserve visible object evidence |
| Occupancy | selects candidates to match object occupancy/effective object count |
| Context | selects candidates using context-compatibility/contrast criteria |

The project helpers compute source crops, box visibility fractions, and a soft effective object count. The scene-compatible wrapper computes per-image descriptors such as relative object size, object spacing, and occupancy, then gates candidates whose descriptors drift outside the positive-scene support. These policy changes affect Mosaic selection, not detector architecture.

## 4. Filtered run inventory

### 4.1 Mosaic-only and Mosaic-policy runs

- `duyle2408/levir-ship-mosaic-policy-matrix`: standard, visibility, occupancy, and context Mosaic; canonical YOLOv8 P3/P4/P5; no OACP.
- `duyle2408/yolov8n-mosaic-resize-policy-matrix-seed42`: LEVIR standard/visibility/occupancy policies and a TinyPerson standard-policy control.
- `duyle2408/tinyperson-yolo11-oacp-mosaic-matrix`: YOLO11n legacy/single-pass OACP versus Mosaic/no-Mosaic matrix.
- `duyle2408/tinyperson-yolov9-yolov10-legacy-oacp-mosaic`: YOLOv9t and YOLOv10n with legacy double-OACP and standard Mosaic.
- `duyle2408/tinyperson-yolov9t-p2p3p4-oacp-budget-density-20260911`: YOLOv9t P2/P3/P4 budget and density OACP with Mosaic.
- `duyle2408/levir-yolov9t-p2p3p4-oacp-density-20260911` and `...oacp-budget-20260911`: YOLOv9t P2/P3/P4 OACP with Mosaic.
- `duyle2408/levirship-yolov8n-oacp-spacing-adaptive-mosaic`, `...load-adaptive-mosaic`, and `...mass-adaptive-mosaic`: YOLOv8n P2/P3/P4 adaptive OACP + Mosaic.

### 4.2 Copy-Paste runs

- `duyle2408/tinyperson-copy-paste`: TinyPerson CP0/CP1/CP2/CP3, no Mosaic.
- `duyle2408/levir-ship-copy-paste`: LEVIR-Ship CP0/CP1/CP2/CP3, no Mosaic.
- `duyle2408/tinyperson-copy-paste-mosaic`: TinyPerson CP3 + Mosaic.
- `duyle2408/levir-ship-copy-paste-mosaic`: LEVIR-Ship CP3 + Mosaic.
- `duyle2408/tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp`: P2/P3/P4 CP3 + Mosaic, no OACP.
- `duyle2408/tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic`: P2/P3/P4 mass-adaptive OACP + CP3, no Mosaic.
- `duyle2408/tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic`: P2/P3/P4 mass-adaptive OACP + CP3 + Mosaic.
- `duyle2408/tinyperson-copy-paste-oacp-mass-cp3-mosaic`: YOLOv8n mass-adaptive OACP + CP3 + Mosaic without the P2/P3/P4 detector.

### 4.3 OACP-only, no-Mosaic, seed, and parameter runs

The supplied list also contains the following OACP families and controls:

- Adaptive OACP: `tinyperson-yolov8n-oacp-spacing-adaptive-022e5d9`, `...load-adaptive-022e5d9`, `...mass-adaptive-022e5d9`; `levir-yolov8n-p2p3p4-oacp-spacing-adaptive-022e5d9`, `...load-adaptive-022e5d9`, `...mass-adaptive-20260910`.
- Budget/density: `tinyperson-yolov8n-p2p3p4-oacp-budget-density-20260910`, `levir-yolov8n-p2p3p4-oacp-budget-density-20260910`, `levir-yolov8n-p2p3p4-oacp-budget-20260911`, `levir-yolov8n-p2p3p4-oacp-density-nomosaic-20260911`, `levir-yolov8n-p2p3p4-oacp-budget-nomosaic-20260911`, `levir-yolov9t-p2p3p4-oacp-density-nomosaic-20260911`, and `levir-yolov9t-p2p3p4-oacp-budget-nomosaic-20260911`.
- No-Mosaic OACP controls and seed sweeps: `tinyperson-yolov8n-p2p3p4-oacp-no-mosaic-seed-sweep-25dcadc`, `...servernew-25dcadc`, `levir-yolov8n-p2-oacp-no-mosaic-2d308a3` plus seeds 43/44, `levir-yolov8n-p2-oacp-no-mosaic-nondeterministic-seed44-1ac1835-server2`, `levir-yolov8n-p2-oacp-ftal-no-mosaic-seed42-62ff114`, and `levir-yolov9t-oacp-no-mosaic-seed42`.
- Fixed-split and parameter comparisons: `tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-0a2ca54-seed42/43/44`, `tinyperson-yolov8n-p2p3p4-oacp-fixedsplit42-9dda1cb-seed42`, `levir-yolov8n-p2-oacp-fixedsplit42-a6af599-seed43/44`, `levir-yolov8n-p2-oacp-ftal-1dfe88c-seed42`, `levir-yolov8n-p2-oacp-b1c490b-seed43/44`, `levir-yolov8n-p2-oacp-combinations-c70fd43`, `levir-yolov8n-p2-oacp-ap50-narrow-expand-20260910`, and `levir-oacp-r2-r6-seed-verify-b53f5ee-seed43/44`.
- Aggressive/corrected OACP sweeps: `levir-oacp-aggressive-singlepass-3fc7b92`, `levir-oacp-strong-ecaec0b`, `levir-oacp-double-compare-9a77f69`, `levir-oacp-r2-r6-seed-verify-b53f5ee-seed43/44`, and historical `levir-yolov8n-p2-oacp-aggressive-no-mosaic-nondeterministic-9fb0758`.
- Baseline controls used for interpretation: `tinyperson-yolo-baselines-no-mosaic`, `tinyperson-yolov8n-baselines`, `levir-yolov8n-p2-baseline-no-aug-no-mosaic-seed42-d1f8206`, and the no-augmentation DMM controls listed in the previous Hugging Face compilation.
- Mixed OACP/model controls: `levir-yolov8n-p2p3p4-samc-oacp-current-seed42`, `levir-yolov8n-p2p3p4-samc-seed42`, `tinyperson-yolov8n-p2p3p4-samc-seed42`, `tinyperson-yolov8n-p2p3p4-oacp-current-seed42`, `tinyperson-yolov9t-cp3-mosaic-no-oacp-rerun`, and `tinyperson-yolov9t-no-oacp-no-mosaic-seed42`. These are retained as related controls, but are not pooled with the clean OACP/Copy-Paste/Mosaic ablations because they also change the detector, attention/module stack, or model family.

- Latest R2 OACP audit repositories: `levir-oacp-context-r2-audit-runs`, `levir-oacp-r2-hardness-adaptive-seed42-runs`, `levir-oacp-r2-curriculum-seed42-runs`, `levir-oacp-r2-load-adaptive-seed42-runs`, `levir-oacp-r2-size-adaptive-seed42-runs`, and `levir-oacp-r2-effect-adaptive-seed42-runs`.
- Latest matched augmentation controls: `levir-augmentation-baselines-seed42-runs` and `tinyperson-augmentation-baselines-seed42-runs`.
- Latest TinyPerson Copy-Paste + Mosaic variants: `tinyperson-copy-paste-mosaic-runs`.

The repository list also includes broader baseline/model runs such as `...samc...`, DETR, and MMDetection repositories. They are not augmentation results and are excluded from the tables below unless they serve as an explicitly named control.

## 4.4 Newly uploaded augmentation results: 2026-09-14 to 2026-09-16

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

**HF detector provenance audit (2026-09-16):** the earlier uncertainty about this family is now resolved. The LEVIR CP0, CP1, CP2, and CP3 manifests in [`duyle2408/stw-yolo-runs`](https://huggingface.co/datasets/duyle2408/stw-yolo-runs/tree/main/copy_paste/levir) all record the same `yolov8n.pt` model path, commit `32f5899cdec987a0b07dbb0e5d558e8b971fed5f`, split seed 42, workers 8, and `mosaic=0.0`. The CP0 checkpoint was downloaded from HF and inspected with the project Ultralytics loader: its serialized `Detect` layer is `f=[15, 18, 21]`, `nl=3`, and strides `[8, 16, 32]`, which is the canonical YOLOv8 **P3/P4/P5** head, not a P2 detector. The CP1-CP3 manifests confirm the same model path and commit, so the corrected LEVIR CP0-CP3 post-hoc table is a matched canonical P3/P4/P5 family. The post-hoc JSON repository contains evaluation files only; model/config provenance comes from `stw-yolo-runs`.

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

### 4.5 Deduplication and baseline-reference audit

The latest list contains related artifacts that should be grouped, but not blindly pooled. The following consolidation is now used throughout this report:

| Consolidated family | Repositories/artifacts | Action | Reason |
|---|---|---|---|
| LEVIR Copy-Paste post-hoc family | `stw-yolo-runs` + `levir-copy-paste-posthoc-test-runs` | **Merge as one family**; use the post-hoc artifact for test metrics and retain the STW repository as checkpoint/config provenance | The post-hoc JSON explicitly evaluates the STW checkpoints on the fixed post-hoc test split. The older manifest-reported CP table in Section 6.4 is retained only as historical evidence and must not be averaged with the post-hoc table. |
| Mosaic policy results | `mosaic-only-yolo-runs` versus `levir-ship-mosaic-policy-matrix` | **Keep separate subfamilies** | Both use canonical YOLOv8-style detection, but the policy sets and uploaded artifacts differ. The newer M2-M5 policies are not duplicate rows of standard/visibility/occupancy/context. |
| Post-Mosaic OACP | `levir-post-mosaic-oacp-a63b4bd` versus earlier adaptive-OACP repositories | **Keep separate subfamilies** | Placement (`post_mosaic`), detector/configuration, and artifact protocol differ. Similar method names do not establish a matched duplicate. |
| Empty TinyPerson Copy-Paste repository | `tinyperson-copy-paste-runs` | **Exclude** | It contains no result artifact, so it cannot be merged with `tinyperson-copy-paste` or used as a zero-valued result. |

This prevents double counting. In particular, the same CP checkpoint is not counted once from `stw-yolo-runs` and again from the post-hoc repository, and the older manifest-only CP0-CP3 values are not mixed with the corrected post-hoc test values.

### 4.6 Newly uploaded R2 audit and matched baseline results

The following repositories were present in the supplied 2026-09-16 list but were not represented by result rows above. Metrics below come from each repository's uploaded `evaluation_metrics.json`; values are fractions. Every row reports validation and test metrics separately. Unless stated otherwise, the manifests record training seed 42, split seed 42, workers 8, 100 epochs, NMS IoU 0.50, and `mosaic=0.0`.

#### LEVIR-Ship matched augmentation baselines

Repository: [`duyle2408/levir-augmentation-baselines-seed42-runs`](https://huggingface.co/datasets/duyle2408/levir-augmentation-baselines-seed42-runs). The canonical rows use the upstream-style YOLOv8 P3/P4/P5 detector. The P2/P3/P4 rows use the explicitly different `yolov8n_p2p3p4_levir_plain.yaml` detector.

| Detector | Method | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---|---:|---:|---:|---:|
| Canonical YOLOv8 P3/P4/P5 | No augmentation, no Mosaic | 0.7524 | 0.2710 | 0.7152 | 0.2615 |
| Canonical YOLOv8 P3/P4/P5 | Standard Mosaic | 0.8044 | 0.3118 | 0.7653 | 0.2876 |
| YOLOv8n P2/P3/P4 | No augmentation, no Mosaic | 0.7286 | 0.2774 | 0.6959 | 0.2583 |
| YOLOv8n P2/P3/P4 | Standard Mosaic | **0.8282** | **0.3366** | **0.8078** | **0.3198** |

The canonical standard-Mosaic row is a new baseline reference for the M2-M5 policy family. It must not be pooled with the earlier `levir-ship-mosaic-policy-matrix` standard row because the uploaded artifacts and evaluation values differ. The P2/P3/P4 rows are useful matched controls for the R2 OACP variants below, but they change the detector architecture relative to the canonical rows.

#### TinyPerson matched augmentation baselines

Repository: [`duyle2408/tinyperson-augmentation-baselines-seed42-runs`](https://huggingface.co/datasets/duyle2408/tinyperson-augmentation-baselines-seed42-runs). The uploaded baseline manifest identifies the primary metrics as the standard Ultralytics test split and also records merged corner-window metrics for TinyPerson. The detector input is `imgsz=640`, while source windows are `640x512`.

| Detector | Method | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---|---:|---:|---:|---:|
| Canonical YOLOv8 | No augmentation, no Mosaic | 0.4330 | 0.1480 | 0.4344 | 0.1470 |
| Canonical YOLOv8 | Standard Mosaic | **0.5245** | **0.1889** | **0.4948** | **0.1750** |
| YOLOv8n P2/P3/P4 | No augmentation, no Mosaic | 0.4827 | 0.1660 | 0.4715 | 0.1620 |
| YOLOv8n P2/P3/P4 | Standard Mosaic | **0.5617** | **0.2006** | **0.5153** | **0.1862** |

These rows provide the missing no-augmentation control for the newer TinyPerson Copy-Paste + Mosaic variants, although the same-detector no-augmentation baseline is still not available for the canonical YOLOv8 Copy-Paste rows.

#### R2 adaptive OACP screening, LEVIR-Ship

Repositories: [`hardness`](https://huggingface.co/datasets/duyle2408/levir-oacp-r2-hardness-adaptive-seed42-runs), [`curriculum`](https://huggingface.co/datasets/duyle2408/levir-oacp-r2-curriculum-seed42-runs), [`load`](https://huggingface.co/datasets/duyle2408/levir-oacp-r2-load-adaptive-seed42-runs), [`size`](https://huggingface.co/datasets/duyle2408/levir-oacp-r2-size-adaptive-seed42-runs), [`effect`](https://huggingface.co/datasets/duyle2408/levir-oacp-r2-effect-adaptive-seed42-runs). All five artifacts use the YOLOv8n P2/P3/P4 plain detector, no Mosaic, `imgsz=512`, and a single training seed 42. The repository names identify the adaptive policy; the minimal uploaded manifests do not expose all policy-specific parameter fields, so the exact policy should be verified from the runner source before publication.

| R2 policy | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test AP75 |
|---|---:|---:|---:|---:|---:|
| Hardness-adaptive | 0.8332 | 0.3241 | 0.8016 | 0.3102 | 0.1059 |
| Curriculum | 0.8082 | 0.3250 | 0.7871 | 0.2992 | 0.0998 |
| Load-adaptive | 0.8337 | 0.3291 | **0.8189** | 0.3094 | **0.1154** |
| Size-adaptive | 0.8257 | **0.3288** | 0.7854 | 0.2977 | 0.1110 |
| Effect-adaptive | 0.8316 | 0.3298 | 0.8074 | 0.3102 | 0.1131 |

The apparent test mAP50-95 difference between load-adaptive (**0.3094**) and effect-adaptive (**0.3102**) is small and should not be treated as a meaningful winner from one seed. Load-adaptive has the highest test AP50 and AP75 in this five-way screen; effect-adaptive has the highest validation mAP50-95. These rows are no-Mosaic controls and therefore are not duplicates of the earlier adaptive-OACP + Mosaic family.

#### Context R2 audit baseline

Repository: [`duyle2408/levir-oacp-context-r2-audit-runs`](https://huggingface.co/datasets/duyle2408/levir-oacp-context-r2-audit-runs). This is a single-pass/current OACP R2 audit using the four-scale historical P2-family YAML `yolov8n_p2_levir_baseline.yaml`, no Mosaic, `imgsz=512`, and seed/split seed 42.

| Method | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test AP75 |
|---|---:|---:|---:|---:|---:|
| Current OACP R2 audit baseline | 0.8348 | 0.3262 | 0.8156 | 0.3113 | 0.1203 |

This four-scale P2-family detector is not the same architecture as the P2/P3/P4 plain detector used by the five adaptive R2 rows. It is retained as a provenance-qualified audit control, not pooled into either adaptive table.

#### Newly completed context-adaptive audits

Repositories: [`C1 strength`](https://huggingface.co/datasets/duyle2408/levir-oacp-context-c1-strength-audit-runs), [`C2 scale`](https://huggingface.co/datasets/duyle2408/levir-oacp-context-c2-scale-audit-runs), and [`C3 expand`](https://huggingface.co/datasets/duyle2408/levir-oacp-context-c3-expand-audit-runs). These repositories were previously listed as empty, but were updated after the earlier audit. They now contain complete evaluation artifacts. All three use the same four-scale historical P2-family YAML `yolov8n_p2_levir_baseline.yaml`, single-pass/current OACP, no Mosaic, `imgsz=512`, workers 8, 100 epochs, training seed 42, split seed 42, and NMS IoU 0.50. The policy-specific manifest fields are: C1 `oacp_strength_policy=context_adaptive`, C2 `oacp_scale_policy=context_adaptive`, and C3 `oacp_protection_policy=contrast_adaptive`.

| Context audit | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 | Test AP75 |
|---|---:|---:|---:|---:|---:|
| C1 strength-adaptive | 0.8280 | 0.3294 | 0.8034 | 0.3123 | 0.1347 |
| C2 scale-adaptive | 0.8330 | 0.3251 | 0.7844 | 0.3051 | 0.1152 |
| C3 expand/protection-adaptive | **0.8444** | **0.3366** | **0.8207** | **0.3145** | 0.1254 |

**Interpretation:** C3 is the best test mAP50-95 and test AP50 in this matched screen, while C1 has the highest test AP75. These are still single-seed results and should be treated as context-policy screening evidence, not a multi-seed final ranking.

#### TinyPerson Copy-Paste + Mosaic variants

Repository: [`duyle2408/tinyperson-copy-paste-mosaic-runs`](https://huggingface.co/datasets/duyle2408/tinyperson-copy-paste-mosaic-runs). These are canonical YOLOv8n, seed 42, `mosaic=1.0`, `close_mosaic=10`, and no OACP. The source windows are `640x512`; detector input is `imgsz=640` and the test artifact is the merged corner-window evaluation.

| Copy-Paste mode | Val AP50 | Val mAP50-95 | Test AP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|
| Crowded, mild overlap | 0.5441 | **0.2013** | 0.5166 | 0.1880 |
| Crowded, moderate overlap | 0.5472 | 0.1994 | **0.5172** | 0.1853 |
| Negative Copy-Paste, offline bank | 0.5467 | 0.2004 | 0.5201 | **0.1885** |
| Scale-matched Copy-Paste | **0.5492** | 0.2030 | 0.5163 | 0.1885 |

The mode names are not CP0-CP3 and should not be merged with the earlier TinyPerson CP3 table. Among these four new rows, negative and scale-matched Copy-Paste tie at the displayed precision for test mAP50-95, with negative Copy-Paste highest by the unrounded value. The differences are single-seed and small.

#### Repositories with no usable result artifact

The following supplied repositories currently contain no manifest, evaluation metrics, or uploaded run result and are excluded rather than treated as zero-valued experiments:

- [`duyle2408/levir-oacp-r2-fixed-seed42-runs`](https://huggingface.co/datasets/duyle2408/levir-oacp-r2-fixed-seed42-runs)
- [`duyle2408/tinyperson-copy-paste-runs`](https://huggingface.co/datasets/duyle2408/tinyperson-copy-paste-runs)

This is an artifact-status statement only. It does not imply failed training or zero performance.

#### Baseline coverage by experiment family

| Experiment family | Baseline currently available | Reference used in this report | Remaining baseline gap |
|---|---|---|---|
| LEVIR canonical Mosaic M2-M5 | **Partial** | Standard Mosaic M0 in the controlled policy matrix, Section 6.1 | No same-repository no-Mosaic/no-augmentation control for M2-M5. Use M0 only as a Mosaic-policy reference, not as a no-augmentation baseline. |
| TinyPerson Mosaic M2-M4 | **Partial** | TinyPerson standard-policy control in the resize-policy matrix and M2-M4 rows above | No same-detector, same-artifact no-Mosaic/no-augmentation control in `mosaic-only-yolo-runs`. |
| LEVIR Copy-Paste CP0-CP3 post-hoc | **Present** | CP0 is the matched no-Copy-Paste, no-Mosaic control in the same post-hoc protocol | No gap for the CP0-CP3 comparison. Crowd/negative/scale variants still use CP0 as the nearest control and should not be interpreted as a complete factorial design. |
| LEVIR post-Mosaic OACP | **Partial** | No-Mosaic OACP rows in the same repository and earlier OACP controls | No pure no-augmentation P2/P3/P4 baseline in the same repository. The no-Mosaic OACP rows are augmentation controls, not no-augmentation controls. |
| LEVIR adaptive OACP + Mosaic | **Partial** | Earlier standard/no-Mosaic OACP family in Section 6.2 | No single matched canonical P2/P3/P4 no-OACP baseline across every adaptive variant. Detector and policy effects remain confounded. |
| TinyPerson CP3/OACP/Mosaic combinations | **Partial** | CP3 + Mosaic without OACP is available for the P2/P3/P4 group | Missing a same P2/P3/P4, no-OACP/no-Copy-Paste/no-Mosaic baseline. The YOLOv8 baseline elsewhere is not an architecture-matched substitute. |

For publication-quality comparisons, the missing controls should be added as explicit rows rather than inferred from another repository. The minimum next baseline set is: canonical detector with no augmentation, canonical detector with standard Mosaic, P2/P3/P4 detector with no augmentation, and P2/P3/P4 detector with standard Mosaic, each using the same dataset split, training seed set, workers, schedule, and evaluation protocol as its augmentation family.

## 5. Per-run settings audit

This section records the augmentation switches explicitly, rather than inferring them from a checkpoint name. `On` means `mosaic=1.0`; `Off` means `mosaic=0.0`. Unless a row says otherwise, Mosaic-enabled runs use `close_mosaic=10`, while the matched Copy-Paste-only protocol uses `close_mosaic=0`.

### 5.1 Copy-Paste settings shared by CP0-CP3 runs

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

### 5.2 OACP and Mosaic settings by named run family

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

### 5.2a Exact detector YAML map

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
4. Where only a Hugging Face result table is available and neither the manifest nor a downloadable checkpoint exposes the detector graph, the report keeps the architecture label provenance-qualified rather than guessing it from the repository name. The CP0-CP3 families documented in Section 4.4 and Section 6.4 are exceptions because their HF checkpoints were directly inspected.

### 5.3 Exact YOLO11 matrix variants

The four YOLO11 variants are not interchangeable:

| Variant | OACP path | OACP parameters | Mosaic |
|---|---|---|---|
| `legacy_default_oacp_mosaic` | legacy double-pass | `p=.20`, strength `.20-.40`, scale `.65-.85`, expand `3.0` | On, `close_mosaic=10` |
| `legacy_default_oacp_no_mosaic` | legacy double-pass | same as above | Off, `mosaic=0.0` |
| `r2_frequent_mild_oacp_mosaic` | corrected single-pass | `p=.40`, strength `.10-.25`, scale `.80-.95`, expand `3.0` | On, `close_mosaic=10` |
| `r2_frequent_mild_oacp_no_mosaic` | corrected single-pass | same as above | Off, `mosaic=0.0` |

All four use YOLO11n, TinyPerson, train seed 42, split seed 42, 100 epochs, patience 0, image size 640, batch size 8, workers 8, and NMS IoU 0.5.

## 6. Results grouped by experiment family

### 6.1 LEVIR-Ship: controlled Mosaic policy matrix

All four rows use the canonical upstream-style YOLOv8 P3/P4/P5 YAML, fixed split seed 42, training seed 42, 100 epochs, and workers 8. Only the Mosaic policy changes.

| Policy | Test mAP50 | Test mAP50-95 | Test AP75 | Val mAP50-95 |
|---|---:|---:|---:|---:|
| Standard Mosaic | 0.7235 | 0.2453 | 0.0710 | 0.2859 |
| Visibility-aware Mosaic | **0.7351** | **0.2580** | 0.0685 | **0.2944** |
| Occupancy-matched Mosaic | 0.7324 | 0.2454 | 0.0653 | 0.2736 |
| Context-contrast Mosaic | 0.7235 | 0.2474 | 0.0675 | 0.2862 |

**Interpretation:** visibility-aware selection is the only policy with a clear primary-metric improvement in this controlled matrix. Occupancy and context policies do not improve test mAP50-95 over standard Mosaic.

### 6.2 LEVIR-Ship: adaptive OACP and detector-scale effects

| Configuration | Mosaic | Test mAP50 | Test mAP50-95 | Test AP75 |
|---|---|---:|---:|---:|
| YOLOv8n P2/P3/P4, spacing-adaptive OACP | On | 0.8079 | 0.3065 | 0.0977 |
| YOLOv8n P2/P3/P4, load-adaptive OACP | On | **0.8204** | **0.3109** | **0.1240** |
| YOLOv8n P2/P3/P4, mass-adaptive OACP | On | 0.7731 | 0.2881 | 0.1017 |
| YOLOv8n P2/P3/P4, density/budget OACP | Off | 0.7960 | 0.3054 | 0.1186 |
| YOLOv9t P2/P3/P4, density/budget OACP | On | 0.7156 | 0.2575 | 0.0936 |
| YOLOv9t P2/P3/P4, density/budget OACP | Off | 0.7265 | 0.2720 | 0.1159 |

**Interpretation:** load-adaptive OACP + Mosaic is the strongest reported LEVIR adaptive-OACP row. The YOLOv9t rows are lower, demonstrating that a newer/larger detector is not automatically better in this small-object setting. The P2/P3/P4 advantage is a detector effect plus augmentation, not an isolated OACP effect.

### 6.3 TinyPerson: adaptive OACP

| Configuration | Test mAP50 | Test mAP50-95 | Validation mAP50-95 |
|---|---:|---:|---:|
| YOLOv8 baseline, no Mosaic/no OACP | 0.4953 | 0.1752 | 0.1796 |
| YOLOv9 baseline, no Mosaic/no OACP | 0.5052 | 0.1793 | 0.1777 |
| YOLOv8n, spacing-adaptive OACP + Mosaic | 0.5252 | 0.1923 | 0.2134 |
| YOLOv8n, load-adaptive OACP + Mosaic | **0.5341** | **0.1955** | 0.2085 |
| YOLOv8n, mass-adaptive OACP + Mosaic | 0.5310 | 0.1940 | **0.2113** |
| YOLOv9t P2/P3/P4, budget/density + Mosaic | 0.4716-0.4744 | 0.1708-0.1720 | 0.1823-0.1831 |

**Interpretation:** adaptive OACP + Mosaic improves over the YOLOv8 no-Mosaic baseline by about **8.8% relative in test mAP50-95** for load-adaptive OACP. The YOLOv9t P2/P3/P4 rows are below the YOLOv8 baseline in this reported setup.

### 6.4 Copy-Paste without Mosaic

Metrics below are manifest-reported because these Copy-Paste repositories do not expose the same explicit evaluation artifact as the Mosaic matrix.

The HF checkpoint audit resolves the architecture for the older no-Mosaic CP ablations. The LEVIR CP0-CP3 checkpoints in `duyle2408/levir-ship-copy-paste` and the TinyPerson CP0 checkpoint in `duyle2408/tinyperson-copy-paste` serialize the same three-output canonical YOLOv8 head, `Detect f=[15, 18, 21]`, with strides `[8, 16, 32]`. These rows are therefore **canonical P3/P4/P5**, not YOLOv8 P2/P3/P4. The older TinyPerson manifests omit a model field, so this conclusion is based on direct checkpoint inspection rather than the manifest name alone.

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

### 6.5 Copy-Paste + OACP + Mosaic combinations

| Repository/run | Detector | Configuration | Reported mAP50 | Reported mAP50-95 |
|---|---|---|---:|---:|
| `tinyperson-yolov8n-p2p3p4-cp3-mosaic-no-oacp` | YOLOv8n P2/P3/P4 | CP3 + Mosaic, no OACP | 0.5529 | 0.1996 |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-no-mosaic` | YOLOv8n P2/P3/P4 | mass OACP + CP3, no Mosaic | 0.5226 | 0.1863 |
| `tinyperson-yolov8n-p2p3p4-oacp-mass-cp3-mosaic` | YOLOv8n P2/P3/P4 | mass OACP + CP3 + Mosaic | **0.5537** | **0.2061** |
| `tinyperson-copy-paste-oacp-mass-cp3-mosaic` | YOLOv8n | mass OACP + CP3 + Mosaic | 0.4990 | 0.1803 |
| `tinyperson-copy-paste-mosaic` | YOLOv8n | CP3 + Mosaic | 0.5056 | 0.1851 |
| `levir-ship-copy-paste-mosaic` | YOLOv8n | CP3 + Mosaic | 0.7689 | 0.2848 |

**Interpretation:** in the matched P2/P3/P4 TinyPerson group, adding Mosaic to mass OACP + CP3 improves reported mAP50-95 from **0.1863** to **0.2061**. However, the comparison with the plain YOLOv8n combinations is confounded by detector architecture, so it should not be described as a pure augmentation gain.

## 7. Seed and provenance notes

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

## 8. Conclusions and recommended next matrix

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

## 9. Source artifacts

- Existing compiled results: [`docs/reports/huggingface_results_20260912.md`](huggingface_results_20260912.md)
- Earlier compiled results: [`docs/reports/huggingface_results_20260910.md`](huggingface_results_20260910.md)
- OACP method/provenance report: [`oacp.md`](../../oacp.md)
- OACP implementation: [`project_ultralytics/context_augment.py`](../../project_ultralytics/context_augment.py)
- Copy-Paste implementation: [`project_ultralytics/copy_paste.py`](../../project_ultralytics/copy_paste.py)
- Mosaic policy helpers: [`project_ultralytics/mosaic_policy.py`](../../project_ultralytics/mosaic_policy.py)
- Scene-compatible Mosaic helpers: [`project_ultralytics/scene_compatible_mosaic.py`](../../project_ultralytics/scene_compatible_mosaic.py)

Metric provenance follows the existing compiled report: explicit evaluation artifacts are labeled as test/validation metrics, while Copy-Paste rows whose repositories upload only manifest metrics are labeled **reported** rather than silently relabeled as test metrics.
