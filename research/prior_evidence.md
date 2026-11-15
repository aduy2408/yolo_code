# Curated Prior Evidence

This file is the active research index extracted from the approach and result
reports. Historical reports remain unchanged and are evidence sources. A method
marked `closed` is not deleted; it is excluded from new experiment proposals
unless new evidence changes the hypothesis.

## Active foundations

| Item | Status | Reuse rule | Evidence |
|---|---|---|---|
| Full P2 detector | `foundation` | Use as the primary small-object detector baseline when the hypothesis concerns tiny-object representation or responsibility. Keep P2/P3/P4/P5 and P2-only as separate protocols. | `approach+results/approach_report.md`, Approach 1 |
| P2-only | `reference` | Use only as a compact representation baseline; do not silently compare it with full P2. | `approach+results/approach_report.md`, Approach 1 |
| GAP Channel Attention | `reference` | Keep as a matched reference because it is one of the strongest and most stable P2 recipes. | `approach+results/approach_report.md`, Approach 5; `report_factorized_tal.md` |
| GAP + FTAL k=1.5, lambda=0.5 | `reference winner` | Use as a reference recipe on the matched GAP/P2 protocol. Do not call FTAL a general TAL replacement. | `approach+results/approach_report_3.md`, Approach 16; `report_factorized_tal.md` |
| Decoupled Detect head | `control` | Keep as the default head for GAP+FTAL. Shared-head and wider-classifier variants are closed. | `report_factorized_tal.md` |

## Active method leads

| Item | Status | What may be reused | Boundary |
|---|---|---|---|
| Scale-conditioned gated residual responsibility | `active lead` | Preserve standard TAL and positive supervision; add a bounded residual only for tiny objects; use warmup/fallback; log oracle gap, responsibility entropy, positive mass, and medium-object metrics. | Current evidence is strongest on matched long YOLOv8 TinyPerson B4/B6. It is not architecture-independent and has not yet passed multi-dataset matched controls. See `tod_research_exploration.md` and `docs/reports/tod_research_direction_analysis.md`. |
| Candidate responsibility diagnostics | `required diagnostic` | Reuse local oracle gap, score-selected vs best-IoU candidate, candidate identity stability, P2/P3/P4 breakdown, and classification-vs-localization failure buckets. | Diagnostics are not themselves a method and do not replace AP. |
| Targeted partial-view clipping | `auxiliary candidate` | Use only when the hypothesis concerns partial visibility or border-truncated objects. Keep it separate from architecture and loss changes. | It improves recall in the reported LEVIR study but is not the main architecture direction. `report_ablation_results.md`, `approach_report_4.md`. |
| OACP | `screening candidate` | Reuse corrected single-pass implementation and matched protocol. Prefer load/context/spacing policies only as separately named hypotheses. | Historical double-OACP results are not interchangeable. P2/P3/P4 results cannot be pooled with canonical P3/P4/P5 results. `augmentation_report.md`. |
| Negative-canvas Copy-Paste | `screening candidate` | Reuse R1/R4 definitions and artifact protocol if testing training-data support for small/negative scenes. | It is augmentation evidence, not proof of a responsibility or architecture mechanism. Keep native and merged TinyPerson metrics separate. `augmentation_report.md`. |
| Independent representation branch | `design principle` | If a future raw-image branch is proposed, it should form its own representation from near raw input before fusion and use a distinct inductive bias. | This is not a trained method result. `approach_report_4.md`, Approach 26. |

## Closed or do-not-reuse as default directions

| Item | Status | Reason |
|---|---|---|
| GCTS sub-cell routing | `closed` | No stable gain; do not increase selector complexity. |
| Offset regression, NUDFL, Pair-Competitive DFL | `closed/deprioritized` | AP75 can move, but total detection metrics are not stable. |
| Spatial attention, KVCA, local attention-radius search | `closed` | More complex attention did not beat the simpler channel/GAP reference. |
| P1 fusion as main architecture | `closed as main` | P1 contains useful detail but also sea texture; gains are not stable across seeds. |
| Fixed edge/high-frequency enhancer | `auxiliary only` | Detail/HF signals can help with GAP but do not justify a main architecture. |
| Raw color/edge/frequency cue concatenation | `closed` | Direct cue exposure did not beat Plain P2. A future raw branch must learn its own representation first. |
| DCFB / same-stream channel isolation | `closed` | Separate processing of the same feature is not independent representation. |
| Native cross-reconstruction / FFM | `closed` | Ancestor/descendant features did not provide complementary information automatically. |
| GGCF refined boxes inside TAL assignment | `closed` | Geometry has signal, but early refined assignment destabilized training. |
| CFR reconstruction as the main method | `closed as main` | Clean rerun was below GAP/GAP+FTAL; retain only as historical auxiliary-loss evidence. |
| DBSS as a general method | `not promoted` | Three-seed mean effect is near zero with high variance; residual separation is a weak proxy and scene leakage is substantial. |
| Size-aware classification loss weighting | `reject` | Increased false positives and reduced precision/mAP. |
| Naive predicted-IoU or sqrt-IoU responsibility replacement | `reject` | Severe TinyPerson degradation relative to standard TAL. |
| Marked-mass normalization | `reject` | Observational mass signal exists, but normalization reduced detection performance. |
| Post-hoc frozen reranking/calibration | `reject` | Held-out calibration did not improve top-candidate IoU consistently. |
| Shared or widened Detect head variants | `closed` | Held-out test metrics degraded against the decoupled GAP+FTAL reference. |
| Amplitude calibration/perturbation as main direction | `closed` | Small or continuation-only gains; insufficient as a primary method. |

## Reusable diagnostics and protocol rules

- Keep the baseline architecture, dataset, split, training seed, schedule, and evaluator fixed for a causal comparison.
- Record `nms_iou: 0.5` and do not mix native `test/*` with TinyPerson `test_merged/*`.
- Keep P2/P3/P4/P5 architecture changes separate from augmentation changes.
- Treat single-seed screens as screening evidence, not stable claims.
- Preserve per-seed results; do not compare one treatment seed against an unmatched multi-seed baseline aggregate.
- For every method, inspect AP50, AP75, mAP50-95, recall/precision, and size buckets where available.
- A proxy improvement without detection improvement is not a promoted method.
- Keep incomplete, interrupted, preflight-only, or unverified runs out of scientific conclusions.

## Augmentation evidence

The augmentation reports are a separate evidence family, not a pool of
architecture results. The final YOLO matrix contains `38/38` verified jobs
(`12` OACP, `10` Mosaic, `16` Copy-Paste), but conclusions remain within
matched detector/protocol families.

| Family | Status | Reuse rule | Boundary |
|---|---|---|---|
| Canonical YOLOv8 P3/P4/P5 + standard Mosaic | `control` | Use as the augmentation control only when detector, optimizer, split, and schedule match. | Do not use it as a control for P2/P3/P4 variants. |
| Visibility-aware Mosaic | `screening candidate` | Retain as a policy hypothesis for preserving visible object evidence. | Its reported LEVIR gain is family-specific and not a general Mosaic claim. |
| Hard-negative Mosaic | `screening candidate` | Retain for a separately matched policy comparison. | LEVIR/TinyPerson/Varroa prefer different policies; no cross-dataset winner. |
| OACP load/effect/context adaptive | `screening candidate` | Reuse corrected single-pass OACP and name the policy explicitly. | Single-seed screens; P2/P3/P4 and four-scale P2 families are not interchangeable. |
| Copy-Paste CP2 | `screening candidate` | Keep as a bounded candidate because it led the corrected LEVIR CP0–CP3 post-hoc family. | It is near the CP0 control; no universal Copy-Paste benefit. |
| NegativeCanvas R1/R4 | `diagnostic/screening` | Reuse as a distinct synthetic-negative-canvas hypothesis with native and merged metrics separated. | R1/R4 do not establish a winner; larger-donor R3 was negative in the reported LEVIR screen. |
| `random_viewport`, `mosaic_occlusion` | `secondary candidate` | `mosaic_occlusion` is the only extra-transform candidate with a near-baseline multi-seed mean. | Neither improved all primary metrics; do not add new primitive sweeps. |
| `viewport_mosaic_occlusion`, resolution-degrade tail | `closed as default` | Retain only as historical schedule evidence. | Seed instability or trade-offs prevented promotion. |
| `small_weight` | `reject` | Do not reuse. | Precision and mAP dropped through false positives. |

Augmentation rules:

- Keep augmentation policy, detector architecture, optimizer, Mosaic state,
  close-Mosaic schedule, split seed, training seed, and evaluator fixed for a
  causal comparison.
- Historical double-OACP and corrected single-pass OACP are different methods.
- The current seed-43/44 Copy-Paste matrix has protocol/evaluator parity
  caveats; do not compare its low size metrics directly with historical
  seed-42 Copy-Paste results until both use one pinned runtime and matched
  augmentation settings.
- A repository with no metric artifact is `unavailable`, not a failed or
  zero-valued experiment.
- The augmentation matrix is complete for artifact coverage; completion does
  not imply that every method is promoted.

Sources: `docs/reports/augmentation_report.md`,
`docs/reports/complete_main_report.md`,
`docs/reports/approach+results/report_260826.md`, and
`docs/reports/copy_paste_hf_audit_20261005.md`.

## Baseline, model-family, and protocol evidence

| Evidence family | Status | Reuse rule |
|---|---|---|
| YOLOv5/v8/v9/v10/YOLO11 baseline matrix | `reference archive` | Use for model-family context and architecture-matched controls, not as a claim that the newest model is the causal improvement. |
| STW YOLO, TPH YOLOv5, FCOS-SET, FCOS-SR-TOD | `external reference` | Cite only with the exact dataset, split, native/merged protocol, and artifact status. |
| DETR/RT-DETR R18 | `external diagnostic` | RT-DETR is substantially stronger than DETR on some datasets but unstable on LEVIR across seeds; never hide the per-seed failure behind a mean. |
| DRENet | `external reference` | Keep as a separate model-family comparison; its low confidence threshold and evaluation implementation are not a YOLO training control. |
| YOLO architecture graph report | `architecture reference` | Use to define matched baseline architecture and avoid choosing a YAML by a familiar filename or weight name. |

Protocol facts that must stay in the loop:

- `nms_iou: 0.5` for final YOLO artifacts.
- LEVIR native evaluation, TinyPerson native corner-window evaluation, and
  TinyPerson merged original-image evaluation are different metric protocols.
- Split-qualified `val/*` and `test/*` must not be inferred from aggregate
  training metrics.
- Missing or incomplete split-qualified artifacts block a scientific claim;
  they do not become zeros.
- Historical FTAL winner reproduction requires
  `factorized_tal_mode=legacy`; `legacy` and `current` semantics cannot share
  one causal table.

Sources: `docs/reports/baseline_main.md`,
`docs/reports/all_datasets_baseline_validation_test.md`,
`docs/reports/SOTA_main_report.md`,
`docs/reports/detr_r18_rtdetr_r18_matrix_four_datasets.md`,
`docs/reports/report_drenet_levir.md`,
`docs/reports/yolo_v9_v10_v11_architecture.md`, and
`docs/reports/approach+results/report_250826.md`.

## Diagnostic-only evidence from non-approach reports

- `detection_results.md`: the LEVIR failure mix is primarily classification
  confidence, dense-scene candidate/NMS interaction, and partial observation;
  regression is not the first intervention target.
- `report_bottleneck_diagnostics.md`: retain raw-candidate oracle gap,
  DFL uncertainty/error, edge-error decomposition, and frequency probes as
  measurements, not as standalone improvements.
- `investigate_pooling.md`: P2 already preserves high-frequency signal; fixed
  high-pass or pooling additions need a causal detector result and must not be
  justified by visualization alone.
- `report_positive_support_dropout.md`: positive-support dropout is an
  implementation/robustness diagnostic; it does not establish a promoted
  training method.
- `report_levir_dbss_generalization.md` and `report_varroa_dbss.md`: DBSS/HIT
  remains an ablation/diagnostic family with near-zero or high-variance
  generalization evidence; keep it out of the main method queue.
- `2026-10-15_r5_paper_story_handoff.md`: R5 is a historical lead; do not claim
  small-object improvement without explicit size-bucket metrics.

## What this changes in the experiment loop

Every new hypothesis must declare one source family:

1. detector/representation;
2. assignment/responsibility;
3. augmentation/data distribution;
4. external model-family control; or
5. diagnostic measurement.

One run may not silently combine families. If it combines them, the manifest
must state the interaction hypothesis and include the matched controls needed
to identify each factor. The designer should reject proposals that only rename
an already-closed method or expand a single-seed screening result into a grid.

## Current recommendation

Do not start a new attention, raw-cue, regression-bin, or mass-normalization
sweep. The next method-level work should stay within one of these bounded paths:

1. matched evaluation of the gated responsibility lead on a second dataset with
   the same diagnostics;
2. architecture-matched controls for any transfer claim;
3. a separately justified augmentation hypothesis using the retained OACP or
   partial-view mechanisms.
