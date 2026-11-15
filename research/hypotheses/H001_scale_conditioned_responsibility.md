# H001: Scale-conditioned tiny-only gated responsibility residual

## Status

**BLOCKED — planning only; not executable.** Execution MUST wait until the exact repository-relative baseline YAML, source SHA, B4 residual formula and semantics, and exact VisDrone dataset YAML/protocol are recovered and reviewed. This card does not authorize implementation, smoke testing, training, or evaluation.

## Evidence family

**Responsibility.**

## Claim

On VisDrone2019-DET tiny objects, a B4-style, tiny-only, uncertainty-gated, bounded residual added to the learned candidate responsibility signal will reduce candidate-score misalignment while preserving the standard TAL assignment, standard positive-supervision fallback, detector architecture, optimizer/data protocol, and inference path.

This is a bounded candidate-ranking intervention, not a claim of an AP improvement. The one planned treatment is intended to make the score-selected candidate closer to the locally best-IoU candidate, while avoiding the broad-object damage seen when an unrestricted residual is applied.

## Single mechanism under test

The mechanism is **tiny-only gated residual responsibility correction**:

1. Keep standard TAL assignment and its positive supervision as the baseline path.
2. Add one bounded residual correction to the responsibility signal only for the recovered B4 tiny-object cohort.
3. Apply the recovered uncertainty gate and explicit B0 fallback semantics so uncertain or out-of-cohort candidates retain standard TAL responsibility.
4. Keep inference unchanged; the learned responsibility affects training-time candidate scoring only.

The exact residual formula, tiny-object scale threshold, uncertainty gate, residual bound, warmup/fallback semantics, and configuration names are **not verified** in the reviewed reports. They MUST be recovered rather than inferred or re-created for this experiment. No alternative formula or second mechanism is proposed here.

## Why this is a valid new test

The prior evidence supports the mechanism but does not establish it on VisDrone:

- The matched candidate probe reports an approximately 0.25-IoU tiny-object local oracle gap on VisDrone, with the score-selected candidate below the best-IoU local candidate.
- Candidate identity is generally stable under brightness/blur perturbations, so the intervention targets score learning rather than an unstable candidate generator.
- Frozen post-hoc calibration/reranking was rejected; the correction must therefore be learned during training.
- Naive predicted-IoU and sqrt-IoU responsibility replacement were rejected, and B3's unrestricted residual was unsafe.
- B4 was the safest residual smoke candidate because its tiny-only gate kept the short-run metrics near the standard-TAL control. The long B4 result is one TinyPerson dataset/seed and lacks the required responsibility diagnostics; it is screening evidence, not a promoted claim.
- VisDrone supplies new evidence for a bounded transfer test: its oracle gap is large, while the exact matched baseline and B4 implementation provenance remain to be recovered.

## Observable predictions

### Primary mechanism diagnostics

For the tiny-object cohort, compared with the exact matched standard-TAL baseline:

- **Local oracle gap:** decreases.
- **Score-selected candidate IoU:** increases toward the locally best-IoU candidate.
- **Candidate ranking quality:** score-vs-IoU rank correlation, top-candidate hit rate, and/or selected-vs-oracle IoU improve in the same direction.
- **Responsibility entropy:** decreases modestly (responsibility becomes less diffuse) without collapsing to a one-candidate/one-hot assignment; positive responsibility mass per ground-truth object remains present.
- **Candidate identity stability:** does not decrease under the declared brightness/blur diagnostic; the selected candidate should remain at least as stable as the matched baseline.

The diagnostic evaluator MUST report these fields separately for tiny and medium objects and, where available, by P2/P3/P4/P5 level. A proxy movement without a complete candidate/ranking diagnostic artifact is inconclusive and cannot support promotion.

### Safety and outcome metrics

The treatment MUST be compared with an exact same-protocol baseline using split-qualified validation and test artifacts:

- Tiny: AP50, AP75, mAP50-95, precision, recall, and the available VisDrone tiny-size AP50 bucket should be no worse than the matched baseline; improvement is welcome but is not assumed.
- Medium: AP50, AP75, mAP50-95, precision, recall, and the available medium-size AP50 bucket MUST not decrease. Medium safety is a hard constraint because the mechanism is gated to tiny objects.
- Overall: standard TAL assignment, the inference path, NMS IoU `0.5`, and the declared dataset split/evaluator MUST remain unchanged.

## Falsification and rejection rule

After provenance and protocol validity are established, **reject** H001 if any of the following is observed in a complete, matched run:

- the tiny local oracle gap does not decrease or the score-selected candidate does not move toward the local oracle;
- candidate ranking quality does not improve, or responsibility entropy becomes more diffuse or collapses without the predicted ranking benefit;
- candidate identity stability decreases materially under the declared perturbation diagnostic;
- any hard medium-object safety metric decreases, or tiny-object metrics fall below the matched baseline without a diagnostic improvement that the predeclared reviewer accepts;
- the residual is active outside the recovered tiny gate, changes TAL assignment/inference, or reproduces B3-like broad-object degradation;
- required split-qualified diagnostics, safety metrics, evaluator provenance, or artifact hashes are missing.

A missing or unverified provenance prerequisite is **BLOCKED**, not a scientific rejection. An interrupted or incomplete future run is not evidence for either decision.

## Minimal controlled experiment

- **Baseline:** immutable standard-TAL VisDrone protocol; exact repository-relative model YAML and dataset YAML are currently unrecovered and MUST be recorded before execution.
- **Treatment:** one B4-style tiny-only gated bounded residual responsibility correction, with the exact recovered formula, threshold, uncertainty gate, and fallback semantics; preserve standard TAL and the standard inference path.
- **Single changed factor:** the training-time candidate responsibility path (standard TAL responsibility versus standard TAL plus the B4-style tiny-only gated residual). No architecture, augmentation, optimizer, schedule, dataset split, or inference change.
- **Dataset:** VisDrone2019-DET official train/validation/test-dev protocol. The exact dataset YAML and protocol artifact are currently unavailable.
- **Dataset provenance:** reviewed VisDrone report records the official split family, 1536px baseline family, workers `8`, and NMS IoU `0.5`; this is insufficient to launch until the exact YAML and split/evaluator provenance are recovered.
- **Training seed:** `42`.
- **Split seed:** `42`.
- **Image size:** `1536` (reported VisDrone baseline setting; exact baseline manifest still requires recovery).
- **Workers:** `8` (reported setting; exact baseline manifest still requires recovery).
- **Smoke schedule:** after provenance recovery only, perform the repository-required import/model-construction/dummy-forward checks, then one 3-epoch treatment smoke at seed 42 against the exact matched baseline. Record per-epoch local oracle gap, selected-vs-oracle IoU, responsibility entropy, positive mass, candidate identity stability, tiny metrics, and medium safety metrics. Do not launch this schedule while blocked.
- **Long schedule:** only if the smoke is valid and meets the diagnostic/safety gate, run the single declared matched VisDrone schedule recovered from the immutable baseline protocol (the exact epoch schedule is currently unavailable; no value is guessed here). This remains one treatment variant, not a grid.
- **NMS IoU:** `0.5`.

## Promotion rule

E001 may be considered a bounded candidate only if the implementation/protocol is valid, all required artifacts are complete, the tiny local oracle gap and candidate-ranking diagnostic move in the predicted direction, responsibility entropy improves without collapse, candidate identity stability is non-decreasing, and no tiny or medium safety metric violates the declared baseline comparison. Because E001 uses one dataset and one training seed, it MUST NOT by itself authorize a promoted method claim; an independent matched replication under the declared research budget is required before promotion.

## Artifact contract

A valid future execution must contain, at minimum:

- immutable `experiment_manifest.json` and the exact recovered baseline/treatment configuration snapshots;
- source SHA, config hashes, dataset YAML/protocol provenance, split seed, training seed, image size, workers, schedule, and NMS IoU;
- split-qualified validation/test metrics with tiny and medium safety buckets, precision, recall, AP50, AP75, and mAP50-95;
- per-epoch responsibility diagnostics: local oracle gap, score-selected and best-IoU candidate identities/IoUs, rank correlation or hit rate, responsibility entropy, positive mass per ground truth, and perturbation identity stability;
- implementation/preflight and smoke records, training/evaluation logs, results CSV, checkpoints, evaluator metadata, failure analysis, and completion marker;
- a provenance record showing the recovered exact baseline YAML, source SHA, residual semantics, and dataset protocol. Until those four items are present, the artifact contract is non-executable and the run remains blocked.

## Decision history

| Experiment | Result | Decision | Reason |
|---|---|---|---|
| — | — | blocked | Exact baseline YAML, source SHA, B4 residual semantics, and VisDrone dataset protocol are not verified. |
