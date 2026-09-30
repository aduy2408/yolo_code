# TOD Research Direction Analysis

## Decision

The strongest direction is **scale-conditioned local responsibility learning under stable candidate identity**.

This is not an evidence-generation problem first. The probes found useful intermediate evidence and a large local candidate oracle gap, especially on VisDrone and TinyPerson. The detector often has a better local candidate than the candidate selected by its score. However, post-hoc reranking and simple frozen calibration failed, so the correction must be learned during training.

## Four original hypotheses

| Hypothesis | Cross-dataset result | Decision |
|---|---|---|
| Causal feature intervention | Context dependence changed substantially by dataset and scale. No stable shortcut pattern appeared across all three domains. | Reject as primary direction |
| Tiny-object information bottleneck | Object-region signal remained in intermediate and late features. Curves were non-monotonic rather than a universal progressive collapse. | Reject the simple compression story |
| Object survival modeling | Survival proxies were useful diagnostics but did not decrease monotonically through the network. | Reject as a universal method hypothesis |
| Gradient optimization | The initial activation-gradient proxy did not establish that unstable true training gradients are the dominant cause. | Reject as first intervention |

## Decisive candidate-alignment evidence

The local candidate probe compared the best-IoU candidate against the score-selected candidate around each ground-truth object. Approximate oracle gaps for tiny objects were:

- LEVIR-Ship: 0.06 IoU
- VisDrone: 0.25 IoU
- TinyPerson: 0.32 IoU

Candidate identity was relatively stable under brightness and blur perturbations, usually around 0.85 to 0.93 for tiny objects. A frozen calibration using score, feature energy, geometry, and interactions did not improve held-out top-candidate IoU. Therefore, inference reranking is not the right intervention.

## Training-feasibility gate

Three 3-epoch TinyPerson runs used the same YOLOv8n P3/P4/P5 baseline, `seed=42`, `split_seed=42`, `workers=8`, 640px inputs, and the generated `sw640/sh512` corner-window test protocol. Only the classification responsibility target changed.

| Variant | Target | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| B0 | Standard TAL (`off`) | 0.2666 | 0.0880 | 0.3123 | 0.0986 |
| B1 | Normalized predicted IoU (`iou`) | 0.0964 | 0.0239 | 0.1059 | 0.0257 |
| B2 | Normalized sqrt-IoU (`iou_sqrt`) | 0.1645 | 0.0432 | 0.1799 | 0.0482 |

B1 and B2 both degraded validation and test metrics. B1 retained only about 34% of baseline test AP50 and 26% of baseline test mAP50-95. B2 retained about 58% and 49%. This rejects the naive claim that current predicted IoU can directly replace standard positive classification responsibility.

Artifacts were uploaded and verified in task-specific repositories:

- `duyle2408/tinyperson-responsibility-off-smoke-v2-runs`
- `duyle2408/tinyperson-responsibility-iou-smoke-v3-runs`
- `duyle2408/tinyperson-responsibility-iou-sqrt-smoke-v3-runs`

The source artifacts are each run's `evaluation_metrics.json`, `experiment_manifest.json`, `results.csv`, and uploaded best/last checkpoints. The exact test protocol and split seed are recorded in each manifest.

## Recommended research mechanism

Do not replace TAL with raw predicted IoU. Instead:

1. Preserve standard TAL assignment and B0 supervision as the fallback path.
2. Add a bounded residual correction to positive responsibility, not a full replacement.
3. Build the correction from detached utility aggregated across stable augmentation views.
4. Add uncertainty and scale gating, with a minimum classification/objectness floor.
5. Measure local oracle gap and responsibility accuracy before measuring AP.
6. Require no medium-object degradation and replication on at least two datasets before a full training sweep.

## Why this is worth pursuing

Existing TOD work commonly adds resolution, attention, feature fusion, or modified losses. The measurements here point to a different unresolved issue: the detector can retain evidence and produce locally useful candidates, but its training signal does not reliably teach the score to assign responsibility to the useful candidate. The negative B1/B2 result makes the contribution more precise: localization quality is informative but too noisy to be used directly as a tiny-object classification target.

The possible contribution is therefore a **safe, scale-conditioned, uncertainty-aware responsibility correction** that changes candidate supervision while leaving the detector architecture and inference path unchanged. The first full method experiment should only begin after a bounded residual version reduces the local oracle gap without reproducing the B1/B2 collapse.

## Final recommendation

Focus future work on the responsibility-target problem, but do not proceed with raw-IoU or sqrt-IoU target replacement. Implement the detached, augmentation-consistent, bounded residual version next. Keep the original four hypotheses as rejected diagnostic explanations, not as method directions.

## B3-B6 follow-up result

The bounded-residual matrix was run on TinyPerson with the same YOLOv8n P3/P4/P5 detector, `seed=42`, `split_seed=42`, 640px inputs, batch 8, workers 8, NMS IoU 0.5, and three epochs. The test protocol was the generated `sw640/sh512` corner-window split with 17,693 test windows.

| Variant | Responsibility mode | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| B0 | standard TAL, `off` | 0.266618 | 0.087976 | 0.312338 | 0.098611 |
| B3 | bounded detached residual | 0.217014 | 0.071044 | 0.219437 | 0.068131 |
| B4 | residual plus tiny-only gate | 0.263660 | 0.081930 | 0.309724 | 0.097945 |
| B5 | residual plus geometric consistency proxy | 0.237847 | 0.074394 | 0.284961 | 0.092150 |
| B6 | curriculum warmup then residual-consistent mode | 0.266618 | 0.087976 | 0.312338 | 0.098611 |

All four task-specific Hugging Face repositories contain verified upload markers for metrics, manifests, results, and both checkpoints:

- `duyle2408/tinyperson-responsibility-b3-smoke-runs`
- `duyle2408/tinyperson-responsibility-b4-smoke-runs`
- `duyle2408/tinyperson-responsibility-b5-smoke-runs`
- `duyle2408/tinyperson-responsibility-b6-smoke-runs`

The merged TinyBenchmark evaluator was unavailable because `pycocotools` was not installed. The reported test values are the regular split-qualified corner-window test metrics, not merged-test values.

### Decision

This matrix does not justify a full method implementation or a multi-seed sweep yet.

- **B3 is rejected as unsafe.** Applying the residual from the start reduced test AP50 by 29.7% relative to B0.
- **B4 is the best non-control safety candidate.** The tiny-only gate kept test AP50 within 0.0026 of B0 and test mAP50-95 within 0.0007. This supports scale gating as a safety mechanism, not as an improvement.
- **B5 is inconclusive and currently negative.** It recovered some of B3's loss but remained below B0. Its consistency signal is a geometric perturbation proxy, not genuine paired augmentation consistency.
- **B6 is safe but inactive at this schedule.** It exactly matched B0 after warmup/ramp. The curriculum avoided collapse, but did not demonstrate useful correction in three epochs.

The justified next step is a narrow diagnostic run using the B4/B6 safety design at a longer schedule, with per-epoch local oracle gap, responsibility entropy, positive mass per ground-truth object, and medium-object metrics. The correction should be retained only if it improves responsibility diagnostics without reducing B0-level validation and test metrics on TinyPerson and VisDrone. Otherwise, discard this direction. Do not claim AP improvement from the current smoke results.

## R1/J1 comparison against the documented baseline

The table below combines the documented B0 control from the three-epoch TinyPerson feasibility gate with the verified 100-epoch R1/J1 uploads in `duyle2408/tinyperson-reranking-runs`. B0 is the closest documented baseline on the same `seed=42`, `split_seed=42`, `sw640/sh512` protocol, but its three-epoch schedule is not a fully matched control for the 100-epoch R1/J1 runs. The deltas are therefore descriptive, not causal evidence of improvement.

| Run | Training schedule | Mode | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | Δ test/AP50 vs B0 | Δ test/mAP50-95 vs B0 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| B0 documented control | 3 epochs | Standard TAL (`off`) | 0.266618 | 0.087976 | 0.312338 | 0.098611 | — | — |
| R1 | 100 epochs | Localization ranking | 0.515954 | 0.184284 | 0.496686 | 0.177407 | +0.184348 | +0.078796 |
| J1 | 100 epochs | Joint correction-only ranking | 0.526127 | 0.188514 | 0.498978 | 0.178672 | +0.186640 | +0.080061 |

**Sources and protocol boundary.** B0 is the existing `B0` row in the three-epoch B3-B6 TinyPerson table above. R1 is uploaded at `runs/localization/yolov8n_base/seed_42`; J1 is uploaded at `runs/joint/yolov8n_base/seed_42`. Both uploads contain verified weights, `results.csv`, manifests, metrics, and `upload_complete.json`. Both metrics files explicitly label `val/AP50`, `val/mAP50-95`, `test/AP50`, and `test/mAP50-95`, and record the TinyPerson corner-window protocol and `corner_manifest.json` source artifact. However, both report `test_merged/available = 0.0`, so the displayed test metrics are the regular split-qualified corner-window test metrics, not successfully computed merged-test metrics.

**Interpretation.** On the observed metrics, J1 is slightly above R1 on all four reported split metrics, but the comparison against B0 is confounded by the three-versus-100-epoch schedule. A fair AP claim requires a 100-epoch B0 control under the same run contract and a working merged-test evaluator.
