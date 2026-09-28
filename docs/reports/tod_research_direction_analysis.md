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
