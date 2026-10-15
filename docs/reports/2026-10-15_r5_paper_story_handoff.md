# YOLO11 R5 Paper-Story Handoff

## Scope

This report records the completed YOLO11 R5-A/B/C LevirShip experiment and the
remaining acceptance gap. The experiment used split seed `42`, training seed
`42`, image size `512`, batch size `8`, workers `8`, NMS IoU `0.50`, and the
seven scale bins:

```text
[0,8), [8,12), [12,16), [16,20), [20,24), [24,28), [28,32]
```

All three modes used the same adaptive negative-canvas Copy-Paste mechanics.
Only the difficulty-control method changed:

- R5-A: `offline_recall`
- R5-B: `online_tal_iou`
- R5-C: `online_probe_recall`

## Native val/test results

Values are fractions from the split-qualified `evaluation_metrics.json` files.

| Run | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---:|---:|---:|---:|
| R5-A offline recall | 0.8192 | 0.3169 | 0.7709 | 0.2892 |
| R5-B online TAL-IoU | 0.8021 | 0.3192 | **0.8046** | **0.3087** |
| R5-C online probe recall | **0.8206** | 0.3168 | 0.7922 | 0.3010 |

The available no-Mosaic YOLO11 baseline summary is approximately `0.8074`
test/AP50 and `0.3122` test/mAP50-95. This is an aggregate comparison, not a
same-seed paired comparison. R5-B is near parity but does not establish an
improvement claim.

## Evidence and artifacts

All three runs have local checkpoints, results, manifests, split-qualified
metrics, and verified uploads.

```text
/marimo/yolo_code_probe/runs/paper_story_r5_seed42/offline_recall/artifacts/offline_recall/
/marimo/yolo_code_probe/runs/paper_story_r5_seed42/online_tal_iou/artifacts/online_tal_iou/
/marimo/yolo_code_probe/runs/paper_story_r5_seed42/online_probe_recall/artifacts/online_probe_recall/
```

Each artifact root contains `weights/best.pt`, `weights/last.pt`,
`results.csv`, `evaluation_metrics.json`, and `experiment_manifest.json`.
The task-specific remote repository is `duyle2408/levirship-r5-paper-runs`,
with prefixes `levirship/{mode}/seed_42`.

## Interpretation

The defensible claim is:

> Scale-conditioned adaptive Copy-Paste can preserve YOLO11 native detection
> performance, with online TAL-IoU giving the strongest test result among the
> three R5 controllers in this single-seed comparison. The current evidence
> does not show a reliable accuracy gain over the existing YOLO11 baseline.

The results do not yet support the stronger claim that R5 improves small-object
accuracy. That claim requires explicit APsmall and scale-bucket metrics.

## Blocked follow-up

The repository evaluator `evaluate_test/size_bucket_evaluator.py` and wrapper
`evaluate_test/evaluate_r5_followup_size_metrics.py` were inspected. The live
bucket-evaluation path was attempted repeatedly through the approved Marimo
connection, but the supplied Marimo server became unreachable after reload.
No partial bucket artifact or upload was accepted.

The remaining hypothesis is therefore unresolved:

> Does the controller that is near-parity on native test mAP also improve the
> `Tiny1`, `Tiny2`, `Tiny3`, and `Small` area buckets?

Do not claim this until the bucket evaluator produces and uploads the explicit
`val_size/*` and `test_size/*` metrics.
