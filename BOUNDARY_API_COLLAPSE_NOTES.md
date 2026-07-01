# Boundary/API Training Collapse Notes

## Observed Run

Run name:

```text
yolov8n_varroa_compare_baseline_p2_boundary_api_baseline_p2_boundary_api
```

Path shown by training:

```text
/marimo/yolo_code/runs/detect/yolo_related/runs/train/yolov8n_varroa_compare_baseline_p2_boundary_api_baseline_p2_boundary_api
```

Setup from log:

```text
epochs=200
imgsz=640
batch appears effectively 16 based on 173 train steps for 2762 images
GPU memory ~15.9G
BoundaryFeatureBlock + AdversarialPerturbationInjection on P2
```

## Symptom

Validation metrics collapse after a few epochs even though training losses keep decreasing.

Epoch summary from user log:

```text
epoch 1: box=1.435,  cls=3.773, dfl=2.341, mAP50=0.617,  mAP50-95=0.211
epoch 2: box=1.701,  cls=2.303, dfl=1.912, mAP50=0.467,  mAP50-95=0.145
epoch 3: box=1.449,  cls=1.635, dfl=1.532, mAP50=0.0991, mAP50-95=0.0243
epoch 4: box=0.4543, cls=0.702, dfl=0.903, mAP50=0,      mAP50-95=0
epoch 5: box=0.3771, cls=0.514, dfl=0.856, mAP50=0.000006, mAP50-95=0.000001
epoch 6: box=0.3564, cls=0.446, dfl=0.843, mAP50=0,      mAP50-95=0
epoch 7: box=0.3588, cls=0.408, dfl=0.837, mAP50=0.0000068, mAP50-95=0.0000034
epoch 8: box=0.3338, cls=0.382, dfl=0.828, mAP50=0,      mAP50-95=0
```

This is suspicious because all three train losses become very small while validation precision/recall go to zero. That usually means the model is learning a degenerate training-time behavior or the train/inference behavior diverges strongly.

## Current Leading Hypotheses

1. Train-only feature path mismatch is too strong.

   `BoundaryFeatureBlock` and API affect P2 during training, but inference/eval path returns identity. If the detector head learns to rely on transformed/perturbed P2 statistics that disappear at validation, validation can collapse.

2. API auxiliary loss may overpower normal YOLO detection training.

   Current API adds auxiliary loss into the cls loss slot. Even with `api_weight=0.25`, the target is dense P2 supervision, so its gradient volume may dominate small-object detection gradients.

3. Boundary block residual may still be too aggressive.

   BoundaryFeatureBlock currently has a bounded residual, but any train-only feature transform before the P2 detect path can create a train/eval domain gap.

4. Foreground/objectness API target may be misaligned with the actual problem.

   The observed issue is bbox under-localization, not object discovery. Foreground aux target can encourage objectness shortcuts rather than better boundary regression.

5. Batch size / learning rate interaction.

   The shown run likely used batch 16, not batch 2, because 2762 images / 16 is about 173 steps. This differs from earlier debug settings and may make the auxiliary branch much stronger in aggregate.

## Immediate Debug Actions

1. Stop this run if it is still running.

2. Run short comparisons with the same settings:

   - `baseline_p2`
   - `baseline_p2_boundary_api`
   - `baseline_p2_boundary_api_ring`

3. For API runs, reduce the strength first:

```yaml
AdversarialPerturbationInjection: rho=0.005, api_weight=0.05
BoundaryFeatureBlock: alpha_init=0.0, alpha_max=0.02
```

4. Consider making `BoundaryFeatureBlock` active in eval as identity-compatible only after training stabilizes, or freeze/delay it for warmup epochs.

5. Add logging for:

   - API auxiliary loss value
   - perturbation norm
   - effective boundary alpha
   - train vs eval prediction count on the same batch

## Likely Fix Direction

For the localization problem, replace foreground API target with boundary-ring API target and weaken the intervention:

```text
P2 -> BoundaryFeatureBlock(alpha_max small) -> API(boundary_ring, rho small, api_weight small) -> Detect
```

The key is avoiding a train-only feature distribution that the Detect head cannot reproduce at validation/inference.
