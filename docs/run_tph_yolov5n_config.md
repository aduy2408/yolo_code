# Custom TPH-YOLOv5n profile

## Provenance

- Control: upstream `models/yolov5l-xs-tph.yaml`
- Upstream source commit: `052dfeb375e51756f17e9ca4f96b7e3e3a7cf3c4`
- Variant: `models_related/models_config/yolov5/tph/yolov5n-xs-tph.yaml`
- Architecture source: original TPH-YOLOv5 backbone, C3STR neck, and TPH prediction path
- No KVCA, CBAM, or project-specific attention blocks are added

## Scaling choice

The upstream `C3STR` implementation chooses attention heads as `hidden_channels // 32`.
A conventional `width_multiple: 0.25` would make the P2 C3STR hidden width 16 and
would create zero attention heads. This profile therefore uses:

```text
depth_multiple = 0.33
width_multiple = 0.50
```

This is the smallest profile that remains valid without modifying the upstream
TPH implementation. It is a custom nano-depth profile, not an official upstream
`yolov5n` checkpoint or paper configuration.

## Runner support

`train_scripts/train_tph_yolov5_multidataset.py` now preserves an explicit
`--model-yaml` instead of replacing it with the default `yolov5l-xs-tph` YAML.
It also records the selected variant and accepts `--weights`. For a from-scratch
run, pass an empty weights value through the Marimo launch command if the upstream
trainer accepts that mode. Do not reuse the official l-xs results as nano results.

## Required validation before training

1. Build the model with the exact pinned upstream TPH checkout.
2. Run a dummy forward at the requested image size.
3. Run a bounded one-batch smoke train and evaluation.
4. Record parameter count, output shapes, source commit, dataset split, and weights source.
5. Only then launch the full upload-required experiment.
