# Current Best: LEVIR-Ship GAP + FTAL

**Status:** canonical reference. Use this document before launching or reporting any new LEVIR-Ship result.

**Last verified:** 2026-08-25

## Canonical result

The current best verified detector is `gap_factorized_k15`:

- Test mAP50, seed 42: **0.8283**
- Test mAP50-95, seed 42: **0.3161**
- Three-seed mAP50-95: **0.3179 ± 0.0017**
- Seeds: 42, 43, 44

The stored reference artifacts are verified in:

- [HF dataset repo](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-gap-factorized-tal-seed42)
- Artifact path: `runs/gap_factorized_k15/seed_<seed>/`
- Report: [`docs/reports/approach+results/report_factorized_tal.md`](reports/approach+results/report_factorized_tal.md)

## Exact architecture

```text
RGB
  -> YOLOv8n P2 backbone/FPN
  -> GAP ChannelAttention on the P2 feature
  -> shared Detect head from P2 only
```

Canonical model file:

```text
models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_channel_only.yaml
```

The `cbam` part is a legacy filename from the attention ablation family. It does **not** mean full CBAM is enabled. The YAML contains only `ChannelAttention` at the P2 output. There is no spatial-attention branch.

Required topology:

- `P2 -> GAP ChannelAttention -> shared Detect`
- Detect input is P2 only
- No GGCF
- No geometry-guided candidate field
- No top-hat or other input cue
- No P2 deep supervision
- No KVCA, CRC, auxiliary loss, cue weighting, or side branch

## Exact FTAL settings

```yaml
factorized_tal_target: true
factorized_tal_tau: 0.75
factorized_tal_kappa: 1.5
factorized_tal_lambda: 0.5
factorized_tal_s_max: 32.0
factorized_tal_p2_only: true
factorized_tal_warmup_start: 5
factorized_tal_warmup_end: 15
```

The documented implementation preserves the original TAL classification normalization (`oldnorm` behavior). Do not switch to a mass-preserving or other target-mode variant when reproducing this reference.

## Exact training/evaluation protocol

```text
epochs: 100
patience: 0
imgsz: 512
batch: 8
split seed: 42
NMS IoU: 0.5
checkpoint: best.pt
```

The fixed LEVIR-Ship split is 2,320 train, 788 validation, and 788 test images.

## Canonical augmentation protocol

The canonical launcher does not override the Ultralytics augmentation hyperparameters, so the reference run uses the
standard defaults from `models_related/ultralytics/ultralytics/cfg/default.yaml`:

```yaml
mosaic: 1.0
close_mosaic: 10
translate: 0.1
scale: 0.5

## Canonical launcher

The exact runner is:

```text
train_all_levir_yolov8n_p2_gap_factorized_tal.py
```

A seed-42 reproduction command is:

```bash
python train_all_levir_yolov8n_p2_gap_factorized_tal.py \\
  --variants gap_factorized_k15 \\
  --seeds 42 \\
  --epochs 100 \\
  --imgsz 512 \\
  --batch-size 8 \\
  --hf-repo-id duyle2408/levir-yolov8n-p2-gap-factorized-tal-seed42
```

For a fresh verification run, use a separate HF repository or a separate artifact path. Do not overwrite the canonical reference without recording the reason.

## Preflight guardrails

Before starting a run, verify all of the following:

1. The model path ends in `yolov8n_p2_fpn_only_cbam_channel_only.yaml`.
2. The variant is exactly `gap_factorized_k15`.
3. `factorized_tal_kappa == 1.5`, `tau == 0.75`, and `lambda == 0.5`.
4. `epochs == 100` and `patience == 0`.
5. Evaluation uses explicit NMS IoU `0.5`.
6. The model has GAP ChannelAttention and a P2-only Detect head.
7. The config and command contain no `ggcf_*` settings.
8. The config and command contain no `InputCueConv`, `top_hat`, or other cue variant.
9. Hugging Face authentication is available before launch.
10. After evaluation, verify `best.pt`, `last.pt`, `results.csv`, `evaluation_metrics.json`, metadata, and the HF upload marker.

## Do not confuse with these experiments

These are different and must not be reported as the canonical GAP + FTAL result:

- `train_levir_scripts/train_all_levir_yolov8n_p2_gap_ftal_ggcf.py`
- `yolov8n_p2_fpn_only_gap_ggcf.yaml`
- `levir-yolov8n-p2-gap-ftal-ggcf`
- Any `top_hat_gap_ftal` or `top_hat_plain_ftal` run
- `gap_factorized_k15_lambda1`
- `gap_ftal_mass_preserve_l1`
- `gap_ftal_geometry_l05`
- `gap_ftal_agreement_gate_l05`

The `gap-ftal-ggcf` and top-hat artifacts created during the mistaken queue are invalid for this reference and must be excluded from tables, reports, and claims.

## Reference metrics

| Seed | Test precision | Test recall | Test mAP50 | Test AP75 | Test mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.8393 | 0.7882 | 0.8283 | 0.1388 | 0.3161 |
| 43 | 0.8488 | 0.7759 | 0.8277 | 0.1291 | 0.3195 |
| 44 | 0.8157 | 0.7759 | 0.8147 | 0.1402 | 0.3180 |
| Mean ± std | 0.8346 ± 0.0170 | 0.7800 ± 0.0071 | 0.8235 ± 0.0076 | 0.1360 ± 0.0060 | 0.3179 ± 0.0017 |
