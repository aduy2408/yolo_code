# Channel Perturbation Report

## MatchedChannelPerturbation, seed 42

Source run:

- Marimo workspace: `/marimo/yolo_code`
- Variant: `matched_channel_perturbation`
- Run: `runs/levir_yolov8n_p2_matched_channel_perturbation/matched_channel_perturbation/seed_42_continue_from_epoch59_41e`
- Checkpoint evaluated: `weights/best.pt`
- Split seed: `42`
- Train seed: `42`
- Image size: `512`
- Batch size: `8`
- NMS IoU: `0.5`

Important caveat: this run is a 41-epoch fine-tune from a prior `last.pt`
checkpoint, not an exact optimizer/scheduler resume.

## Results

Validation/test evaluation with NMS IoU `0.5`, from
`seed_42_continue_from_epoch59_41e/evaluation_metrics.json`:

| split | precision | recall | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| val | 0.8631 | 0.7821 | 0.8357 | 0.1424 | 0.3313 |
| test | 0.8274 | 0.7644 | 0.7982 | 0.1051 | 0.2985 |

## Interpretation

The fine-tuned checkpoint reaches test `mAP50-95 = 0.2985`. Because the
continuation is not an exact optimizer resume, treat this as a useful
directional continuation result, not a clean 100-epoch training run.

## RepDW5-P2 + GAP, seeds 42/43/44

Source run:

- Hugging Face dataset: `duyle2408/levir-yolov8n-p2-repdw5-gap-seed42`
- Variant: `repdw5_gap`
- Remote path pattern: `runs/repdw5_gap/seed_<seed>/`
- Topology: `P2 -> ResidualDWConv5(alpha=0.1) -> ChannelAttention(avg) -> Detect([P2])`
- Split seed: `42`
- Train seeds: `42`, `43`, `44`
- Epochs: `100`
- Image size: `512`
- Batch size: `8`
- Checkpoint evaluated: `weights/best.pt`
- NMS IoU: `0.5`

All three seeds have verified remote `best.pt`, `last.pt`, `results.csv`,
`args.yaml`, `evaluation_metrics.json`, `config.yaml`,
`experiment_manifest.json`, and `upload_complete.json`.

### Results

Held-out test evaluation with NMS IoU `0.5`:

| seed | precision | recall | mAP50 | mAP75 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.8256 | 0.8003 | 0.8255 | 0.1184 | 0.3044 |
| 43 | 0.8234 | 0.7428 | 0.7968 | 0.1387 | 0.3069 |
| 44 | 0.8258 | 0.7833 | 0.7999 | 0.1282 | 0.3132 |
| mean +/- std | 0.8249 +/- 0.0014 | 0.7755 +/- 0.0295 | 0.8074 +/- 0.0157 | 0.1284 +/- 0.0102 | 0.3082 +/- 0.0045 |

Validation mean over the same three seeds:

| precision | recall | mAP50 | mAP75 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.8652 | 0.7821 | 0.8425 | 0.1498 | 0.3337 |

### Interpretation

The seed-42 AP50 spike was not fully stable: the 3-seed test AP50 mean is
`0.8074 +/- 0.0157`. Test `mAP50-95` is steadier at `0.3082 +/- 0.0045`,
but AP75 remains low (`0.1284 +/- 0.0102`). This supports the narrow reading
that local depthwise mixing before GAP can help detectability/score behavior,
but it does not fix tight localization.
