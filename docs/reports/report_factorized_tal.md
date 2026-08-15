# Factorized TAL Target Cho GAP P2

## Protocol

- Model: `yolov8n_p2_fpn_only_cbam_channel_only.yaml`
- Topology: `P2 -> GAP ChannelAttention -> shared Detect`
- Code commits: `a68b00e` for seed-42 `k15/k20`; `fca5b73` for component ablations; `f64d762` for seed 43/44 diagnostic resume; `57e6593` for TAL ranking reliability diagnostic
- Dataset: LEVIR-Ship fixed split seed 42
- Train: 100 epochs, `imgsz=512`, `batch=8`; seed 42 for all variants, plus seed 43/44 for `gap_factorized_k15`
- Eval: `best.pt`, val/test, NMS `iou=0.5`
- HF artifact repo: `duyle2408/levir-yolov8n-p2-gap-factorized-tal-seed42`

Factorized TAL keeps the original TAL denominator for classification loss and only changes the positive classification target after TAL:

```text
q_i = u_max * r_i
q_new = u_max^tau * r_i^kappa
q_final = q_i + lambda * (q_new - q_i)
```

Run variants:

| Variant | tau | kappa | lambda | size gate |
| :--- | ---: | ---: | ---: | :--- |
| `gap_factorized_ceiling` | 0.75 | 1.0 | 0.5 | GT size < 32 px |
| `gap_factorized_separation` | 1.0 | 1.5 | 0.5 | GT size < 32 px |
| `gap_factorized_k15` | 0.75 | 1.5 | 0.5 | GT size < 32 px |
| `gap_factorized_k20` | 0.75 | 2.0 | 0.5 | GT size < 32 px |

Baseline GAP was not rerun in this job. The reference GAP seed-42 result below is from `report_yolov8n_p2_attention.md`.

## Test Metrics

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GAP baseline | 0.8266 | 0.7701 | 0.8162 | 0.1305 | 0.3106 |
| Scale-temper mild | 0.8283 | 0.7744 | 0.8084 | 0.1213 | 0.3053 |
| Scale-temper medium | 0.8235 | 0.7615 | 0.7898 | 0.1111 | 0.2891 |
| Scale-temper strong | 0.8292 | 0.7950 | 0.8154 | 0.1144 | 0.3027 |
| Factorized TAL ceiling-only | 0.8310 | 0.7787 | 0.8189 | 0.1280 | 0.3157 |
| Factorized TAL separation-only | **0.8527** | 0.7572 | 0.8204 | 0.1216 | 0.3145 |
| Factorized TAL k=1.5 | 0.8393 | 0.7882 | **0.8283** | **0.1388** | **0.3161** |
| Factorized TAL k=2.0 | 0.8321 | 0.7773 | 0.8208 | 0.1188 | 0.3091 |

`gap_factorized_k15` is the clean detector-level winner. It improves AP50, AP75, mAP, precision, and recall over the GAP seed-42 baseline, while avoiding the AP75 drop seen in scale-temper strong.

`gap_factorized_k20` still improves AP50 over GAP, but AP75 and mAP fall back near baseline/scale-temper territory. The stronger ranking exponent looks too suppressive for this setup.

The component ablation clarifies that neither piece alone reproduces the full k=1.5 gain. Ceiling-only nearly matches full mAP50-95 but loses AP50/AP75, while separation-only raises precision and AP50 but costs recall and AP75. The full variant is the only one that improves AP50, AP75, mAP, precision, and recall together over GAP.

## Multi-Seed k=1.5

`gap_factorized_k15` was extended to seeds 43/44 under the same protocol. Seed 43 initially stopped after val/test because the raw P2 analyzer still hard-coded expected checkpoint seed 42; commit `f64d762` made the expected seed configurable, then the run resumed, reused completed training/evaluation, wrote diagnostics, uploaded seed 43, and trained/evaluated/uploaded seed 44.

| Seed | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.8393 | 0.7882 | 0.8283 | 0.1388 | 0.3161 |
| 43 | **0.8488** | 0.7759 | 0.8277 | 0.1291 | **0.3195** |
| 44 | 0.8157 | 0.7759 | 0.8147 | **0.1402** | 0.3180 |
| Mean ± std | 0.8346 ± 0.0170 | 0.7800 ± 0.0071 | 0.8235 ± 0.0076 | 0.1360 ± 0.0060 | 0.3179 ± 0.0017 |

The 3-seed result is stable on mAP50-95 (`0.3179 ± 0.0017`) and keeps AP75 above the seed-42 GAP reference on mean (`0.1360` vs `0.1305`). AP50 improves on seeds 42/43 and is slightly below the seed-42 GAP reference on seed 44, so the AP50 claim should be framed as a mean trend for k15, not a per-seed guarantee until paired GAP seeds 43/44 are available.

## Raw P2 Diagnostics

These diagnostics use decoded raw P2 candidates before score thresholding and before NMS. The test split has 696 GT boxes in the diagnostic set.

| Variant | raw P2 best IoU >= 0.5 | conf < 0.25 among good raw candidates | conf < 0.10 among good raw candidates | median good-candidate conf | max good-candidate conf |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Scale-temper mild | 681 | 447 | - | 0.1466 | - |
| Scale-temper medium | 676 | 426 | - | 0.1698 | - |
| Scale-temper strong | 679 | 340 | - | 0.2500 | - |
| Factorized TAL ceiling-only | 676 | 457 | 301 | 0.1442 | 0.6466 |
| Factorized TAL separation-only | 679 | 528 | 344 | 0.0970 | 0.6038 |
| Factorized TAL k=1.5 | 677 | 493 | 343 | 0.0984 | 0.6448 |
| Factorized TAL k=1.5 seed 43 | 675 | 500 | 346 | 0.0933 | 0.6104 |
| Factorized TAL k=1.5 seed 44 | 674 | 475 | 249 | 0.1605 | 0.6385 |
| Factorized TAL k=2.0 | 679 | 511 | 340 | 0.0997 | 0.5920 |

Ceiling-only raises median good-candidate confidence from the full factorized variants' `~0.10` to `0.1442`, but this does not translate into the best AP75. Separation-only keeps confidence low and increases the low-confidence count, yet still improves AP50 over GAP. Confidence ceiling is therefore not sufficient as the mechanism.

## Ranking Diagnostics

| Variant | IoU_topscore mean | IoU_best mean | rank_gap mean | Spearman(conf, IoU) mean | best-IoU conf median | best-IoU score rank median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Factorized TAL ceiling-only | 0.6682 | 0.7931 | 0.1249 | 0.4064 | 0.1351 | 11 |
| Factorized TAL separation-only | 0.6701 | **0.7964** | 0.1264 | 0.3866 | 0.0940 | 11 |
| Factorized TAL k=1.5 | 0.6699 | 0.7925 | 0.1226 | 0.3866 | 0.0969 | 11 |
| Factorized TAL k=1.5 seed 43 | 0.6606 | 0.7907 | 0.1301 | 0.4041 | 0.0923 | 11 |
| Factorized TAL k=1.5 seed 44 | 0.6636 | 0.7838 | 0.1201 | 0.4448 | 0.1539 | 9 |
| Factorized TAL k=2.0 | **0.6747** | 0.7933 | **0.1186** | **0.4845** | 0.0975 | 11 |

Ranking metrics do not explain the detector-level winner. Separation-only has the best raw best-IoU mean, and k=2.0 has the best raw ranking/correlation, but both lose AP75/mAP relative to full k=1.5. This supports the view that final detection needs both calibrated target scale and a not-too-collapsed positive support distribution.

## Target Support Diagnostic

This diagnostic recomputes TAL positives on the held-out test split from each variant's `best.pt`, then applies the intended full-strength target transform and summarizes P2 positives per GT.

| Variant | target mass M mean | N_eff mean | top1 / M mean | entropy mean |
| :--- | ---: | ---: | ---: | ---: |
| GAP baseline | 4.6199 | 8.2229 | 0.1900 | 2.1488 |
| Factorized TAL ceiling-only | **4.7273** | 8.2066 | 0.1915 | 2.1475 |
| Factorized TAL separation-only | 4.2320 | 7.7624 | 0.2106 | 2.1081 |
| Factorized TAL k=1.5 | 4.3343 | 7.7669 | 0.2097 | 2.1090 |
| Factorized TAL k=2.0 | 4.0339 | 7.4444 | **0.2280** | 2.0815 |

The pattern matches the expected mechanism. Ceiling-only mostly raises total supervision mass while preserving support shape. Separation-only and full k=1.5 reduce mass and increase concentration moderately. k=2.0 concentrates supervision the most and has the lowest effective support/entropy, consistent with over-suppressing positive support.

## Plain P2 + Factorized TAL

Plain P2 means `yolov8n_p2_fpn_only_plain.yaml`: `Plain P2 -> Detect`, Detect input `[18]`, stride `[4.0]`, no GAP/ChannelAttention. The new run used the same Factorized TAL setting as GAP k=1.5: `tau=0.75`, `kappa=1.5`, `lambda=0.5`, seed 42, 100 epochs, `best.pt`, val/test with NMS `iou=0.5`.

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Plain P2 baseline | **0.8049** | 0.6223 | 0.7213 | **0.1353** | **0.2815** |
| Plain P2 + Factorized TAL k=1.5 | 0.7937 | **0.7342** | **0.7621** | 0.0985 | 0.2701 |

Plain P2 + Factorized TAL is not a clean win over plain P2. It increases recall and AP50 substantially, but precision, AP75, and mAP50-95 drop. In contrast, GAP + Factorized TAL k=1.5 improves GAP on AP50/AP75/mAP. This points to a GAP-loss interaction/co-design rather than a standalone general TAL improvement.

## TAL Ranking Reliability: Plain vs GAP

This diagnostic recomputes TAL positives on the held-out test split from seed-42 `best.pt` checkpoints, using P2 positives only. For each GT, it compares TAL relative target ranking `r_i = q_i / max(q)` against decoded-box IoU `u_i`.

| Variant | support count mean | Spearman(r, IoU) mean | top-r IoU mean | oracle IoU mean | top-r regret mean | top1 agreement |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain P2 baseline | 9.6437 | 0.7038 | **0.7730** | **0.7933** | **0.0203** | 0.4799 |
| GAP baseline | **9.6451** | **0.7090** | 0.7653 | 0.7865 | 0.0212 | **0.5129** |

The reliability hypothesis is only partially supported. GAP gives a small increase in Spearman alignment and top1 agreement, which suggests TAL ordering is slightly more consistent. But top-r IoU is lower, oracle IoU is lower, and regret is not reduced. This is not strong enough to claim that GAP improves the geometry reliability of TAL positive ordering as the main mechanism.

Current interpretation should stay narrower:

> GAP + Factorized TAL is a real observed interaction, but the gain is not fully explained by better `r_i -> IoU` ranking reliability.

## Lambda 1.0 Check

One final Marimo run tested the same winning formula at full blend strength:
`tau=0.75`, `kappa=1.5`, `lambda=1.0`, GAP P2-only, seed 42, 100 epochs,
`best.pt`, val/test with explicit NMS `iou=0.5`.

Run path:
`/marimo/yolo_code/runs/levir_yolov8n_p2_gap_factorized_tal_lambda1/gap_factorized_k15_lambda1/seed_42`.
The process finished training/evaluation and remained as a zombie; no upload
manifest was present because it was launched with `--no-upload`.

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GAP + FTAL k=1.5, lambda=0.5 | 0.8393 | 0.7882 | **0.8283** | 0.1388 | **0.3161** |
| GAP + FTAL k=1.5, lambda=1.0 | 0.8250 | 0.7585 | 0.7956 | **0.1421** | 0.3107 |
| Delta lambda=1.0 - lambda=0.5 | -0.0143 | -0.0297 | -0.0327 | +0.0033 | -0.0054 |

Validation for lambda=1.0 was P/R/AP50/AP75/mAP50-95:
`0.8357 / 0.7928 / 0.8248 / 0.1392 / 0.3314`.

Readout: lambda=1.0 does not pass the detector-level gate. The slight AP75
increase does not offset the AP50, recall, and mAP50-95 drop. Do not sweep more
lambda values from this result alone; keep lambda=0.5 as the current setting.

## Target-Mode Checks

Three seed-42 checks tested whether the lambda=1 failure was due to target-mass
loss, TAL-relative ranking, or unsafe sharpening when TAL and IoU disagree. All
use the same GAP P2-only config, `tau=0.75`, `kappa=1.5`, `s_max=32`, 100
epochs, `best.pt`, and explicit NMS `iou=0.5`.

Run path:
`/marimo/yolo_code/runs/levir_yolov8n_p2_gap_ftal_target_modes`.

HF artifact repo:
`duyle2408/levir-yolov8n-p2-gap-ftal-target-modes-seed42`.

### Held-Out Test

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GAP + FTAL k=1.5, lambda=0.5 | 0.8393 | **0.7882** | **0.8283** | **0.1388** | **0.3161** |
| GAP + FTAL k=1.5, lambda=1.0 | 0.8250 | 0.7585 | 0.7956 | 0.1421 | 0.3107 |
| Mass-preserving FTAL, lambda=1.0 | **0.8431** | 0.7658 | 0.8144 | 0.1195 | 0.3062 |
| Geometry-decoupled FTAL, lambda=0.5 | 0.8216 | 0.7739 | 0.8035 | 0.1052 | 0.2941 |
| Agreement-gated FTAL, lambda=0.5 | 0.8338 | 0.7495 | 0.7981 | 0.1223 | 0.3151 |

### Validation

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Mass-preserving FTAL, lambda=1.0 | **0.8612** | 0.7776 | 0.8308 | 0.1379 | 0.3304 |
| Geometry-decoupled FTAL, lambda=0.5 | 0.8209 | 0.7579 | 0.8123 | 0.1182 | 0.3216 |
| Agreement-gated FTAL, lambda=0.5 | 0.8563 | **0.7843** | **0.8342** | **0.1433** | **0.3330** |

Readout:

- Mass preservation does not rescue the lambda=1 run. It recovers AP50 versus
  lambda=1 but loses AP75 and mAP50-95 versus the current lambda=0.5 baseline.
- Geometry-decoupled ranking is worse on every main detector metric.
- Agreement gating is closest on mAP50-95, but it still loses AP50, AP75, and
  recall versus the current baseline.

Decision: keep GAP + FTAL k=1.5 lambda=0.5 as the baseline. Do not promote
`mass_preserve`, `geometry`, or `agreement_gate` from this seed.

## Shared / Late-Decoupled Detect Head Check

This run tested whether the GAP + FTAL gain depends on forcing classification
and regression to share more late head computation. No new context, gate,
descriptor, residual verifier, or loss change was added.

Protocol:

- Base: GAP + FTAL k=1.5, lambda=0.5.
- Dataset: LEVIR-Ship fixed split seed 42.
- Train: seed 42, 100 epochs, `imgsz=512`, batch 8.
- Eval: `best.pt`, val/test, explicit NMS `iou=0.5`.
- Run path: `/marimo/yolo_code/runs/levir_yolov8n_p2_gap_ftal_shared_head`.
- HF artifact repo: `duyle2408/levir-yolov8n-p2-gap-ftal-shared-head-seed42`.

Variants:

| Variant | Head topology |
| :--- | :--- |
| `gap_ftal_decoupled` | current YOLOv8-style separate box and cls towers |
| `gap_ftal_share1` | one shared 3x3 block, then separate box/cls blocks |
| `gap_ftal_fully_shared` | two shared 3x3 blocks, then final box/cls predictors |

### Held-Out Test

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 | Params |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gap_ftal_decoupled` | **0.8393** | **0.7882** | **0.8283** | **0.1388** | **0.3161** | 1.60M |
| `gap_ftal_share1` | 0.7647 | 0.7126 | 0.7420 | 0.0985 | 0.2730 | 1.62M |
| `gap_ftal_fully_shared` | 0.8083 | 0.7471 | 0.7846 | 0.0948 | 0.2863 | 1.58M |

### Validation

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `gap_ftal_decoupled` | **0.8605** | **0.7743** | **0.8361** | **0.1515** | **0.3338** |
| `gap_ftal_share1` | 0.8429 | 0.7428 | 0.8198 | 0.1255 | 0.3149 |
| `gap_ftal_fully_shared` | 0.8446 | 0.7655 | 0.8180 | 0.1236 | 0.3100 |

Decision gate:

- `share1` fails: test mAP50-95 drops `0.3161 -> 0.2730`, AP75 drops
  `0.1388 -> 0.0985`, recall drops `0.7882 -> 0.7126`.
- `fully_shared` fails: test mAP50-95 drops `0.3161 -> 0.2863`, AP75 drops
  `0.1388 -> 0.0948`, recall drops `0.7882 -> 0.7471`.

Readout: late sharing is not a useful topology change for the current
GAP+FTAL recipe. The standard decoupled Detect head remains the reference.
This closes the shared-head line for now; if continuing the subtractive
architecture direction, the cleaner next test is learned low-rank P2
compression before Detect, not another head-sharing variant.

## Classifier-Capacity Detect Head Check

This run tested a narrower explanation for the failed shared-head result:
the default Detect head may be asymmetric because the regression tower uses
width 64 from `4 * reg_max`, while the one-class classification tower can be
only width 32. Regression depth and semantics were unchanged; only classifier
capacity was changed.

Protocol:

- Base: GAP + FTAL k=1.5, lambda=0.5.
- Dataset: LEVIR-Ship fixed split seed 42.
- Train: seed 42, 100 epochs, `imgsz=512`, batch 8.
- Eval: `best.pt`, val/test, explicit NMS `iou=0.5`.
- Run path: `/marimo/yolo_code/runs/levir_yolov8n_p2_gap_ftal_cls_capacity`.
- HF artifact repo: `duyle2408/levir-yolov8n-p2-gap-ftal-cls-capacity-seed42`.

Variants:

| Variant | Head topology |
| :--- | :--- |
| `gap_ftal_decoupled` | current reference Detect head |
| `gap_ftal_reg64_cls64_dw` | keep regression tower width 64; set cls tower width 64 with DW+PW blocks |
| `gap_ftal_reg64_cls64_fullconv` | keep regression tower width 64; set cls tower width 64 with dense Conv3x3 blocks |

### Held-Out Test

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 | Params |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gap_ftal_decoupled` | **0.8393** | **0.7882** | **0.8283** | **0.1388** | **0.3161** | 1.60M |
| `gap_ftal_reg64_cls64_dw` | 0.8306 | 0.7701 | 0.7945 | 0.1136 | 0.2922 | 1.59M |
| `gap_ftal_reg64_cls64_fullconv` | 0.8108 | 0.7513 | 0.7924 | 0.1092 | 0.2918 | 1.64M |

### Validation

| Variant | Precision | Recall | AP50 | AP75 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `gap_ftal_decoupled` | 0.8605 | 0.7743 | **0.8361** | **0.1515** | 0.3338 |
| `gap_ftal_reg64_cls64_dw` | **0.8606** | **0.7845** | 0.8337 | 0.1494 | **0.3347** |
| `gap_ftal_reg64_cls64_fullconv` | 0.8530 | 0.7458 | 0.8291 | 0.1439 | 0.3219 |

Decision gate:

- `cls64-DW` looks harmless or slightly better on validation mAP50-95
  (`0.3338 -> 0.3347`) but fails on held-out test: AP50 drops
  `0.8283 -> 0.7945`, AP75 drops `0.1388 -> 0.1136`, and mAP50-95 drops
  `0.3161 -> 0.2922`.
- `cls64-fullConv` also fails on held-out test: AP50 drops
  `0.8283 -> 0.7924`, AP75 drops `0.1388 -> 0.1092`, and mAP50-95 drops
  `0.3161 -> 0.2918`.

Readout: simply increasing the one-class classification tower capacity does
not solve the GAP+FTAL classification/ranking bottleneck, and dense Conv3x3
classification is worse than the DW+PW width-64 control. The current Detect
classifier width/separability is not the next useful architecture lever.

## Decision

Factorized TAL k=1.5 passes the main detector-level gate:

- AP50 increases over GAP: `0.8162 -> 0.8283`
- AP75 increases over GAP: `0.1305 -> 0.1388`
- mAP50-95 increases over GAP: `0.3106 -> 0.3161`
- Recall increases without precision collapse: recall `0.7701 -> 0.7882`, precision `0.8266 -> 0.8393`

Mechanism gate is mixed:

- It does not reduce low-confidence good raw P2 candidates.
- It does not raise median best-candidate confidence.
- The gain therefore should not be claimed as "confidence ceiling fixed" yet.

Component ablation:

- Ceiling-only improves mAP over GAP (`0.3106 -> 0.3157`) but loses AP75 (`0.1305 -> 0.1280`) and barely improves recall.
- Separation-only improves AP50 (`0.8162 -> 0.8204`) and precision (`0.8266 -> 0.8527`) but recall drops (`0.7701 -> 0.7572`) and AP75 drops to `0.1216`.
- Full k=1.5 remains the only clean detector-level win: AP50 `0.8283`, AP75 `0.1388`, mAP50-95 `0.3161`.

Best current interpretation:

> Factorized TAL k=1.5 is not just fixing an absolute confidence ceiling, and not just sharpening positive ranking. The useful behavior appears to be a GAP-loss interaction: ceiling-only improves target scale but lacks enough localization-quality gain; separation-only makes the classifier selective but loses support/recall; full k=1.5 balances both only when paired with GAP. The new TAL reliability diagnostic shows only a weak ordering-reliability advantage for GAP, so that should be treated as a possible contributor, not the settled mechanism.

## Component Ablation Status

Component ablations and k15 seed 43/44 completed on Marimo and uploaded verified artifacts to `duyle2408/levir-yolov8n-p2-gap-factorized-tal-seed42`:

- `runs/gap_factorized_ceiling/seed_42/...`
- `runs/gap_factorized_separation/seed_42/...`
- `runs/gap_factorized_k15/seed_43/...`
- `runs/gap_factorized_k15/seed_44/...`

The plain P2 run completed on Marimo and uploaded verified artifacts to `duyle2408/levir-yolov8n-p2-plain-factorized-tal-seed42`:

- `runs/plain_p2_factorized_k15/seed_42/...`

The TAL ranking reliability diagnostic completed on Marimo:

- `/marimo/yolo_code/runs/levir_yolov8n_p2_gap_factorized_tal/tal_r_iou_reliability/tal_r_iou_reliability_per_gt.csv`
- `/marimo/yolo_code/runs/levir_yolov8n_p2_gap_factorized_tal/tal_r_iou_reliability/tal_r_iou_reliability_summary.json`

Do not run k=2.5/3.0 or k=1.25. Because plain P2 + Factorized TAL does not improve mAP/AP75, do not claim a general TAL loss method yet. The shared-head and classifier-capacity checks both failed, so do not pursue more Detect-head topology or width variants without a new diagnostic reason.
