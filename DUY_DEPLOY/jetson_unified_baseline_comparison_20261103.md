# Jetson unified evaluation — coverage and baseline check


Generated from `metrics_all.json` after excluding the standalone smoke result. `nms_iou=0.5`; Jetson protocol: `rect=False`, `batch=1`, `half=False`, `augment=False`; image sizes LEVIR 512, TinyPerson/Varroa 640.

## Coverage audit

- Planned: 30 baseline jobs (5 models × 3 datasets × seeds 42/43) + 14 requested augmentation jobs (YOLOv9 excluded).
- Completed: **43 jobs / 86 split rows**: 29 baseline + 14 augmentation.
- Missing: `009_baseline_levirship_yolo11n_s43`; source repo has no `runs/levirship/yolov11/seed_43/weights/best.pt`; seed44 was not substituted.
- Reused rather than downloaded from the current HF tree: TinyPerson YOLOv8 Negative Canvas `nc2` and `nc4`; these use the existing Jetson checkpoint files recorded in `checkpoint_inventory.json`.
- No missing Varroa baseline or requested augmentation job.

## Baseline comparison

Baseline reference is `docs/reports/baseline_main.md` Section 3. Queue-to-reference modes: LEVIR-Ship = No Mosaic; TinyPerson = Mosaic; Varroa = Mosaic. Reference means may use 3 seeds for LEVIR/TinyPerson YOLOv5–10, while this queue has seeds 42/43; YOLO11 reference is n=2 for these modes. Values below are mAP50 and mAP50-95; `Δ` = Jetson queue mean − report mean.

| Dataset / model | Queue n | Val AP50 | Ref | Δ | Val mAP | Ref | Δ | Test AP50 | Ref | Δ | Test mAP | Ref | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| levirship YOLOv5 (No Mosaic) | 2 | 0.7936 | 0.8252 | -0.0316 | 0.3114 | 0.3175 | -0.0061 | 0.7987 | 0.7985 | 0.0002 | 0.3041 | 0.3020 | 0.0021 |
| levirship YOLOv8 (No Mosaic) | 2 | 0.7975 | 0.8319 | -0.0344 | 0.3131 | 0.3253 | -0.0122 | 0.7910 | 0.7956 | -0.0046 | 0.3051 | 0.3022 | 0.0029 |
| levirship YOLOv9 (No Mosaic) | 2 | 0.8170 | 0.8311 | -0.0141 | 0.3236 | 0.3284 | -0.0048 | 0.7837 | 0.8077 | -0.0240 | 0.2971 | 0.3075 | -0.0104 |
| levirship YOLOv10 (No Mosaic) | 2 | 0.7029 | 0.7883 | -0.0854 | 0.2741 | 0.3243 | -0.0502 | 0.6976 | 0.7493 | -0.0517 | 0.2731 | 0.3089 | -0.0358 |
| levirship YOLO11 (No Mosaic) | 1 | 0.7911 | 0.8373 | -0.0462 | 0.3054 | 0.3317 | -0.0263 | 0.7775 | 0.8074 | -0.0299 | 0.2930 | 0.3122 | -0.0192 |
| tinyperson YOLOv5 (Mosaic) | 2 | 0.5530 | 0.5770 | -0.0240 | 0.2060 | 0.2211 | -0.0151 | 0.4828 | 0.4945 | -0.0117 | 0.1705 | 0.1760 | -0.0055 |
| tinyperson YOLOv8 (Mosaic) | 2 | 0.5747 | 0.6033 | -0.0286 | 0.2231 | 0.2395 | -0.0164 | 0.4971 | 0.5036 | -0.0065 | 0.1765 | 0.1796 | -0.0031 |
| tinyperson YOLOv9 (Mosaic) | 2 | 0.5368 | 0.5684 | -0.0316 | 0.1977 | 0.2177 | -0.0200 | 0.5016 | 0.5097 | -0.0081 | 0.1816 | 0.1844 | -0.0028 |
| tinyperson YOLOv10 (Mosaic) | 2 | 0.5535 | 0.5751 | -0.0216 | 0.2190 | 0.2315 | -0.0125 | 0.4822 | 0.4814 | 0.0008 | 0.1803 | 0.1803 | -0.0000 |
| tinyperson YOLO11 (Mosaic) | 2 | 0.5868 | 0.5918 | -0.0050 | 0.2253 | 0.2248 | 0.0005 | 0.5203 | 0.5211 | -0.0008 | 0.1877 | 0.1872 | 0.0005 |
| varroa YOLOv5 (Mosaic) | 2 | 0.9202 | 0.9225 | -0.0023 | 0.3317 | 0.3310 | 0.0007 | 0.9083 | 0.9113 | -0.0030 | 0.3367 | 0.3349 | 0.0018 |
| varroa YOLOv8 (Mosaic) | 2 | 0.9138 | 0.9109 | 0.0029 | 0.3295 | 0.3287 | 0.0008 | 0.9009 | 0.9023 | -0.0014 | 0.3322 | 0.3313 | 0.0009 |
| varroa YOLOv9 (Mosaic) | 2 | 0.8995 | 0.8967 | 0.0028 | 0.3274 | 0.3259 | 0.0015 | 0.8915 | 0.8865 | 0.0050 | 0.3197 | 0.3183 | 0.0014 |
| varroa YOLOv10 (Mosaic) | 2 | 0.8480 | 0.8519 | -0.0039 | 0.3257 | 0.3273 | -0.0016 | 0.8647 | 0.8614 | 0.0033 | 0.3224 | 0.3212 | 0.0012 |
| varroa YOLO11 (Mosaic) | 2 | 0.9110 | 0.9131 | -0.0021 | 0.3323 | 0.3341 | -0.0018 | 0.8977 | 0.9022 | -0.0045 | 0.3271 | 0.3293 | -0.0022 |

## Interpretation

- **Varroa:** queue and report are close for all five models (absolute mAP50-95 deltas ≤ 0.0022 on val/test). This is the strongest protocol/provenance match.
- **TinyPerson:** YOLO11 matches the report closely; YOLOv5–10 are lower mainly because the queue has only seeds 42/43 while the report mean uses three seeds. Native test values are being compared; the report also provides a separate merged-corner test view, which this queue did not run.
- **LEVIR-Ship:** YOLOv5/8/9 are broadly close on native test mAP, but YOLOv10 is materially lower (about −0.036 test mAP and −0.050 val mAP), so that row needs a checkpoint/provenance or evaluator investigation before treating it as verified. YOLO11 is incomplete (only seed42).
- This comparison is a sanity check against aggregate report means, not a replacement for per-seed artifact equality. Exact equality requires using the same evaluator, dataset manifest, and test protocol as each source artifact.

## Required follow-up before calling the matrix complete

1. Supply or regenerate LEVIR YOLO11 seed43 from the exact source checkpoint.
2. If the deliverable requires the authoritative TinyPerson merged-corner metric, run the merged evaluator and add `test_merged` rows; current Jetson results are native full-image `test` only.
3. Investigate LEVIR YOLOv10 (and, secondarily, the seed-count differences) using the exact source `evaluation_metrics.json` and a raw-output comparison on one fixed image.

## YOLOv10 LEVIR diagnosis checkpoint

- This is **PyTorch `.pt` evaluation**, not TensorRT, so the discrepancy is not an FP16/TRT issue.
- The exact seed42 checkpoint SHA matches the Hugging Face inventory (`2270da...597b1e`).
- HF source metrics: seed42 `val mAP50-95=0.333025`, `test=0.312856`; seed43 `val=0.320893`, `test=0.308342`.
- Jetson unified metrics: seed42 `val=0.268835`, `test=0.260745`; seed43 `val=0.279336`, `test=0.285519`.
- Both seeds are low, so this is not explained by seed averaging.
- Source `args.yaml` confirms the same relevant protocol settings: `imgsz=512`, `rect=false`, `mosaic=0.0`, YOLOv10n, one-class LEVIR data. Jetson runtime is Ultralytics `8.4.137`, matching the pinned project version.
- Initial conclusion was too narrow: seed44 shows the mismatch is not isolated to YOLOv10. LEVIR YOLOv9 and YOLO11 also diverge, while YOLOv5 is close and YOLOv8 is moderately lower. Keep YOLOv10 as the largest outlier, but investigate the LEVIR evaluation path globally.


Seed44 changes the diagnosis: the LEVIR discrepancy is broader than YOLOv10. YOLOv9 has a test mAP delta of `−0.030635`, YOLOv10 `−0.033547`, and YOLO11 `−0.014744`; YOLOv5 is close. TinyPerson seed44 native test remains close to HF despite higher validation values. The next comparison should use one fixed LEVIR image and raw outputs for YOLOv5/v9/v10, not YOLOv10 alone.

## Artifacts

- Raw metrics: `DUY_DEPLOY/results/unified_pt_jetson_seed42_43_20261103/metrics_all.json` (43 jobs, 86 rows).
- Detailed summary: `DUY_DEPLOY/results/unified_pt_jetson_seed42_43_20261103/metrics_summary.md`.
- Checkpoint provenance: `DUY_DEPLOY/results/unified_pt_jetson_seed42_43_20261103/checkpoint_inventory.json`.
- Run contract: `DUY_DEPLOY/results/unified_pt_jetson_seed42_43_20261103/run_manifest.json`.

## Seed44 follow-up (2026-11-03)

Seed44 was evaluated wherever the source checkpoint exists: **9 jobs / 18 split rows**. LEVIR-Ship has all five seed44 checkpoints; TinyPerson has YOLOv5/8/9/10. TinyPerson YOLO11 and all Varroa Mosaic seed44 checkpoints are absent from their HF source repositories.

| Dataset / model | Jetson val mAP | HF val mAP | Δ | Jetson test mAP | HF test mAP | Δ |
|---|---:|---:|---:|---:|---:|---:|
| levirship YOLOv5 | 0.310676 | 0.312277 | -0.001600 | 0.301903 | 0.304093 | -0.002191 |
| levirship YOLOv8 | 0.302756 | 0.325764 | -0.023008 | 0.294141 | 0.299493 | -0.005352 |
| levirship YOLOv9 | 0.292489 | 0.330783 | -0.038294 | 0.286058 | 0.316693 | -0.030635 |
| levirship YOLOv10 | 0.271649 | 0.319102 | -0.047453 | 0.271937 | 0.305483 | -0.033547 |
| levirship YOLO11 | 0.312117 | 0.334766 | -0.022649 | 0.297469 | 0.312213 | -0.014744 |
| tinyperson YOLOv5 | 0.255246 | 0.194066 | +0.061180 | 0.188995 | 0.187764 | +0.001231 |
| tinyperson YOLOv8 | 0.272852 | 0.196754 | +0.076098 | 0.188405 | 0.188055 | +0.000350 |
| tinyperson YOLOv9 | 0.261198 | 0.193327 | +0.067871 | 0.191103 | 0.190121 | +0.000982 |
| tinyperson YOLOv10 | 0.259181 | 0.189137 | +0.070044 | 0.181470 | 0.181789 | -0.000319 |

Detailed seed44 artifact: `DUY_DEPLOY/results/unified_pt_jetson_seed44_20261103/metrics_summary.md`.

## Backend parity status

- The current matrix is `cloned_ultralytics.pt` via `evaluate_with_cloned_ultralytics()`, using `model.val()` with `rect=False`, `batch=1`, `half=False`, `augment=False`.
- The historical TensorRT bundle was inspected and then deleted: it had 29 `DRY-RUN` logs, empty accuracy directories, and zero JSON metric files. It was a command record, not a completed TensorRT accuracy result.
- Therefore no numeric TRT-vs-PT delta can be computed from the saved artifacts. Backend/evaluator parity is **not proven**; a real `.pt`/`.engine` run with the same YAML, splits, and saved raw predictions is required.

## Backend comparison audit

See `DUY_DEPLOY/backend_comparison_audit_20261104.md` for the artifact-level evidence and exact missing deliverables.
