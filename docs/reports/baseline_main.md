# Baseline Main Report: YOLOv5/v8/v9/v10/v11

**Updated:** 2026-10-05  
**Scope:** Verified YOLO baseline re-evaluation on LEVIR-Ship, TinyPerson, and Varroa, covering Mosaic and no-Mosaic/MuSGD artifacts.

## 1. Executive summary

- Local evaluation bundles verified: **73/73** requested baseline jobs.
- Hugging Face completion, metrics, and manifests independently verified: **73/73** each.
- Every bundle contains split-qualified validation and test metrics:
  `val/test AP50`, `val/test AP75`, `val/test mAP50-95`, and native size-bucket `AP50/AP75`.
- TinyPerson `test/*` uses the official `sw640/sh512` corner-window native test protocol. Merged-corner metrics remain a separate protocol.
- Two custom P2/P2-offset checkpoints were excluded because they are not part of the requested YOLOv5/v8/v9/v10/v11 baseline families.
- A live Hugging Face audit confirms that Varroa Mosaic baseline training archives do exist, but their source repositories contain training logs and plots rather than reusable `best.pt` checkpoints or split-qualified evaluation JSON. They were therefore not part of the 73-job re-evaluation queue.

## 2. Verified coverage

| Dataset | Mode | YOLOv5 | YOLOv8 | YOLOv9 | YOLOv10 | YOLO11 | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | Mosaic | 3 | 3 | 3 | 3 | 3 | 15 |
| LEVIR-Ship | No Mosaic | 3 | 3 | 3 | 3 | 2 | 14 |
| TinyPerson | Mosaic | 3 | 3 | 3 | 3 | 2 | 14 |
| TinyPerson | No Mosaic | 3 | 3 | 3 | 3 | 3 | 15 |
| Varroa | Mosaic, source-confirmed | 3 | 3 | 3 | 3 | 3 | 15 |
| Varroa | No Mosaic / MuSGD, re-evaluated | 3 | 3 | 3 | 3 | 3 | 15 |
| **Re-evaluated total** |  |  |  |  |  |  | **73** |

The current re-evaluation queue has no Varroa Mosaic checkpoint evaluation bundle, but Varroa Mosaic training archives are real and complete for the requested five families and seeds. They are split across `duyle2408/varroa-yolo-baselines-part1-full`, `duyle2408/varroa-yolo-baselines-part2-full`, and `duyle2408/varroa-yolo-baselines-missing-part3-missing`. Their `args.yaml` files record `mosaic: 1.0`, `close_mosaic: 10`, `imgsz: 640`, and seeds `42, 43, 44` for YOLOv5n, YOLOv8n, YOLOv9t, YOLOv10n, and YOLO11n. The 73-job count below remains the count of bundles re-evaluated with the new val/test/AP75/size evaluator, not the count of all source training archives.

## 3. Mean ± std across available seeds

Sample standard deviation across the available verified seeds. `n` is the number of seed artifacts in each group. Values are fractions, not percentages.

| Dataset | Mode | Model | n | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-Small | Val AP75-Small | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-Small | Test AP75-Small |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | Mosaic | YOLOv5 | 3 | 0.7525 ± 0.0295 | 0.0973 ± 0.0220 | 0.2714 ± 0.0135 | 0.7923 ± 0.0434 | 0.0899 ± 0.0277 | 0.6928 ± 0.0587 | 0.0724 ± 0.0089 | 0.2347 ± 0.0218 | 0.7433 ± 0.0608 | 0.0748 ± 0.0120 |
| LEVIR-Ship | Mosaic | YOLOv8 | 3 | 0.8202 ± 0.0345 | 0.1267 ± 0.0248 | 0.3087 ± 0.0211 | 0.8504 ± 0.0340 | 0.1390 ± 0.0257 | 0.7457 ± 0.0400 | 0.0991 ± 0.0256 | 0.2642 ± 0.0253 | 0.7998 ± 0.0153 | 0.1023 ± 0.0180 |
| LEVIR-Ship | Mosaic | YOLOv9 | 3 | 0.8200 ± 0.0393 | 0.1426 ± 0.0226 | 0.3168 ± 0.0254 | 0.8555 ± 0.0605 | 0.1622 ± 0.0342 | 0.7737 ± 0.0460 | 0.0939 ± 0.0059 | 0.2766 ± 0.0229 | 0.8122 ± 0.0706 | 0.1063 ± 0.0068 |
| LEVIR-Ship | Mosaic | YOLOv10 | 3 | 0.8057 ± 0.0429 | 0.1833 ± 0.0076 | 0.3372 ± 0.0157 | 0.7995 ± 0.0531 | 0.1563 ± 0.0181 | 0.7353 ± 0.0421 | 0.1445 ± 0.0082 | 0.2940 ± 0.0200 | 0.7687 ± 0.0518 | 0.1267 ± 0.0083 |
| LEVIR-Ship | Mosaic | YOLO11 | 3 | 0.7952 ± 0.0118 | 0.1422 ± 0.0057 | 0.3049 ± 0.0077 | 0.8326 ± 0.0132 | 0.1564 ± 0.0184 | 0.7392 ± 0.0232 | 0.0921 ± 0.0093 | 0.2645 ± 0.0150 | 0.8046 ± 0.0219 | 0.1165 ± 0.0115 |
| LEVIR-Ship | No Mosaic | YOLOv5 | 3 | 0.8252 ± 0.0071 | 0.1379 ± 0.0073 | 0.3175 ± 0.0065 | 0.8291 ± 0.0100 | 0.1324 ± 0.0014 | 0.7985 ± 0.0134 | 0.1120 ± 0.0079 | 0.3020 ± 0.0066 | 0.8256 ± 0.0058 | 0.1248 ± 0.0122 |
| LEVIR-Ship | No Mosaic | YOLOv8 | 3 | 0.8319 ± 0.0087 | 0.1442 ± 0.0062 | 0.3253 ± 0.0067 | 0.8117 ± 0.0221 | 0.1394 ± 0.0131 | 0.7956 ± 0.0084 | 0.1226 ± 0.0071 | 0.3022 ± 0.0090 | 0.8108 ± 0.0199 | 0.1196 ± 0.0105 |
| LEVIR-Ship | No Mosaic | YOLOv9 | 3 | 0.8311 ± 0.0107 | 0.1543 ± 0.0115 | 0.3284 ± 0.0098 | 0.8476 ± 0.0273 | 0.1290 ± 0.0262 | 0.8077 ± 0.0175 | 0.1180 ± 0.0093 | 0.3075 ± 0.0080 | 0.8185 ± 0.0175 | 0.1143 ± 0.0179 |
| LEVIR-Ship | No Mosaic | YOLOv10 | 3 | 0.7883 ± 0.0087 | 0.1788 ± 0.0239 | 0.3243 ± 0.0076 | 0.7181 ± 0.0368 | 0.0874 ± 0.0112 | 0.7493 ± 0.0124 | 0.1520 ± 0.0057 | 0.3089 ± 0.0037 | 0.7317 ± 0.0299 | 0.0928 ± 0.0272 |
| LEVIR-Ship | No Mosaic | YOLO11 | 2 | 0.8373 ± 0.0074 | 0.1700 ± 0.0149 | 0.3317 ± 0.0043 | 0.8443 ± 0.0034 | 0.1391 ± 0.0127 | 0.8074 ± 0.0059 | 0.1248 ± 0.0048 | 0.3122 ± 0.0001 | 0.8284 ± 0.0213 | 0.1310 ± 0.0131 |
| TinyPerson | Mosaic | YOLOv5 | 3 | 0.5770 ± 0.1043 | 0.1253 ± 0.0528 | 0.2211 ± 0.0556 | 0.5747 ± 0.1516 | 0.1252 ± 0.0502 | 0.4945 ± 0.0411 | 0.0811 ± 0.0144 | 0.1760 ± 0.0198 | 0.4986 ± 0.0440 | 0.0797 ± 0.0138 |
| TinyPerson | Mosaic | YOLOv8 | 3 | 0.6033 ± 0.0955 | 0.1462 ± 0.0555 | 0.2395 ± 0.0563 | 0.5927 ± 0.1616 | 0.1443 ± 0.0590 | 0.5036 ± 0.0295 | 0.0843 ± 0.0131 | 0.1796 ± 0.0139 | 0.5001 ± 0.0476 | 0.0828 ± 0.0134 |
| TinyPerson | Mosaic | YOLOv9 | 3 | 0.5684 ± 0.0566 | 0.1239 ± 0.0404 | 0.2177 ± 0.0373 | 0.5635 ± 0.1027 | 0.1252 ± 0.0426 | 0.5097 ± 0.0225 | 0.0869 ± 0.0047 | 0.1844 ± 0.0081 | 0.5137 ± 0.0374 | 0.0850 ± 0.0053 |
| TinyPerson | Mosaic | YOLOv10 | 3 | 0.5751 ± 0.0607 | 0.1405 ± 0.0429 | 0.2315 ± 0.0407 | 0.5604 ± 0.1028 | 0.1220 ± 0.0390 | 0.4814 ± 0.0022 | 0.0923 ± 0.0021 | 0.1803 ± 0.0021 | 0.4768 ± 0.0072 | 0.0783 ± 0.0012 |
| TinyPerson | Mosaic | YOLO11 | 2 | 0.5918 ± 0.0669 | 0.1207 ± 0.0314 | 0.2248 ± 0.0378 | 0.5316 ± 0.0642 | 0.1205 ± 0.0302 | 0.5211 ± 0.0004 | 0.0889 ± 0.0017 | 0.1872 ± 0.0012 | 0.5132 ± 0.0015 | 0.0888 ± 0.0014 |
| TinyPerson | No Mosaic | YOLOv5 | 3 | 0.4923 ± 0.0419 | 0.0798 ± 0.0097 | 0.1707 ± 0.0171 | 0.4525 ± 0.0483 | 0.0759 ± 0.0128 | 0.4809 ± 0.0326 | 0.0756 ± 0.0113 | 0.1689 ± 0.0153 | 0.4730 ± 0.0355 | 0.0737 ± 0.0120 |
| TinyPerson | No Mosaic | YOLOv8 | 3 | 0.4963 ± 0.0311 | 0.0840 ± 0.0093 | 0.1751 ± 0.0126 | 0.4454 ± 0.0507 | 0.0831 ± 0.0108 | 0.4812 ± 0.0265 | 0.0758 ± 0.0110 | 0.1693 ± 0.0126 | 0.4737 ± 0.0312 | 0.0749 ± 0.0102 |
| TinyPerson | No Mosaic | YOLOv9 | 3 | 0.4955 ± 0.0090 | 0.0883 ± 0.0010 | 0.1771 ± 0.0027 | 0.4516 ± 0.0124 | 0.0860 ± 0.0010 | 0.4954 ± 0.0042 | 0.0822 ± 0.0026 | 0.1766 ± 0.0022 | 0.4971 ± 0.0042 | 0.0807 ± 0.0018 |
| TinyPerson | No Mosaic | YOLOv10 | 3 | 0.4645 ± 0.0092 | 0.0792 ± 0.0018 | 0.1676 ± 0.0020 | 0.3717 ± 0.0243 | 0.0672 ± 0.0034 | 0.4439 ± 0.0035 | 0.0816 ± 0.0010 | 0.1637 ± 0.0011 | 0.4332 ± 0.0069 | 0.0676 ± 0.0011 |
| TinyPerson | No Mosaic | YOLO11 | 3 | 0.5196 ± 0.0077 | 0.0910 ± 0.0044 | 0.1827 ± 0.0006 | 0.4492 ± 0.0284 | 0.0895 ± 0.0044 | 0.4889 ± 0.0042 | 0.0798 ± 0.0012 | 0.1725 ± 0.0013 | 0.4838 ± 0.0074 | 0.0788 ± 0.0010 |
| Varroa | No Mosaic | YOLOv5 | 3 | 0.9074 ± 0.0101 | 0.1024 ± 0.0091 | 0.3261 ± 0.0087 | 0.8246 ± 0.0296 | 0.0967 ± 0.0066 | 0.8940 ± 0.0089 | 0.0998 ± 0.0059 | 0.3289 ± 0.0049 | 0.8045 ± 0.0151 | 0.0954 ± 0.0045 |
| Varroa | No Mosaic | YOLOv8 | 3 | 0.8952 ± 0.0057 | 0.1034 ± 0.0070 | 0.3256 ± 0.0033 | 0.8441 ± 0.0328 | 0.0980 ± 0.0093 | 0.8973 ± 0.0161 | 0.1094 ± 0.0115 | 0.3294 ± 0.0054 | 0.8393 ± 0.0421 | 0.1042 ± 0.0105 |
| Varroa | No Mosaic | YOLOv9 | 3 | 0.9144 ± 0.0091 | 0.1031 ± 0.0040 | 0.3303 ± 0.0054 | 0.8742 ± 0.0174 | 0.0956 ± 0.0042 | 0.8995 ± 0.0024 | 0.1137 ± 0.0072 | 0.3304 ± 0.0021 | 0.8619 ± 0.0171 | 0.1076 ± 0.0151 |
| Varroa | No Mosaic | YOLOv10 | 3 | 0.8285 ± 0.0258 | 0.1128 ± 0.0150 | 0.3043 ± 0.0146 | 0.7202 ± 0.0477 | 0.0790 ± 0.0206 | 0.8327 ± 0.0141 | 0.1316 ± 0.0026 | 0.3161 ± 0.0102 | 0.6972 ± 0.0510 | 0.0919 ± 0.0058 |
| Varroa | No Mosaic | YOLO11 | 3 | 0.8907 ± 0.0264 | 0.1111 ± 0.0112 | 0.3230 ± 0.0059 | 0.7763 ± 0.0591 | 0.0990 ± 0.0121 | 0.8954 ± 0.0112 | 0.1138 ± 0.0136 | 0.3294 ± 0.0082 | 0.7851 ± 0.0657 | 0.1046 ± 0.0058 |

## 4. Per-seed artifact source

The complete per-seed table is preserved by the verified artifacts in the task-specific Hugging Face repository. Each row is stored under the source repository/prefix in:

- `evaluation_metrics.json`
- `evaluation_manifest.json`
- `evaluation_complete.json`

The HF audit found `73/73` of each artifact type.

## 5. Provenance and protocols

- Split seed: `42`; NMS IoU: `0.50`.
- Image sizes: LEVIR-Ship `512`, TinyPerson `640`, Varroa `640`.
- Source repository and prefix are preserved in every evaluation manifest.
- Output repository: `duyle2408/yolo-baseline-valtest-ap75-small-runs`.
- Varroa Mosaic source archives: `duyle2408/varroa-yolo-baselines-part1-full`, `duyle2408/varroa-yolo-baselines-part2-full`, and `duyle2408/varroa-yolo-baselines-missing-part3-missing`. Their uploaded `args.yaml` files confirm `mosaic: 1.0` and `close_mosaic: 10` for the requested nano/tiny families and seeds.
- No-Mosaic/MuSGD source: `duyle2408/yolo-baselines-no-mosaic-musgd-runs`.
- Limitation: the Varroa Mosaic source archives currently expose `args.yaml`, `results.csv`, test summaries, and plots, but no reusable `weights/best.pt`. The full split-qualified AP75 and size-bucket re-evaluation for that slice remains pending rather than being reconstructed from plots or unlabeled summaries.

## 6. Artifact acceptance

- Local marker: `evaluation_complete.json`.
- Local metrics: `evaluation_metrics.json`.
- HF manifest: `evaluation_manifest.json`.
- Final HF audit: `73/73` completion markers, `73/73` metric files, and `73/73` manifests.

## 7. Related reports

- [`yolo_baselines_three_datasets_mosaic_matrix.md`](yolo_baselines_three_datasets_mosaic_matrix.md)
- [`complete_main_report.md`](complete_main_report.md)
- [`augmentation_report.md`](augmentation_report.md)
- [`baseline_validation_test_recovered.md`](baseline_validation_test_recovered.md)
