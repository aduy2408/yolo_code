# Baseline Main Report: YOLOv5/v8/v9/v10/v11

**Updated:** 2026-10-05  
**Scope:** Verified YOLO baseline re-evaluation on LEVIR-Ship, TinyPerson, and Varroa, covering Mosaic and no-Mosaic/MuSGD artifacts.

## 1. Executive summary

- Local evaluation bundles verified: **83/83** requested baseline jobs, including the recovered Varroa Mosaic two-seed slice.
- Hugging Face completion, metrics, and manifests independently verified: **83/83** for the consolidated baseline report, with **10/10** additional Varroa Mosaic jobs verified in `duyle2408/varroa-yolo-baselines-mosaic-two-seed-runs`.
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
| Varroa | Mosaic, re-evaluated | 2 | 2 | 2 | 2 | 2 | 10 |
| Varroa | No Mosaic / MuSGD, re-evaluated | 3 | 3 | 3 | 3 | 3 | 15 |
| **Re-evaluated total** |  |  |  |  |  |  | **83** |

The Varroa Mosaic two-seed recovery queue is now complete for the requested five model families. It uses `duyle2408/varroa-yolo-baselines-mosaic-two-seed-runs`, with seeds `42, 43`, split seed `42`, `mosaic: 1.0`, `close_mosaic: 10`, `imgsz: 640`, and native held-out Varroa test evaluation. All 10 prefixes have split-qualified metrics and verified upload markers.

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
| Varroa | Mosaic | YOLOv5 | 2 | 0.9225 ± 0.0141 | 0.1025 ± 0.0106 | 0.3310 ± 0.0037 | 0.8961 ± 0.0115 | 0.0961 ± 0.0140 | 0.9113 ± 0.0037 | 0.1186 ± 0.0082 | 0.3349 ± 0.0001 | 0.8704 ± 0.0023 | 0.1076 ± 0.0132 |
| Varroa | Mosaic | YOLOv8 | 2 | 0.9109 ± 0.0117 | 0.0949 ± 0.0082 | 0.3287 ± 0.0037 | 0.8922 ± 0.0040 | 0.0933 ± 0.0017 | 0.9023 ± 0.0090 | 0.1031 ± 0.0072 | 0.3313 ± 0.0082 | 0.8727 ± 0.0221 | 0.0973 ± 0.0139 |
| Varroa | Mosaic | YOLOv9 | 2 | 0.8967 ± 0.0007 | 0.1057 ± 0.0043 | 0.3259 ± 0.0035 | 0.8825 ± 0.0231 | 0.0954 ± 0.0072 | 0.8865 ± 0.0001 | 0.0968 ± 0.0123 | 0.3183 ± 0.0106 | 0.8685 ± 0.0124 | 0.0909 ± 0.0070 |
| Varroa | Mosaic | YOLOv10 | 2 | 0.8519 ± 0.0594 | 0.1371 ± 0.0071 | 0.3273 ± 0.0159 | 0.7791 ± 0.1125 | 0.0967 ± 0.0114 | 0.8614 ± 0.0283 | 0.1314 ± 0.0105 | 0.3212 ± 0.0014 | 0.7752 ± 0.0710 | 0.0965 ± 0.0016 |
| Varroa | Mosaic | YOLO11 | 2 | 0.9131 ± 0.0044 | 0.1116 ± 0.0000 | 0.3341 ± 0.0072 | 0.8866 ± 0.0033 | 0.1053 ± 0.0084 | 0.9022 ± 0.0125 | 0.1084 ± 0.0070 | 0.3293 ± 0.0014 | 0.8624 ± 0.0012 | 0.1035 ± 0.0014 |

## 4. Per-seed artifact source

The complete per-seed table is preserved by the verified artifacts in the task-specific Hugging Face repository. Each row is stored under the source repository/prefix in:

- `evaluation_metrics.json`
- `evaluation_manifest.json`
- `evaluation_complete.json`

The HF audit found `83/83` completion, metric, and manifest artifacts in the consolidated report scope. The recovered Varroa Mosaic repository independently contains `10/10` `evaluation_metrics.json`, `10/10` `experiment_manifest.json`, and `10/10` `upload_complete.json` files.

## 5. Provenance and protocols

- Split seed: `42`; NMS IoU: `0.50`.
- Image sizes: LEVIR-Ship `512`, TinyPerson `640`, Varroa `640`.
- Source repository and prefix are preserved in every evaluation manifest.
- Output repository: `duyle2408/yolo-baseline-valtest-ap75-small-runs`.
- Varroa Mosaic re-evaluation repository: `duyle2408/varroa-yolo-baselines-mosaic-two-seed-runs`.
- Varroa Mosaic source archives: `duyle2408/varroa-yolo-baselines-part1-full`, `duyle2408/varroa-yolo-baselines-part2-full`, and `duyle2408/varroa-yolo-baselines-missing-part3-missing`. Their uploaded `args.yaml` files confirm `mosaic: 1.0` and `close_mosaic: 10` for the requested nano/tiny families and seeds.
- No-Mosaic/MuSGD source: `duyle2408/yolo-baselines-no-mosaic-musgd-runs`.
- Varroa Mosaic metrics were read from uploaded `evaluation_metrics.json` files, not reconstructed from plots. All ten files contain the required `val/test` AP50, AP75, mAP50-95, and native size-bucket AP50/AP75 fields.

## 6. Artifact acceptance

- Local marker: `evaluation_complete.json`.
- Local metrics: `evaluation_metrics.json`.
- HF manifest: `evaluation_manifest.json`.
- Final HF audit: `83/83` completion markers, `83/83` metric files, and `83/83` manifests in the consolidated report scope. The Varroa Mosaic recovery slice is `10/10` for each artifact type.

## 7. Related reports

- [`yolo_baselines_three_datasets_mosaic_matrix.md`](yolo_baselines_three_datasets_mosaic_matrix.md)
- [`complete_main_report.md`](complete_main_report.md)
- [`augmentation_report.md`](augmentation_report.md)
- [`baseline_validation_test_recovered.md`](baseline_validation_test_recovered.md)

## 8. MMDetection baseline supplement

This supplement records the independently evaluated MMDetection baseline jobs for the same three datasets. It does not change the YOLO baseline count in Sections 1-6. The evaluation completed **26/26 jobs**, with all requested validation and test fields present: AP50, AP75, mAP50-95, and AP50-Small.

**Coverage note:** the first table is the newly evaluated missing-row supplement. It adds 24 Varroa rows, one LEVIR-Ship row (`seed42/no_mosaic/retinanet`), and one TinyPerson row (`seed43/no_mosaic/retinanet`). The merged seed statistics in Section 8.2 combine these new rows with the pre-existing LEVIR-Ship and TinyPerson rows from [`mmdetection_baseline_runs_report.md`](mmdetection_baseline_runs_report.md).
| Dataset | Protocol | Seed | Model | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-Small | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-Small |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | no_mosaic | 42 | retinanet | 0.6613 | 0.0774 | 0.2314 | 0.6618 | 0.7074 | 0.0901 | 0.2563 | 0.7034 |
| TinyPerson | no_mosaic | 43 | retinanet | 0.5591 | 0.0781 | 0.1892 | 0.5277 | 0.0567 | 0.0045 | 0.0172 | 0.0476 |
| Varroa | mosaic | 42 | atss | 0.8792 | 0.0810 | 0.3043 | 0.8457 | 0.8778 | 0.0932 | 0.3218 | 0.8434 |
| Varroa | mosaic | 42 | cascade_rcnn | 0.9007 | 0.0817 | 0.3105 | 0.8788 | 0.8929 | 0.1084 | 0.3157 | 0.8709 |
| Varroa | mosaic | 42 | faster_rcnn | 0.9183 | 0.0959 | 0.3249 | 0.9015 | 0.8886 | 0.1013 | 0.3216 | 0.8764 |
| Varroa | mosaic | 42 | fcos | 0.8899 | 0.0887 | 0.3102 | 0.8745 | 0.8939 | 0.1089 | 0.3207 | 0.8671 |
| Varroa | mosaic | 42 | retinanet | 0.8937 | 0.0994 | 0.3205 | 0.8780 | 0.8938 | 0.1105 | 0.3316 | 0.8630 |
| Varroa | mosaic | 42 | rtmdet | 0.9053 | 0.1261 | 0.3398 | 0.9094 | 0.9051 | 0.1257 | 0.3447 | 0.8795 |
| Varroa | mosaic | 43 | atss | 0.8976 | 0.0897 | 0.3110 | 0.8783 | 0.8927 | 0.0837 | 0.3132 | 0.8615 |
| Varroa | mosaic | 43 | cascade_rcnn | 0.9087 | 0.0911 | 0.3214 | 0.8918 | 0.8913 | 0.1134 | 0.3267 | 0.8743 |
| Varroa | mosaic | 43 | faster_rcnn | 0.9098 | 0.0982 | 0.3119 | 0.8865 | 0.8857 | 0.1036 | 0.3198 | 0.8741 |
| Varroa | mosaic | 43 | fcos | 0.8872 | 0.0842 | 0.3133 | 0.8472 | 0.9007 | 0.1162 | 0.3247 | 0.8623 |
| Varroa | mosaic | 43 | retinanet | 0.8898 | 0.0925 | 0.3210 | 0.8848 | 0.8954 | 0.0930 | 0.3352 | 0.8757 |
| Varroa | mosaic | 43 | rtmdet | 0.9233 | 0.1069 | 0.3340 | 0.9079 | 0.9134 | 0.1166 | 0.3289 | 0.8991 |
| Varroa | no_mosaic | 42 | atss | 0.8821 | 0.0865 | 0.3156 | 0.8532 | 0.8892 | 0.1021 | 0.3169 | 0.8547 |
| Varroa | no_mosaic | 42 | cascade_rcnn | 0.8819 | 0.0768 | 0.3055 | 0.8720 | 0.8802 | 0.1006 | 0.3128 | 0.8534 |
| Varroa | no_mosaic | 42 | faster_rcnn | 0.9016 | 0.0976 | 0.3282 | 0.8777 | 0.8929 | 0.0919 | 0.3241 | 0.8750 |
| Varroa | no_mosaic | 42 | fcos | 0.9170 | 0.0825 | 0.3195 | 0.8973 | 0.8987 | 0.0997 | 0.3257 | 0.8875 |
| Varroa | no_mosaic | 42 | retinanet | 0.8953 | 0.1076 | 0.3246 | 0.8845 | 0.8819 | 0.1028 | 0.3268 | 0.8727 |
| Varroa | no_mosaic | 42 | rtmdet | 0.9316 | 0.1147 | 0.3442 | 0.9303 | 0.9048 | 0.1088 | 0.3387 | 0.8931 |
| Varroa | no_mosaic | 43 | atss | 0.8799 | 0.1054 | 0.3132 | 0.8593 | 0.8793 | 0.0859 | 0.3086 | 0.8588 |
| Varroa | no_mosaic | 43 | cascade_rcnn | 0.9007 | 0.0830 | 0.3108 | 0.8840 | 0.8774 | 0.0879 | 0.3085 | 0.8408 |
| Varroa | no_mosaic | 43 | faster_rcnn | 0.9031 | 0.1031 | 0.3171 | 0.8857 | 0.8854 | 0.0910 | 0.3108 | 0.8676 |
| Varroa | no_mosaic | 43 | fcos | 0.9168 | 0.1025 | 0.3229 | 0.9002 | 0.9133 | 0.0983 | 0.3257 | 0.8985 |
| Varroa | no_mosaic | 43 | retinanet | 0.8941 | 0.0971 | 0.3211 | 0.8862 | 0.8923 | 0.1036 | 0.3303 | 0.8673 |
| Varroa | no_mosaic | 43 | rtmdet | 0.9392 | 0.1181 | 0.3506 | 0.9321 | 0.9191 | 0.1193 | 0.3427 | 0.8986 |

### 8.1 MMDetection provenance

- Evaluation manifest: `/marimo/mmdet_eval_20261005/results_combined/metrics_manifest.json`.
- Detailed result roots: `/marimo/mmdet_eval_20261005/results_retry3/` and `/marimo/mmdet_eval_20261005/results_retry6/`.
- Evaluator: `mmdetection/evaluate_missing_baseline_metrics.py`, source commit `5ae6ca63`.
- Runner: `/marimo/mmdet-venv/bin/python` through `utils.marimo_ops`.
- Split seed: `42`. Training seeds represented: `42` and `43`. NMS IoU: `0.50`.
- Metric source: MMDetection inference output converted to COCO predictions and scored with `pycocotools`. TinyPerson test used the same external COCO scorer because the native evaluator had a category mapping mismatch.
- This evaluation was local to the Marimo server. No Hugging Face upload was performed.

### 8.2 Merged LEVIR-Ship and TinyPerson seed statistics

These mean ± sample standard deviation values use both training seeds (`42`, `43`). They merge the pre-existing rows from `mmdetection_baseline_runs_report.md` with the two newly evaluated RetinaNet rows. Values are fractions, not percentages.

| Dataset | Protocol | Model | n | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-Small | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-Small |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | mosaic | atss | 2 | 0.7548 ± 0.0087 | 0.0911 ± 0.0061 | 0.2631 ± 0.0022 | 0.7526 ± 0.0070 | 0.7514 ± 0.0059 | 0.1169 ± 0.0060 | 0.2842 ± 0.0034 | 0.7490 ± 0.0103 |
| LEVIR-Ship | mosaic | cascade_rcnn | 2 | 0.7476 ± 0.0170 | 0.1163 ± 0.0018 | 0.2775 ± 0.0105 | 0.7503 ± 0.0156 | 0.6804 ± 0.0459 | 0.1096 ± 0.0180 | 0.2515 ± 0.0238 | 0.6750 ± 0.0425 |
| LEVIR-Ship | mosaic | faster_rcnn | 2 | 0.7415 ± 0.0217 | 0.1239 ± 0.0051 | 0.2806 ± 0.0042 | 0.7445 ± 0.0206 | 0.7046 ± 0.0037 | 0.1006 ± 0.0133 | 0.2551 ± 0.0013 | 0.7018 ± 0.0054 |
| LEVIR-Ship | mosaic | fcos | 2 | 0.7339 ± 0.0047 | 0.1008 ± 0.0144 | 0.2652 ± 0.0105 | 0.7383 ± 0.0066 | 0.7104 ± 0.0160 | 0.0847 ± 0.0013 | 0.2475 ± 0.0029 | 0.7063 ± 0.0164 |
| LEVIR-Ship | mosaic | retinanet | 2 | 0.7115 ± 0.0177 | 0.1003 ± 0.0028 | 0.2547 ± 0.0051 | 0.7135 ± 0.0199 | 0.6988 ± 0.0552 | 0.1130 ± 0.0120 | 0.2669 ± 0.0226 | 0.6993 ± 0.0563 |
| LEVIR-Ship | mosaic | rtmdet | 2 | 0.7274 ± 0.0124 | 0.1267 ± 0.0088 | 0.2827 ± 0.0031 | 0.7328 ± 0.0079 | 0.6949 ± 0.0861 | 0.0979 ± 0.0116 | 0.2605 ± 0.0412 | 0.6991 ± 0.0834 |
| LEVIR-Ship | no_mosaic | atss | 2 | 0.7318 ± 0.0074 | 0.1021 ± 0.0012 | 0.2603 ± 0.0011 | 0.7330 ± 0.0059 | 0.7614 ± 0.0141 | 0.0979 ± 0.0033 | 0.2803 ± 0.0012 | 0.7579 ± 0.0142 |
| LEVIR-Ship | no_mosaic | cascade_rcnn | 2 | 0.7365 ± 0.0141 | 0.1077 ± 0.0170 | 0.2683 ± 0.0163 | 0.7415 ± 0.0141 | 0.6539 ± 0.0024 | 0.0940 ± 0.0028 | 0.2366 ± 0.0079 | 0.6475 ± 0.0053 |
| LEVIR-Ship | no_mosaic | faster_rcnn | 2 | 0.7484 ± 0.0091 | 0.1237 ± 0.0115 | 0.2857 ± 0.0015 | 0.7509 ± 0.0070 | 0.7226 ± 0.0168 | 0.1057 ± 0.0340 | 0.2636 ± 0.0159 | 0.7200 ± 0.0185 |
| LEVIR-Ship | no_mosaic | fcos | 2 | 0.7139 ± 0.0124 | 0.0917 ± 0.0039 | 0.2486 ± 0.0001 | 0.7132 ± 0.0127 | 0.7343 ± 0.0366 | 0.0815 ± 0.0245 | 0.2594 ± 0.0394 | 0.7314 ± 0.0380 |
| LEVIR-Ship | no_mosaic | retinanet | 2 | 0.6899 ± 0.0403 | 0.0807 ± 0.0047 | 0.2445 ± 0.0184 | 0.6909 ± 0.0412 | 0.7075 ± 0.0001 | 0.0919 ± 0.0025 | 0.2571 ± 0.0011 | 0.7064 ± 0.0042 |
| LEVIR-Ship | no_mosaic | rtmdet | 2 | 0.6469 ± 0.0839 | 0.1425 ± 0.0066 | 0.2660 ± 0.0317 | 0.6492 ± 0.0852 | 0.6258 ± 0.0642 | 0.0990 ± 0.0266 | 0.2410 ± 0.0351 | 0.6323 ± 0.0660 |
| TinyPerson | mosaic | atss | 2 | 0.4268 ± 0.0043 | 0.0636 ± 0.0034 | 0.1490 ± 0.0014 | 0.5595 ± 0.0565 | 0.4409 ± 0.0086 | 0.0702 ± 0.0007 | 0.1533 ± 0.0024 | 0.5770 ± 0.0303 |
| TinyPerson | mosaic | cascade_rcnn | 2 | 0.5222 ± 0.0071 | 0.0985 ± 0.0008 | 0.1946 ± 0.0049 | 0.5698 ± 0.0400 | 0.4997 ± 0.0265 | 0.0965 ± 0.0030 | 0.1874 ± 0.0076 | 0.5932 ± 0.0094 |
| TinyPerson | mosaic | faster_rcnn | 2 | 0.2976 ± 0.0103 | 0.0523 ± 0.0012 | 0.1102 ± 0.0025 | 0.5862 ± 0.0057 | 0.3624 ± 0.0139 | 0.0710 ± 0.0041 | 0.1366 ± 0.0053 | 0.6212 ± 0.0037 |
| TinyPerson | mosaic | fcos | 2 | 0.4054 ± 0.0008 | 0.0559 ± 0.0004 | 0.1374 ± 0.0028 | 0.5772 ± 0.0127 | 0.4370 ± 0.0064 | 0.0658 ± 0.0049 | 0.1505 ± 0.0025 | 0.5910 ± 0.0100 |
| TinyPerson | mosaic | retinanet | 2 | 0.4506 ± 0.0013 | 0.0520 ± 0.0016 | 0.1464 ± 0.0028 | 0.5055 ± 0.0050 | 0.4415 ± 0.0100 | 0.0493 ± 0.0021 | 0.1421 ± 0.0023 | 0.5114 ± 0.0190 |
| TinyPerson | mosaic | rtmdet | 2 | 0.4982 ± 0.0102 | 0.0860 ± 0.0001 | 0.1823 ± 0.0020 | 0.5719 ± 0.0133 | 0.5197 ± 0.0009 | 0.0972 ± 0.0012 | 0.1905 ± 0.0004 | 0.6362 ± 0.0020 |
| TinyPerson | no_mosaic | atss | 2 | 0.4361 ± 0.0043 | 0.0719 ± 0.0018 | 0.1533 ± 0.0012 | 0.5802 ± 0.0262 | 0.4380 ± 0.0047 | 0.0692 ± 0.0037 | 0.1533 ± 0.0028 | 0.5862 ± 0.0134 |
| TinyPerson | no_mosaic | cascade_rcnn | 2 | 0.5157 ± 0.0071 | 0.0985 ± 0.0088 | 0.1925 ± 0.0030 | 0.5709 ± 0.0460 | 0.5167 ± 0.0040 | 0.0962 ± 0.0075 | 0.1917 ± 0.0030 | 0.6058 ± 0.0047 |
| TinyPerson | no_mosaic | faster_rcnn | 2 | 0.2994 ± 0.0139 | 0.0581 ± 0.0033 | 0.1104 ± 0.0001 | 0.5779 ± 0.0247 | 0.3588 ± 0.0141 | 0.0661 ± 0.0030 | 0.1323 ± 0.0054 | 0.6255 ± 0.0033 |
| TinyPerson | no_mosaic | fcos | 2 | 0.4212 ± 0.0081 | 0.0611 ± 0.0004 | 0.1448 ± 0.0052 | 0.5555 ± 0.0151 | 0.4290 ± 0.0021 | 0.0661 ± 0.0016 | 0.1477 ± 0.0011 | 0.5659 ± 0.0009 |
| TinyPerson | no_mosaic | retinanet | 2 | 0.5191 ± 0.0567 | 0.0665 ± 0.0164 | 0.1703 ± 0.0267 | 0.5403 ± 0.0179 | 0.2618 ± 0.2900 | 0.0254 ± 0.0296 | 0.0834 ± 0.0937 | 0.2929 ± 0.3470 |
| TinyPerson | no_mosaic | rtmdet | 2 | 0.5016 ± 0.0064 | 0.0876 ± 0.0004 | 0.1803 ± 0.0021 | 0.5933 ± 0.0271 | 0.5083 ± 0.0037 | 0.0963 ± 0.0014 | 0.1867 ± 0.0032 | 0.6143 ± 0.0035 |

### 8.3 Full LEVIR-Ship and TinyPerson per-seed rows

The following table makes the full baseline-model coverage visible in `baseline_main`. These are the existing verified rows plus the two newly evaluated RetinaNet rows. The 26-row missing-row table above is not the complete LEVIR-Ship or TinyPerson matrix by itself.

| Dataset | Protocol | Seed | Model | Val AP50 | Val AP75 | Val mAP50-95 | Val AP50-Small | Test AP50 | Test AP75 | Test mAP50-95 | Test AP50-Small |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-Ship | no_mosaic | 42 | fcos | 0.7227 | 0.0945 | 0.2487 | 0.7222 | 0.7602 | 0.0988 | 0.2872 | 0.7583 |
| LEVIR-Ship | no_mosaic | 42 | faster_rcnn | 0.7420 | 0.1318 | 0.2846 | 0.7459 | 0.7344 | 0.1297 | 0.2749 | 0.7331 |
| LEVIR-Ship | no_mosaic | 42 | cascade_rcnn | 0.7265 | 0.0957 | 0.2568 | 0.7316 | 0.6556 | 0.0959 | 0.2310 | 0.6512 |
| LEVIR-Ship | no_mosaic | 42 | rtmdet | 0.5875 | 0.1378 | 0.2435 | 0.5889 | 0.5804 | 0.0802 | 0.2162 | 0.5857 |
| LEVIR-Ship | no_mosaic | 42 | atss | 0.7370 | 0.1029 | 0.2611 | 0.7372 | 0.7714 | 0.1002 | 0.2795 | 0.7679 |
| LEVIR-Ship | no_mosaic | 42 | retinanet | 0.6899 | 0.0807 | 0.2445 | 0.6909 | 0.7075 | 0.0919 | 0.2571 | 0.7064 |
| LEVIR-Ship | mosaic | 42 | fcos | 0.7306 | 0.1109 | 0.2726 | 0.7336 | 0.6991 | 0.0837 | 0.2455 | 0.6947 |
| LEVIR-Ship | mosaic | 42 | faster_rcnn | 0.7262 | 0.1275 | 0.2776 | 0.7299 | 0.7073 | 0.1100 | 0.2560 | 0.7056 |
| LEVIR-Ship | mosaic | 42 | cascade_rcnn | 0.7596 | 0.1151 | 0.2850 | 0.7614 | 0.6479 | 0.0968 | 0.2347 | 0.6450 |
| LEVIR-Ship | mosaic | 42 | rtmdet | 0.7186 | 0.1205 | 0.2805 | 0.7272 | 0.6341 | 0.0897 | 0.2313 | 0.6402 |
| LEVIR-Ship | mosaic | 42 | atss | 0.7486 | 0.0954 | 0.2647 | 0.7476 | 0.7556 | 0.1212 | 0.2866 | 0.7563 |
| LEVIR-Ship | mosaic | 42 | retinanet | 0.6990 | 0.1022 | 0.2511 | 0.6994 | 0.7378 | 0.1215 | 0.2829 | 0.7391 |
| LEVIR-Ship | no_mosaic | 43 | fcos | 0.7051 | 0.0890 | 0.2485 | 0.7042 | 0.7084 | 0.0642 | 0.2315 | 0.7046 |
| LEVIR-Ship | no_mosaic | 43 | faster_rcnn | 0.7548 | 0.1156 | 0.2867 | 0.7558 | 0.7107 | 0.0816 | 0.2524 | 0.7069 |
| LEVIR-Ship | no_mosaic | 43 | cascade_rcnn | 0.7465 | 0.1197 | 0.2798 | 0.7515 | 0.6522 | 0.0920 | 0.2422 | 0.6437 |
| LEVIR-Ship | no_mosaic | 43 | rtmdet | 0.7062 | 0.1472 | 0.2884 | 0.7094 | 0.6712 | 0.1178 | 0.2658 | 0.6790 |
| LEVIR-Ship | no_mosaic | 43 | atss | 0.7265 | 0.1012 | 0.2596 | 0.7289 | 0.7514 | 0.0956 | 0.2812 | 0.7478 |
| LEVIR-Ship | no_mosaic | 43 | retinanet | 0.7184 | 0.0840 | 0.2575 | 0.7200 | 0.7076 | 0.0936 | 0.2579 | 0.7093 |
| LEVIR-Ship | mosaic | 43 | fcos | 0.7372 | 0.0906 | 0.2578 | 0.7430 | 0.7217 | 0.0856 | 0.2496 | 0.7179 |
| LEVIR-Ship | mosaic | 43 | faster_rcnn | 0.7569 | 0.1203 | 0.2836 | 0.7591 | 0.7020 | 0.0912 | 0.2542 | 0.6980 |
| LEVIR-Ship | mosaic | 43 | cascade_rcnn | 0.7355 | 0.1176 | 0.2701 | 0.7393 | 0.7128 | 0.1223 | 0.2684 | 0.7051 |
| LEVIR-Ship | mosaic | 43 | rtmdet | 0.7362 | 0.1329 | 0.2849 | 0.7384 | 0.7558 | 0.1061 | 0.2896 | 0.7581 |
| LEVIR-Ship | mosaic | 43 | atss | 0.7609 | 0.0868 | 0.2616 | 0.7575 | 0.7472 | 0.1127 | 0.2818 | 0.7417 |
| LEVIR-Ship | mosaic | 43 | retinanet | 0.7241 | 0.0983 | 0.2583 | 0.7276 | 0.6598 | 0.1046 | 0.2509 | 0.6595 |
| TinyPerson | no_mosaic | 42 | fcos | 0.4154 | 0.0608 | 0.1411 | 0.5449 | 0.4305 | 0.0672 | 0.1484 | 0.5666 |
| TinyPerson | no_mosaic | 42 | faster_rcnn | 0.3092 | 0.0557 | 0.1105 | 0.5954 | 0.3688 | 0.0682 | 0.1361 | 0.6278 |
| TinyPerson | no_mosaic | 42 | cascade_rcnn | 0.5106 | 0.0923 | 0.1903 | 0.6034 | 0.5139 | 0.0909 | 0.1896 | 0.6025 |
| TinyPerson | no_mosaic | 42 | rtmdet | 0.5061 | 0.0873 | 0.1818 | 0.6124 | 0.5056 | 0.0953 | 0.1844 | 0.6118 |
| TinyPerson | no_mosaic | 42 | atss | 0.4392 | 0.0706 | 0.1525 | 0.5617 | 0.4413 | 0.0718 | 0.1553 | 0.5768 |
| TinyPerson | no_mosaic | 42 | retinanet | 0.4790 | 0.0549 | 0.1514 | 0.5530 | 0.4668 | 0.0463 | 0.1497 | 0.5383 |
| TinyPerson | mosaic | 42 | fcos | 0.4060 | 0.0557 | 0.1394 | 0.5862 | 0.4415 | 0.0693 | 0.1523 | 0.5981 |
| TinyPerson | mosaic | 42 | faster_rcnn | 0.2904 | 0.0515 | 0.1084 | 0.5822 | 0.3526 | 0.0681 | 0.1329 | 0.6185 |
| TinyPerson | mosaic | 42 | cascade_rcnn | 0.5272 | 0.0979 | 0.1981 | 0.5981 | 0.4810 | 0.0944 | 0.1820 | 0.5865 |
| TinyPerson | mosaic | 42 | rtmdet | 0.5054 | 0.0861 | 0.1837 | 0.5813 | 0.5191 | 0.0980 | 0.1902 | 0.6348 |
| TinyPerson | mosaic | 42 | atss | 0.4298 | 0.0660 | 0.1500 | 0.5995 | 0.4469 | 0.0707 | 0.1550 | 0.5984 |
| TinyPerson | mosaic | 42 | retinanet | 0.4515 | 0.0532 | 0.1484 | 0.5020 | 0.4344 | 0.0478 | 0.1405 | 0.4980 |
| TinyPerson | no_mosaic | 43 | fcos | 0.4269 | 0.0614 | 0.1485 | 0.5662 | 0.4276 | 0.0650 | 0.1469 | 0.5653 |
| TinyPerson | no_mosaic | 43 | faster_rcnn | 0.2895 | 0.0604 | 0.1103 | 0.5605 | 0.3488 | 0.0640 | 0.1285 | 0.6231 |
| TinyPerson | no_mosaic | 43 | cascade_rcnn | 0.5207 | 0.1047 | 0.1946 | 0.5384 | 0.5195 | 0.1015 | 0.1938 | 0.6092 |
| TinyPerson | no_mosaic | 43 | rtmdet | 0.4971 | 0.0878 | 0.1788 | 0.5741 | 0.5109 | 0.0973 | 0.1889 | 0.6168 |
| TinyPerson | no_mosaic | 43 | atss | 0.4331 | 0.0732 | 0.1542 | 0.5988 | 0.4347 | 0.0666 | 0.1514 | 0.5957 |
| TinyPerson | no_mosaic | 43 | retinanet | 0.5591 | 0.0781 | 0.1892 | 0.5277 | 0.0567 | 0.0045 | 0.0172 | 0.0476 |
| TinyPerson | mosaic | 43 | fcos | 0.4048 | 0.0562 | 0.1354 | 0.5682 | 0.4325 | 0.0623 | 0.1488 | 0.5839 |
| TinyPerson | mosaic | 43 | faster_rcnn | 0.3049 | 0.0532 | 0.1120 | 0.5902 | 0.3722 | 0.0739 | 0.1404 | 0.6238 |
| TinyPerson | mosaic | 43 | cascade_rcnn | 0.5172 | 0.0991 | 0.1912 | 0.5416 | 0.5185 | 0.0986 | 0.1928 | 0.5998 |
| TinyPerson | mosaic | 43 | rtmdet | 0.4910 | 0.0859 | 0.1809 | 0.5625 | 0.5204 | 0.0963 | 0.1907 | 0.6376 |
| TinyPerson | mosaic | 43 | atss | 0.4237 | 0.0612 | 0.1480 | 0.5196 | 0.4348 | 0.0697 | 0.1516 | 0.5555 |
| TinyPerson | mosaic | 43 | retinanet | 0.4496 | 0.0509 | 0.1444 | 0.5091 | 0.4485 | 0.0508 | 0.1437 | 0.5249 |
