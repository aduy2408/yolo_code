# TAL / RANKING session audit report

**Audit window:** 2026-09-28 through 2026-09-30  
**Prepared:** 2026-10-01  
**Scope:** Các session gần đây có nội dung về Task-Aligned Learning/Assignment, responsibility target, ranking loss, R1/J1, và selector probes.  
**Primary session:** `session_bug_1790644563239_21e423159b4d78ff`  
**Related sessions:** `session_goat_1790530029129_fd1fd0993d3be8cc`, `session_horse_1790600460330_9e8d4cf1b1ac9731`, `session_otter_1790770140377_e96f5567febc0b0b`.

## 1. Executive summary

1. Hai run responsibility/KL đã hoàn tất trên TinyPerson. `K1 kl_tiny` tốt hơn `K2 kl_curriculum` trên cả bốn metric standard split, nhưng cả hai không phải là hướng ranking chính hiện tại.
2. R1 và J1 đã được implement, sửa theo hypothesis correction-only, test/compile đầy đủ, rồi train/evaluate trên YOLOv8n P3/P4/P5. J1 cao hơn R1 rất nhẹ trên cả bốn metric standard split.
3. Probe A/B và Probe C/C2 cho thấy bottleneck không còn nằm chủ yếu ở việc P2 candidate tốt bị loại khỏi TAL top-k. Với pool đông hơn 10 candidate, failure chính là final classification ranking.
4. Probe C hậu-TAL cho thấy frozen ranker chỉ cải thiện rất nhỏ hoặc không ổn định. Probe C2 trước top-k xác nhận candidate pool P2/P3/P4 có coverage tốt hơn rõ rệt nhưng TAL vẫn không phải bottleneck chính trong cohort `eligible <= 10`.
5. P2/P3/P4 R1 đã được launch lại sau khi sửa confirmation gate. R1 P2/P3/P4 cuối cùng hoàn tất với standard metrics `val/AP50=0.55368`, `val/mAP50-95=0.20086`, `test/AP50=0.51816`, `test/mAP50-95=0.18917`, và merged `test_merged/AP50-Small=0.66977`. Đây là detector P2/P3/P4, không được trộn với R1/J1 P3/P4/P5.
6. J1 P2/P3/P4 đã được setup/preflight nhưng trong session audit không có kết quả train/evaluation cuối cùng đã verify. Không được ghi J1 P2/P3/P4 như một run hoàn tất.

## 2. Provenance chung của TinyPerson ranking runs

- Dataset: TinyPerson.
- Split seed: `42`; training seed: `42`.
- Standard corner-window protocol: `sw640/sh512`, input 640.
- Standard evaluation: NMS IoU `0.50`.
- Các run train chính: 100 epochs, patience `0`, workers `8`, batch `8`.
- Với R1/J1 P3/P4/P5: detector YOLOv8n canonical P3/P4/P5, non-mosaic.
- Với R1/J1 P2/P3/P4: YAML `yolov8n_tinyperson_p2p3p4_plain.yaml`, non-mosaic.
- Tiny-object gate của ranking loss: `max(width, height) <= 16` ở input 640.
- TAL parameters trong các probe: `topk=10`, `alpha=0.5`, `beta=6.0`.
- Các metric standard được ghi tách rõ `val/*` và `test/*`. Merged metrics chỉ ghi khi artifact evaluator thật sự có mặt.

## 3. K1/K2 responsibility experiments

### 3.1 Kết quả cuối đã verify

| Run | Mode | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| K1 | `kl_tiny` | **0.486215** | **0.170398** | **0.478620** | **0.168225** |
| K2 | `kl_curriculum` | 0.479548 | 0.168147 | 0.475990 | 0.168085 |

K1 cao hơn K2 trên cả bốn metric:

- `val/AP50`: +0.006667
- `val/mAP50-95`: +0.002251
- `test/AP50`: +0.002630
- `test/mAP50-95`: +0.000140

### 3.2 Execution/provenance

- Cả hai run chạy đủ 100 epochs.
- Artifacts, evaluation metrics, test protocol và HF upload được verify.
- K2 dùng curriculum riêng, không còn rơi về default warmup/ramp `1/1`.
- `responsibility_metrics` được ghi vào `results.csv` với prefix `resp_*`.
- Trong session có một K1 v2 bị dừng theo yêu cầu trước khi có artifact hoàn chỉnh. Run đó là interrupted/unverified và không được dùng làm kết quả.
- Các sửa implementation liên quan đã được validate bằng compile và 11 focused tests trước khi chạy lại.

## 4. R1/J1 implementation and correction-only revision

### 4.1 Hypothesis và code path

- **R1:** localization-only ranking, dùng IoU để học ordering.
- **J1:** joint correction-only ranking, dùng teacher classification score detached cộng correction localization:

```text
R_i = z_i.detach() + lambda_loc * A_i^loc
```

- Không thay đổi TAL assignment.
- Không thay đổi detector architecture trong nhánh P3/P4/P5.
- Tiny-object gate giữ `max_dim <= 16`.
- Các tham số chính của R1/J1 P3/P4/P5: `rank_loss=0.05`, `rank_tau=0.25`, `lambda_loc=0.25`, `topk=10`.

### 4.2 Validation của implementation

- Initial R1/J1 interface/config/runner validation: **14 tests passed**, compile, CLI help và diff check passed, commit `13076fe`.
- Sau review, J1 được đổi sang correction-only pairing. Mechanism tests xác nhận:
  - ranking sai nhưng localization sửa được thì loss có gradient correction;
  - ranking đã đúng thì correction loss bằng 0.
- Correction-only revision: **16 tests passed**, compile và runner/config checks passed.
- Serialization bug của `write_summaries()` được sửa ở commit `4067a21`:
  - giữ metadata string `test/protocol` và `test/source_artifact` trong `summary_runs.csv`;
  - chỉ aggregate numeric metrics;
  - 5 ranking tests, smoke test và `py_compile` passed.

## 5. R1/J1 completed runs: YOLOv8n P3/P4/P5

### 5.1 Standard split metrics

Nguồn: HF task-specific repository `duyle2408/tinyperson-reranking-runs`, prefixes:

- `runs/localization/yolov8n_base/seed_42`
- `runs/joint/yolov8n_base/seed_42`

| Run | Setup | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---|---:|---:|---:|---:|
| R1 | localization-only ranking | 0.515954 | 0.184284 | 0.496686 | 0.177407 |
| J1 | joint correction-only ranking | **0.526127** | **0.188514** | **0.498978** | **0.178672** |

J1 trội hơn R1 trên cả bốn field:

- `val/AP50`: +0.010173
- `val/mAP50-95`: +0.004230
- `test/AP50`: +0.002293
- `test/mAP50-95`: +0.001265

### 5.2 Baseline comparison caveat

Recent standard YOLOv8n baseline aggregate từ `duyle2408/tinyperson-yolo-baselines`:

| Run | Seeds | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n baseline aggregate | 42,43,44 | 0.6236 ± 0.0522 | 0.1795 ± 0.0153 | 0.4990 ± 0.0265 | 0.1782 ± 0.0129 |

Không được dùng aggregate baseline 3 seeds để claim matched-seed improvement so với R1/J1 seed 42. Report `docs/reports/tinyperson_yolov8_r1_j1_comparison.md` đã ghi caveat này.

Merged baseline artifact seed 43 có:

| Artifact | AP50 | AP75 | mAP50-75 | AP-Tiny1 | AP-Tiny2 | AP-Tiny3 | AP-Small | AP-Medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline seed 43 | 0.5369 | 0.0945 | 0.3176 | 0.1595 | 0.3084 | 0.3684 | 0.4390 | 0.4352 |

R1/J1 P3/P4/P5 trong artifact này có `test_merged/available=0.0`, nên không được điền AP-Small/AP-Tiny bằng suy luận từ standard test.

## 6. Probe A/B: stage tracing

### 6.1 Protocol

- Checkpoint: YOLOv8 baseline, R1, J1.
- TinyPerson validation: 808 images.
- Tiny GT filter: `max_dim <= 16`.
- Pools được trace: `local -> eligible -> TAL-topk -> final`.
- Metrics: oracle gap, score-top, joint-top, hit rate, Spearman score/IoU.
- Probe đã từng fail vì raw class score có shape `[classes, candidates]` nhưng bị đọc ngược thành `[candidates, classes]`. Lỗi đã sửa, smoke 20 images chạy return code 0, sau đó full 808-image run hoàn tất.

### 6.2 Kết quả chính

- Local score gap:
  - YOLOv8: `0.1657`
  - J1: `0.1616`
- Joint criterion có local gap khoảng `0.1556 -> 0.1426` trong bảng diagnostic tương ứng.
- J1 cải thiện frozen local/eligible ranking quality và tăng oracle-hit rate, nhưng chưa tạo ra thay đổi đủ lớn ở candidate được chọn cuối cùng trong TAL-positive pool.

Diễn giải: J1 có tín hiệu cơ chế đúng ở ranking quality, nhưng AP gain không thể quy toàn bộ cho việc đổi candidate cuối cùng. Bottleneck cần xem tiếp là score learning sau assignment, đặc biệt trong pool đông.

## 7. Probe C: frozen post-selection learnability

### 7.1 Kết quả cũ bị bác bỏ

Probe C cũ trên baseline có 221 held-out groups:

- Raw score top IoU: `0.3808`
- Calibrated feature-ranker IoU: `0.3652`
- Delta: `-0.0156`

Vì ranker làm kết quả xấu hơn, không dùng probe cũ để claim selector có ích.

### 7.2 Probe C rerun trên baseline/R1/J1

- Commit: `126a542`.
- PID Marimo: `24420`.
- 808 validation images.
- Even/odd held-out split trên TAL-positive candidates.

| Model | Raw top IoU | Learned top IoU | Delta IoU | Raw hit | Learned hit | Delta hit |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8 | 0.56719 | 0.56956 | +0.00237 | 0.5626 | 0.5573 | -0.0053 |
| R1 | 0.57577 | 0.57365 | -0.00212 | 0.5788 | — | — |
| J1 | — | — | — | — | — | — |

Bảng session output bị cắt ở phần J1/hit đầy đủ, vì vậy chỉ các ô có giá trị quan sát trực tiếp mới được ghi. Kết luận an toàn là frozen post-selection ranker không cho thấy cải thiện ổn định đủ lớn để biện minh cho learnable selector hậu TAL.

## 8. Probe C2: pre-top-k eligible pool

### 8.1 Protocol và execution

- Script: `analysis/probing/pretopk_learnability_probe.py`.
- Commit: `e0c9019`.
- PID Marimo: `29435`.
- Mục tiêu: kiểm tra rankability trên đúng eligible mask trước TAL top-k, không phải post-positive pool.

### 8.2 Coverage

| Model | Eligible mean | Eligible max | GT có eligible < topk=10 | TAL overlap |
|---|---:|---:|---:|---:|
| YOLOv8 P3/P4/P5 | 3.75 | 6 | 100% | 1.00 |
| R1 P3/P4/P5 | 3.75 | 6 | 100% | 1.00 |
| J1 P3/P4/P5 | 3.75 | 6 | 100% | 1.00 |

Đây là geometry của cùng detector P3/P4/P5 nên candidate coverage không đổi giữa checkpoint.

Với historical P2/P3/P4 baseline từ HF `duyle2408/tinyperson-yolov8n-baselines`, probe cho:

| Metric | P3/P4/P5 | P2/P3/P4 historical |
|---|---:|---:|
| Eligible candidates mean | 3.75 | **7.44** |
| Eligible candidates max | 6 | **21** |
| GT có eligible < topk=10 | 100% | **72.1%** |
| GT thực sự bị top-k lọc | 0% | **27.9%** |
| TAL best IoU@10 | 0.6104 | **0.6908** |
| TAL oracle recall@10 | 1.000 | **1.000** |

P2/P3/P4 tăng candidate coverage rõ rệt, nhưng coverage tốt không đồng nghĩa final score learning đã tốt.

### 8.3 Local full-budget replication

Đã chạy lại public entrypoint thật bằng:

```text
PYTHONPATH=. conda run -n ml2 python analysis/probing/pretopk_learnability_probe.py
```

Input là local TinyPerson split `datasets/tinyperson_split_42_corner_sw640_sh512/tinyperson.yaml`, checkpoint local `runs/tinyperson_yolo_baselines/yolov8/seed_42_corner_sw640_sh512/weights/best.pt`, CPU, đủ 808 validation images. Smoke 20 ảnh exit 0 nhưng bị `insufficient_split`; full run đã đạt `status=ok` với artifact `probe_c2_summary.json`.

| Metric | Full local baseline |
|---|---:|
| Images | 808 |
| Train candidates | 2098 |
| Test candidates | 2292 |
| Test groups | 611 |
| Eligible mean | 3.751227 |
| Eligible max | 6 |
| Eligible `< topk=10` | 100% |
| TAL best IoU | 0.610408 |
| Learned best IoU | 0.610408 |
| Delta best IoU | 0.000000 |
| TAL oracle recall | 1.000000 |
| Learned oracle recall | 1.000000 |
| TAL rank correlation | 0.977424 |
| Learned rank correlation | 0.509125 |
| Top-k overlap | 1.000000 |

Kết quả local tái lập đúng kết luận pre-top-k của session Marimo cho P3/P4/P5: toàn bộ group có dưới 10 eligible candidate, TAL đã đạt cùng best IoU/oracle recall với learned ranker, và learned ranker không thay đổi top-k. Đây là kiểm tra end-to-end trên public probe và artifact thật, không phải fixture tổng hợp. Nó không kiểm tra được cohort `eligible >10` của P2/P3/P4, vì checkpoint P2/P3/P4 historical và artifact Marimo tương ứng không có sẵn local trong workspace.

## 9. Post-selection trace probe trên P2/P3/P4

### 9.1 Protocol

- Script: `analysis/probing/post_selection_trace_probe.py`.
- Dataset split: TinyPerson `seed_42_corner_sw640_sh512`.
- 808 validation images.
- 1102 qualifying GT mỗi model, tổng 3306 trace rows.
- Checkpoints: historical P2/P3/P4 baseline, R1, J1.
- Artifact remote:
  - `/marimo/yolo_code/runs/tod_post_selection_trace_p2_good_retry2/post_selection_trace.json`
  - `/marimo/yolo_code/runs/tod_post_selection_trace_p2_good_retry2/post_selection_trace_rows.jsonl`

### 9.2 Failure buckets

| Model | Cohort | N | Retained | Top-k | Conflict | Low responsibility | Score learning |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | eligible <=10 | 841 | 99.29% | 0.24% | 0.12% | 0.36% | 0% |
| R1 | eligible <=10 | 841 | 99.41% | 0% | 0% | 0.59% | 0% |
| J1 | eligible <=10 | 841 | 99.05% | 0% | 0.59% | 0.36% | 0% |
| baseline | eligible >10 | 261 | 90.80% | 0% | 0% | 0% | 9.20% |
| R1 | eligible >10 | 261 | 90.04% | 0% | 0% | 0.38% | 9.58% |
| J1 | eligible >10 | 261 | 90.80% | 0.38% | 0% | 0% | 8.81% |

### 9.3 Rank and target statistics

Cohort `eligible <=10`:

| Model | Mean IoU rank | Mean TAL rank | Mean cls rank | Mean target mass | Mean responsibility share |
|---|---:|---:|---:|---:|---:|
| baseline | 1.0297 | 1.2533 | 2.7551 | 0.6470 | 0.8420 |
| R1 | 1.0273 | 1.1902 | 2.7979 | 0.6444 | 0.5026 |
| J1 | 1.0226 | 1.2140 | 2.7515 | 0.6414 | 0.5117 |

Cohort `eligible >10`:

| Model | Mean IoU rank | Mean TAL rank | Mean cls rank | Mean target mass | Mean responsibility share |
|---|---:|---:|---:|---:|---:|
| baseline | 1.0153 | 1.6513 | 6.2874 | 0.7053 | 0.2860 |
| R1 | 1.0613 | 1.5824 | 6.3257 | 0.7042 | 0.2847 |
| J1 | 1.0268 | 1.5824 | 5.7433 | 0.6974 | 0.3057 |

### 9.4 Readout

- `eligible <=10`: P2 candidate tốt gần như luôn qua TAL, không bị top-k hoặc conflict loại đáng kể, target mass không thấp, final classification rank thường vẫn ở top.
- `eligible >10`: failure chính là final classification ranking. TAL rank khoảng `1.58-1.65`, target mass tuyệt đối vẫn cao, nhưng class rank tụt khoảng `5.74-6.33`; khoảng `8.81%-9.58%` không còn là top candidate theo class score.
- J1 cải thiện nhẹ mean class rank trong pool đông, nhưng chưa loại được failure bucket.

## 10. R1/J1 P2/P3/P4 follow-up

### 10.1 Compatibility/setup

- Detector: YOLOv8n P2/P3/P4 plain.
- YAML: `yolov8n_tinyperson_p2p3p4_plain.yaml`.
- R1: `ranking_mode=localization`.
- J1: `ranking_mode=joint`.
- `rank_loss=0.05`, `rank_tau=0.25`, `lambda_loc=0.25`, `topk=10`.
- Seed/split seed: `42/42`.
- Epochs/patience: `100/0`.
- Workers: `8`.
- HF repository: `duyle2408/tinyperson-p2p4-reranking-runs`.

### 10.2 Attempts and final known state

1. R1 first launch failed immediately because an old confirmation gate rejected the P2/P3/P4 variant. Exit code `1`; no checkpoint or metrics. Đây là failed run, không dùng số liệu.
2. J1 setup trên server thứ hai pass full preflight với commit `1f3f5ed`, nhưng tại thời điểm audit chưa có final train/evaluation metrics được verify.
3. Confirmation gate được sửa ở commit `1f3f5ed`, server được đồng bộ clean, và R1 được launch lại thành công. Một thời điểm R1 chạy với PID `47160`; sau đó parallel rerun được ghi nhận với PID `40261` trên server `sb-2c3b504d3a5a486b.sb.molab.run`.
4. Kết quả cuối đã được ghi nhận cho R1 P2/P3/P4:

| Run | val/AP50 | val/mAP50-95 | test/AP50 | test/mAP50-95 | merged test |
|---|---:|---:|---:|---:|---:|
| R1 P2/P3/P4 | 0.55368 | 0.20086 | 0.51816 | 0.18917 | `test_merged/AP50-Small=0.66977` |

Merged `AP50-Small=0.66977` thuộc **R1 P2/P3/P4, seed 42, no Mosaic**, không phải R1 P3/P4/P5 và không phải các run Mosaic seed 43/44.

Các giá trị `0.689397` và `0.681343` là R1 seed 43/44 dùng canonical P3/P4/P5 với Mosaic trong study khác. Không trộn chúng vào bảng P2/P3/P4.

### 10.3 Related session with an overloaded `R1` label

Session `session_otter_1790770140377_e96f5567febc0b0b` có các kết quả dùng tên R1/R4 nhưng đây là **Negative-canvas Copy-Paste augmentation**, không phải ranking-loss R1 `ranking_mode=localization` trong các section trên. Ghi lại để tránh bỏ sót và tránh nhập nhầm protocol:

| Dataset / study | Result | Provenance boundary |
|---|---|---|
| LEVIR-Ship | R1 `test/AP50=0.8222`, `test/mAP50-95=0.3156`; R4 `test/mAP50-95=0.3158` | YOLOv8 canonical P3/P4/P5 + Negative-canvas augmentation |
| TinyPerson augmentation matrix | Best reported method: P2/P3/P4 + mass-adaptive OACP + clustered CP3 + Mosaic, `test/mAP50-95=0.2061` | Different augmentation study, not TAL ranking R1/J1 |
| TinyPerson R1 seed 43 | `test_merged/AP50-Small=0.689397` | P3/P4/P5 + Mosaic + Negative-canvas |
| TinyPerson R1 seed 44 | `test_merged/AP50-Small=0.681343` | P3/P4/P5 + Mosaic + Negative-canvas |

Các metric trên không được dùng làm bằng chứng cho ranking loss R1/J1. Session `session_dromedary_1790668453013_fa076d346a3855e2` cũng nằm trong cùng ngày nhưng chủ yếu là VisDrone DETR/RT-DETR, không thuộc TAL/RANKING scope nên không đưa vào bảng kết quả này.

## 11. Run status ledger

| Item | Status | Có metric cuối? | Ghi chú |
|---|---|---:|---|
| K1 `kl_tiny` | completed/verified | Có | 100 epochs, upload verified |
| K2 `kl_curriculum` | completed/verified | Có | 100 epochs, upload verified |
| R1 P3/P4/P5 | completed/verified | Có | standard val/test, merged unavailable |
| J1 P3/P4/P5 | completed/verified | Có | standard val/test, merged unavailable |
| Probe A/B | completed/verified | Diagnostic | 808 images, corrected tensor orientation |
| Probe C cũ | completed/negative | Diagnostic | 221 groups, learned ranker worse |
| Probe C rerun | completed/verified | Diagnostic | frozen ranker gain negligible/unstable |
| Probe C2 | completed/verified | Diagnostic | pre-top-k eligible pool |
| P2/P3/P4 baseline probe | completed/verified | Diagnostic | HF historical checkpoint |
| Post-selection trace | completed/verified | Diagnostic | 3306 rows, 1102 GT/model |
| R1 P2/P3/P4 first attempt | failed | Không | confirmation gate rejected architecture |
| J1 P2/P3/P4 setup | preflight/setup only | Không | no verified final metrics in audit |
| R1 P2/P3/P4 retry | completed/verified | Có | standard + merged AP50-Small |

## 12. Final scientific conclusion

- Candidate coverage của P2/P3/P4 là tốt hơn P3/P4/P5: eligible pool lớn hơn, best IoU@10 cao hơn, và oracle recall vẫn 1.0.
- Với `eligible <=10`, top-k không phải bottleneck chính vì candidate P2 tốt gần như luôn được giữ.
- Với `eligible >10`, failure chính là classification score learning sau assignment, không phải thiếu geometry candidate và cũng không phải target mass tuyệt đối quá thấp.
- J1 có cải thiện nhỏ ở frozen ranking/classification rank, nhưng chưa chứng minh được một learnable selector hậu TAL là hướng đúng.
- Không nên tiếp tục thay top-k hoặc xây selector post-hoc trước khi xử lý score learning trong eligible pool đông.
- Hướng tiếp theo hợp lý là bounded, scale-conditioned responsibility correction trong training, giữ TAL assignment làm control, và đánh giá riêng cohort `eligible >10`.

## 13. Existing reports updated/used as sources

- [`docs/reports/tinyperson_yolov8_r1_j1_comparison.md`](tinyperson_yolov8_r1_j1_comparison.md)
- [`docs/reports/tod_research_direction_analysis.md`](tod_research_direction_analysis.md)
- [`docs/reports/2026-09-30.md`](2026-09-30.md)
- [`docs/reports/approach+results/report_250926.md`](approach+results/report_250926.md)
- [`docs/reports/approach+results/report_factorized_tal.md`](approach+results/report_factorized_tal.md)

This audit intentionally does not treat incomplete checkpoints, interrupted processes, preflight-only setups, or unavailable merged evaluators as completed experimental results.
