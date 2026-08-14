# DRENet trên LEVIR-Ship degraded

Report này ghi lại run DRENet seed 42 trên fixed LEVIR split seed 42. Khác với các run YOLOv8n P2 trong repo, input ở đây là ảnh đã degrade theo workflow DRENet, nên số liệu không nên so trực tiếp như một ablation cùng distribution với LEVIR gốc.

## Protocol

| Mục | Giá trị |
| :--- | :--- |
| Model | DRENet |
| Config | `/marimo/yolo_code/DRENet/models/DRENet.yaml` |
| Dataset | `/marimo/yolo_code/datasets_drenet/levir_ship_yolo_seed42/levir_ship.yaml` |
| Split | train/val/test = 2320/788/788 |
| Classes | 1 (`ship`) |
| Train seed | 42 |
| Split seed | 42 |
| Image size | 512 |
| Batch size | 16 |
| Workers | 8 |
| Epochs | 400 total |
| Checkpoint evaluated | `best.pt` |
| Eval NMS IoU | DRENet default `0.6` |
| Eval confidence | DRENet default `0.001` |

Training bắt đầu bằng 100 epoch, sau đó resume từ `last.pt` tới tổng 400 epoch. Run hoàn tất đủ 400 dòng trong `results.txt`.

Remote artifacts:

- Run dir: `/marimo/yolo_code/runs/levir_drenet_100ep_exactdeg/seed_42`
- Train log: `/marimo/yolo_code/runs/levir_drenet_100ep_exactdeg/resume_400_seed42_retry.log`
- Eval summary: `/marimo/yolo_code/runs/levir_drenet_100ep_exactdeg/drenet_400_eval_summary.json`
- Val eval log: `/marimo/yolo_code/runs/levir_drenet_100ep_exactdeg/eval_400_val_best.log`
- Test eval log: `/marimo/yolo_code/runs/levir_drenet_100ep_exactdeg/eval_400_test_best.log`

## Validation và test

Metrics dưới đây là re-evaluation của `best.pt`, không phải chỉ lấy dòng validation history trong train log.

| Split | Images | Targets | Precision | Recall | AP50 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | 788 | 661 | 0.517 | 0.823 | 0.780 | 0.290 |
| Test | 788 | 696 | 0.532 | 0.815 | 0.742 | 0.276 |

## Training history checkpoints

| Selection | Epoch | Precision | Recall | AP50 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Best train-history AP50 | 280 | 0.5143 | 0.8502 | 0.7950 | 0.2795 |
| Best train-history mAP50-95 | 268 | 0.5157 | 0.8215 | 0.7770 | 0.2886 |
| Final epoch | 399 | 0.5298 | 0.8351 | 0.7824 | 0.2865 |

## Automatic weighting run, 500 epochs

Theo README official, code public mặc định dùng fixed weight balance; muốn dùng automatic weight balance cần bật `weightOptimizer` trong `train.py` và các dòng `ForAuto` trong `utils/loss.py`. Một run mới được chạy theo hướng này:

| Mục | Giá trị |
| :--- | :--- |
| Run dir | `/marimo/yolo_code/runs/levir_drenet_paper500_auto/seed_42` |
| Train log | `/marimo/yolo_code/runs/levir_drenet_paper500_auto/train_seed42.log` |
| Epochs | 500 |
| Batch size | 16 |
| Momentum | 0.99 |
| Network optimizer | SGD |
| Loss-weight optimizer | Adam, lr 0.01 |
| Loss weighting | `AutomaticWeightedLoss(2)` over `(det_loss, dre_loss)` |

Run hoàn tất đủ 500 dòng trong `results.txt`, nhưng training bị collapse: sau khoảng epoch 16 loss bắt đầu ra `nan`, final row là `nan` và final validation trả về zero metrics. Vì vậy bảng dưới đây là re-evaluation của `best.pt`, tương ứng checkpoint tốt nhất trước khi collapse.

| Split | Images | Targets | Precision | Recall | AP50 | mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | 788 | 661 | 0.136 | 0.720 | 0.453 | 0.136 |
| Test | 788 | 696 | 0.135 | 0.728 | 0.428 | 0.130 |

Training-history best nằm ở epoch 16: Val P `0.1355`, R `0.7186`, AP50 `0.4523`, mAP50-95 `0.1361`. Kết quả này thấp hơn rõ rệt so với fixed/default 400-epoch run ở trên.

## Notes

- DRENet eval emitted OpenCV plot-thread errors because plotting expects `uint8` images, but both val and test commands exited with code 0 and produced metrics.
- Precision is lower than the YOLOv8n P2 reports partly because DRENet eval uses very low default `conf_thres=0.001`; AP metrics are still threshold-swept.
- This run uses degraded images. Treat comparison against normal LEVIR YOLOv8n runs as cross-protocol evidence, not a controlled architecture comparison.
