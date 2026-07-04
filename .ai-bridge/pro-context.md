# Varroa experiment inventory

Generated: 2026-07-04T02:49:50.656Z
Workspace: /mnt/data/varroa/yolo_related
Workspace ID: ws_9d5e8cd2d1d155051e793f22
Write mode: workspace
Bash mode: safe
Tool mode: standard

Purpose: paste this bundle into a high-context ChatGPT model when that model cannot call the CodexPro MCP tools directly.
Instruction for ChatGPT: use this as repository context, produce a narrow Codex execution plan, and avoid inventing files or runtime facts not shown here.

## Repository Tree

.
├── __pycache__/
│   ├── __init__.cpython-310.pyc
│   ├── __init__.cpython-314.pyc
│   ├── custom_blocks.cpython-310.pyc
│   ├── prepare_dataset.cpython-310.pyc
│   ├── prepare_dataset.cpython-314.pyc
│   ├── summarize_hf_baselines.cpython-314.pyc
│   ├── train_all_2.cpython-310.pyc
│   ├── train_all_full.cpython-310.pyc
│   ├── train_all_missing.cpython-310.pyc
│   ├── train_all.cpython-314.pyc
│   ├── train_star_boundary.cpython-310.pyc
│   ├── train_star_boundary.cpython-314.pyc
│   ├── train_star_lqm.cpython-310.pyc
│   ├── train_star_lqm.cpython-314.pyc
│   └── train.cpython-310.pyc
├── baseline_reuslts/
│   ├── varroa-yolo-baselines-missing-part1-missing/
│   │   ├── test/
│   │   │   ├── yolov5mu_seed42/
│   │   │   ├── yolov5mu_seed43/
│   │   │   ├── yolov5mu_seed44/
│   │   │   ├── yolov5xu_seed42/
│   │   │   ├── yolov5xu_seed43/
│   │   │   ├── yolov5xu_seed44/
│   │   │   ├── yolov8m_seed42/
│   │   │   ├── yolov8m_seed43/
│   │   │   ├── yolov8m_seed44/
│   │   │   └── test_summary_missing.csv
│   │   └── train/
│   │       ├── yolov5mu_seed42/
│   │       ├── yolov5mu_seed43/
│   │       ├── yolov5mu_seed44/
│   │       ├── yolov5xu_seed42/
│   │       ├── yolov5xu_seed43/
│   │       ├── yolov5xu_seed44/
│   │       ├── yolov8m_seed42/
│   │       ├── yolov8m_seed43/
│   │       └── yolov8m_seed44/
│   ├── varroa-yolo-baselines-missing-part2-missing/
│   │   ├── test/
│   │   │   ├── yolov10m_seed42/
│   │   │   ├── yolov10m_seed43/
│   │   │   ├── yolov10m_seed44/
│   │   │   ├── yolov10x_seed42/
│   │   │   ├── yolov10x_seed43/
│   │   │   ├── yolov10x_seed44/
│   │   │   ├── yolov9e_seed42/
│   │   │   ├── yolov9e_seed43/
│   │   │   ├── yolov9e_seed44/
│   │   │   ├── yolov9m_seed42/
│   │   │   ├── yolov9m_seed43/
│   │   │   ├── yolov9m_seed44/
│   │   │   └── test_summary_missing.csv
│   │   └── train/
│   │       ├── yolov10m_seed42/
│   │       ├── yolov10m_seed43/
│   │       ├── yolov10m_seed44/
│   │       ├── yolov10x_seed42/
│   │       ├── yolov10x_seed43/
│   │       ├── yolov10x_seed44/
│   │       ├── yolov9e_seed42/
│   │       ├── yolov9e_seed43/
│   │       ├── yolov9e_seed44/
│   │       ├── yolov9m_seed42/
│   │       ├── yolov9m_seed43/
│   │       └── yolov9m_seed44/
│   ├── varroa-yolo-baselines-missing-part3-missing/
│   │   ├── test/
│   │   │   ├── yolo11m_seed42/
│   │   │   ├── yolo11m_seed43/
│   │   │   ├── yolo11m_seed44/
│   │   │   ├── yolo11x_seed42/
│   │   │   ├── yolo11x_seed43/
│   │   │   ├── yolo11x_seed44/
│   │   │   ├── yolov8x_seed42/
│   │   │   ├── yolov8x_seed43/
│   │   │   ├── yolov8x_seed44/
│   │   │   └── test_summary_missing.csv
│   │   └── train/
│   │       ├── yolo11m_seed42/
│   │       ├── yolo11m_seed43/
│   │       ├── yolo11m_seed44/
│   │       ├── yolo11n_seed42/
│   │       ├── yolo11n_seed43/
│   │       ├── yolo11n_seed44/
│   │       ├── yolo11s_seed42/
│   │       ├── yolo11x_seed42/
│   │       ├── yolo11x_seed43/
│   │       ├── yolo11x_seed44/
│   │       ├── yolov8x_seed42/
│   │       ├── yolov8x_seed43/
│   │       └── yolov8x_seed44/
│   ├── varroa-yolo-baselines-part1-full/
│   │   ├── test/
│   │   │   ├── yolov5lu_seed42/
│   │   │   ├── yolov5lu_seed43/
│   │   │   ├── yolov5lu_seed44/
│   │   │   ├── yolov5nu_seed42/
│   │   │   ├── yolov5nu_seed43/
│   │   │   ├── yolov5nu_seed44/
│   │   │   ├── yolov5su_seed42/
│   │   │   ├── yolov5su_seed43/
│   │   │   ├── yolov5su_seed44/
│   │   │   ├── yolov8l_seed42/
│   │   │   ├── yolov8l_seed43/
│   │   │   ├── yolov8l_seed44/
│   │   │   ├── yolov8n_seed42/
│   │   │   ├── yolov8n_seed43/
│   │   │   ├── yolov8n_seed44/
│   │   │   ├── yolov8s_seed42/
│   │   │   ├── yolov8s_seed43/
│   │   │   ├── yolov8s_seed44/
│   │   │   ├── yolov9t_seed42/
│   │   │   ├── yolov9t_seed43/
│   │   │   ├── yolov9t_seed44/
│   │   │   └── test_summary_full.csv
│   │   └── train/
│   │       ├── yolov5lu_seed42/
│   │       ├── yolov5lu_seed43/
│   │       ├── yolov5lu_seed44/
│   │       ├── yolov5nu_seed42/
│   │       ├── yolov5nu_seed43/
│   │       ├── yolov5nu_seed44/
│   │       ├── yolov5su_seed42/
│   │       ├── yolov5su_seed43/
│   │       ├── yolov5su_seed44/
│   │       ├── yolov8l_seed42/
│   │       ├── yolov8l_seed43/
│   │       ├── yolov8l_seed44/
│   │       ├── yolov8n_seed42/
│   │       ├── yolov8n_seed43/
│   │       ├── yolov8n_seed44/
│   │       ├── yolov8s_seed42/
│   │       ├── yolov8s_seed43/
│   │       ├── yolov8s_seed44/
│   │       ├── yolov9t_seed42/
│   │       ├── yolov9t_seed43/
│   │       └── yolov9t_seed44/
│   ├── varroa-yolo-baselines-part2-full/
│   │   ├── test/
│   │   │   ├── yolo11l_seed42/
│   │   │   ├── yolo11l_seed43/
│   │   │   ├── yolo11l_seed44/
│   │   │   ├── yolo11n_seed42/
│   │   │   ├── yolo11n_seed43/
│   │   │   ├── yolo11n_seed44/
│   │   │   ├── yolo11s_seed42/
│   │   │   ├── yolo11s_seed43/
│   │   │   ├── yolo11s_seed44/
│   │   │   ├── yolov10l_seed42/
│   │   │   ├── yolov10l_seed43/
│   │   │   ├── yolov10l_seed44/
│   │   │   ├── yolov10n_seed42/
│   │   │   ├── yolov10n_seed43/
│   │   │   ├── yolov10n_seed44/
│   │   │   ├── yolov10s_seed42/
│   │   │   ├── yolov10s_seed43/
│   │   │   ├── yolov10s_seed44/
│   │   │   ├── yolov9c_seed42/
│   │   │   ├── yolov9c_seed43/
│   │   │   ├── yolov9c_seed44/
│   │   │   ├── yolov9s_seed42/
│   │   │   ├── yolov9s_seed43/
│   │   │   ├── yolov9s_seed44/
│   │   │   └── test_summary_full.csv
│   │   └── train/
│   │       ├── yolo11l_seed42/
│   │       ├── yolo11l_seed43/
│   │       ├── yolo11l_seed44/
│   │       ├── yolo11n_seed42/
│   │       ├── yolo11n_seed43/
│   │       ├── yolo11n_seed44/
│   │       ├── yolo11s_seed42/
│   │       ├── yolo11s_seed43/
│   │       ├── yolo11s_seed44/
│   │       ├── yolov10l_seed42/
│   │       ├── yolov10l_seed43/
│   │       ├── yolov10l_seed44/
│   │       ├── yolov10n_seed42/
│   │       ├── yolov10n_seed43/
│   │       ├── yolov10n_seed44/
│   │       ├── yolov10s_seed42/
│   │       ├── yolov10s_seed43/
│   │       ├── yolov10s_seed44/
│   │       ├── yolov9c_seed42/
│   │       ├── yolov9c_seed43/
│   │       ├── yolov9c_seed44/
│   │       ├── yolov9s_seed42/
│   │       ├── yolov9s_seed43/
│   │       └── yolov9s_seed44/
│   ├── BASELINE_RESULTS_SUMMARY.md
│   ├── BASELINE_RESULTS_TABLES.aux
│   ├── BASELINE_RESULTS_TABLES.fdb_latexmk
│   ├── BASELINE_RESULTS_TABLES.fls
│   ├── BASELINE_RESULTS_TABLES.log
│   ├── BASELINE_RESULTS_TABLES.pdf
│   ├── BASELINE_RESULTS_TABLES.synctex.gz
│   └── BASELINE_RESULTS_TABLES.tex
├── data/
│   ├── test/
│   │   ├── labels/
│   │   │   ├── 2017-09-01_10-54-26/
│   │   │   └── 2017-10-17_1-39-36/
│   │   ├── videos/
│   │   │   ├── 2017-09-01_10-54-26/
│   │   │   └── 2017-10-17_1-39-36/
│   │   ├── gt_filtered.csv
│   │   └── gt_one.csv
│   ├── train/
│   │   ├── labels/
│   │   │   ├── 2017-08-28_09-30-00-1_500_dirty_glass/
│   │   │   ├── 2017-08-28_16-31-55/
│   │   │   ├── 2017-08-30_15-42-59/
│   │   │   ├── 2017-09-20_19-24-55/
│   │   │   ├── 2017-09-25_16-03-38-2/
│   │   │   └── 2017-10-17_16-41-10/
│   │   ├── videos/
│   │   │   ├── 2017-08-28_09-30-00-1_500_dirty_glass/
│   │   │   ├── 2017-08-28_16-31-55/
│   │   │   ├── 2017-08-30_15-42-59/
│   │   │   ├── 2017-09-20_19-24-55/
│   │   │   ├── 2017-09-25_16-03-38-2/
│   │   │   └── 2017-10-17_16-41-10/
│   │   ├── gt_filtered.csv
│   │   └── gt_one.csv
│   ├── val/
│   │   ├── labels/
│   │   │   ├── 2017-08-28_16-32-55_30_sec/
│   │   │   ├── 2017-09-01_20-00-17/
│   │   │   ├── 2017-09-01_3-01-01/
│   │   │   ├── 2017-09-25_16-03-38/
│   │   │   └── 2017-09-29_15-31-49/
│   │   ├── videos/
│   │   │   ├── 2017-08-28_16-32-55_30_sec/
│   │   │   ├── 2017-09-01_20-00-17/
│   │   │   ├── 2017-09-01_3-01-01/
│   │   │   ├── 2017-09-25_16-03-38/
│   │   │   └── 2017-09-29_15-31-49/
│   │   ├── gt_filtered.csv
│   │   └── gt_one.csv
│   └── yolo_related/
│       └── datasets/
│           └── varroa_probe/
├── datasets/
│   └── varroa_yolo/
│       ├── images/
│       │   ├── test/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── test/
│       │   ├── train/
│       │   ├── val/
│       │   ├── test.cache
│       │   ├── train.cache
│       │   └── val.cache
│       └── varroa.yaml
├── misc/
│   ├── __pycache__/
│   │   ├── api.cpython-310.pyc
│   │   ├── kvca_localization_probe.cpython-310.pyc
│   │   ├── prepare_dataset.cpython-310.pyc
│   │   ├── quality_probe_compare.cpython-310.pyc
│   │   ├── train_api_boxgrad_boundary_ring.cpython-310.pyc
│   │   ├── train_boundary_api_compare.cpython-310.pyc
│   │   ├── train_boundary_api_late.cpython-310.pyc
│   │   ├── train_edge.cpython-310.pyc
│   │   ├── train_only_p3_edge.cpython-310.pyc
│   │   ├── train_vfl_tal_a0.cpython-310.pyc
│   │   └── vfl_feature_iou_linear_probe.cpython-310.pyc
│   ├── api.py
│   ├── check_tensors.py
│   ├── hbs.py
│   ├── hf_download.txt
│   ├── kvca_localization_probe.py
│   ├── nguyen_blocks.py
│   ├── plot_bee_gt_grid.py
│   ├── prepare_dataset.py
│   ├── quality_probe_compare.py
│   ├── some_custom_modules.py
│   ├── summarize_hf_baselines.py
│   ├── sweep.py
│   ├── train_all_2.py
│   ├── train_all_full.py
│   ├── train_all_missing.py
│   ├── train_all.py
│   ├── train_api_boxgrad_boundary_ring.py
│   ├── train_boundary_api_compare.py
│   ├── train_boundary_api_late.py
│   ├── train_edge.py
│   ├── train_kvca_sweep.py
│   ├── train_lm_net_varroa.py
│   ├── train_only_p3_edge.py
│   ├── train_p3_api_sweep.py
│   ├── train_star_boundary.py
│   ├── train_star_lqm.py
│   ├── train_vfl_tal_a0.py
│   ├── upload_hf.py
│   └── vfl_feature_iou_linear_probe.py
├── models_related/
│   ├── custom_blocks/
│   │   └── custom_blocks.py
│   ├── models_config/
│   │   ├── yolov5/
│   │   │   └── yolov5-tph-kvca-cbam-p2.yaml
│   │   └── yolov8/
│   │       ├── kvca_sweep/
│   │       ├── tried/
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam_late.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_quality_iou.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_add.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_concat.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b2.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b4.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b6.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_train.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad_ensimam.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_boundary_api_ring.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_boundary_api.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_boundary_late.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2_boundary.yaml
│   │       ├── yolov8_varroa_compare_baseline_p2p3_edge_pooling.yaml
│   │       ├── yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_ensimam.yaml
│   │       ├── yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml
│   │       ├── yolov8_varroa_compare_baseline_p3_api_boxgrad_ensimam.yaml
│   │       ├── yolov8_varroa_compare_baseline_p3_edge_pooling_quality_iou.yaml
│   │       ├── yolov8_varroa_compare_baseline_p3_edge_pooling.yaml
│   │       ├── yolov8_varroa_compare_p2p3_api_boxgrad_edge_pooling.yaml
│   │       ├── yolov8_varroa_compare_p3_api_boxgrad_edge_pooling.yaml
│   │       └── yolov8n_varroa_compare_baseline_p2_boundary_api.yaml
│   ├── train_eval/
│   │   ├── __pycache__/
│   │   │   ├── eval.cpython-310.pyc
│   │   │   ├── eval.cpython-314.pyc
│   │   │   ├── train.cpython-310.pyc
│   │   │   └── train.cpython-314.pyc
│   │   ├── eval.py
│   │   └── train.py
│   └── ultralytics/
│       ├── docker/
│       │   ├── Dockerfile
│       │   ├── Dockerfile-arm64
│       │   ├── Dockerfile-conda
│       │   ├── Dockerfile-cpu
│       │   ├── Dockerfile-export
│       │   ├── Dockerfile-jetson-jetpack4
│       │   ├── Dockerfile-jetson-jetpack5
│       │   ├── Dockerfile-jetson-jetpack6
│       │   ├── Dockerfile-jupyter
│       │   ├── Dockerfile-nvidia-arm64
│       │   ├── Dockerfile-python
│       │   ├── Dockerfile-python-export
│       │   ├── Dockerfile-runner
│       │   └── Dockerfile-runner-cpu
│       ├── docs/
│       │   ├── en/
│       │   ├── macros/
│       │   ├── overrides/
│       │   ├── build_docs.py
│       │   ├── build_reference.py
│       │   ├── coming_soon_template.md
│       │   ├── mkdocs_github_authors.yaml
│       │   ├── model_data.py
│       │   └── README.md
│       ├── examples/
│       │   ├── RTDETR-ONNXRuntime-Python/
│       │   ├── YOLO-Axelera-Python/
│       │   ├── YOLO-Interactive-Tracking-UI/
│       │   ├── YOLO-Series-ONNXRuntime-Rust/
│       │   ├── YOLO11-Triton-CPP/
│       │   ├── YOLOv8-Action-Recognition/
│       │   ├── YOLOv8-CPP-Inference/
│       │   ├── YOLOv8-LibTorch-CPP-Inference/
│       │   ├── YOLOv8-MNN-CPP/
│       │   ├── YOLOv8-ONNXRuntime/
│       │   ├── YOLOv8-ONNXRuntime-CPP/
│       │   ├── YOLOv8-ONNXRuntime-Rust/
│       │   ├── YOLOv8-OpenCV-ONNX-Python/
│       │   ├── YOLOv8-OpenVINO-CPP-Inference/
│       │   ├── YOLOv8-Region-Counter/
│       │   ├── YOLOv8-SAHI-Inference-Video/
│       │   ├── YOLOv8-Segmentation-ONNXRuntime-Python/
│       │   ├── YOLOv8-TFLite-Python/
│       │   ├── heatmaps.ipynb
│       │   ├── hub.ipynb
│       │   ├── object_counting.ipynb
│       │   ├── object_tracking.ipynb
│       │   ├── README.md
│       │   └── tutorial.ipynb
│       ├── tests/
│       │   ├── __pycache__/
│       │   ├── __init__.py
│       │   ├── cache_test_assets.py
│       │   ├── conftest.py
│       │   ├── test_cli.py
│       │   ├── test_cuda.py
│       │   ├── test_engine.py
│       │   ├── test_exports.py
│       │   ├── test_integrations.py
│       │   ├── test_nms.py
│       │   ├── test_python.py
│       │   └── test_solutions.py
│       ├── ultralytics/
│       │   ├── __pycache__/
│       │   ├── assets/
│       │   ├── cfg/
│       │   ├── data/
│       │   ├── engine/
│       │   ├── hub/
│       │   ├── models/
│       │   ├── nn/
│       │   ├── optim/
│       │   ├── solutions/
│       │   ├── trackers/
│       │   ├── utils/
│       │   ├── __init__.py
│       │   └── py.typed
│       ├── CITATION.cff
│       ├── CONTRIBUTING.md
│       ├── LICENSE
│       ├── mkdocs.yml
│       ├── pyproject.toml
│       ├── README.md
│       ├── README.zh-CN.md
│       └── upload_hf.py
├── runs/
│   └── detect/
│       ├── box_voting_ablation_local/
│       │   ├── hard/
│       │   ├── hard_boxvote/
│       │   ├── soft_linear/
│       │   └── soft_linear_boxvote/
│       ├── runs/
│       │   └── detect/
│       ├── train/
│       ├── train-2/
│       ├── val/
│       └── yolo_related/
│           └── runs/
├── SKILLS/
│   ├── test_results/
│   │   ├── results(3)_kvca_before_heads.csv
│   │   └── results(4)_p2_custom.csv
│   ├── clean_architecture_SKILL.md
│   └── clean_code_SKILL.md
├── test_results/
│   ├── EXPERIMENT_LOG.md
│   ├── placement_summary_shard1_of_2.csv
│   ├── summary-1.csv
│   └── summary.csv
├── Understand-Anything/
│   ├── assets/
│   │   ├── hero.png
│   │   └── overview.png
│   ├── docs/
│   │   └── superpowers/
│   │       ├── plans/
│   │       └── specs/
│   ├── homepage/
│   │   ├── public/
│   │   │   ├── fonts/
│   │   │   ├── images/
│   │   │   ├── CNAME
│   │   │   ├── favicon.ico
│   │   │   ├── favicon.png
│   │   │   └── favicon.svg
│   │   ├── src/
│   │   │   ├── components/
│   │   │   ├── layouts/
│   │   │   ├── pages/
│   │   │   └── styles/
│   │   ├── astro.config.mjs
│   │   ├── package.json
│   │   ├── README.md
│   │   └── tsconfig.json
│   ├── READMEs/
│   │   ├── README.es-ES.md
│   │   ├── README.ja-JP.md
│   │   ├── README.ko-KR.md
│   │   ├── README.ru-RU.md
│   │   ├── README.tr-TR.md
│   │   ├── README.zh-CN.md
│   │   └── README.zh-TW.md
│   ├── scripts/
│   │   └── generate-large-graph.mjs
│   ├── tests/
│   │   └── skill/
│   │       └── understand/
│   ├── understand-anything-plugin/
│   │   ├── agents/
│   │   │   ├── architecture-analyzer.md
│   │   │   ├── article-analyzer.md
│   │   │   ├── assemble-reviewer.md
│   │   │   ├── domain-analyzer.md
│   │   │   ├── file-analyzer.md
│   │   │   ├── graph-reviewer.md
│   │   │   ├── knowledge-graph-guide.md
│   │   │   ├── project-scanner.md
│   │   │   └── tour-builder.md
│   │   ├── hooks/
│   │   │   ├── auto-update-prompt.md
│   │   │   └── hooks.json
│   │   ├── packages/
│   │   │   ├── core/
│   │   │   ├── dashboard/
│   │   │   ├── tree-sitter-dart-wasm/
│   │   │   └── tree-sitter-swift-wasm/
│   │   ├── skills/
│   │   │   ├── understand/
│   │   │   ├── understand-chat/
│   │   │   ├── understand-dashboard/
│   │   │   ├── understand-diff/
│   │   │   ├── understand-domain/
│   │   │   ├── understand-explain/
│   │   │   ├── understand-knowledge/
│   │   │   └── understand-onboard/
│   │   ├── src/
│   │   │   ├── __tests__/
│   │   │   ├── context-builder.ts
│   │   │   ├── diff-analyzer.ts
│   │   │   ├── explain-builder.ts
│   │   │   ├── index.ts
│   │   │   ├── onboard-builder.ts
│   │   │   └── understand-chat.ts
│   │   ├── package.json
│   │   ├── pnpm-lock.yaml
│   │   ├── pnpm-workspace.yaml
│   │   ├── tsconfig.json
│   │   └── vitest.config.ts
│   ├── CLAUDE.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── eslint.config.mjs
│   ├── install.ps1
│   ├── install.sh
│   ├── LICENSE
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── pnpm-workspace.yaml
│   ├── README.md
│   ├── SECURITY.md
│   ├── tsconfig.json
│   └── vitest.config.ts
├── VarifocalNet/
│   ├── configs/
│   │   ├── _base_/
│   │   │   ├── datasets/
│   │   │   ├── models/
│   │   │   ├── schedules/
│   │   │   ├── default_runtime.py
│   │   │   └── swa.py
│   │   ├── albu_example/
│   │   │   ├── mask_rcnn_r50_fpn_albu_1x_coco.py
│   │   │   └── README.md
│   │   ├── atss/
│   │   │   ├── atss_gfl_r50_fpn_1x_coco.py
│   │   │   ├── atss_r101_fpn_1x_coco.py
│   │   │   ├── atss_r50_fpn_1x_coco.py
│   │   │   ├── atss_raw_r50_fpn_1x_coco.py
│   │   │   ├── atss_vfl_r50_fpn_1x_coco.py
│   │   │   └── README.md
│   │   ├── carafe/
│   │   │   ├── faster_rcnn_r50_fpn_carafe_1x_coco.py
│   │   │   ├── mask_rcnn_r50_fpn_carafe_1x_coco.py
│   │   │   └── README.md
│   │   ├── cascade_rcnn/
│   │   │   ├── cascade_mask_rcnn_r101_caffe_fpn_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_r101_fpn_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_r101_fpn_20e_coco.py
│   │   │   ├── cascade_mask_rcnn_r50_caffe_fpn_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_r50_fpn_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_r50_fpn_20e_coco.py
│   │   │   ├── cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_x101_32x4d_fpn_20e_coco.py
│   │   │   ├── cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py
│   │   │   ├── cascade_rcnn_r101_caffe_fpn_1x_coco.py
│   │   │   ├── cascade_rcnn_r101_fpn_1x_coco.py
│   │   │   ├── cascade_rcnn_r101_fpn_20e_coco.py
│   │   │   ├── cascade_rcnn_r50_caffe_fpn_1x_coco.py
│   │   │   ├── cascade_rcnn_r50_fpn_1x_coco.py
│   │   │   ├── cascade_rcnn_r50_fpn_20e_coco.py
│   │   │   ├── cascade_rcnn_x101_32x4d_fpn_1x_coco.py
│   │   │   ├── cascade_rcnn_x101_32x4d_fpn_20e_coco.py
│   │   │   ├── cascade_rcnn_x101_64x4d_fpn_1x_coco.py
│   │   │   ├── cascade_rcnn_x101_64x4d_fpn_20e_coco.py
│   │   │   └── README.md
│   │   ├── cascade_rpn/
│   │   │   ├── crpn_fast_rcnn_r50_caffe_fpn_1x_coco.py
│   │   │   ├── crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py
│   │   │   ├── crpn_r50_caffe_fpn_1x_coco.py
│   │   │   └── README.md
│   │   ├── centripetalnet/
│   │   │   ├── centripetalnet_hourglass104_mstest_16x6_210e_coco.py
│   │   │   └── README.md
│   │   ├── cityscapes/
│   │   │   ├── faster_rcnn_r50_fpn_1x_cityscapes.py
│   │   │   ├── mask_rcnn_r50_fpn_1x_cityscapes.py
│   │   │   └── README.md
│   │   ├── cornernet/
│   │   │   ├── cornernet_hourglass104_mstest_10x5_210e_coco.py
│   │   │   ├── cornernet_hourglass104_mstest_32x3_210e_coco.py
│   │   │   ├── cornernet_hourglass104_mstest_8x6_210e_coco.py
│   │   │   └── README.md
│   │   ├── dcn/
│   │   │   ├── cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_dpool_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_mdpool_1x_coco.py
│   │   │   ├── faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py
│   │   │   ├── mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py
│   │   │   └── README.md
│   │   ├── deepfashion/
│   │   │   ├── mask_rcnn_r50_fpn_15e_deepfashion.py
│   │   │   └── README.md
│   │   ├── detectors/
│   │   │   ├── cascade_rcnn_r50_rfp_1x_coco.py
│   │   │   ├── cascade_rcnn_r50_sac_1x_coco.py
│   │   │   ├── detectors_cascade_rcnn_r50_1x_coco.py
│   │   │   ├── detectors_htc_r50_1x_coco.py
│   │   │   ├── htc_r50_rfp_1x_coco.py
│   │   │   ├── htc_r50_sac_1x_coco.py
│   │   │   └── README.md
│   │   ├── detr/
│   │   │   ├── detr_r50_8x2_150e_coco.py
│   │   │   └── README.md
│   │   ├── double_heads/
│   │   │   ├── dh_faster_rcnn_r50_fpn_1x_coco.py
│   │   │   └── README.md
│   │   ├── dynamic_rcnn/
│   │   │   ├── dynamic_rcnn_r50_fpn_1x.py
│   │   │   └── README.md
│   │   ├── empirical_attention/
│   │   │   ├── faster_rcnn_r50_fpn_attention_0010_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_attention_1111_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco.py
│   │   │   └── README.md
│   │   ├── fast_rcnn/
│   │   │   ├── fast_rcnn_r101_caffe_fpn_1x_coco.py
│   │   │   ├── fast_rcnn_r101_fpn_1x_coco.py
│   │   │   ├── fast_rcnn_r101_fpn_2x_coco.py
│   │   │   ├── fast_rcnn_r50_caffe_fpn_1x_coco.py
│   │   │   ├── fast_rcnn_r50_fpn_1x_coco.py
│   │   │   ├── fast_rcnn_r50_fpn_2x_coco.py
│   │   │   └── README.md
│   │   ├── faster_rcnn/
│   │   │   ├── faster_rcnn_r101_caffe_fpn_1x_coco.py
│   │   │   ├── faster_rcnn_r101_fpn_1x_coco.py
│   │   │   ├── faster_rcnn_r101_fpn_2x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_c4_1x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_dc5_1x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_dc5_mstrain_3x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_fpn_1x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person-bicycle-car.py
│   │   │   ├── faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py
│   │   │   ├── faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_fpn_mstrain_2x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py
│   │   │   ├── faster_rcnn_r50_caffe_fpn_mstrain_90k_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_2x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_bounded_iou_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_giou_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_iou_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_ohem_1x_coco.py
│   │   │   ├── faster_rcnn_r50_fpn_soft_nms_1x_coco.py
│   │   │   ├── faster_rcnn_x101_32x4d_fpn_1x_coco.py
│   │   │   ├── faster_rcnn_x101_32x4d_fpn_2x_coco.py
│   │   │   ├── faster_rcnn_x101_64x4d_fpn_1x_coco.py
│   │   │   ├── faster_rcnn_x101_64x4d_fpn_2x_coco.py
│   │   │   └── README.md
│   │   ├── fcos/
│   │   │   ├── fcos_center_r50_caffe_fpn_gn-head_1x_coco.py
│   │   │   ├── fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py
│   │   │   ├── fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco.py
│   │   │   ├── fcos_r101_caffe_fpn_gn-head_1x_coco.py
│   │   │   ├── fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py
│   │   │   ├── fcos_r50_caffe_fpn_gn-head_1x_coco.py
│   │   │   ├── fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py
│   │   │   ├── fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py
│   │   │   ├── fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py
│   │   │   └── README.md
│   │   ├── foveabox/
...[tree truncated after 700 entries]

## Git Status

```text
## main...origin/main
 M VARROA_POOLING_BOX_ERROR_DIAGNOSTIC.md
 M models_related/ultralytics/tests/test_nms.py
 M models_related/ultralytics/ultralytics/cfg/__init__.py
 M models_related/ultralytics/ultralytics/cfg/default.yaml
 M models_related/ultralytics/ultralytics/engine/model.py
 M models_related/ultralytics/ultralytics/models/yolo/detect/predict.py
 M models_related/ultralytics/ultralytics/models/yolo/detect/val.py
 M models_related/ultralytics/ultralytics/utils/nms.py
?? .ai-bridge/
```

## Recent Commits

```text
4295925 (HEAD -> main, origin/main) .
b747cae .
182a53d .
addde0f .
845d8a7 .
d323322 .
bba99b4 .
abacb73 .
```

## Selected Files

Changed files detected: VARROA_POOLING_BOX_ERROR_DIAGNOSTIC.md, models_related/ultralytics/tests/test_nms.py, models_related/ultralytics/ultralytics/cfg/__init__.py, models_related/ultralytics/ultralytics/cfg/default.yaml, models_related/ultralytics/ultralytics/engine/model.py, models_related/ultralytics/ultralytics/models/yolo/detect/predict.py, models_related/ultralytics/ultralytics/models/yolo/detect/val.py, models_related/ultralytics/ultralytics/utils/nms.py, .ai-bridge/
Auto-include important root files: no
Auto-include changed files: yes
Explicit selected paths: VARROA_POOLING_BOX_ERROR_DIAGNOSTIC.md, VARROA_POOLING_BOX_ERROR_DIAGNOSTIC_APPENDIX.md, models_related/ultralytics/ultralytics/utils/nms.py, models_related/ultralytics/ultralytics/cfg/default.yaml, models_related/ultralytics/ultralytics/cfg/__init__.py
Extra globs: models_related/models_config/yolov8/*.yaml, misc/train*.py, test_results/*.csv, test_results/*.md
Files included below: .ai-bridge/, misc/train_all_2.py, misc/train_all_full.py, misc/train_all_missing.py, misc/train_all.py, misc/train_api_boxgrad_boundary_ring.py, misc/train_boundary_api_compare.py, misc/train_boundary_api_late.py, misc/train_edge.py, misc/train_kvca_sweep.py, misc/train_lm_net_varroa.py, misc/train_only_p3_edge.py, misc/train_p3_api_sweep.py, misc/train_star_boundary.py, misc/train_star_lqm.py, misc/train_vfl_tal_a0.py, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam_late.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_quality_iou.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_add.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_concat.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b2.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b4.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b6.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_train.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_ensimam.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_api_ring.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_api.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_late.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2p3_edge_pooling.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_ensimam.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_api_boxgrad_ensimam.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling_quality_iou.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_p2p3_api_boxgrad_edge_pooling.yaml, models_related/models_config/yolov8/yolov8_varroa_compare_p3_api_boxgrad_edge_pooling.yaml, models_related/models_config/yolov8/yolov8n_varroa_compare_baseline_p2_boundary_api.yaml, models_related/ultralytics/tests/test_nms.py, models_related/ultralytics/ultralytics/cfg/__init__.py, models_related/ultralytics/ultralytics/cfg/default.yaml, models_related/ultralytics/ultralytics/engine/model.py, models_related/ultralytics/ultralytics/models/yolo/detect/predict.py, models_related/ultralytics/ultralytics/models/yolo/detect/val.py, models_related/ultralytics/ultralytics/utils/nms.py

## File Contents

### misc/train_all_2.py

Bytes: 6184
SHA-256: a36679db67b8468e505bfd01ae7c33cfbe18842efe7236a9ee6a4c8bfa3e4a3b
Lines: 1-211 of 211

```python
  1 | #!/usr/bin/env python3
  2 | """Train and test the two TPH/KVCA Varroa configs."""
  3 | 
  4 | from __future__ import annotations
  5 | 
  6 | import csv
  7 | import os
  8 | import shutil
  9 | import sys
 10 | from pathlib import Path
 11 | 
 12 | ROOT = Path(__file__).resolve().parent
 13 | LOCAL_ULTRALYTICS = ROOT / "models_related" / "ultralytics"
 14 | 
 15 | # ==========================================
 16 | # EDIT CONFIG HERE
 17 | # ==========================================
 18 | MODEL_SCALE = "n"  # n, s, m, l, x
 19 | EPOCHS = 200
 20 | IMGSZ = 640
 21 | BATCH = 16
 22 | WORKERS = 4
 23 | DEVICE = "cuda"
 24 | PATIENCE = 30
 25 | 
 26 | RUN_PREPARE_DATA = True
 27 | DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
 28 | DATASET_OUT_DIR = ROOT / "datasets" / "varroa_yolo"
 29 | DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"
 30 | 
 31 | PROJECT = ROOT / "runs" / "detect" / "yolo_related" / "runs" / "train"
 32 | TEST_PROJECT = ROOT / "runs" / "detect" / "yolo_related" / "runs" / "test"
 33 | GENERATED_CONFIG_DIR = Path("/tmp") / "varroa_train_all_2_configs"
 34 | 
 35 | CONFIGS_TO_RUN = [
 36 |     ROOT / "models_related" / "models_config" / "yolov8" / "yolov8_varroa_tph_kvencoder_repc2f_ensimam_p2.yaml",
 37 |     ROOT / "models_related" / "models_config" / "yolov5" / "yolov5-tph-kvca-cbam-p2.yaml",
 38 | ]
 39 | # ==========================================
 40 | 
 41 | 
 42 | def prefer_local_ultralytics() -> None:
 43 |     """Force imports to use the repo's local Ultralytics fork."""
 44 |     sys.path.insert(0, str(LOCAL_ULTRALYTICS))
 45 |     for name in list(sys.modules):
 46 |         if name == "ultralytics" or name.startswith("ultralytics."):
 47 |             del sys.modules[name]
 48 | 
 49 | 
 50 | prefer_local_ultralytics()
 51 | 
 52 | from ultralytics import YOLO
 53 | 
 54 | 
 55 | def prepare_data() -> None:
 56 |     """Prepare the Varroa YOLO dataset."""
 57 |     from misc.prepare_dataset import prepare_dataset
 58 | 
 59 |     print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
 60 |     prepare_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))
 61 | 
 62 | 
 63 | def config_for_scale(config: Path) -> Path:
 64 |     """Copy a config and inject the selected model scale."""
 65 |     text = config.read_text()
 66 |     out_lines = []
 67 |     inserted = False
 68 | 
 69 |     for line in text.splitlines():
 70 |         if line.startswith("scale:"):
 71 |             out_lines.append(f"scale: {MODEL_SCALE}")
 72 |             inserted = True
 73 |             continue
 74 | 
 75 |         out_lines.append(line)
 76 |         if not inserted and line.startswith("nc:"):
 77 |             out_lines.append(f"scale: {MODEL_SCALE}")
 78 |             inserted = True
 79 | 
 80 |     if not inserted:
 81 |         out_lines.insert(0, f"scale: {MODEL_SCALE}")
 82 | 
 83 |     GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
 84 |     scaled_config = GENERATED_CONFIG_DIR / f"{config.stem}_{MODEL_SCALE}{config.suffix}"
 85 |     scaled_config.write_text("\n".join(out_lines) + "\n")
 86 |     return scaled_config
 87 | 
 88 | 
 89 | def pretrained_for(config: Path) -> str:
 90 |     """Return matching pretrained weights for a config family."""
 91 |     return f"yolov5{MODEL_SCALE}.pt" if "yolov5" in config.parts else f"yolov8{MODEL_SCALE}.pt"
 92 | 
 93 | 
 94 | def train_config(config: Path) -> None:
 95 |     """Train one config."""
 96 |     if not config.is_file():
 97 |         raise FileNotFoundError(f"Missing config: {config}")
 98 | 
 99 |     model_yaml = config_for_scale(config)
100 |     run_name = f"{config.stem}_{MODEL_SCALE}"
101 | 
102 |     print("\n" + "=" * 60)
103 |     print(f"Training: {run_name}")
104 |     print(f"Config:   {model_yaml}")
105 |     print("=" * 60)
106 | 
107 |     model = YOLO(str(model_yaml))
108 |     model.load(pretrained_for(config), smart_transfer=True)
109 | 
110 |     model.train(
111 |         data=str(DATA_PATH),
112 |         epochs=EPOCHS,
113 |         imgsz=IMGSZ,
114 |         batch=BATCH,
115 |         workers=WORKERS,
116 |         device=DEVICE,
117 |         patience=PATIENCE,
118 |         bbox_iou_loss="wiou",
119 |         wiou_monotonous=False,
120 |         # EXPERIMENTAL: boundary-aware contrastive localization ablation.
121 |         # boundary_contrast=0.05,
122 |         # boundary_levels=2,
123 |         # boundary_ring=1.0,
124 |         # boundary_samples=16,
125 |         # boundary_tau=0.2,
126 |         project=str(PROJECT),
127 |         name=run_name,
128 |     )
129 | 
130 | 
131 | def find_best_weights() -> list[Path]:
132 |     """Find best.pt files for this script's run names."""
133 |     run_names = {f"{config.stem}_{MODEL_SCALE}" for config in CONFIGS_TO_RUN}
134 |     best_paths = []
135 |     for best_path in PROJECT.glob("*/weights/best.pt"):
136 |         if best_path.parents[1].name in run_names:
137 |             best_paths.append(best_path)
138 |     return sorted(best_paths)
139 | 
140 | 
141 | def run_test_inference() -> Path | None:
142 |     """Evaluate trained checkpoints on the test split and save a summary CSV."""
143 |     best_paths = find_best_weights()
144 |     print(f"\nFound {len(best_paths)} trained checkpoints for testing")
145 |     if not best_paths:
146 |         return None
147 | 
148 |     results = []
149 |     for best_path in best_paths:
150 |         run_name = best_path.parents[1].name
151 | 
152 |         print("\n" + "=" * 60)
153 |         print(f"Testing: {run_name}")
154 |         print(f"Weight:  {best_path}")
155 |         print("=" * 60)
156 | 
157 |         model = YOLO(str(best_path))
158 |         metrics = model.val(
159 |             data=str(DATA_PATH),
160 |             split="test",
161 |             imgsz=IMGSZ,
162 |             batch=BATCH,
163 |             device=DEVICE,
164 |             conf=0.001,
165 |             iou=0.5,
166 |             project=str(TEST_PROJECT),
167 |             name=run_name,
168 |             exist_ok=True,
169 |         )
170 | 
171 |         row = (
172 |             run_name,
173 |             float(metrics.box.map50),
174 |             float(metrics.box.map),
175 |             float(metrics.box.mp),
176 |             float(metrics.box.mr),
177 |         )
178 |         results.append(row)
179 |         print(f"mAP50={row[1]:.4f} | mAP50-95={row[2]:.4f} | P={row[3]:.4f} | R={row[4]:.4f}")
180 | 
181 |     results.sort(key=lambda x: x[1], reverse=True)
182 |     TEST_PROJECT.mkdir(parents=True, exist_ok=True)
183 |     summary_path = TEST_PROJECT / "test_summary.csv"
184 |     with summary_path.open("w", newline="") as f:
185 |         writer = csv.writer(f)
186 |         writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
187 |         writer.writerows(results)
188 | 
189 |     print(f"\nSaved test summary: {summary_path}")
190 |     return TEST_PROJECT
191 | 
192 | 
193 | def main() -> None:
194 |     if RUN_PREPARE_DATA:
195 |         prepare_data()
196 | 
197 |     if not DATA_PATH.is_file():
198 |         raise FileNotFoundError(f"Dataset YAML not found: {DATA_PATH}")
199 | 
200 |     if GENERATED_CONFIG_DIR.exists():
201 |         shutil.rmtree(GENERATED_CONFIG_DIR)
202 | 
203 |     for config in CONFIGS_TO_RUN:
204 |         train_config(config)
205 | 
206 |     run_test_inference()
207 | 
208 | 
209 | if __name__ == "__main__":
210 |     main()
211 | 
```

### misc/train_all_missing.py

Bytes: 7875
SHA-256: 1b864442525acf80c779c9f3db89037fde3ba52c0b2f0c56a82dca0846618692
Lines: 1-237 of 237

```python
  1 | import os
  2 | import sys
  3 | import subprocess
  4 | import csv
  5 | import argparse
  6 | from pathlib import Path
  7 | 
  8 | ROOT = Path("/marimo/yolo_code").resolve()
  9 | LOCAL = ROOT / "models_related" / "ultralytics"
 10 | 
 11 | # Đưa local ultralytics lên đầu sys.path
 12 | sys.path.insert(0, str(LOCAL))
 13 | 
 14 | # Xóa toàn bộ module ultralytics đã cache
 15 | for k in list(sys.modules.keys()):
 16 |     if k == "ultralytics" or k.startswith("ultralytics."):
 17 |         del sys.modules[k]
 18 | 
 19 | # Import lại từ đầu
 20 | import ultralytics
 21 | from ultralytics import YOLO
 22 | 
 23 | print("ultralytics path:", ultralytics.__file__)
 24 | 
 25 | subprocess.run(
 26 |     [
 27 |         "python",
 28 |         "/marimo/yolo_code/prepare_dataset.py",
 29 |         "--root", "/marimo/data",
 30 |         "--out-dir", "datasets/varroa_yolo",
 31 |     ],
 32 |     check=True,
 33 | )
 34 | 
 35 | def find_best_weights(root: Path) -> list[Path]:
 36 |     """Find trained best.pt files across known Ultralytics output roots."""
 37 |     train_roots = [
 38 |         root / "runs/detect/runs/detect/yolo_related/runs/train",
 39 |         root / "runs/detect/yolo_related/runs/train",
 40 |         root / "runs/detect/train",
 41 |         Path("runs/detect/yolo_related/runs/train"),
 42 |         Path("runs/detect/train"),
 43 |     ]
 44 |     best_paths = []
 45 |     seen = set()
 46 |     for train_root in train_roots:
 47 |         for best_path in train_root.glob("*/weights/best.pt"):
 48 |             resolved = best_path.resolve()
 49 |             if resolved not in seen:
 50 |                 seen.add(resolved)
 51 |                 best_paths.append(best_path)
 52 |     return sorted(best_paths)
 53 | 
 54 | 
 55 | def run_test_inference(root: Path, data_test: str) -> Path | None:
 56 |     """Evaluate all discovered best.pt checkpoints on the test split and save a summary CSV."""
 57 |     test_root = root / "runs/detect/yolo_related/runs/test"
 58 |     best_paths = find_best_weights(root)
 59 | 
 60 |     print(f"Found {len(best_paths)} models")
 61 |     if not best_paths:
 62 |         return None
 63 | 
 64 |     results = []
 65 |     for best_path in best_paths:
 66 |         run_name = best_path.parents[1].name
 67 | 
 68 |         print("\n" + "=" * 60)
 69 |         print(f"Testing: {run_name}")
 70 |         print(f"Weight: {best_path}")
 71 |         print("=" * 60)
 72 | 
 73 |         model = YOLO(str(best_path))
 74 |         metrics = model.val(
 75 |             data=data_test,
 76 |             split="test",
 77 |             imgsz=640,
 78 |             batch=16,
 79 |             device="cuda",
 80 |             conf=0.001,
 81 |             iou=0.5,
 82 |             project=str(test_root),
 83 |             name=run_name,
 84 |             exist_ok=True,
 85 |         )
 86 | 
 87 |         map50 = float(metrics.box.map50)
 88 |         map5095 = float(metrics.box.map)
 89 |         precision = float(metrics.box.mp)
 90 |         recall = float(metrics.box.mr)
 91 | 
 92 |         results.append((run_name, map50, map5095, precision, recall))
 93 | 
 94 |         print(f"mAP50:    {map50:.4f}")
 95 |         print(f"mAP50-95: {map5095:.4f}")
 96 |         print(f"Precision:{precision:.4f}")
 97 |         print(f"Recall:   {recall:.4f}")
 98 | 
 99 |     print("\n\n===== SUMMARY =====")
100 |     results = sorted(results, key=lambda x: x[1], reverse=True)
101 | 
102 |     for run_name, map50, map5095, precision, recall in results:
103 |         print(
104 |             f"{run_name:45s} | "
105 |             f"mAP50={map50:.4f} | "
106 |             f"mAP50-95={map5095:.4f} | "
107 |             f"P={precision:.4f} | "
108 |             f"R={recall:.4f}"
109 |         )
110 | 
111 |     test_root.mkdir(parents=True, exist_ok=True)
112 |     summary_path = test_root / "test_summary_missing.csv"
113 |     with summary_path.open("w", newline="") as f:
114 |         writer = csv.writer(f)
115 |         writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
116 |         writer.writerows(results)
117 |     print(f"\nSaved test summary: {summary_path}")
118 |     return test_root
119 | 
120 | 
121 | def upload_runs_to_hf(root: Path, part_name: str, hf_token: str | None = None, repo_id: str | None = None) -> None:
122 |     """Upload train/test outputs to Hugging Face when HF_TOKEN is available."""
123 |     hf_token = hf_token or os.environ.get("HF_TOKEN")
124 |     if not hf_token:
125 |         print("HF_TOKEN is not set; skipping Hugging Face upload.")
126 |         return
127 | 
128 |     try:
129 |         from huggingface_hub import HfApi
130 |     except ImportError:
131 |         print("huggingface_hub is not installed; skipping Hugging Face upload.")
132 |         return
133 | 
134 |     repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-{part_name}-missing")
135 |     folder_path = root / "runs/detect/yolo_related/runs"
136 |     if not folder_path.exists():
137 |         print(f"Upload folder does not exist: {folder_path}")
138 |         return
139 | 
140 |     api = HfApi(token=hf_token)
141 |     api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
142 | 
143 |     print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id} (excluding .pt files)")
144 |     if hasattr(api, "upload_large_folder"):
145 |         api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset", ignore_patterns=["*.pt"])
146 |     else:
147 |         api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset", ignore_patterns=["*.pt"])
148 |     print("Hugging Face upload complete.")
149 | 
150 | 
151 | data_path = "/marimo/data/datasets/varroa_yolo/varroa.yaml"
152 | 
153 | parser = argparse.ArgumentParser()
154 | parser.add_argument(
155 |     "part",
156 |     nargs="?",
157 |     default="1",
158 |     choices=("1", "2", "3"),
159 |     help="Phần models để chạy (1, 2 hoặc 3 cho 3 máy).",
160 | )
161 | parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
162 | parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
163 | args = parser.parse_args()
164 | part = args.part
165 | 
166 | # Danh sách các mô hình nguyên bản còn thiếu từ train_all_full.py.
167 | # train_all_full.py đã có n/s/l cho YOLOv5, YOLOv8, YOLOv10, YOLO11
168 | # và t/s/c cho YOLOv9. Script này bổ sung các scale còn thiếu có sẵn
169 | # trong Ultralytics: m/x cho v5, v8, v10; m/e cho v9.
170 | # Riêng YOLO11 chỉ chạy m/x vì n/s/l đã có trong train_all_full.py.
171 | # YOLOv9 không có weight x chuẩn trong repo Ultralytics này.
172 | all_models = [
173 |     # YOLOv5 (Sử dụng bản 'u' - updated cho tương thích Ultralytics)
174 |     "yolov5mu.pt", "yolov5xu.pt",
175 |     # YOLOv8
176 |     "yolov8m.pt", "yolov8x.pt",
177 |     # YOLOv9
178 |     "yolov9m.pt", "yolov9e.pt",
179 |     # YOLOv10
180 |     "yolov10m.pt", "yolov10x.pt",
181 |     # YOLO11
182 |     "yolo11m.pt", "yolo11x.pt",
183 | ]
184 | 
185 | model_parts = {
186 |     "1": [
187 |         "yolov5mu.pt", "yolov5xu.pt",
188 |         "yolov8m.pt",
189 |     ],
190 |     "2": [
191 |         "yolov9m.pt", "yolov9e.pt",
192 |         "yolov10m.pt", "yolov10x.pt",
193 |     ],
194 |     "3": [
195 |         "yolov8x.pt",
196 |         "yolo11m.pt", "yolo11x.pt",
197 |     ],
198 | }
199 | models_to_run = model_parts[part]
200 | print(f"[*] ĐANG CHẠY MÁY {part} (Phần {part}/3): {len(models_to_run)} models")
201 | 
202 | print(f"[*] Danh sách: {models_to_run}")
203 | 
204 | seeds = [42, 43, 44]
205 | 
206 | for model_name in models_to_run:
207 |     for seed in seeds:
208 |         print("\n" + "="*50)
209 |         print(f"[*] Bắt đầu training với model: {model_name} | Seed: {seed}")
210 |         print("="*50 + "\n")
211 |         
212 |         # Khởi tạo mô hình mới mỗi lần để tránh rò rỉ state
213 |         model = YOLO(model_name)
214 |         
215 |         # Tên run sẽ là tên file bỏ đuôi .pt, ví dụ: yolov8n_seed42
216 |         run_name = f"{model_name.replace('.pt', '')}_seed{seed}"
217 |         
218 |         # Train mô hình với settings default (như yêu cầu), 100 epoch, early stop 20
219 |         model.train(
220 |             data=data_path,
221 |             epochs=100,
222 |             patience=20,     # Early stopping = 20 epochs
223 |             imgsz=640,
224 |             batch=8,
225 |             workers=4,
226 |             device="cuda",
227 |             seed=seed,
228 |             project=str(ROOT / "runs/detect/yolo_related/runs/train"),
229 |             name=run_name,
230 |         )
231 | 
232 | print("\n[*] Hoàn thành training tất cả các mô hình. Đang chạy test inference...")
233 | test_output_root = run_test_inference(ROOT, data_path)
234 | 
235 | if test_output_root is not None:
236 |     upload_runs_to_hf(ROOT, f"baselines-missing-part{part}", hf_token=args.hf_token, repo_id=args.hf_repo_id)
237 | 
```

### misc/train_api_boxgrad_boundary_ring.py

Bytes: 6546
SHA-256: bd2a663a764e8c8a7e1a5898dcae0d68cb15522c077e8db022a0086015e7352d
Lines: 1-187 of 187

```python
  1 | import argparse
  2 | import csv
  3 | import os
  4 | import shutil
  5 | import sys
  6 | from pathlib import Path
  7 | 
  8 | 
  9 | ROOT = Path(__file__).resolve().parents[1]
 10 | LOCAL = ROOT / "models_related" / "ultralytics"
 11 | 
 12 | sys.path.insert(0, str(ROOT))
 13 | sys.path.insert(0, str(LOCAL))
 14 | for k in list(sys.modules.keys()):
 15 |     if k == "ultralytics" or k.startswith("ultralytics."):
 16 |         del sys.modules[k]
 17 | 
 18 | from ultralytics import YOLO
 19 | 
 20 | 
 21 | MODEL_SCALE = "n"
 22 | CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
 23 | DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
 24 | DATASET_OUT_DIR = ROOT / "datasets" / "varroa_yolo"
 25 | DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"
 26 | TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train_api_boxgrad_boundary_ring"
 27 | TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_api_boxgrad_boundary_ring"
 28 | 
 29 | EXPERIMENTS = (
 30 |     {
 31 |         "key": "api_boxgrad",
 32 |         "config": "yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml",
 33 |     },
 34 |     {
 35 |         "key": "api_boxgrad_ensimam",
 36 |         "config": "yolov8_varroa_compare_baseline_p2_api_boxgrad_ensimam.yaml",
 37 |     },
 38 | )
 39 | 
 40 | 
 41 | def parse_args() -> argparse.Namespace:
 42 |     parser = argparse.ArgumentParser(
 43 |         description="Train API boxgrad configs with boundary-ring contrastive localization loss."
 44 |     )
 45 |     parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
 46 |     parser.add_argument("--epochs", type=int, default=100)
 47 |     parser.add_argument("--imgsz", type=int, default=640)
 48 |     parser.add_argument("--batch", type=int, default=16)
 49 |     parser.add_argument("--workers", type=int, default=4)
 50 |     parser.add_argument("--device", default="cuda")
 51 |     parser.add_argument("--patience", type=int, default=20)
 52 |     parser.add_argument("--boundary-contrast", type=float, default=0.02)
 53 |     parser.add_argument("--boundary-ring", type=float, default=1.0)
 54 |     parser.add_argument("--boundary-samples", type=int, default=16)
 55 |     parser.add_argument("--boundary-tau", type=float, default=0.2)
 56 |     parser.add_argument(
 57 |         "--only",
 58 |         choices=tuple(exp["key"] for exp in EXPERIMENTS),
 59 |         default=None,
 60 |         help="Train only one experiment instead of both configs.",
 61 |     )
 62 |     parser.add_argument("--test", action="store_true", help="Evaluate best.pt from runs on the test split.")
 63 |     parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
 64 |     parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics training plots.")
 65 |     parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run names.")
 66 |     parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_dataset.py.")
 67 |     return parser.parse_args()
 68 | 
 69 | 
 70 | def prepare_dataset(skip: bool) -> None:
 71 |     if skip:
 72 |         return
 73 |     from misc.prepare_dataset import prepare_dataset as build_varroa_dataset
 74 | 
 75 |     print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
 76 |     build_varroa_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))
 77 | 
 78 | 
 79 | def scaled_yaml(config_name: str, scale: str) -> Path:
 80 |     src = CONFIG_DIR / config_name
 81 |     dst = CONFIG_DIR / config_name.replace("yolov8_", f"yolov8{scale}_")
 82 |     shutil.copy(src, dst)
 83 |     return dst
 84 | 
 85 | 
 86 | def train_one(exp: dict, args: argparse.Namespace) -> str:
 87 |     model_yaml = scaled_yaml(exp["config"], args.scale)
 88 |     run_name = f"{model_yaml.stem}_bcon{int(args.boundary_contrast * 1000):03d}_ring{args.boundary_ring:g}"
 89 | 
 90 |     model = YOLO(str(model_yaml))
 91 |     model.load(f"yolov8{args.scale}.pt", smart_transfer=True)
 92 | 
 93 |     train_kwargs = {
 94 |         "data": DATA_PATH,
 95 |         "epochs": args.epochs,
 96 |         "imgsz": args.imgsz,
 97 |         "batch": args.batch,
 98 |         "workers": args.workers,
 99 |         "device": args.device,
100 |         "patience": args.patience,
101 |         "bbox_iou_loss": "wiou",
102 |         "wiou_monotonous": False,
103 |         "boundary_contrast": args.boundary_contrast,
104 |         "boundary_levels": 2,
105 |         "boundary_ring": args.boundary_ring,
106 |         "boundary_samples": args.boundary_samples,
107 |         "boundary_tau": args.boundary_tau,
108 |         "project": str(TRAIN_PROJECT),
109 |         "name": run_name,
110 |         "exist_ok": args.exist_ok,
111 |         "val": not args.no_val,
112 |         "plots": not args.no_plots,
113 |     }
114 | 
115 |     print("\n" + "=" * 80)
116 |     print(f"Training {exp['key']}: {run_name}")
117 |     print(f"YAML: {model_yaml}")
118 |     print(f"boundary_contrast={args.boundary_contrast}, boundary_ring={args.boundary_ring}")
119 |     print("=" * 80)
120 |     model.train(**train_kwargs)
121 |     return run_name
122 | 
123 | 
124 | def evaluate_runs(run_names: list[str], args: argparse.Namespace) -> Path:
125 |     rows = []
126 |     for run_name in run_names:
127 |         best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
128 |         if not best.exists():
129 |             print(f"Missing best.pt, skipping test: {best}")
130 |             continue
131 | 
132 |         print("\n" + "=" * 80)
133 |         print(f"Testing {run_name}")
134 |         print("=" * 80)
135 |         metrics = YOLO(str(best)).val(
136 |             data=DATA_PATH,
137 |             split="test",
138 |             imgsz=args.imgsz,
139 |             batch=args.batch,
140 |             device=args.device,
141 |             conf=0.001,
142 |             iou=0.5,
143 |             project=str(TEST_PROJECT),
144 |             name=run_name,
145 |             exist_ok=True,
146 |         )
147 |         rows.append(
148 |             {
149 |                 "run_name": run_name,
150 |                 "mAP50": float(metrics.box.map50),
151 |                 "mAP50-95": float(metrics.box.map),
152 |                 "precision": float(metrics.box.mp),
153 |                 "recall": float(metrics.box.mr),
154 |             }
155 |         )
156 | 
157 |     TEST_PROJECT.mkdir(parents=True, exist_ok=True)
158 |     summary = TEST_PROJECT / "summary.csv"
159 |     with summary.open("w", newline="") as f:
160 |         writer = csv.DictWriter(f, fieldnames=["run_name", "mAP50", "mAP50-95", "precision", "recall"])
161 |         writer.writeheader()
162 |         writer.writerows(rows)
163 | 
164 |     print("\n===== TEST SUMMARY =====")
165 |     for row in sorted(rows, key=lambda x: x["mAP50"], reverse=True):
166 |         print(
167 |             f"{row['run_name']:70s} | "
168 |             f"mAP50={row['mAP50']:.4f} | "
169 |             f"mAP50-95={row['mAP50-95']:.4f} | "
170 |             f"P={row['precision']:.4f} | R={row['recall']:.4f}"
171 |         )
172 |     print(f"Saved summary: {summary}")
173 |     return summary
174 | 
175 | 
176 | def main() -> None:
177 |     args = parse_args()
178 |     prepare_dataset(args.skip_prepare)
179 |     experiments = [exp for exp in EXPERIMENTS if args.only in (None, exp["key"])]
180 |     run_names = [train_one(exp, args) for exp in experiments]
181 |     if args.test:
182 |         evaluate_runs(run_names, args)
183 | 
184 | 
185 | if __name__ == "__main__":
186 |     main()
187 | 
```

### misc/train_boundary_api_compare.py

Bytes: 6701
SHA-256: 6ddbe802c1f56535da8f077b0235d4a9c3b4d708e21f7ca1ce3bf9997a98af6c
Lines: 1-200 of 200

```python
  1 | import argparse
  2 | import csv
  3 | import sys
  4 | from pathlib import Path
  5 | import os
  6 | import shutil
  7 | 
  8 | 
  9 | ROOT = Path(__file__).resolve().parents[1]
 10 | LOCAL = ROOT / "models_related" / "ultralytics"
 11 | 
 12 | sys.path.insert(0, str(ROOT))
 13 | sys.path.insert(0, str(LOCAL))
 14 | for k in list(sys.modules.keys()):
 15 |     if k == "ultralytics" or k.startswith("ultralytics."):
 16 |         del sys.modules[k]
 17 | 
 18 | from ultralytics import YOLO
 19 | 
 20 | 
 21 | MODEL_SCALE = "n"
 22 | CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
 23 | DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
 24 | DATASET_OUT_DIR = ROOT / "datasets" / "varroa_yolo"
 25 | DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"
 26 | TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train"
 27 | TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_boundary_api_compare"
 28 | 
 29 | EXPERIMENTS = (
 30 |     {
 31 |         "key": "baseline",
 32 |         "config": "yolov8_varroa_compare_baseline.yaml",
 33 |         "suffix": "baseline",
 34 |         "use_boundary_api": False,
 35 |     },
 36 |     {
 37 |         "key": "baseline_p2",
 38 |         "config": "yolov8_varroa_compare_baseline_p2.yaml",
 39 |         "suffix": "baseline_p2",
 40 |         "use_boundary_api": False,
 41 |     },
 42 |     {
 43 |         "key": "baseline_p2_boundary",
 44 |         "config": "yolov8_varroa_compare_baseline_p2_boundary.yaml",
 45 |         "suffix": "baseline_p2_boundary",
 46 |         "use_boundary_api": True,
 47 |     },
 48 |     {
 49 |         "key": "baseline_p2_api_boxgrad",
 50 |         "config": "yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml",
 51 |         "suffix": "baseline_p2_api_boxgrad",
 52 |         "use_boundary_api": True,
 53 |     },
 54 |     {
 55 |         "key": "baseline_p2_boundary_api_boxgrad",
 56 |         "config": "yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad.yaml",
 57 |         "suffix": "baseline_p2_boundary_api_boxgrad",
 58 |         "use_boundary_api": True,
 59 |     },
 60 | )
 61 | 
 62 | 
 63 | def parse_args() -> argparse.Namespace:
 64 |     parser = argparse.ArgumentParser(
 65 |         description="Train baseline/P2 comparisons plus boundary and localization-gradient P2 API perturbation."
 66 |     )
 67 |     parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
 68 |     parser.add_argument("--epochs", type=int, default=200)
 69 |     parser.add_argument("--imgsz", type=int, default=640)
 70 |     parser.add_argument("--batch", type=int, default=16)
 71 |     parser.add_argument("--workers", type=int, default=4)
 72 |     parser.add_argument("--device", default="cuda")
 73 |     parser.add_argument("--patience", type=int, default=40)
 74 |     parser.add_argument(
 75 |         "--only",
 76 |         choices=tuple(exp["key"] for exp in EXPERIMENTS),
 77 |         default=None,
 78 |         help="Train only one experiment instead of the full comparison.",
 79 |     )
 80 |     parser.add_argument("--test", action="store_true", help="Evaluate best.pt from both runs on the test split.")
 81 |     parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
 82 |     parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics training plots.")
 83 |     parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run names.")
 84 |     parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_dataset.py.")
 85 |     return parser.parse_args()
 86 | 
 87 | def prepare_dataset(skip: bool) -> None:
 88 |     if skip:
 89 |         return
 90 |     from misc.prepare_dataset import prepare_dataset as build_varroa_dataset
 91 | 
 92 |     print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
 93 |     build_varroa_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))
 94 | 
 95 | 
 96 | def scaled_yaml(config_name: str, scale: str) -> Path:
 97 |     src = CONFIG_DIR / config_name
 98 |     dst = CONFIG_DIR / config_name.replace("yolov8_", f"yolov8{scale}_")
 99 |     shutil.copy(src, dst)
100 |     return dst
101 | 
102 | 
103 | def train_one(exp: dict, args: argparse.Namespace) -> str:
104 |     model_yaml = scaled_yaml(exp["config"], args.scale)
105 |     run_name = model_yaml.stem
106 | 
107 |     model = YOLO(str(model_yaml))
108 |     model.load(f"yolov8{args.scale}.pt", smart_transfer=True)
109 | 
110 |     train_kwargs = {
111 |         "data": DATA_PATH,
112 |         "epochs": args.epochs,
113 |         "imgsz": args.imgsz,
114 |         "batch": args.batch,
115 |         "workers": args.workers,
116 |         "device": args.device,
117 |         "patience": args.patience,
118 |         "bbox_iou_loss": "wiou",
119 |         "wiou_monotonous": False,
120 |         "project": str(TRAIN_PROJECT),
121 |         "name": run_name,
122 |         "exist_ok": args.exist_ok,
123 |         "val": not args.no_val,
124 |         "plots": not args.no_plots,
125 |     }
126 |     train_kwargs["boundary_contrast"] = 0.0
127 | 
128 |     print("\n" + "=" * 80)
129 |     print(f"Training {exp['key']}: {run_name}")
130 |     print(f"YAML: {model_yaml}")
131 |     print(f"Boundary/API module: {exp['use_boundary_api']}")
132 |     print("=" * 80)
133 |     model.train(**train_kwargs)
134 |     return run_name
135 | 
136 | 
137 | def evaluate_runs(run_names: list[str], args: argparse.Namespace) -> Path:
138 |     rows = []
139 |     for run_name in run_names:
140 |         best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
141 |         if not best.exists():
142 |             print(f"Missing best.pt, skipping test: {best}")
143 |             continue
144 | 
145 |         print("\n" + "=" * 80)
146 |         print(f"Testing {run_name}")
147 |         print("=" * 80)
148 |         metrics = YOLO(str(best)).val(
149 |             data=DATA_PATH,
150 |             split="test",
151 |             imgsz=args.imgsz,
152 |             batch=args.batch,
153 |             device=args.device,
154 |             conf=0.001,
155 |             iou=0.5,
156 |             project=str(TEST_PROJECT),
157 |             name=run_name,
158 |             exist_ok=True,
159 |         )
160 |         rows.append(
161 |             {
162 |                 "run_name": run_name,
163 |                 "mAP50": float(metrics.box.map50),
164 |                 "mAP50-95": float(metrics.box.map),
165 |                 "precision": float(metrics.box.mp),
166 |                 "recall": float(metrics.box.mr),
167 |             }
168 |         )
169 | 
170 |     TEST_PROJECT.mkdir(parents=True, exist_ok=True)
171 |     summary = TEST_PROJECT / "summary.csv"
172 |     with summary.open("w", newline="") as f:
173 |         writer = csv.DictWriter(f, fieldnames=["run_name", "mAP50", "mAP50-95", "precision", "recall"])
174 |         writer.writeheader()
175 |         writer.writerows(rows)
176 | 
177 |     print("\n===== TEST SUMMARY =====")
178 |     for row in sorted(rows, key=lambda x: x["mAP50"], reverse=True):
179 |         print(
180 |             f"{row['run_name']:60s} | "
181 |             f"mAP50={row['mAP50']:.4f} | "
182 |             f"mAP50-95={row['mAP50-95']:.4f} | "
183 |             f"P={row['precision']:.4f} | R={row['recall']:.4f}"
184 |         )
185 |     print(f"Saved summary: {summary}")
186 |     return summary
187 | 
188 | 
189 | def main() -> None:
190 |     args = parse_args()
191 |     prepare_dataset(args.skip_prepare)
192 |     experiments = [exp for exp in EXPERIMENTS if args.only in (None, exp["key"])]
193 |     run_names = [train_one(exp, args) for exp in experiments]
194 |     if args.test:
195 |         evaluate_runs(run_names, args)
196 | 
197 | 
198 | if __name__ == "__main__":
199 |     main()
200 | 
```

### misc/train_boundary_api_late.py

Bytes: 6247
SHA-256: fbb3d631161893a5447d8ac05bdbe202ed3a3f3fb87469a977303094e3ec61c0
Lines: 1-187 of 187

```python
  1 | import argparse
  2 | import csv
  3 | import os
  4 | import shutil
  5 | import sys
  6 | from pathlib import Path
  7 | 
  8 | 
  9 | ROOT = Path(__file__).resolve().parents[1]
 10 | LOCAL = ROOT / "models_related" / "ultralytics"
 11 | 
 12 | sys.path.insert(0, str(ROOT))
 13 | sys.path.insert(0, str(LOCAL))
 14 | for k in list(sys.modules.keys()):
 15 |     if k == "ultralytics" or k.startswith("ultralytics."):
 16 |         del sys.modules[k]
 17 | 
 18 | from ultralytics import YOLO
 19 | 
 20 | 
 21 | MODEL_SCALE = "n"
 22 | CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
 23 | DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
 24 | DATASET_OUT_DIR = ROOT / "datasets" / "varroa_yolo"
 25 | DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"
 26 | TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train_boundary_api_late"
 27 | TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_boundary_api_late"
 28 | 
 29 | EXPERIMENTS = (
 30 |     {
 31 |         "key": "baseline_p2_boundary",
 32 |         "config": "yolov8_varroa_compare_baseline_p2_boundary.yaml",
 33 |     },
 34 |     {
 35 |         "key": "baseline_p2_boundary_api_boxgrad",
 36 |         "config": "yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad.yaml",
 37 |     },
 38 |     {
 39 |         "key": "baseline_p2_boundary_late",
 40 |         "config": "yolov8_varroa_compare_baseline_p2_boundary_late.yaml",
 41 |     },
 42 |     {
 43 |         "key": "baseline_p2_boundary_api_boxgrad_late",
 44 |         "config": "yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad_late.yaml",
 45 |     },
 46 | )
 47 | 
 48 | 
 49 | def parse_args() -> argparse.Namespace:
 50 |     parser = argparse.ArgumentParser(
 51 |         description="Train early vs late Boundary/API P2 placement comparisons."
 52 |     )
 53 |     parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
 54 |     parser.add_argument("--epochs", type=int, default=200)
 55 |     parser.add_argument("--imgsz", type=int, default=640)
 56 |     parser.add_argument("--batch", type=int, default=16)
 57 |     parser.add_argument("--workers", type=int, default=4)
 58 |     parser.add_argument("--device", default="cuda")
 59 |     parser.add_argument("--patience", type=int, default=20)
 60 |     parser.add_argument(
 61 |         "--only",
 62 |         choices=tuple(exp["key"] for exp in EXPERIMENTS),
 63 |         default=None,
 64 |         help="Train only one experiment instead of the full late-placement comparison.",
 65 |     )
 66 |     parser.add_argument("--test", action="store_true", help="Evaluate best.pt from runs on the test split.")
 67 |     parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
 68 |     parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics training plots.")
 69 |     parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run names.")
 70 |     parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_dataset.py.")
 71 |     return parser.parse_args()
 72 | 
 73 | 
 74 | def prepare_dataset(skip: bool) -> None:
 75 |     if skip:
 76 |         return
 77 |     from misc.prepare_dataset import prepare_dataset as build_varroa_dataset
 78 | 
 79 |     print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
 80 |     build_varroa_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))
 81 | 
 82 | 
 83 | def scaled_yaml(config_name: str, scale: str) -> Path:
 84 |     src = CONFIG_DIR / config_name
 85 |     dst = CONFIG_DIR / config_name.replace("yolov8_", f"yolov8{scale}_")
 86 |     shutil.copy(src, dst)
 87 |     return dst
 88 | 
 89 | 
 90 | def train_one(exp: dict, args: argparse.Namespace) -> str:
 91 |     model_yaml = scaled_yaml(exp["config"], args.scale)
 92 |     run_name = model_yaml.stem
 93 | 
 94 |     model = YOLO(str(model_yaml))
 95 |     model.load(f"yolov8{args.scale}.pt", smart_transfer=True)
 96 | 
 97 |     train_kwargs = {
 98 |         "data": DATA_PATH,
 99 |         "epochs": args.epochs,
100 |         "imgsz": args.imgsz,
101 |         "batch": args.batch,
102 |         "workers": args.workers,
103 |         "device": args.device,
104 |         "patience": args.patience,
105 |         "bbox_iou_loss": "wiou",
106 |         "wiou_monotonous": False,
107 |         "project": str(TRAIN_PROJECT),
108 |         "name": run_name,
109 |         "exist_ok": args.exist_ok,
110 |         "val": not args.no_val,
111 |         "plots": not args.no_plots,
112 |         "boundary_contrast": 0.0,
113 |     }
114 | 
115 |     print("\n" + "=" * 80)
116 |     print(f"Training {exp['key']}: {run_name}")
117 |     print(f"YAML: {model_yaml}")
118 |     print(f"Patience: {args.patience}")
119 |     print("=" * 80)
120 |     model.train(**train_kwargs)
121 |     return run_name
122 | 
123 | 
124 | def evaluate_runs(run_names: list[str], args: argparse.Namespace) -> Path:
125 |     rows = []
126 |     for run_name in run_names:
127 |         best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
128 |         if not best.exists():
129 |             print(f"Missing best.pt, skipping test: {best}")
130 |             continue
131 | 
132 |         print("\n" + "=" * 80)
133 |         print(f"Testing {run_name}")
134 |         print("=" * 80)
135 |         metrics = YOLO(str(best)).val(
136 |             data=DATA_PATH,
137 |             split="test",
138 |             imgsz=args.imgsz,
139 |             batch=args.batch,
140 |             device=args.device,
141 |             conf=0.001,
142 |             iou=0.5,
143 |             project=str(TEST_PROJECT),
144 |             name=run_name,
145 |             exist_ok=True,
146 |         )
147 |         rows.append(
148 |             {
149 |                 "run_name": run_name,
150 |                 "mAP50": float(metrics.box.map50),
151 |                 "mAP50-95": float(metrics.box.map),
152 |                 "precision": float(metrics.box.mp),
153 |                 "recall": float(metrics.box.mr),
154 |             }
155 |         )
156 | 
157 |     TEST_PROJECT.mkdir(parents=True, exist_ok=True)
158 |     summary = TEST_PROJECT / "summary.csv"
159 |     with summary.open("w", newline="") as f:
160 |         writer = csv.DictWriter(f, fieldnames=["run_name", "mAP50", "mAP50-95", "precision", "recall"])
161 |         writer.writeheader()
162 |         writer.writerows(rows)
163 | 
164 |     print("\n===== TEST SUMMARY =====")
165 |     for row in sorted(rows, key=lambda x: x["mAP50"], reverse=True):
166 |         print(
167 |             f"{row['run_name']:70s} | "
168 |             f"mAP50={row['mAP50']:.4f} | "
169 |             f"mAP50-95={row['mAP50-95']:.4f} | "
170 |             f"P={row['precision']:.4f} | R={row['recall']:.4f}"
171 |         )
172 |     print(f"Saved summary: {summary}")
173 |     return summary
174 | 
175 | 
176 | def main() -> None:
177 |     args = parse_args()
178 |     prepare_dataset(args.skip_prepare)
179 |     experiments = [exp for exp in EXPERIMENTS if args.only in (None, exp["key"])]
180 |     run_names = [train_one(exp, args) for exp in experiments]
181 |     if args.test:
182 |         evaluate_runs(run_names, args)
183 | 
184 | 
185 | if __name__ == "__main__":
186 |     main()
187 | 
```

### misc/train_edge.py

Bytes: 7276
SHA-256: 307c6159b2bc745bbbdffdacf01dc8565624e3e4606437be5f706356acea7aca
Lines: 1-211 of 211

```python
  1 | """Train, test, and upload the YOLOv8 edge RepC2f Varroa configs."""
  2 | 
  3 | import argparse
  4 | import csv
  5 | import os
  6 | import shutil
  7 | import subprocess
  8 | import sys
  9 | from pathlib import Path
 10 | 
 11 | ROOT = Path(os.environ.get("YOLO_CODE_ROOT", "/marimo/yolo_code")).resolve()
 12 | LOCAL = ROOT / "models_related" / "ultralytics"
 13 | 
 14 | sys.path.insert(0, str(LOCAL))
 15 | 
 16 | for key in list(sys.modules.keys()):
 17 |     if key == "ultralytics" or key.startswith("ultralytics."):
 18 |         del sys.modules[key]
 19 | 
 20 | import ultralytics
 21 | from ultralytics import YOLO
 22 | from ultralytics.utils import DEFAULT_CFG_DICT
 23 | 
 24 | print("ultralytics path:", ultralytics.__file__)
 25 | print("bbox_iou_loss:", "bbox_iou_loss" in DEFAULT_CFG_DICT)
 26 | print("wiou_monotonous:", "wiou_monotonous" in DEFAULT_CFG_DICT)
 27 | 
 28 | PREPARE_DATASET = ROOT / "prepare_dataset.py"
 29 | if not PREPARE_DATASET.exists():
 30 |     PREPARE_DATASET = ROOT / "misc" / "prepare_dataset.py"
 31 | 
 32 | subprocess.run(
 33 |     [
 34 |         "python",
 35 |         str(PREPARE_DATASET),
 36 |         "--root",
 37 |         "/marimo/data",
 38 |         "--out-dir",
 39 |         "datasets/varroa_yolo",
 40 |     ],
 41 |     check=True,
 42 | )
 43 | 
 44 | MODEL_SCALE = "n"
 45 | CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
 46 | DATA_PATH = "/marimo/data/datasets/varroa_yolo/varroa.yaml"
 47 | TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train"
 48 | TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_edge"
 49 | 
 50 | EDGE_CONFIGS = [
 51 |     "yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam.yaml",
 52 |     "yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling.yaml",
 53 |     "yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam_late.yaml",
 54 | ]
 55 | 
 56 | 
 57 | def scaled_config_name(config_name: str, scale: str) -> str:
 58 |     """Return the temporary scale-specific YAML name used for a training run."""
 59 |     return config_name.replace("yolov8_", f"yolov8{scale}_", 1)
 60 | 
 61 | 
 62 | def find_best_weights(run_names: list[str]) -> list[Path]:
 63 |     """Find best.pt files for the requested edge runs."""
 64 |     best_paths = []
 65 |     for run_name in run_names:
 66 |         best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
 67 |         if best.exists():
 68 |             best_paths.append(best)
 69 |         else:
 70 |             print(f"Missing best.pt, skipping test: {best}")
 71 |     return best_paths
 72 | 
 73 | 
 74 | def run_test_inference(run_names: list[str], data_test: str) -> Path | None:
 75 |     """Evaluate trained edge checkpoints on the test split and save a summary CSV."""
 76 |     best_paths = find_best_weights(run_names)
 77 |     print(f"\nFound {len(best_paths)} edge models")
 78 |     if not best_paths:
 79 |         return None
 80 | 
 81 |     results = []
 82 |     for best_path in best_paths:
 83 |         run_name = best_path.parents[1].name
 84 | 
 85 |         print("\n" + "=" * 60)
 86 |         print(f"Testing: {run_name}")
 87 |         print(f"Weight: {best_path}")
 88 |         print("=" * 60)
 89 | 
 90 |         metrics = YOLO(str(best_path)).val(
 91 |             data=data_test,
 92 |             split="test",
 93 |             imgsz=640,
 94 |             batch=16,
 95 |             device="cuda",
 96 |             conf=0.001,
 97 |             iou=0.5,
 98 |             project=str(TEST_PROJECT),
 99 |             name=run_name,
100 |             exist_ok=True,
101 |         )
102 | 
103 |         map50 = float(metrics.box.map50)
104 |         map5095 = float(metrics.box.map)
105 |         precision = float(metrics.box.mp)
106 |         recall = float(metrics.box.mr)
107 |         results.append((run_name, map50, map5095, precision, recall))
108 | 
109 |         print(f"mAP50:    {map50:.4f}")
110 |         print(f"mAP50-95: {map5095:.4f}")
111 |         print(f"Precision:{precision:.4f}")
112 |         print(f"Recall:   {recall:.4f}")
113 | 
114 |     results = sorted(results, key=lambda row: row[1], reverse=True)
115 |     print("\n\n===== EDGE SUMMARY =====")
116 |     for run_name, map50, map5095, precision, recall in results:
117 |         print(
118 |             f"{run_name:60s} | "
119 |             f"mAP50={map50:.4f} | "
120 |             f"mAP50-95={map5095:.4f} | "
121 |             f"P={precision:.4f} | "
122 |             f"R={recall:.4f}"
123 |         )
124 | 
125 |     TEST_PROJECT.mkdir(parents=True, exist_ok=True)
126 |     summary_path = TEST_PROJECT / "test_summary_edge.csv"
127 |     with summary_path.open("w", newline="") as handle:
128 |         writer = csv.writer(handle)
129 |         writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
130 |         writer.writerows(results)
131 |     print(f"\nSaved test summary: {summary_path}")
132 |     return TEST_PROJECT
133 | 
134 | 
135 | def upload_runs_to_hf(hf_token: str | None = None, repo_id: str | None = None) -> None:
136 |     """Upload edge train/test outputs to Hugging Face when HF_TOKEN is available."""
137 |     hf_token = hf_token or os.environ.get("HF_TOKEN")
138 |     if not hf_token:
139 |         print("HF_TOKEN is not set; skipping Hugging Face upload.")
140 |         return
141 | 
142 |     try:
143 |         from huggingface_hub import HfApi
144 |     except ImportError:
145 |         print("huggingface_hub is not installed; skipping Hugging Face upload.")
146 |         return
147 | 
148 |     folder_path = ROOT / "runs/detect/yolo_related/runs"
149 |     if not folder_path.exists():
150 |         print(f"Upload folder does not exist: {folder_path}")
151 |         return
152 | 
153 |     repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-edge-yolov8{MODEL_SCALE}")
154 |     api = HfApi(token=hf_token)
155 |     api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
156 | 
157 |     print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id}")
158 |     if hasattr(api, "upload_large_folder"):
159 |         api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
160 |     else:
161 |         api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
162 |     print("Hugging Face upload complete.")
163 | 
164 | 
165 | parser = argparse.ArgumentParser()
166 | parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"), help="YOLOv8 model scale.")
167 | parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
168 | parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
169 | parser.add_argument("--no-test", action="store_true", help="Skip test split evaluation after training.")
170 | parser.add_argument("--no-upload", action="store_true", help="Skip Hugging Face upload after testing.")
171 | args = parser.parse_args()
172 | MODEL_SCALE = args.scale
173 | 
174 | run_names = []
175 | print(f"[*] ĐANG CHẠY EDGE YOLOv8{MODEL_SCALE}: {len(EDGE_CONFIGS)} configs")
176 | 
177 | for config_name in EDGE_CONFIGS:
178 |     print("\n" + "=" * 50)
179 |     print(f"[*] Bắt đầu training với config: {config_name}")
180 |     print("=" * 50 + "\n")
181 | 
182 |     original_yaml = CONFIG_DIR / config_name
183 |     new_config_name = scaled_config_name(config_name, MODEL_SCALE)
184 |     model_yaml = CONFIG_DIR / new_config_name
185 |     shutil.copy(original_yaml, model_yaml)
186 | 
187 |     model = YOLO(str(model_yaml))
188 |     model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)
189 | 
190 |     run_name = new_config_name.replace(".yaml", "")
191 |     run_names.append(run_name)
192 | 
193 |     model.train(
194 |         data=DATA_PATH,
195 |         epochs=200,
196 |         imgsz=640,
197 |         batch=16,
198 |         workers=4,
199 |         device="cuda",
200 |         patience=40,
201 |         bbox_iou_loss="wiou",
202 |         wiou_monotonous=False,
203 |         project=str(TRAIN_PROJECT),
204 |         name=run_name,
205 |     )
206 | 
207 | if not args.no_test:
208 |     test_output_root = run_test_inference(run_names, DATA_PATH)
209 |     if test_output_root is not None and not args.no_upload:
210 |         upload_runs_to_hf(hf_token=args.hf_token, repo_id=args.hf_repo_id)
211 | 
```

### misc/train_only_p3_edge.py

Bytes: 7318
SHA-256: 1c98d8abdf149b6538b30dcff029f6cf96bd5a95b8c1719f2e44b3eccba058e8
Lines: 1-211 of 211

```python
  1 | """Train, test, and upload the YOLOv8 P3-only edge Varroa configs."""
  2 | 
  3 | import argparse
  4 | import csv
  5 | import os
  6 | import shutil
  7 | import subprocess
  8 | import sys
  9 | from pathlib import Path
 10 | 
 11 | ROOT = Path(os.environ.get("YOLO_CODE_ROOT", "/marimo/yolo_code")).resolve()
 12 | LOCAL = ROOT / "models_related" / "ultralytics"
 13 | 
 14 | sys.path.insert(0, str(LOCAL))
 15 | 
 16 | for key in list(sys.modules.keys()):
 17 |     if key == "ultralytics" or key.startswith("ultralytics."):
 18 |         del sys.modules[key]
 19 | 
 20 | import ultralytics
 21 | from ultralytics import YOLO
 22 | from ultralytics.utils import DEFAULT_CFG_DICT
 23 | 
 24 | print("ultralytics path:", ultralytics.__file__)
 25 | print("bbox_iou_loss:", "bbox_iou_loss" in DEFAULT_CFG_DICT)
 26 | print("wiou_monotonous:", "wiou_monotonous" in DEFAULT_CFG_DICT)
 27 | 
 28 | PREPARE_DATASET = ROOT / "prepare_dataset.py"
 29 | if not PREPARE_DATASET.exists():
 30 |     PREPARE_DATASET = ROOT / "misc" / "prepare_dataset.py"
 31 | 
 32 | subprocess.run(
 33 |     [
 34 |         "python",
 35 |         str(PREPARE_DATASET),
 36 |         "--root",
 37 |         "/marimo/data",
 38 |         "--out-dir",
 39 |         "datasets/varroa_yolo",
 40 |     ],
 41 |     check=True,
 42 | )
 43 | 
 44 | MODEL_SCALE = "n"
 45 | CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
 46 | DATA_PATH = "/marimo/data/datasets/varroa_yolo/varroa.yaml"
 47 | TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train"
 48 | TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_p3_edge"
 49 | 
 50 | P3_EDGE_CONFIGS = [
 51 |     "yolov8_varroa_compare_baseline_p3_api_boxgrad_ensimam.yaml",
 52 |     "yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_ensimam.yaml",
 53 |     "yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml",
 54 | ]
 55 | 
 56 | 
 57 | def scaled_config_name(config_name: str, scale: str) -> str:
 58 |     """Return the temporary scale-specific YAML name used for a training run."""
 59 |     return config_name.replace("yolov8_", f"yolov8{scale}_", 1)
 60 | 
 61 | 
 62 | def find_best_weights(run_names: list[str]) -> list[Path]:
 63 |     """Find best.pt files for the requested P3-only runs."""
 64 |     best_paths = []
 65 |     for run_name in run_names:
 66 |         best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
 67 |         if best.exists():
 68 |             best_paths.append(best)
 69 |         else:
 70 |             print(f"Missing best.pt, skipping test: {best}")
 71 |     return best_paths
 72 | 
 73 | 
 74 | def run_test_inference(run_names: list[str], data_test: str) -> Path | None:
 75 |     """Evaluate trained P3-only checkpoints on the test split and save a summary CSV."""
 76 |     best_paths = find_best_weights(run_names)
 77 |     print(f"\nFound {len(best_paths)} P3-only edge models")
 78 |     if not best_paths:
 79 |         return None
 80 | 
 81 |     results = []
 82 |     for best_path in best_paths:
 83 |         run_name = best_path.parents[1].name
 84 | 
 85 |         print("\n" + "=" * 60)
 86 |         print(f"Testing: {run_name}")
 87 |         print(f"Weight: {best_path}")
 88 |         print("=" * 60)
 89 | 
 90 |         metrics = YOLO(str(best_path)).val(
 91 |             data=data_test,
 92 |             split="test",
 93 |             imgsz=640,
 94 |             batch=16,
 95 |             device="cuda",
 96 |             conf=0.001,
 97 |             iou=0.5,
 98 |             project=str(TEST_PROJECT),
 99 |             name=run_name,
100 |             exist_ok=True,
101 |         )
102 | 
103 |         map50 = float(metrics.box.map50)
104 |         map5095 = float(metrics.box.map)
105 |         precision = float(metrics.box.mp)
106 |         recall = float(metrics.box.mr)
107 |         results.append((run_name, map50, map5095, precision, recall))
108 | 
109 |         print(f"mAP50:    {map50:.4f}")
110 |         print(f"mAP50-95: {map5095:.4f}")
111 |         print(f"Precision:{precision:.4f}")
112 |         print(f"Recall:   {recall:.4f}")
113 | 
114 |     results = sorted(results, key=lambda row: row[1], reverse=True)
115 |     print("\n\n===== P3 EDGE SUMMARY =====")
116 |     for run_name, map50, map5095, precision, recall in results:
117 |         print(
118 |             f"{run_name:60s} | "
119 |             f"mAP50={map50:.4f} | "
120 |             f"mAP50-95={map5095:.4f} | "
121 |             f"P={precision:.4f} | "
122 |             f"R={recall:.4f}"
123 |         )
124 | 
125 |     TEST_PROJECT.mkdir(parents=True, exist_ok=True)
126 |     summary_path = TEST_PROJECT / "test_summary_p3_edge.csv"
127 |     with summary_path.open("w", newline="") as handle:
128 |         writer = csv.writer(handle)
129 |         writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
130 |         writer.writerows(results)
131 |     print(f"\nSaved test summary: {summary_path}")
132 |     return TEST_PROJECT
133 | 
134 | 
135 | def upload_runs_to_hf(hf_token: str | None = None, repo_id: str | None = None) -> None:
136 |     """Upload P3-only edge train/test outputs to Hugging Face when HF_TOKEN is available."""
137 |     hf_token = hf_token or os.environ.get("HF_TOKEN")
138 |     if not hf_token:
139 |         print("HF_TOKEN is not set; skipping Hugging Face upload.")
140 |         return
141 | 
142 |     try:
143 |         from huggingface_hub import HfApi
144 |     except ImportError:
145 |         print("huggingface_hub is not installed; skipping Hugging Face upload.")
146 |         return
147 | 
148 |     folder_path = ROOT / "runs/detect/yolo_related/runs"
149 |     if not folder_path.exists():
150 |         print(f"Upload folder does not exist: {folder_path}")
151 |         return
152 | 
153 |     repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-p3-edge-yolov8{MODEL_SCALE}")
154 |     api = HfApi(token=hf_token)
155 |     api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
156 | 
157 |     print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id}")
158 |     if hasattr(api, "upload_large_folder"):
159 |         api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
160 |     else:
161 |         api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
162 |     print("Hugging Face upload complete.")
163 | 
164 | 
165 | parser = argparse.ArgumentParser()
166 | parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"), help="YOLOv8 model scale.")
167 | parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
168 | parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
169 | parser.add_argument("--no-test", action="store_true", help="Skip test split evaluation after training.")
170 | parser.add_argument("--no-upload", action="store_true", help="Skip Hugging Face upload after testing.")
171 | args = parser.parse_args()
172 | MODEL_SCALE = args.scale
173 | 
174 | run_names = []
175 | print(f"[*] ĐANG CHẠY P3-ONLY EDGE YOLOv8{MODEL_SCALE}: {len(P3_EDGE_CONFIGS)} configs")
176 | 
177 | for config_name in P3_EDGE_CONFIGS:
178 |     print("\n" + "=" * 50)
179 |     print(f"[*] Bắt đầu training với config: {config_name}")
180 |     print("=" * 50 + "\n")
181 | 
182 |     original_yaml = CONFIG_DIR / config_name
183 |     new_config_name = scaled_config_name(config_name, MODEL_SCALE)
184 |     model_yaml = CONFIG_DIR / new_config_name
185 |     shutil.copy(original_yaml, model_yaml)
186 | 
187 |     model = YOLO(str(model_yaml))
188 |     model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)
189 | 
190 |     run_name = new_config_name.replace(".yaml", "")
191 |     run_names.append(run_name)
192 | 
193 |     model.train(
194 |         data=DATA_PATH,
195 |         epochs=200,
196 |         imgsz=640,
197 |         batch=16,
198 |         workers=4,
199 |         device="cuda",
200 |         patience=40,
201 |         bbox_iou_loss="wiou",
202 |         wiou_monotonous=False,
203 |         project=str(TRAIN_PROJECT),
204 |         name=run_name,
205 |     )
206 | 
207 | if not args.no_test:
208 |     test_output_root = run_test_inference(run_names, DATA_PATH)
209 |     if test_output_root is not None and not args.no_upload:
210 |         upload_runs_to_hf(hf_token=args.hf_token, repo_id=args.hf_repo_id)
211 | 
```

### misc/train_star_boundary.py

Bytes: 1562
SHA-256: d67b457a78c48cc918db1a538ee5081e788d960e07164b8f0b40b2129c82aac6
Lines: 1-64 of 64

```python
 1 | import sys
 2 | import shutil
 3 | import subprocess
 4 | from pathlib import Path
 5 | 
 6 | 
 7 | ROOT = Path("/marimo/yolo_code").resolve()
 8 | LOCAL = ROOT / "models_related" / "ultralytics"
 9 | 
10 | sys.path.insert(0, str(LOCAL))
11 | for k in list(sys.modules.keys()):
12 |     if k == "ultralytics" or k.startswith("ultralytics."):
13 |         del sys.modules[k]
14 | 
15 | subprocess.run(
16 |     [
17 |         "python",
18 |         str(ROOT / "prepare_dataset.py"),
19 |         "--root",
20 |         "/marimo/data",
21 |         "--out-dir",
22 |         "datasets/varroa_yolo",
23 |     ],
24 |     check=True,
25 | )
26 | 
27 | from ultralytics import YOLO
28 | 
29 | 
30 | MODEL_SCALE = "n"
31 | config_dir = ROOT / "models_related" / "models_config"
32 | data_path = "/marimo/data/datasets/varroa_yolo/varroa.yaml"
33 | 
34 | config_name = "yolov8_varroa_kvheads_repc2f_bifpn_asf_ensimam_star.yaml"
35 | original_yaml = config_dir / config_name
36 | new_config_name = config_name.replace("yolov8_", f"yolov8{MODEL_SCALE}_")
37 | model_yaml = config_dir / new_config_name
38 | shutil.copy(original_yaml, model_yaml)
39 | 
40 | model = YOLO(str(model_yaml))
41 | model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)
42 | 
43 | run_name = new_config_name.replace(".yaml", "") + "_bcon005"
44 | 
45 | model.train(
46 |     data=data_path,
47 |     epochs=200,
48 |     imgsz=640,
49 |     batch=16,
50 |     workers=4,
51 |     device="cuda",
52 |     patience=40,
53 |     bbox_iou_loss="wiou",
54 |     wiou_monotonous=False,
55 |     # EXPERIMENTAL: boundary-aware contrastive localization ablation.
56 |     boundary_contrast=0.05,
57 |     boundary_levels=2,
58 |     boundary_ring=1.0,
59 |     boundary_samples=16,
60 |     boundary_tau=0.2,
61 |     project=str(ROOT / "runs/detect/yolo_related/runs/train"),
62 |     name=run_name,
63 | )
64 | 
```

### misc/train_star_lqm.py

Bytes: 3327
SHA-256: d96fb7d6349aa768baee12b9b252bfc6c1f17a85544a2b1e025a859f0b6cd78c
Lines: 1-105 of 105

```python
  1 | import argparse
  2 | import os
  3 | import sys
  4 | import shutil
  5 | import subprocess
  6 | from pathlib import Path
  7 | 
  8 | 
  9 | ROOT = Path("/marimo/yolo_code").resolve()
 10 | LOCAL = ROOT / "models_related" / "ultralytics"
 11 | 
 12 | sys.path.insert(0, str(LOCAL))
 13 | for k in list(sys.modules.keys()):
 14 |     if k == "ultralytics" or k.startswith("ultralytics."):
 15 |         del sys.modules[k]
 16 | 
 17 | subprocess.run(
 18 |     [
 19 |         "python",
 20 |         str(ROOT / "prepare_dataset.py"),
 21 |         "--root",
 22 |         "/marimo/data",
 23 |         "--out-dir",
 24 |         "datasets/varroa_yolo",
 25 |     ],
 26 |     check=True,
 27 | )
 28 | 
 29 | from ultralytics import YOLO
 30 | 
 31 | 
 32 | MODEL_SCALE = "n"
 33 | config_dir = ROOT / "models_related" / "models_config"
 34 | data_path = "/marimo/data/datasets/varroa_yolo/varroa.yaml"
 35 | 
 36 | 
 37 | def upload_runs_to_hf(root: Path, hf_token: str | None = None, repo_id: str | None = None) -> None:
 38 |     """Upload LQM train outputs to Hugging Face when HF_TOKEN is available."""
 39 |     hf_token = hf_token or os.environ.get("HF_TOKEN")
 40 |     if not hf_token:
 41 |         print("HF_TOKEN is not set; skipping Hugging Face upload.")
 42 |         return
 43 | 
 44 |     try:
 45 |         from huggingface_hub import HfApi
 46 |     except ImportError:
 47 |         print("huggingface_hub is not installed; skipping Hugging Face upload.")
 48 |         return
 49 | 
 50 |     repo_id = repo_id or os.environ.get("HF_REPO_ID", f"duyle2408/varroa-yolo-star-lqm-yolov8{MODEL_SCALE}")
 51 |     folder_path = root / "runs/detect/yolo_related/runs"
 52 |     if not folder_path.exists():
 53 |         print(f"Upload folder does not exist: {folder_path}")
 54 |         return
 55 | 
 56 |     api = HfApi(token=hf_token)
 57 |     api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
 58 | 
 59 |     print(f"Uploading {folder_path} to https://huggingface.co/datasets/{repo_id}")
 60 |     if hasattr(api, "upload_large_folder"):
 61 |         api.upload_large_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
 62 |     else:
 63 |         api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, repo_type="dataset")
 64 |     print("Hugging Face upload complete.")
 65 | 
 66 | 
 67 | parser = argparse.ArgumentParser()
 68 | parser.add_argument("--hf-token", default=None, help="Hugging Face token for upload. Falls back to HF_TOKEN env var.")
 69 | parser.add_argument("--hf-repo-id", default=None, help="Target Hugging Face repo id. Falls back to HF_REPO_ID or auto name.")
 70 | args = parser.parse_args()
 71 | 
 72 | config_name = "yolov8_varroa_kvheads_repc2f_bifpn_asf_ensimam_star.yaml"
 73 | original_yaml = config_dir / config_name
 74 | new_config_name = config_name.replace("yolov8_", f"yolov8{MODEL_SCALE}_")
 75 | model_yaml = config_dir / new_config_name
 76 | shutil.copy(original_yaml, model_yaml)
 77 | 
 78 | model = YOLO(str(model_yaml))
 79 | model.load(f"yolov8{MODEL_SCALE}.pt", smart_transfer=True)
 80 | 
 81 | run_name = new_config_name.replace(".yaml", "") + "_lqm005"
 82 | 
 83 | model.train(
 84 |     data=data_path,
 85 |     epochs=200,
 86 |     imgsz=640,
 87 |     batch=16,
 88 |     workers=4,
 89 |     device="cuda",
 90 |     patience=40,
 91 |     bbox_iou_loss="wiou",
 92 |     wiou_monotonous=False,
 93 |     # EXPERIMENTAL: train-only Localization Quality Map supervision.
 94 |     loc_quality=0.05,
 95 |     loc_quality_levels=2,
 96 |     loc_quality_sigma=0.45,
 97 |     loc_quality_loss="mse",
 98 |     # Keep boundary contrast disabled for a clean LQM ablation.
 99 |     boundary_contrast=0.0,
100 |     project=str(ROOT / "runs/detect/yolo_related/runs/train"),
101 |     name=run_name,
102 | )
103 | 
104 | upload_runs_to_hf(ROOT, hf_token=args.hf_token, repo_id=args.hf_repo_id)
105 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam_late.yaml

Bytes: 1697
SHA-256: f148c06372160096778542af36ad7bde32cda8ffbab2ab68fbef8e2a5075abb4
Lines: 1-54 of 54

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+late API box-gradient+edge EnSimAM.
 2 | # Applies API only to the final P2 Detect input, after bottom-up PAN.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, EnSimAMEdgeRepC2f, [128, True]] # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, EnSimAMEdgeRepC2f, [256]] # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, EnSimAMEdgeRepC2f, [128]] # 18-P2/4
36 | 
37 |   - [-1, 1, Conv, [128, 3, 2]]
38 |   - [[-1, 15], 1, Concat, [1]]
39 |   - [-1, 3, EnSimAMEdgeRepC2f, [256]] # 21-P3/8
40 | 
41 |   - [-1, 1, Conv, [256, 3, 2]]
42 |   - [[-1, 12], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 24-P4/16
44 | 
45 |   - [-1, 1, Conv, [512, 3, 2]]
46 |   - [[-1, 9], 1, Concat, [1]]
47 |   - [-1, 3, RepC2f, [1024]]         # 27-P5/32
48 | 
49 |   - [18, 1, EnSimAM, []]            # 28-P2 final
50 |   - [21, 1, EnSimAM, []]            # 29-P3 final
51 |   - [28, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 30-late P2 localization API
52 | 
53 |   - [[30, 29, 24, 27], 1, Detect, [nc]]
54 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_ensimam.yaml

Bytes: 1705
SHA-256: 854c8d90c867b169466af7acdcafb43679e79bc85bdad8dba73fc590e62b6c57
Lines: 1-54 of 54

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+edge EnSimAM.
 2 | # Adds EnSimAMEdgeRepC2f on selected RepC2f branches while keeping final P2/P3 EnSimAM.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, EnSimAMEdgeRepC2f, [128, True]] # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, EnSimAMEdgeRepC2f, [256]] # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, EnSimAMEdgeRepC2f, [128]] # 18-P2/4
36 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 19-P2 localization API
37 | 
38 |   - [-1, 1, Conv, [128, 3, 2]]
39 |   - [[-1, 15], 1, Concat, [1]]
40 |   - [-1, 3, EnSimAMEdgeRepC2f, [256]] # 22-P3/8
41 | 
42 |   - [-1, 1, Conv, [256, 3, 2]]
43 |   - [[-1, 12], 1, Concat, [1]]
44 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
45 | 
46 |   - [-1, 1, Conv, [512, 3, 2]]
47 |   - [[-1, 9], 1, Concat, [1]]
48 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
49 | 
50 |   - [19, 1, EnSimAM, []]            # 29-P2 final
51 |   - [22, 1, EnSimAM, []]            # 30-P3 final
52 | 
53 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
54 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_quality_iou.yaml

Bytes: 2168
SHA-256: ba4c3d377056efac7b5467a01917b546cacc8946f6f88e025027839830a0827b
Lines: 1-70 of 70

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+pooling edge with box-aware IoU quality head.
 2 | # Quality predicts per-anchor localization tightness from feature maps plus DFL/box summary cues.
 3 | 
 4 | nc: 1
 5 | quality_head: true
 6 | quality_box_features: true
 7 | quality_box_detach: true
 8 | quality_loss: bce_balanced
 9 | quality_gain: 0.5
10 | quality_neg_gain: 0.10
11 | quality_pos_iou_thr: 0.5
12 | quality_hard_neg_iou_thr: 0.3
13 | quality_hard_neg_score_thr: 0.05
14 | quality_target_mode: ap75_ramp
15 | quality_ramp_low: 0.50
16 | quality_ramp_high: 0.75
17 | quality_neg_mode: hard
18 | quality_score_mode: sqrt_cls_mul_q
19 | quality_detach_target: true
20 | 
21 | scales:
22 |   n: [0.33, 0.25, 1024]
23 |   s: [0.33, 0.50, 1024]
24 |   m: [0.67, 0.75, 768]
25 |   l: [1.00, 1.00, 512]
26 |   x: [1.00, 1.25, 512]
27 | 
28 | backbone:
29 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
30 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
31 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
32 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
33 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
34 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
35 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
36 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
37 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
38 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
39 | 
40 | head:
41 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
42 |   - [[-1, 6], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
44 | 
45 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
46 |   - [[-1, 4], 1, Concat, [1]]
47 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
48 | 
49 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
50 |   - [[-1, 2], 1, Concat, [1]]
51 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
52 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad, True]] # 19-P2 localization API (partial_forward=True)
53 | 
54 |   - [-1, 1, Conv, [128, 3, 2]]
55 |   - [[-1, 15], 1, Concat, [1]]
56 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 22-P3/8
57 | 
58 |   - [-1, 1, Conv, [256, 3, 2]]
59 |   - [[-1, 12], 1, Concat, [1]]
60 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
61 | 
62 |   - [-1, 1, Conv, [512, 3, 2]]
63 |   - [[-1, 9], 1, Concat, [1]]
64 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
65 | 
66 |   - [19, 1, EnSimAM, []]            # 29-P2 final
67 |   - [22, 1, EnSimAM, []]            # 30-P3 final
68 | 
69 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
70 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_add.yaml

Bytes: 2060
SHA-256: 438b60ae12770626ef1c097a96646389f790f97a4bdfb36cbc2f16b014e1e05b
Lines: 1-69 of 69

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+pooling edge + VFL + cls geometry add fusion.
 2 | # Same as the P2/P3/P4/P5 pooling-edge VFL architecture, but without the P2 API box-gradient block.
 3 | # Detect cls head receives DFL expected l/t/r/b geometry through additive fusion.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | # Loss/training knobs. These are consumed when passed as train overrides/cfg.
14 | vfl: true
15 | vfl_alpha: 0.75
16 | vfl_gamma: 2.0
17 | vfl_iou_detach: true
18 | vfl_pos_q_weight: true
19 | tal_topk: 10
20 | tal_beta: 6.0
21 | vfl_weight_box_by_q: false
22 | box_q_weight_min: 0.25
23 | cls_geometry_fuse: true
24 | cls_geometry_mode: add
25 | cls_geometry_detach: true
26 | cls_deform_geometry: false
27 | 
28 | backbone:
29 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
30 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
31 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
32 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
33 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
34 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
35 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
36 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
37 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
38 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
39 | 
40 | head:
41 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
42 |   - [[-1, 6], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
44 | 
45 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
46 |   - [[-1, 4], 1, Concat, [1]]
47 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
48 | 
49 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
50 |   - [[-1, 2], 1, Concat, [1]]
51 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
52 | 
53 |   - [-1, 1, Conv, [128, 3, 2]]
54 |   - [[-1, 15], 1, Concat, [1]]
55 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 21-P3/8
56 | 
57 |   - [-1, 1, Conv, [256, 3, 2]]
58 |   - [[-1, 12], 1, Concat, [1]]
59 |   - [-1, 3, RepC2f, [512]]          # 24-P4/16
60 | 
61 |   - [-1, 1, Conv, [512, 3, 2]]
62 |   - [[-1, 9], 1, Concat, [1]]
63 |   - [-1, 3, RepC2f, [1024]]         # 27-P5/32
64 | 
65 |   - [18, 1, EnSimAM, []]            # 28-P2 final
66 |   - [21, 1, EnSimAM, []]            # 29-P3 final
67 | 
68 |   - [[28, 29, 24, 27], 1, Detect, [nc]]
69 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_clsgeom_concat.yaml

Bytes: 2064
SHA-256: 229179c45f7d60bd39f338737162c50ad6dd14283b447569aad2ce658cbbb442
Lines: 1-69 of 69

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+pooling edge + VFL + cls geometry concat fusion.
 2 | # Same as the P2/P3/P4/P5 pooling-edge VFL architecture, but without the P2 API box-gradient block.
 3 | # Detect cls head receives DFL expected l/t/r/b geometry through concat fusion.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | # Loss/training knobs. These are consumed when passed as train overrides/cfg.
14 | vfl: true
15 | vfl_alpha: 0.75
16 | vfl_gamma: 2.0
17 | vfl_iou_detach: true
18 | vfl_pos_q_weight: true
19 | tal_topk: 10
20 | tal_beta: 6.0
21 | vfl_weight_box_by_q: false
22 | box_q_weight_min: 0.25
23 | cls_geometry_fuse: true
24 | cls_geometry_mode: concat
25 | cls_geometry_detach: true
26 | cls_deform_geometry: false
27 | 
28 | backbone:
29 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
30 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
31 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
32 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
33 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
34 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
35 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
36 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
37 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
38 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
39 | 
40 | head:
41 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
42 |   - [[-1, 6], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
44 | 
45 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
46 |   - [[-1, 4], 1, Concat, [1]]
47 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
48 | 
49 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
50 |   - [[-1, 2], 1, Concat, [1]]
51 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
52 | 
53 |   - [-1, 1, Conv, [128, 3, 2]]
54 |   - [[-1, 15], 1, Concat, [1]]
55 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 21-P3/8
56 | 
57 |   - [-1, 1, Conv, [256, 3, 2]]
58 |   - [[-1, 12], 1, Concat, [1]]
59 |   - [-1, 3, RepC2f, [512]]          # 24-P4/16
60 | 
61 |   - [-1, 1, Conv, [512, 3, 2]]
62 |   - [[-1, 9], 1, Concat, [1]]
63 |   - [-1, 3, RepC2f, [1024]]         # 27-P5/32
64 | 
65 |   - [18, 1, EnSimAM, []]            # 28-P2 final
66 |   - [21, 1, EnSimAM, []]            # 29-P3 final
67 | 
68 |   - [[28, 29, 24, 27], 1, Detect, [nc]]
69 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b2.yaml

Bytes: 2070
SHA-256: a20f905e76fadbf3c3265743ee68c703cc2fb2ae868377b79a1ac3bfc0eb72be
Lines: 1-67 of 67

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+pooling edge + VFL TAL alpha=0 beta=2.0.
 2 | # Same architecture as yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl.yaml.
 3 | # Ablates TaskAlignedAssigner classification feedback by using tal_alpha=0.0.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | # Loss/training knobs. These are consumed when passed as train overrides/cfg.
14 | vfl: true
15 | vfl_alpha: 0.75
16 | vfl_gamma: 2.0
17 | vfl_iou_detach: true
18 | vfl_pos_q_weight: true
19 | tal_topk: 10
20 | tal_alpha: 0.0
21 | tal_beta: 2.0
22 | vfl_weight_box_by_q: false
23 | box_q_weight_min: 0.25
24 | 
25 | backbone:
26 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
27 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
28 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
29 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
30 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
31 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
32 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
33 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
34 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
35 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
36 | 
37 | head:
38 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
39 |   - [[-1, 6], 1, Concat, [1]]
40 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
41 | 
42 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
43 |   - [[-1, 4], 1, Concat, [1]]
44 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
45 | 
46 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
47 |   - [[-1, 2], 1, Concat, [1]]
48 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
49 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 19-P2 localization API
50 | 
51 |   - [-1, 1, Conv, [128, 3, 2]]
52 |   - [[-1, 15], 1, Concat, [1]]
53 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 22-P3/8
54 | 
55 |   - [-1, 1, Conv, [256, 3, 2]]
56 |   - [[-1, 12], 1, Concat, [1]]
57 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
58 | 
59 |   - [-1, 1, Conv, [512, 3, 2]]
60 |   - [[-1, 9], 1, Concat, [1]]
61 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
62 | 
63 |   - [19, 1, EnSimAM, []]            # 29-P2 final
64 |   - [22, 1, EnSimAM, []]            # 30-P3 final
65 | 
66 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
67 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b4.yaml

Bytes: 2070
SHA-256: 4e019dd5d3d27e10f904a3e829ad6e149863ac5700dfbcf5b03f979652455c4f
Lines: 1-67 of 67

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+pooling edge + VFL TAL alpha=0 beta=4.0.
 2 | # Same architecture as yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl.yaml.
 3 | # Ablates TaskAlignedAssigner classification feedback by using tal_alpha=0.0.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | # Loss/training knobs. These are consumed when passed as train overrides/cfg.
14 | vfl: true
15 | vfl_alpha: 0.75
16 | vfl_gamma: 2.0
17 | vfl_iou_detach: true
18 | vfl_pos_q_weight: true
19 | tal_topk: 10
20 | tal_alpha: 0.0
21 | tal_beta: 4.0
22 | vfl_weight_box_by_q: false
23 | box_q_weight_min: 0.25
24 | 
25 | backbone:
26 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
27 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
28 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
29 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
30 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
31 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
32 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
33 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
34 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
35 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
36 | 
37 | head:
38 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
39 |   - [[-1, 6], 1, Concat, [1]]
40 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
41 | 
42 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
43 |   - [[-1, 4], 1, Concat, [1]]
44 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
45 | 
46 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
47 |   - [[-1, 2], 1, Concat, [1]]
48 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
49 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 19-P2 localization API
50 | 
51 |   - [-1, 1, Conv, [128, 3, 2]]
52 |   - [[-1, 15], 1, Concat, [1]]
53 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 22-P3/8
54 | 
55 |   - [-1, 1, Conv, [256, 3, 2]]
56 |   - [[-1, 12], 1, Concat, [1]]
57 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
58 | 
59 |   - [-1, 1, Conv, [512, 3, 2]]
60 |   - [[-1, 9], 1, Concat, [1]]
61 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
62 | 
63 |   - [19, 1, EnSimAM, []]            # 29-P2 final
64 |   - [22, 1, EnSimAM, []]            # 30-P3 final
65 | 
66 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
67 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_tal_a0_b6.yaml

Bytes: 2070
SHA-256: 3f3df6481c8551790b834921c57792ad86cb16ae6ec9da94dc3d9eb0602b5fa1
Lines: 1-67 of 67

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+pooling edge + VFL TAL alpha=0 beta=6.0.
 2 | # Same architecture as yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl.yaml.
 3 | # Ablates TaskAlignedAssigner classification feedback by using tal_alpha=0.0.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | # Loss/training knobs. These are consumed when passed as train overrides/cfg.
14 | vfl: true
15 | vfl_alpha: 0.75
16 | vfl_gamma: 2.0
17 | vfl_iou_detach: true
18 | vfl_pos_q_weight: true
19 | tal_topk: 10
20 | tal_alpha: 0.0
21 | tal_beta: 6.0
22 | vfl_weight_box_by_q: false
23 | box_q_weight_min: 0.25
24 | 
25 | backbone:
26 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
27 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
28 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
29 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
30 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
31 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
32 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
33 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
34 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
35 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
36 | 
37 | head:
38 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
39 |   - [[-1, 6], 1, Concat, [1]]
40 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
41 | 
42 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
43 |   - [[-1, 4], 1, Concat, [1]]
44 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
45 | 
46 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
47 |   - [[-1, 2], 1, Concat, [1]]
48 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
49 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 19-P2 localization API
50 | 
51 |   - [-1, 1, Conv, [128, 3, 2]]
52 |   - [[-1, 15], 1, Concat, [1]]
53 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 22-P3/8
54 | 
55 |   - [-1, 1, Conv, [256, 3, 2]]
56 |   - [[-1, 12], 1, Concat, [1]]
57 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
58 | 
59 |   - [-1, 1, Conv, [512, 3, 2]]
60 |   - [[-1, 9], 1, Concat, [1]]
61 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
62 | 
63 |   - [19, 1, EnSimAM, []]            # 29-P2 final
64 |   - [22, 1, EnSimAM, []]            # 30-P3 final
65 | 
66 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
67 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_train.yaml

Bytes: 792
SHA-256: 13b90f9a66e47c9f3549a9c69175abff9b099993e7abea0edd4a6c1fdc637386
Lines: 1-31 of 31

```yaml
 1 | # Ultralytics train config for P2 API boxgrad edge pooling with VFL enabled.
 2 | # Use with:
 3 | # yolo detect train cfg=models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl_train.yaml
 4 | 
 5 | model: models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl.yaml
 6 | data: datasets/varroa_yolo/varroa.yaml
 7 | task: detect
 8 | mode: train
 9 | 
10 | epochs: 200
11 | imgsz: 640
12 | batch: 16
13 | workers: 4
14 | device: cuda
15 | patience: 40
16 | project: runs/detect/yolo_related/runs/train
17 | name: yolov8n_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl
18 | 
19 | bbox_iou_loss: wiou
20 | wiou_monotonous: false
21 | 
22 | vfl: true
23 | vfl_alpha: 0.75
24 | vfl_gamma: 2.0
25 | vfl_iou_detach: true
26 | vfl_pos_q_weight: true
27 | tal_topk: 10
28 | tal_beta: 6.0
29 | vfl_weight_box_by_q: false
30 | box_q_weight_min: 0.25
31 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling_vfl.yaml

Bytes: 2035
SHA-256: 14c51778246cb4c9312b8680567dd1f4bd03f130565360ecbdd1ae951a6cb62f
Lines: 1-66 of 66

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+pooling edge + VFL.
 2 | # Same architecture as yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling.yaml.
 3 | # VFL makes classification targets quality-aware without adding an inference head.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | # Loss/training knobs. These are consumed when passed as train overrides/cfg.
14 | vfl: true
15 | vfl_alpha: 0.75
16 | vfl_gamma: 2.0
17 | vfl_iou_detach: true
18 | vfl_pos_q_weight: true
19 | tal_topk: 10
20 | tal_beta: 6.0
21 | vfl_weight_box_by_q: false
22 | box_q_weight_min: 0.25
23 | 
24 | backbone:
25 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
26 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
27 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
28 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
29 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
30 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
31 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
32 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
33 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
34 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
35 | 
36 | head:
37 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
38 |   - [[-1, 6], 1, Concat, [1]]
39 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
40 | 
41 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
42 |   - [[-1, 4], 1, Concat, [1]]
43 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
44 | 
45 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
46 |   - [[-1, 2], 1, Concat, [1]]
47 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
48 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 19-P2 localization API
49 | 
50 |   - [-1, 1, Conv, [128, 3, 2]]
51 |   - [[-1, 15], 1, Concat, [1]]
52 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 22-P3/8
53 | 
54 |   - [-1, 1, Conv, [256, 3, 2]]
55 |   - [[-1, 12], 1, Concat, [1]]
56 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
57 | 
58 |   - [-1, 1, Conv, [512, 3, 2]]
59 |   - [[-1, 9], 1, Concat, [1]]
60 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
61 | 
62 |   - [19, 1, EnSimAM, []]            # 29-P2 final
63 |   - [22, 1, EnSimAM, []]            # 30-P3 final
64 | 
65 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
66 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_edge_pooling.yaml

Bytes: 1734
SHA-256: 56e9e0ddcea2d42c8b0306423164f74bc2886b6dc6b0d5656e61022f2cc240e1
Lines: 1-54 of 54

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+pooling edge.
 2 | # Adds PoolingEdgeRepC2f on selected RepC2f branches while keeping final P2/P3 EnSimAM.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
36 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad, True]] # 19-P2 localization API (partial_forward=True)
37 | 
38 |   - [-1, 1, Conv, [128, 3, 2]]
39 |   - [[-1, 15], 1, Concat, [1]]
40 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 22-P3/8
41 | 
42 |   - [-1, 1, Conv, [256, 3, 2]]
43 |   - [[-1, 12], 1, Concat, [1]]
44 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
45 | 
46 |   - [-1, 1, Conv, [512, 3, 2]]
47 |   - [[-1, 9], 1, Concat, [1]]
48 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
49 | 
50 |   - [19, 1, EnSimAM, []]            # 29-P2 final
51 |   - [22, 1, EnSimAM, []]            # 30-P3 final
52 | 
53 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
54 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad_ensimam.yaml

Bytes: 1674
SHA-256: fe06e7f33dba917ffe9e5025bb52e3947a6d3b4cc3824d20322a52014740fb63
Lines: 1-54 of 54

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient+EnSimAM.
 2 | # Replaces KVCompressedAttention on final P2/P3 Detect inputs with EnSimAM.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, RepC2f, [256]]          # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
36 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 19-P2 localization API
37 | 
38 |   - [-1, 1, Conv, [128, 3, 2]]
39 |   - [[-1, 15], 1, Concat, [1]]
40 |   - [-1, 3, RepC2f, [256]]          # 22-P3/8
41 | 
42 |   - [-1, 1, Conv, [256, 3, 2]]
43 |   - [[-1, 12], 1, Concat, [1]]
44 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
45 | 
46 |   - [-1, 1, Conv, [512, 3, 2]]
47 |   - [[-1, 9], 1, Concat, [1]]
48 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
49 | 
50 |   - [19, 1, EnSimAM, []]            # 29-P2 final
51 |   - [22, 1, EnSimAM, []]            # 30-P3 final
52 | 
53 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
54 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_api_boxgrad.yaml

Bytes: 1678
SHA-256: b0e2b660061dfb03d006eb145f511d05c155ee5af8436a50bd7da0c808d4b58b
Lines: 1-54 of 54

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+API box-gradient.
 2 | # API perturbs P2 from box+dfl gradients and regularizes localization only.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, RepC2f, [256]]          # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
36 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 19-P2 localization API
37 | 
38 |   - [-1, 1, Conv, [128, 3, 2]]
39 |   - [[-1, 15], 1, Concat, [1]]
40 |   - [-1, 3, RepC2f, [256]]          # 22-P3/8
41 | 
42 |   - [-1, 1, Conv, [256, 3, 2]]
43 |   - [[-1, 12], 1, Concat, [1]]
44 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
45 | 
46 |   - [-1, 1, Conv, [512, 3, 2]]
47 |   - [[-1, 9], 1, Concat, [1]]
48 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
49 | 
50 |   - [19, 1, KVCompressedAttention, [128, 4, 8, dwconv]]
51 |   - [22, 1, KVCompressedAttention, [256, 4, 4, dwconv]]
52 | 
53 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
54 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad.yaml

Bytes: 1458
SHA-256: 0ee5c67da2fee7de0dc65a544f729f3191d534fd3970617b0b421dd68f6758b6
Lines: 1-48 of 48

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+boundary+API box-gradient.
 2 | # Boundary is train-only and API regularizes box+dfl localization only.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]
14 |   - [-1, 1, Conv, [128, 3, 2]]
15 |   - [-1, 3, RepC2f, [128, True]]
16 |   - [-1, 1, Conv, [256, 3, 2]]
17 |   - [-1, 6, RepC2f, [256, True]]
18 |   - [-1, 1, Conv, [512, 3, 2]]
19 |   - [-1, 6, RepC2f, [512, True]]
20 |   - [-1, 1, Conv, [1024, 3, 2]]
21 |   - [-1, 3, RepC2f, [1024, True]]
22 |   - [-1, 1, SPPF, [1024, 5]]
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]
28 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
29 |   - [[-1, 4], 1, Concat, [1]]
30 |   - [-1, 3, RepC2f, [256]]
31 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
32 |   - [[-1, 2], 1, Concat, [1]]
33 |   - [-1, 3, RepC2f, [128]]
34 |   - [-1, 1, BoundaryFeatureBlock, [1.0, 0.25, 4, 0.0, 0.01]]
35 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]]
36 |   - [-1, 1, Conv, [128, 3, 2]]
37 |   - [[-1, 15], 1, Concat, [1]]
38 |   - [-1, 3, RepC2f, [256]]
39 |   - [-1, 1, Conv, [256, 3, 2]]
40 |   - [[-1, 12], 1, Concat, [1]]
41 |   - [-1, 3, RepC2f, [512]]
42 |   - [-1, 1, Conv, [512, 3, 2]]
43 |   - [[-1, 9], 1, Concat, [1]]
44 |   - [-1, 3, RepC2f, [1024]]
45 |   - [20, 1, KVCompressedAttention, [128, 4, 8, dwconv]]
46 |   - [23, 1, KVCompressedAttention, [256, 4, 4, dwconv]]
47 |   - [[30, 31, 26, 29], 1, Detect, [nc]]
48 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_api_ring.yaml

Bytes: 1783
SHA-256: 53bde744d7fb04879a6d2e6fba02a947f436da2f29f60a938754e15cbcd4032f
Lines: 1-55 of 55

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+boundary/API-ring.
 2 | # Adds train-only BoundaryFeatureBlock and boundary-ring API after P2.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, RepC2f, [256]]          # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
36 |   - [-1, 1, BoundaryFeatureBlock, [1.0, 0.25, 4, 0.0, 0.05]] # 19-P2 train-only boundary feature block
37 |   - [-1, 1, AdversarialPerturbationInjection, [0.02, 0.25, boundary_ring]] # 20-P2 boundary-ring API
38 | 
39 |   - [-1, 1, Conv, [128, 3, 2]]
40 |   - [[-1, 15], 1, Concat, [1]]
41 |   - [-1, 3, RepC2f, [256]]          # 23-P3/8
42 | 
43 |   - [-1, 1, Conv, [256, 3, 2]]
44 |   - [[-1, 12], 1, Concat, [1]]
45 |   - [-1, 3, RepC2f, [512]]          # 26-P4/16
46 | 
47 |   - [-1, 1, Conv, [512, 3, 2]]
48 |   - [[-1, 9], 1, Concat, [1]]
49 |   - [-1, 3, RepC2f, [1024]]         # 29-P5/32
50 | 
51 |   - [20, 1, KVCompressedAttention, [128, 4, 8, dwconv]]
52 |   - [23, 1, KVCompressedAttention, [256, 4, 4, dwconv]]
53 | 
54 |   - [[30, 31, 26, 29], 1, Detect, [nc]]
55 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_api.yaml

Bytes: 1792
SHA-256: a4cb639329a0f30e0ec0e453fe8662b4ec5a4bb4767042a71435cbb71ca1ebd6
Lines: 1-55 of 55

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+boundary/API.
 2 | # Adds train-only BoundaryFeatureBlock and AdversarialPerturbationInjection after P2.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, RepC2f, [256]]          # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
36 |   - [-1, 1, BoundaryFeatureBlock, [1.0, 0.25, 4, 0.0, 0.05]] # 19-P2 train-only boundary feature block
37 |   - [-1, 1, AdversarialPerturbationInjection, [0.02, 0.25]] # 20-P2 gradient-based API perturbation
38 | 
39 |   - [-1, 1, Conv, [128, 3, 2]]
40 |   - [[-1, 15], 1, Concat, [1]]
41 |   - [-1, 3, RepC2f, [256]]          # 23-P3/8
42 | 
43 |   - [-1, 1, Conv, [256, 3, 2]]
44 |   - [[-1, 12], 1, Concat, [1]]
45 |   - [-1, 3, RepC2f, [512]]          # 26-P4/16
46 | 
47 |   - [-1, 1, Conv, [512, 3, 2]]
48 |   - [[-1, 9], 1, Concat, [1]]
49 |   - [-1, 3, RepC2f, [1024]]         # 29-P5/32
50 | 
51 |   - [20, 1, KVCompressedAttention, [128, 4, 8, dwconv]]
52 |   - [23, 1, KVCompressedAttention, [256, 4, 4, dwconv]]
53 | 
54 |   - [[30, 31, 26, 29], 1, Detect, [nc]]
55 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary_late.yaml

Bytes: 1687
SHA-256: b3091e3be36bfcfd0ead04e17527129ef07058d8511b90175dd3cad5a9da050c
Lines: 1-54 of 54

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+late-boundary.
 2 | # Boundary regularization is applied only to the final P2 Detect input.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, RepC2f, [256]]          # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
36 | 
37 |   - [-1, 1, Conv, [128, 3, 2]]
38 |   - [[-1, 15], 1, Concat, [1]]
39 |   - [-1, 3, RepC2f, [256]]          # 21-P3/8
40 | 
41 |   - [-1, 1, Conv, [256, 3, 2]]
42 |   - [[-1, 12], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 24-P4/16
44 | 
45 |   - [-1, 1, Conv, [512, 3, 2]]
46 |   - [[-1, 9], 1, Concat, [1]]
47 |   - [-1, 3, RepC2f, [1024]]         # 27-P5/32
48 | 
49 |   - [18, 1, KVCompressedAttention, [128, 4, 8, dwconv]] # 28-P2 final
50 |   - [21, 1, KVCompressedAttention, [256, 4, 4, dwconv]] # 29-P3 final
51 |   - [28, 1, BoundaryFeatureBlock, [1.0, 0.25, 4, 0.0, 0.01]] # 30-late P2 boundary
52 | 
53 |   - [[30, 29, 24, 27], 1, Detect, [nc]]
54 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2_boundary.yaml

Bytes: 1371
SHA-256: 0f18ea52bdc873bf7041a74cb3e7e2dc43ff7093106099edaae045e1445c6dcf
Lines: 1-47 of 47

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+boundary.
 2 | # Train-only boundary feature regularization with a small residual cap.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]
14 |   - [-1, 1, Conv, [128, 3, 2]]
15 |   - [-1, 3, RepC2f, [128, True]]
16 |   - [-1, 1, Conv, [256, 3, 2]]
17 |   - [-1, 6, RepC2f, [256, True]]
18 |   - [-1, 1, Conv, [512, 3, 2]]
19 |   - [-1, 6, RepC2f, [512, True]]
20 |   - [-1, 1, Conv, [1024, 3, 2]]
21 |   - [-1, 3, RepC2f, [1024, True]]
22 |   - [-1, 1, SPPF, [1024, 5]]
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]
28 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
29 |   - [[-1, 4], 1, Concat, [1]]
30 |   - [-1, 3, RepC2f, [256]]
31 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
32 |   - [[-1, 2], 1, Concat, [1]]
33 |   - [-1, 3, RepC2f, [128]]
34 |   - [-1, 1, BoundaryFeatureBlock, [1.0, 0.25, 4, 0.0, 0.01]]
35 |   - [-1, 1, Conv, [128, 3, 2]]
36 |   - [[-1, 15], 1, Concat, [1]]
37 |   - [-1, 3, RepC2f, [256]]
38 |   - [-1, 1, Conv, [256, 3, 2]]
39 |   - [[-1, 12], 1, Concat, [1]]
40 |   - [-1, 3, RepC2f, [512]]
41 |   - [-1, 1, Conv, [512, 3, 2]]
42 |   - [[-1, 9], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [1024]]
44 |   - [19, 1, KVCompressedAttention, [128, 4, 8, dwconv]]
45 |   - [22, 1, KVCompressedAttention, [256, 4, 4, dwconv]]
46 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
47 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p2p3_edge_pooling.yaml

Bytes: 1615
SHA-256: 6af61985969c7576bac6dbcfcd8cc8b2a0bd03d736f1398bad11d83aa2e22f80
Lines: 1-53 of 53

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+P3 pooling edge (no API).
 2 | # PoolingEdgeRepC2f on both P2 and P3 neck branches. No AdversarialPerturbationInjection.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, PoolingEdgeRepC2f, [256, True]] # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
36 | 
37 |   - [-1, 1, Conv, [128, 3, 2]]
38 |   - [[-1, 15], 1, Concat, [1]]
39 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 21-P3/8
40 | 
41 |   - [-1, 1, Conv, [256, 3, 2]]
42 |   - [[-1, 12], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 24-P4/16
44 | 
45 |   - [-1, 1, Conv, [512, 3, 2]]
46 |   - [[-1, 9], 1, Concat, [1]]
47 |   - [-1, 3, RepC2f, [1024]]         # 27-P5/32
48 | 
49 |   - [18, 1, EnSimAM, []]            # 28-P2 final
50 |   - [21, 1, EnSimAM, []]            # 29-P3 final
51 | 
52 |   - [[28, 29, 24, 27], 1, Detect, [nc]]
53 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_ensimam.yaml

Bytes: 1405
SHA-256: 13d256bc666f9200389c5f7596de1e26a0d70866d1cfba4c233c9c8067a7ed44
Lines: 1-45 of 45

```yaml
 1 | # YOLOv8 Varroa compare baseline+P3+API box-gradient+edge EnSimAM.
 2 | # Detects on P3/P4/P5 only; edge is attached to selected P3 RepC2f branches.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, EnSimAMEdgeRepC2f, [256, True]] # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, EnSimAMEdgeRepC2f, [256]] # 15-P3/8
32 | 
33 |   - [-1, 1, Conv, [256, 3, 2]]
34 |   - [[-1, 12], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [512]]          # 18-P4/16
36 | 
37 |   - [-1, 1, Conv, [512, 3, 2]]
38 |   - [[-1, 9], 1, Concat, [1]]
39 |   - [-1, 3, RepC2f, [1024]]         # 21-P5/32
40 | 
41 |   - [15, 1, EnSimAM, []]            # 22-P3 final
42 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 23-P3 localization API
43 | 
44 |   - [[23, 18, 21], 1, Detect, [nc]]
45 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_api_boxgrad_edge_pooling.yaml

Bytes: 1405
SHA-256: 655e0c67442b5e69ec90095757a720e13808b1544fc131d199d3092c2d08cc44
Lines: 1-45 of 45

```yaml
 1 | # YOLOv8 Varroa compare baseline+P3+API box-gradient+pooling edge.
 2 | # Detects on P3/P4/P5 only; edge is attached to selected P3 RepC2f branches.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, PoolingEdgeRepC2f, [256, True]] # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
32 | 
33 |   - [-1, 1, Conv, [256, 3, 2]]
34 |   - [[-1, 12], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [512]]          # 18-P4/16
36 | 
37 |   - [-1, 1, Conv, [512, 3, 2]]
38 |   - [[-1, 9], 1, Concat, [1]]
39 |   - [-1, 3, RepC2f, [1024]]         # 21-P5/32
40 | 
41 |   - [15, 1, EnSimAM, []]            # 22-P3 final
42 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 23-P3 localization API
43 | 
44 |   - [[23, 18, 21], 1, Detect, [nc]]
45 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_api_boxgrad_ensimam.yaml

Bytes: 1386
SHA-256: b31e4fc8478b0527d4a26ae1e9c50e7ef6e31d5a1c7dc370a9405a59ed20ecd1
Lines: 1-45 of 45

```yaml
 1 | # YOLOv8 Varroa compare baseline+P3+API box-gradient+EnSimAM.
 2 | # Detects on P3/P4/P5 only; API is applied to the final P3 Detect input.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, RepC2f, [256]]          # 15-P3/8
32 | 
33 |   - [-1, 1, Conv, [256, 3, 2]]
34 |   - [[-1, 12], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [512]]          # 18-P4/16
36 | 
37 |   - [-1, 1, Conv, [512, 3, 2]]
38 |   - [[-1, 9], 1, Concat, [1]]
39 |   - [-1, 3, RepC2f, [1024]]         # 21-P5/32
40 | 
41 |   - [15, 1, EnSimAM, []]            # 22-P3 final
42 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad]] # 23-P3 localization API
43 | 
44 |   - [[23, 18, 21], 1, Detect, [nc]]
45 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling_quality_iou.yaml

Bytes: 1481
SHA-256: 7213bce4521f428b45801c40e888dca49f16232fa77d8d6955664ba90ce7f505
Lines: 1-52 of 52

```yaml
 1 | # YOLOv8 Varroa compare baseline+P3 pooling edge with IoU quality head.
 2 | # Detects on P3/P4 only. No P2 Detect branch and no API block.
 3 | 
 4 | nc: 1
 5 | quality_head: true
 6 | quality_box_features: true
 7 | quality_box_detach: true
 8 | quality_loss: bce_balanced
 9 | quality_gain: 0.5
10 | quality_neg_gain: 0.10
11 | quality_pos_iou_thr: 0.5
12 | quality_hard_neg_iou_thr: 0.3
13 | quality_hard_neg_score_thr: 0.05
14 | quality_target_mode: ap75_ramp
15 | quality_ramp_low: 0.50
16 | quality_ramp_high: 0.75
17 | quality_neg_mode: hard
18 | quality_score_mode: sqrt_cls_mul_q
19 | quality_detach_target: true
20 | 
21 | scales:
22 |   n: [0.33, 0.25, 1024]
23 |   s: [0.33, 0.50, 1024]
24 |   m: [0.67, 0.75, 768]
25 |   l: [1.00, 1.00, 512]
26 |   x: [1.00, 1.25, 512]
27 | 
28 | backbone:
29 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
30 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
31 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]]    # 2-P2/4
32 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
33 |   - [-1, 6, PoolingEdgeRepC2f, [256, True]] # 4-P3/8
34 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
35 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
36 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
37 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
38 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
39 | 
40 | head:
41 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
42 |   - [[-1, 6], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
44 | 
45 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
46 |   - [[-1, 4], 1, Concat, [1]]
47 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
48 | 
49 |   - [15, 1, EnSimAM, []]            # 16-P3 final
50 | 
51 |   - [[16, 12], 1, Detect, [nc]]
52 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_baseline_p3_edge_pooling.yaml

Bytes: 1633
SHA-256: 9171359e24f7dd58a46224ce6d7921f4b4b7c08f0fce9a7290b53df34f09451e
Lines: 1-53 of 53

```yaml
 1 | # YOLOv8 Varroa compare baseline+P3 pooling edge only (no API).
 2 | # PoolingEdgeRepC2f only on P3 neck branches. P2 uses standard RepC2f. No AdversarialPerturbationInjection.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, PoolingEdgeRepC2f, [256, True]] # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
36 | 
37 |   - [-1, 1, Conv, [128, 3, 2]]
38 |   - [[-1, 15], 1, Concat, [1]]
39 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 21-P3/8
40 | 
41 |   - [-1, 1, Conv, [256, 3, 2]]
42 |   - [[-1, 12], 1, Concat, [1]]
43 |   - [-1, 3, RepC2f, [512]]          # 24-P4/16
44 | 
45 |   - [-1, 1, Conv, [512, 3, 2]]
46 |   - [[-1, 9], 1, Concat, [1]]
47 |   - [-1, 3, RepC2f, [1024]]         # 27-P5/32
48 | 
49 |   - [18, 1, EnSimAM, []]            # 28-P2 final
50 |   - [21, 1, EnSimAM, []]            # 29-P3 final
51 | 
52 |   - [[28, 29, 24, 27], 1, Detect, [nc]]
53 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_p2p3_api_boxgrad_edge_pooling.yaml

Bytes: 1887
SHA-256: 51599bee99432d0fe1919b899a80c2e9ccd3901a6d714d96d3d30d0f5e1d4f6a
Lines: 1-56 of 56

```yaml
 1 | # YOLOv8 Varroa compare P2+P3 API box-gradient+pooling edge.
 2 | # PoolingEdgeRepC2f on both P2 and P3 neck branches.
 3 | # AdversarialPerturbationInjection applied at both P2 and P3.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | backbone:
14 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
15 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
16 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
17 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
18 |   - [-1, 6, PoolingEdgeRepC2f, [256, True]] # 4-P3/8
19 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
20 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
21 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
22 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
23 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
24 | 
25 | head:
26 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
27 |   - [[-1, 6], 1, Concat, [1]]
28 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
29 | 
30 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
31 |   - [[-1, 4], 1, Concat, [1]]
32 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
33 | 
34 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
35 |   - [[-1, 2], 1, Concat, [1]]
36 |   - [-1, 3, PoolingEdgeRepC2f, [128]] # 18-P2/4
37 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad, True]] # 19-P2 localization API (partial_forward=True)
38 | 
39 |   - [-1, 1, Conv, [128, 3, 2]]
40 |   - [[-1, 15], 1, Concat, [1]]
41 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 22-P3/8
42 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad, True]] # 23-P3 localization API (partial_forward=True)
43 | 
44 |   - [-1, 1, Conv, [256, 3, 2]]
45 |   - [[-1, 12], 1, Concat, [1]]
46 |   - [-1, 3, RepC2f, [512]]          # 26-P4/16
47 | 
48 |   - [-1, 1, Conv, [512, 3, 2]]
49 |   - [[-1, 9], 1, Concat, [1]]
50 |   - [-1, 3, RepC2f, [1024]]         # 29-P5/32
51 | 
52 |   - [19, 1, EnSimAM, []]            # 30-P2 final
53 |   - [23, 1, EnSimAM, []]            # 31-P3 final
54 | 
55 |   - [[30, 31, 26, 29], 1, Detect, [nc]]
56 | 
```

### models_related/models_config/yolov8/yolov8_varroa_compare_p3_api_boxgrad_edge_pooling.yaml

Bytes: 1774
SHA-256: 2775f0ece0736e58ac1cdbcee51142cf59a97a3c1d49c6bcad87e51fe097f1e9
Lines: 1-55 of 55

```yaml
 1 | # YOLOv8 Varroa compare P3-only API box-gradient+pooling edge.
 2 | # PoolingEdgeRepC2f only on P3 neck branches. P2 uses standard RepC2f.
 3 | # AdversarialPerturbationInjection applied only at P3.
 4 | 
 5 | nc: 1
 6 | scales:
 7 |   n: [0.33, 0.25, 1024]
 8 |   s: [0.33, 0.50, 1024]
 9 |   m: [0.67, 0.75, 768]
10 |   l: [1.00, 1.00, 512]
11 |   x: [1.00, 1.25, 512]
12 | 
13 | backbone:
14 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
15 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
16 |   - [-1, 3, PoolingEdgeRepC2f, [128, True]] # 2-P2/4
17 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
18 |   - [-1, 6, PoolingEdgeRepC2f, [256, True]] # 4-P3/8
19 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
20 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
21 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
22 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
23 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
24 | 
25 | head:
26 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
27 |   - [[-1, 6], 1, Concat, [1]]
28 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
29 | 
30 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
31 |   - [[-1, 4], 1, Concat, [1]]
32 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 15-P3/8
33 | 
34 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
35 |   - [[-1, 2], 1, Concat, [1]]
36 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
37 | 
38 |   - [-1, 1, Conv, [128, 3, 2]]
39 |   - [[-1, 15], 1, Concat, [1]]
40 |   - [-1, 3, PoolingEdgeRepC2f, [256]] # 21-P3/8
41 |   - [-1, 1, AdversarialPerturbationInjection, [0.005, 0.05, boxgrad, True]] # 22-P3 localization API (partial_forward=True)
42 | 
43 |   - [-1, 1, Conv, [256, 3, 2]]
44 |   - [[-1, 12], 1, Concat, [1]]
45 |   - [-1, 3, RepC2f, [512]]          # 25-P4/16
46 | 
47 |   - [-1, 1, Conv, [512, 3, 2]]
48 |   - [[-1, 9], 1, Concat, [1]]
49 |   - [-1, 3, RepC2f, [1024]]         # 28-P5/32
50 | 
51 |   - [18, 1, EnSimAM, []]            # 29-P2 final
52 |   - [22, 1, EnSimAM, []]            # 30-P3 final
53 | 
54 |   - [[29, 30, 25, 28], 1, Detect, [nc]]
55 | 
```

### models_related/models_config/yolov8/yolov8n_varroa_compare_baseline_p2_boundary_api.yaml

Bytes: 1792
SHA-256: a4cb639329a0f30e0ec0e453fe8662b4ec5a4bb4767042a71435cbb71ca1ebd6
Lines: 1-55 of 55

```yaml
 1 | # YOLOv8 Varroa compare baseline+P2+boundary/API.
 2 | # Adds train-only BoundaryFeatureBlock and AdversarialPerturbationInjection after P2.
 3 | 
 4 | nc: 1
 5 | scales:
 6 |   n: [0.33, 0.25, 1024]
 7 |   s: [0.33, 0.50, 1024]
 8 |   m: [0.67, 0.75, 768]
 9 |   l: [1.00, 1.00, 512]
10 |   x: [1.00, 1.25, 512]
11 | 
12 | backbone:
13 |   - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
14 |   - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
15 |   - [-1, 3, RepC2f, [128, True]]    # 2-P2/4
16 |   - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
17 |   - [-1, 6, RepC2f, [256, True]]    # 4-P3/8
18 |   - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
19 |   - [-1, 6, RepC2f, [512, True]]    # 6-P4/16
20 |   - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
21 |   - [-1, 3, RepC2f, [1024, True]]   # 8-P5/32
22 |   - [-1, 1, SPPF, [1024, 5]]        # 9-P5/32
23 | 
24 | head:
25 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
26 |   - [[-1, 6], 1, Concat, [1]]
27 |   - [-1, 3, RepC2f, [512]]          # 12-P4/16
28 | 
29 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
30 |   - [[-1, 4], 1, Concat, [1]]
31 |   - [-1, 3, RepC2f, [256]]          # 15-P3/8
32 | 
33 |   - [-1, 1, nn.Upsample, [None, 2, nearest]]
34 |   - [[-1, 2], 1, Concat, [1]]
35 |   - [-1, 3, RepC2f, [128]]          # 18-P2/4
36 |   - [-1, 1, BoundaryFeatureBlock, [1.0, 0.25, 4, 0.0, 0.05]] # 19-P2 train-only boundary feature block
37 |   - [-1, 1, AdversarialPerturbationInjection, [0.02, 0.25]] # 20-P2 gradient-based API perturbation
38 | 
39 |   - [-1, 1, Conv, [128, 3, 2]]
40 |   - [[-1, 15], 1, Concat, [1]]
41 |   - [-1, 3, RepC2f, [256]]          # 23-P3/8
42 | 
43 |   - [-1, 1, Conv, [256, 3, 2]]
44 |   - [[-1, 12], 1, Concat, [1]]
45 |   - [-1, 3, RepC2f, [512]]          # 26-P4/16
46 | 
47 |   - [-1, 1, Conv, [512, 3, 2]]
48 |   - [[-1, 9], 1, Concat, [1]]
49 |   - [-1, 3, RepC2f, [1024]]         # 29-P5/32
50 | 
51 |   - [20, 1, KVCompressedAttention, [128, 4, 8, dwconv]]
52 |   - [23, 1, KVCompressedAttention, [256, 4, 4, dwconv]]
53 | 
54 |   - [[30, 31, 26, 29], 1, Detect, [nc]]
55 | 
```

### models_related/ultralytics/tests/test_nms.py

Bytes: 5909
SHA-256: f347e4afe4805610183648389438fe985bc6222f961e8eb5fc21c86119fbe7f1
Lines: 1-161 of 161

```python
  1 | # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
  2 | 
  3 | import pytest
  4 | import torch
  5 | 
  6 | from models_related.ultralytics.ultralytics.cfg import get_cfg
  7 | from models_related.ultralytics.ultralytics.utils import DEFAULT_CFG
  8 | from models_related.ultralytics.ultralytics.utils.nms import TorchNMS, non_max_suppression
  9 | 
 10 | 
 11 | def test_torchnms_soft_nms_linear_keeps_decayed_box():
 12 |     """Test linear Soft-NMS keeps an overlapping box when its decayed score remains above min_score."""
 13 |     boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]])
 14 |     hard_scores = torch.tensor([0.9, 0.8, 0.7])
 15 |     soft_scores = hard_scores.clone()
 16 | 
 17 |     hard_keep = TorchNMS.nms(boxes, hard_scores, 0.5)
 18 |     soft_keep = TorchNMS.soft_nms(boxes, soft_scores, 0.5, method="linear", min_score=0.001)
 19 | 
 20 |     assert hard_keep.tolist() == [0, 2]
 21 |     assert soft_keep.tolist() == [0, 2, 1]
 22 |     assert soft_scores[1] < hard_scores[1]
 23 | 
 24 | 
 25 | def test_torchnms_soft_nms_gaussian_is_deterministic():
 26 |     """Test Gaussian Soft-NMS produces stable valid indices."""
 27 |     boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]])
 28 |     scores = torch.tensor([0.9, 0.8, 0.7])
 29 | 
 30 |     keep = TorchNMS.soft_nms(boxes, scores, 0.5, method="gaussian", sigma=0.5, min_score=0.001)
 31 | 
 32 |     assert keep.tolist() == [0, 2, 1]
 33 |     assert set(keep.tolist()) == {0, 1, 2}
 34 | 
 35 | 
 36 | def test_non_max_suppression_soft_linear_keeps_more_boxes_than_hard():
 37 |     """Test Soft-NMS keeps more high-overlap detections than default hard NMS."""
 38 |     prediction = torch.zeros((1, 5, 3))
 39 |     prediction[0, :4] = torch.tensor(
 40 |         [[5.0, 6.0, 25.0], [5.0, 6.0, 25.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
 41 |     )
 42 |     prediction[0, 4] = torch.tensor([0.9, 0.8, 0.7])
 43 | 
 44 |     hard = non_max_suppression(prediction.clone(), conf_thres=0.001, iou_thres=0.5, nc=1)
 45 |     soft = non_max_suppression(
 46 |         prediction.clone(),
 47 |         conf_thres=0.001,
 48 |         iou_thres=0.5,
 49 |         nc=1,
 50 |         nms_method="soft-linear",
 51 |         soft_nms_min_score=0.001,
 52 |     )
 53 |     limited = non_max_suppression(
 54 |         prediction.clone(),
 55 |         conf_thres=0.001,
 56 |         iou_thres=0.5,
 57 |         nc=1,
 58 |         max_det=2,
 59 |         nms_method="soft-linear",
 60 |         soft_nms_min_score=0.001,
 61 |     )
 62 | 
 63 |     assert len(hard[0]) == 2
 64 |     assert len(soft[0]) == 3
 65 |     assert len(limited[0]) == 2
 66 |     assert soft[0][:, 4].tolist() == pytest.approx([0.9, 0.7, 0.2555], abs=1e-4)
 67 | 
 68 | 
 69 | def test_non_max_suppression_soft_nms_return_idxs():
 70 |     """Test return_idxs maps Soft-NMS results back to original prediction indices."""
 71 |     prediction = torch.zeros((1, 5, 3))
 72 |     prediction[0, :4] = torch.tensor(
 73 |         [[5.0, 6.0, 25.0], [5.0, 6.0, 25.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
 74 |     )
 75 |     prediction[0, 4] = torch.tensor([0.9, 0.8, 0.7])
 76 | 
 77 |     output, keep_idxs = non_max_suppression(
 78 |         prediction,
 79 |         conf_thres=0.001,
 80 |         iou_thres=0.5,
 81 |         nc=1,
 82 |         nms_method="soft-linear",
 83 |         soft_nms_min_score=0.001,
 84 |         return_idxs=True,
 85 |     )
 86 | 
 87 |     assert len(output[0]) == 3
 88 |     assert keep_idxs[0].tolist() == [0, 2, 1]
 89 | 
 90 | 
 91 | def test_non_max_suppression_box_voting_is_opt_in():
 92 |     """Test box voting leaves default NMS unchanged and refines coordinates only when enabled."""
 93 |     prediction = torch.zeros((1, 5, 2))
 94 |     prediction[0, :4] = torch.tensor([[5.0, 5.5], [5.0, 5.0], [10.0, 10.0], [10.0, 10.0]])
 95 |     prediction[0, 4] = torch.tensor([0.9, 0.8])
 96 | 
 97 |     plain = non_max_suppression(prediction.clone(), conf_thres=0.001, iou_thres=0.5, nc=1)
 98 |     voted = non_max_suppression(
 99 |         prediction.clone(),
100 |         conf_thres=0.001,
101 |         iou_thres=0.5,
102 |         nc=1,
103 |         box_voting=True,
104 |         box_voting_iou=0.5,
105 |     )
106 | 
107 |     assert plain[0].shape == voted[0].shape == (1, 6)
108 |     assert plain[0][0, :4].tolist() == pytest.approx([0.0, 0.0, 10.0, 10.0])
109 |     assert voted[0][0, :4].tolist() != pytest.approx(plain[0][0, :4].tolist())
110 |     assert voted[0][0, 4:].tolist() == pytest.approx(plain[0][0, 4:].tolist())
111 | 
112 | 
113 | def test_torchnms_box_voting_weight_modes_differ():
114 |     """Test squared IoU weighting changes refined coordinates on asymmetric candidates."""
115 |     boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [3.0, 0.0, 13.0, 10.0]])
116 |     scores = torch.tensor([0.9, 0.8, 0.7])
117 |     classes = torch.zeros(3)
118 |     keep = torch.tensor([0])
119 | 
120 |     score_iou = TorchNMS.box_voting(boxes, scores, classes, keep, 0.5, "score_iou")
121 |     score_iou2 = TorchNMS.box_voting(boxes, scores, classes, keep, 0.5, "score_iou2")
122 | 
123 |     assert score_iou.shape == score_iou2.shape == (1, 4)
124 |     assert score_iou.flatten().tolist() != pytest.approx(score_iou2.flatten().tolist())
125 | 
126 | 
127 | def test_non_max_suppression_soft_nms_box_voting_runs():
128 |     """Test Soft-NMS and box voting compose without index or shape errors."""
129 |     prediction = torch.zeros((1, 5, 3))
130 |     prediction[0, :4] = torch.tensor(
131 |         [[5.0, 6.0, 25.0], [5.0, 6.0, 25.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
132 |     )
133 |     prediction[0, 4] = torch.tensor([0.9, 0.8, 0.7])
134 | 
135 |     output = non_max_suppression(
136 |         prediction,
137 |         conf_thres=0.001,
138 |         iou_thres=0.5,
139 |         nc=1,
140 |         nms_method="soft-linear",
141 |         soft_nms_min_score=0.001,
142 |         box_voting=True,
143 |         box_voting_iou=0.5,
144 |     )
145 | 
146 |     assert output[0].shape == (3, 6)
147 | 
148 | 
149 | def test_nms_method_config_validation():
150 |     """Test valid and invalid NMS method config overrides."""
151 |     cfg = get_cfg(DEFAULT_CFG, {"nms_method": "soft-gaussian", "box_voting": True, "box_voting_weight": "score_iou2"})
152 |     assert cfg.nms_method == "soft-gaussian"
153 |     assert cfg.box_voting is True
154 |     assert cfg.box_voting_weight == "score_iou2"
155 | 
156 |     with pytest.raises(ValueError, match="nms_method=foo"):
157 |         get_cfg(DEFAULT_CFG, {"nms_method": "foo"})
158 | 
159 |     with pytest.raises(ValueError, match="box_voting_weight=foo"):
160 |         get_cfg(DEFAULT_CFG, {"box_voting_weight": "foo"})
161 | 
```

### models_related/ultralytics/ultralytics/models/yolo/detect/predict.py

Bytes: 5819
SHA-256: 572214b9409e6d650f9ee6e5b9e37a6e8b7182c85ae141371d8b1316f62e139c
Lines: 1-130 of 130

```python
  1 | # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
  2 | 
  3 | from ultralytics.engine.predictor import BasePredictor
  4 | from ultralytics.engine.results import Results
  5 | from ultralytics.utils import ops
  6 | from ultralytics.utils import nms
  7 | 
  8 | 
  9 | class DetectionPredictor(BasePredictor):
 10 |     """A class extending the BasePredictor class for prediction based on a detection model.
 11 | 
 12 |     This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
 13 |     with bounding boxes and class predictions.
 14 | 
 15 |     Attributes:
 16 |         args (namespace): Configuration arguments for the predictor.
 17 |         model (nn.Module): The detection model used for inference.
 18 |         batch (list): Batch of images and metadata for processing.
 19 | 
 20 |     Methods:
 21 |         postprocess: Process raw model predictions into detection results.
 22 |         construct_results: Build Results objects from processed predictions.
 23 |         construct_result: Create a single Result object from a prediction.
 24 |         get_obj_feats: Extract object features from the feature maps.
 25 | 
 26 |     Examples:
 27 |         >>> from ultralytics.utils import ASSETS
 28 |         >>> from ultralytics.models.yolo.detect import DetectionPredictor
 29 |         >>> args = dict(model="yolo26n.pt", source=ASSETS)
 30 |         >>> predictor = DetectionPredictor(overrides=args)
 31 |         >>> predictor.predict_cli()
 32 |     """
 33 | 
 34 |     def postprocess(self, preds, img, orig_imgs, **kwargs):
 35 |         """Post-process predictions and return a list of Results objects.
 36 | 
 37 |         This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
 38 |         further analysis.
 39 | 
 40 |         Args:
 41 |             preds (torch.Tensor): Raw predictions from the model.
 42 |             img (torch.Tensor): Processed input image tensor in model input format.
 43 |             orig_imgs (torch.Tensor | list): Original input images before preprocessing.
 44 |             **kwargs (Any): Additional keyword arguments.
 45 | 
 46 |         Returns:
 47 |             (list): List of Results objects containing the post-processed predictions.
 48 | 
 49 |         Examples:
 50 |             >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
 51 |             >>> results = predictor.predict("path/to/image.jpg")
 52 |             >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
 53 |         """
 54 |         save_feats = getattr(self, "_feats", None) is not None
 55 |         preds = nms.non_max_suppression(
 56 |             preds,
 57 |             self.args.conf,
 58 |             kwargs.pop("iou", self.args.iou),  # allow callers (e.g. TrackTrack loose-NMS recovery) to override IoU
 59 |             self.args.classes,
 60 |             self.args.agnostic_nms,
 61 |             max_det=self.args.max_det,
 62 |             nc=0 if self.args.task == "detect" else len(self.model.names),
 63 |             end2end=getattr(self.model, "end2end", False),
 64 |             rotated=self.args.task == "obb",
 65 |             return_idxs=save_feats,
 66 |             nms_method=self.args.nms_method,
 67 |             soft_nms_sigma=self.args.soft_nms_sigma,
 68 |             soft_nms_min_score=self.args.soft_nms_min_score,
 69 |             box_voting=self.args.box_voting,
 70 |             box_voting_iou=self.args.box_voting_iou,
 71 |             box_voting_weight=self.args.box_voting_weight,
 72 |         )
 73 | 
 74 |         if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
 75 |             orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
 76 | 
 77 |         if save_feats:
 78 |             obj_feats = self.get_obj_feats(self._feats, preds[1])
 79 |             preds = preds[0]
 80 | 
 81 |         results = self.construct_results(preds, img, orig_imgs, **kwargs)
 82 | 
 83 |         if save_feats:
 84 |             for r, f in zip(results, obj_feats):
 85 |                 r.feats = f  # add object features to results
 86 | 
 87 |         return results
 88 | 
 89 |     @staticmethod
 90 |     def get_obj_feats(feat_maps, idxs):
 91 |         """Extract object features from the feature maps."""
 92 |         import torch
 93 | 
 94 |         s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
 95 |         obj_feats = torch.cat(
 96 |             [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
 97 |         )  # mean reduce all vectors to same length
 98 |         return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch
 99 | 
100 |     def construct_results(self, preds, img, orig_imgs):
101 |         """Construct a list of Results objects from model predictions.
102 | 
103 |         Args:
104 |             preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
105 |             img (torch.Tensor): Batch of preprocessed images used for inference.
106 |             orig_imgs (list[np.ndarray]): List of original images before preprocessing.
107 | 
108 |         Returns:
109 |             (list[Results]): List of Results objects containing detection information for each image.
110 |         """
111 |         return [
112 |             self.construct_result(pred, img, orig_img, img_path)
113 |             for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
114 |         ]
115 | 
116 |     def construct_result(self, pred, img, orig_img, img_path):
117 |         """Construct a single Results object from one image prediction.
118 | 
119 |         Args:
120 |             pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
121 |             img (torch.Tensor): Preprocessed image tensor used for inference.
122 |             orig_img (np.ndarray): Original image before preprocessing.
123 |             img_path (str): Path to the original image file.
124 | 
125 |         Returns:
126 |             (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
127 |         """
128 |         pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
129 |         return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
130 | 
```

## Skipped Files

- .ai-bridge/ [not a file]
- misc/train_all_full.py [File is too large (8501 bytes). Limit: 8000 bytes.]
- misc/train_all.py [File is too large (9403 bytes). Limit: 8000 bytes.]
- misc/train_kvca_sweep.py [File is too large (8361 bytes). Limit: 8000 bytes.]
- misc/train_lm_net_varroa.py [File is too large (32578 bytes). Limit: 8000 bytes.]
- misc/train_p3_api_sweep.py [File is too large (11822 bytes). Limit: 8000 bytes.]
- misc/train_vfl_tal_a0.py [File is too large (8365 bytes). Limit: 8000 bytes.]
- models_related/ultralytics/ultralytics/cfg/__init__.py [File is too large (44531 bytes). Limit: 8000 bytes.]
- models_related/ultralytics/ultralytics/cfg/default.yaml [File is too large (14912 bytes). Limit: 8000 bytes.]
- models_related/ultralytics/ultralytics/engine/model.py [File is too large (55674 bytes). Limit: 8000 bytes.]
- models_related/ultralytics/ultralytics/models/yolo/detect/val.py [File is too large (42086 bytes). Limit: 8000 bytes.]
- models_related/ultralytics/ultralytics/utils/nms.py [File is too large (20185 bytes). Limit: 8000 bytes.]
