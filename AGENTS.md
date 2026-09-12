# Repository agent instructions

## Marimo training is workflow-gated

When a user asks to train, evaluate, or upload through Marimo, the agent MUST:

1. Read `.agents/workflows/marimo-train.md` before touching the live runtime.
2. Use `utils/marimo_ops.py` for preflight, detached launch, status, and artifact
   checks. Do not recreate those operations in scratchpad code.
3. Run the complete preflight before spawning any process. This includes the
   exact checkout, executable, dataset, runner import, requested epochs,
   patience, seed, variants, HF repository, and authentication.
4. Launch only through `python -m utils.marimo_ops launch`. Direct
   `subprocess.Popen`, `nohup`, or ad-hoc background launches are forbidden for
   training jobs.
5. Stop at the first failed gate. Never add `--no-upload`, change the dataset
   root, or substitute a repository to make a run proceed.
6. After each run, verify local artifacts and the remote upload before starting
   the next variant.

The policy is enforced at runtime too: upload-required runners must receive the
`MARIMO_TRAIN_WORKFLOW=1` context injected by the shared launch helper and must
have a valid `HF_TOKEN`. `AGENTS.md` is not a substitute for those fail-closed
checks.

## Local Python testing

When local smoke tests need PyTorch or Ultralytics dependencies, use the Conda environment `ml2`:

```bash
conda run -n ml2 python ...
```

Do not assume the system Python has the project ML dependencies.

## Repository topology and Ultralytics boundary

- `vendor/ultralytics_upstream/` is a clean pinned upstream submodule. Never edit it directly.
- `project_ultralytics/` is the destination for new custom modules, losses, registries, adapters, and training glue.
- `models_related/ultralytics/` is the legacy compatibility fork for historical experiments. Do not modify it casually.
- `models_related/models_config/` contains historical model and experiment YAMLs. A YAML load failure may be a legacy runtime or dependency issue, not a malformed YAML.
- Use `PYTHONPATH=models_related/ultralytics` only for historical reproduction. Use `PYTHONPATH=.` for project-owned tests.
- Keep unrelated dirty files and active training scripts untouched. Check Git status before staging or committing.

## Detector baseline and Mosaic protocol

- Do not infer the detector architecture from the checkpoint name. A baseline
  `yolov8n.pt` checkpoint does **not** make a custom YAML a baseline model.
- The default detector for the Mosaic policy matrix is the upstream-style
  YOLOv8 default YAML:
  `models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml`.
  It is the P3/P4/P5 model and must not be replaced with a P2, CBAM,
  ChannelAttention, GAP, FPN-only, or other project YAML unless the experiment
  explicitly names that architecture.
- `models_related/models_config/` is a historical experiment corpus, not the
  default detector registry. Do not select a file from that tree for a
  baseline run based on a filename such as `plain`, `baseline`, or `yolov8n`.
  Check the canonical config path and its contents first.
- The custom Mosaic experiment changes augmentation policy, not detector
  architecture. The runner must pass `mosaic_policy` as one of
  `standard`, `visibility`, `occupancy_match`, or `context_contrast`, with
  `mosaic=1.0` as the enable gate and `close_mosaic=10` as the final-epoch
  shutdown. `mosaic=1.0` does not mean the default Ultralytics Mosaic
  implementation is being used.
- Before launching, verify the manifest records the exact YAML path, commit,
  seed, split seed, worker count, and policy. For a baseline run, reject any
  manifest or YAML containing P2, CBAM, ChannelAttention, GAP, or another
  unrequested architecture change.
- For this matrix, use `workers=8` unless the user explicitly requests a
  different value. Keep the detector YAML, augmentation policy, and worker
  count identical across matched runs.

## Seed-sweep provenance

- Dataset split seeds and training seeds are separate experiment parameters.
- Across all matched experiments, the split seed is a fixed data-provenance
  parameter, normally `42`. Do not derive it from, reuse, or silently change it
  with the model/training seed.
- For matched YOLOv8n P2 OACP seed comparisons, keep `--split-seed 42` fixed and vary only `--seeds` (for example `43 44`).
- Never pass the training seed into `prepare_levir_ship.prepare` for this comparison. That changes the train/val/test assignment and confounds seed stability with data-split variance.
- The sweep runner writes the fixed `split_seed` into each manifest and reuses one dataset output root for all training seeds.
- A different split seed is allowed only for an explicitly named split-sensitivity
  experiment, and that experiment must record the split seed separately in its
  manifest. Otherwise, a missing `split_seed` is a provenance failure, not an
  invitation to use the training seed.
