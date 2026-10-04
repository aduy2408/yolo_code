# Repository agent instructions

**Scope:** This file governs work inside `/mnt/data/varroa/yolo_related`. YOLO is the active project here. Do not inspect or modify MMDetection unless the request explicitly names it or asks for a cross-project change. The parent contract at `/mnt/data/varroa/AGENTS.md` only supplies shared rules such as baseline provenance, Marimo auth, and mandatory uploads.

## Marimo training is workflow-gated

When a user asks to train, evaluate, or upload through Marimo, the agent MUST:

1. Read `.agents/workflows/marimo-train.md` before touching the live runtime.
2. Use `utils/marimo_ops.py` for preflight, detached launch, status, and artifact
   checks. Do not recreate those operations in scratchpad code.
3. Run the complete preflight before spawning any process. This includes the
   exact checkout, executable, dataset, runner import, requested epochs,
   patience, seed, variants, HF repository, and authentication.
   The dataset check must record and validate the exact `data_root` and
   `dataset_yaml`, including existing `train`, `val`, and `test` paths. Do not
   guess a familiar path or silently substitute another dataset.
   Persistent Marimo source mounts are canonical: `/marimo/LevirShip`,
   `/marimo/Varroa`, and `/marimo/TinyPerson` (with the known Levir nested
   layout `/marimo/LevirShip/LevirShipData`).
4. If the user's original request explicitly includes training, evaluation, or
   upload, a successful setup/preflight is authorization to continue directly
   to launch. Do not ask for a second confirmation between setup and launch.
   This automatic continuation is fail-closed: stop on any failed or ambiguous
   checkout, executable, dataset, runner, seed, artifact, authentication, or
   task-specific HF repository gate. A setup-only request does not authorize a
   training launch.
5. Launch only through `python -m utils.marimo_ops launch`. Direct
   `subprocess.Popen`, `nohup`, or ad-hoc background launches are forbidden for
   training jobs.
6. Stop at the first failed shared gate. For independent multi-server runs,
   stop only the affected slot at a per-run failure and continue healthy slots.
   Never add `--no-upload`, change the dataset root, or substitute a repository
   to make a run proceed.
7. After preflight, schedule runs according to the number of independent live
   Marimo servers explicitly supplied or successfully discovered:
   - With one usable server, keep the queue sequential. Verify local artifacts
     and remote upload before starting the next variant.
   - With two or more usable servers, launch independent queue items
     concurrently, one isolated run per server. Do not hold a pending run behind
     an unrelated run on another server merely because that run has not finished.
   - Preflight the complete queue once before the first launch. All concurrent
     runs must use the same immutable commit and declared dataset provenance
     unless the experiment explicitly requests otherwise.
   - Give every run its own run directory, PID/state/log files, artifact root,
     contract, and task-specific HF repository or collision-free remote prefix.
     Never let concurrent runs share mutable output or upload paths.
   - Before assigning a slot, inspect its `state.json`, PID, command identity,
     `run_contract.json`, and artifact timestamps. Do not relaunch a run that is
     already alive or already verified. If a server is idle and a queued run has
     not started, launch that pending run there immediately after its own gates
     pass.
   - When a slot becomes free, verify that slot's local artifacts,
     split-qualified metrics, and remote upload, then backfill it with the next
     unstarted run without waiting for other servers. A failed or ambiguous run
     blocks only that slot and its dependent items, not independent runs on
     healthy servers.
   - Persist per-server slot ownership and queue progress. Stop only the
     affected slot at the first failed or ambiguous gate, and stop the entire
     queue only for a shared provenance, authentication, dataset, repository,
     or other global gate failure.

Progress and continuation checks must use the run's `state.json`, PID, command,
log/artifact timestamps, and `run_contract.json`. A checkpoint without test
metrics is an evaluation-pending run, not permission to retrain. A dead PID
with incomplete artifacts must be classified as interrupted or unverified.

**Concrete progress reporting contract:** When the user asks for `progress`,
report numeric workflow progress from the relevant artifact set, not only
liveness. Include `completed/total`, the number still pending, the current
PID/state, the newest log or artifact timestamp, and the exact artifact marker
being counted. For an evaluation queue, count per-prefix completion markers
only after the required split-qualified validation/test metrics and bucket
artifacts are present. Report upload progress separately as
`local_complete/total` and `HF_verified/total`. If a helper's process identity
heuristic conflicts with direct `/proc` liveness and the exact command in
`state.json`, report both observations and do not reduce the concrete artifact
count to zero. Never answer a progress question with only “alive” or “running”.

8. Create and use a **task-specific Hugging Face repository** for every
   experiment. Never use a generic shared repository such as
   `stw-yolo-runs`. Prefer an explicit ID following
   `<hf-user>/<dataset>-<experiment>-runs`, for example
   `duyle2408/levir-copy-paste-runs`. If no explicit ID is available, set
   `MARIMO_TASK_NAME` so the shared helper derives a task-specific repository,
   and fail preflight if neither is configured.

9. Every evaluation and report MUST include both validation and test metrics,
   explicitly labeled by split: `AP50`, `mAP50-95`, and the corresponding
   `AP50`/`mAP50-95` values for **both** `val` and `test`. Never present a
   validation metric as a test result, and never report an unlabeled AP/mAP
   number. If a dataset has a merged or corner-window test protocol, record
   that exact test protocol and its source artifact alongside the metrics.

The policy is enforced at runtime too: upload-required runners must receive the
`MARIMO_TRAIN_WORKFLOW=1` context injected by the shared launch helper and must
have a valid `HF_TOKEN` and a task-specific HF repository ID. `AGENTS.md` is
not a substitute for those fail-closed checks.

The minimum metric handoff is:

```text
val/AP50
val/mAP50-95
test/AP50

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
