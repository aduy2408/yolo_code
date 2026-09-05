---
description: Canonical local-to-Marimo training workflow with deterministic preflight, detached launch, monitoring, recovery, and upload verification
---
# Canonical Local → Marimo Training Workflow

This workflow is the **policy layer**. Reusable checks live in
`utils/marimo_ops.py`; do not re-invent PID, Git SHA, artifact, or detached
launch logic in every experiment runner.

## Operating model

Keep these states separate:

```text
local implementation
→ local validation
→ pushed commit
→ exact remote checkout
→ Marimo environment verified
→ smoke passed
→ detached training progressing
→ training complete
→ evaluation complete
→ per-run upload verified
→ report updated
```

A live Marimo kernel, a live PID, a local marker, or one successful upload call
is not completion evidence by itself.

## Agent decision rules

- If the user asks for Marimo execution, do not train/evaluate/upload locally.
- Use local only for implementation, static checks, focused tests, and bounded smoke tests.
- Use the workflow when the task involves a long-running remote experiment.
- Use the `marimo-pair` skill only for live-kernel interaction. Keep orchestration
  logic in repository code, not in ad-hoc scratchpad snippets.
- Connect to the live Marimo runtime through the repository command-line helper
  (`utils/marimo_run.py`) and its configured API credentials. Do not open or
  probe the Marimo URL through a browser as a substitute for that connection.
  Browser interaction is out of scope unless the user explicitly requests it.
- Prefer Git synchronization over direct file patching on `/marimo`.
- Never print, echo, commit, or store auth tokens in notebook cells, logs, or
  process arguments. Pass them through the live kernel environment to the
  detached child process.
- **HF_TOKEN is provided by the live Marimo global kernel namespace, not by
  `os.environ` in the notebook scratchpad.** Read it from the kernel's global
  state and pass it only through the detached child-process environment. Never
  print the token or persist it in notebook/file output.
- Do not overwrite a dirty remote checkout blindly. Use a clean checkout or
  stop and report the conflict.

## 1. Local experiment contract

Before coding, record:

```text
hypothesis:
control:
variant:
primary metric and decision gate:
secondary metrics:
fixed split and seeds:
image size / batch / epochs / patience:
NMS IoU: 0.5
non-goals:
```

Use the smallest matched experiment that answers one question. Do not change
architecture, optimizer, resolution, augmentation, and schedule together
unless the experiment explicitly studies that interaction.

## 2. Local validation gates

Run in order, stopping at the first failure:

```text
syntax/compile
→ local import and registration
→ YAML/model construction
→ dummy forward and output shape
→ focused unit/synthetic test
→ one-batch or bounded smoke train
→ smoke evaluation with iou=0.5
→ git diff --check
```

For custom Ultralytics modules, check the full registration path:

```text
models_related/ultralytics/ultralytics/nn/modules/block.py
models_related/ultralytics/ultralytics/nn/modules/__init__.py
models_related/ultralytics/ultralytics/nn/tasks.py
model YAML
runner
focused test
```

Do not call a partial CPU smoke run a successful full validation.

## 3. Commit and handoff

Commit only the experiment files and required tests/docs. Record the SHA:

```bash
git status --short
git diff --check
git diff --stat
git add <experiment-files> <tests> <docs>
git commit -m "<scoped message>"
git push origin <branch>
git rev-parse HEAD
```

The handoff record must include:

```text
experiment name
commit SHA
runner and model YAML
Python executable expected on Marimo
data root and fixed split
variants and seeds
full command
epochs and patience
HF repo and remote prefix
required artifacts
NMS IoU=0.5
known partial checks
```

## 4. Marimo preflight

Run the following **inside the live Marimo environment**, not local:

```bash
"$MARIMO_PYTHON" -m utils.marimo_ops preflight \
  --repo /marimo/yolo_code \
  --expected-sha "$EXPECTED_SHA" \
  --python "$MARIMO_PYTHON" \
  --epochs 100 \
  --patience 0 \
  --upload-required \
  --hf-repo-id "$HF_REPO_ID"
```

If the utility is not importable from the remote checkout, run it by path or
set the repository root in `PYTHONPATH`. The output must show:

```text
correct executable
correct Git SHA
clean worktree
requested epochs
requested patience
upload required
HF repository configured
```

The Python executable is workload-dependent. For MMDetection or models that
depend on MMDetection, use the fixed `/marimo/mmdet-venv/bin/python` and fail if
it is unavailable. For Ultralytics-only work, including this repository's
forked `models_related/ultralytics`, use the already provisioned Marimo base
environment instead. In that case, discover and record the base interpreter
from the live kernel, pass it explicitly to preflight and launch, and do not
silently substitute another interpreter after preflight. Also inspect CUDA,
dataset, checkpoint, and dependencies.

## 5. Runner contract

Every multi-run runner must:

- process one `variant/seed` at a time unless parallelism is explicitly tested;
- use explicit `epochs` and `patience` values;
- use explicit `iou=0.5` for validation, test, and inference;
- fail closed if upload is required but auth/repository is absent;
- write a manifest containing commit SHA, command, seed, split, and NMS IoU;
- be restart-safe and skip only after verifying artifacts and remote paths;
- upload and verify each completed run before starting the next run.

Minimum local run contract:

```text
weights/best.pt
weights/last.pt
results.csv
evaluation_metrics.json
args.yaml or equivalent manifest
```

Minimum remote completion contract:

```text
all required local artifacts uploaded
remote paths listed and verified
upload_complete.json written only after verification
```

Selected Marimo runners must use the shared helper. A direct background launch
from a notebook cell is a policy violation:

```python
from utils.marimo_ops import artifacts, launch_detached, preflight, status
```

The helper injects `MARIMO_TRAIN_WORKFLOW=1`; upload-required runners reject
direct invocation without that marker. This makes the workflow requirement a
runtime gate rather than an instruction the agent can accidentally skip.

## 6. Detached launch

Never attach a long training job to the request stream. Use the shared helper:

```bash
"$MARIMO_PYTHON" -m utils.marimo_ops launch \
  --cwd /marimo/yolo_code \
  --run-dir /marimo/yolo_code/runs/<experiment> \
  -- \
"$MARIMO_PYTHON" train_all_<experiment>.py \
  --epochs 100 --patience 0 --upload
```

The helper creates durable:

```text
train.pid
train.log
state.json
```

It refuses to launch if the recorded PID is still alive.

## 7. Initial health check

After launch, check briefly. Do not repeatedly stream long logs:

```bash
/marimo/mmdet-venv/bin/python -m utils.marimo_ops status \
  --run-dir /marimo/yolo_code/runs/<experiment>
```

A useful status report distinguishes:

```text
process_alive
observed_status
process_command
latest_artifact_mtime
log_mtime
required_artifacts
upload_verified
```

`process_alive=true` does not mean training is progressing. When the PID is
dead, use `observed_status` rather than trusting a stale `state.json`:
`not_running_unverified` is not completion. Check epoch/log or artifact
timestamps when diagnosing a stall. Check `nvidia-smi` only when GPU state is
relevant.

## 8. Recovery rules

- Kernel dead, detached process progressing: reconnect and inspect; do not
  restart blindly.
- PID alive, logs/artifacts stale: classify as stalled/unknown; inspect process
  tree and GPU before deciding.
- State says `running`, PID dead: reconcile as interrupted, not complete.
- Training complete, test/summary failed: reuse checkpoint with test-only or
  summary-only; do not retrain automatically.
- Local artifacts complete, upload marker absent: verify remote paths and upload
  before continuing.
- Upload/verification fails: stop the matrix. Do not silently continue.
- Wrong SHA or dirty checkout: stop before launch.

## 9. Completion and post-run

Only report a run complete when all are true:

```text
clean training exit
+ evaluation complete
+ required artifacts present
+ metrics finite and protocol recorded
+ upload_complete.json exists
+ remote paths verified
```

Then update the report with:

```text
variant/seed
commit SHA
split and data provenance
checkpoint path
NMS IoU
val/test metrics
HF artifact path
limitations and failed/partial runs
```

Do not choose a scientific winner from validation alone when a matched test
result is available.
