# Seed-44 OACP/NC parity runs on Marimo

Saved command for the 12-run queue requested for LEVIR-Ship and TinyPerson.

## Queue

- Models: `yolov8`, `yolov11`
- Datasets: `levir`, `tinyperson`
- Methods: OACP-R2, `negative_canvas_r2`, `negative_canvas_r4`
- Training seed: `44`
- Dataset split seed: `42`
- LEVIR: `no_mosaic`
- TinyPerson: `mosaic` only
- `--baseline-augmentation-parity`
- `--full-bucket-metrics`
- `--model-prefix`
- Expected jobs: `2 models × 2 datasets × 3 variants = 12`

## Rerun inside the Marimo server

Do not put an HF token in this file. The Marimo runtime must already provide
`HF_TOKEN` and the configured training context must allow the parity repository.

```python
from pathlib import Path
import os
import subprocess

root = Path("/marimo/yolo_code")
log = Path("/marimo/oacp_nc_seed44_parity.log")
project = Path("/marimo/oacp_nc_seed44_runs")
dataset_root = Path("/marimo/oacp_nc_seed44_dataset")
repo = "duyle2408/oacp-r2-parity-molab-sb-a0ca44f32af730b4-seed43-44"

if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("HF_TOKEN is not available in the Marimo runtime")

cmd = [
    "python", "train_scripts/train_all_augmentation_matrix.py",
    "--dataset-root", str(dataset_root),
    "--project", str(project),
    "--datasets", "levir", "tinyperson",
    "--models", "yolov8", "yolov11",
    "--methods", "oacp", "copy_paste",
    "--oacp-variants", "r2",
    "--copy-paste-variants", "negative_canvas_r2", "negative_canvas_r4",
    "--tinyperson-mosaic-modes", "mosaic",
    "--seeds", "44",
    "--split-seed", "42",
    "--baseline-augmentation-parity",
    "--full-bucket-metrics",
    "--model-prefix",
    "--hf-repo-oacp", repo,
    "--hf-repo-copy-paste", repo,
    "--hf-repo-mosaic", repo,
    "--confirm-settings",
]

with log.open("a", encoding="utf-8") as fh:
    fh.write("\nCOMMAND " + " ".join(cmd) + "\n")
    fh.flush()
    process = subprocess.Popen(
        cmd,
        cwd=root,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )

print({"pid": process.pid, "log": str(log), "repo": repo})
```

## Verify the queue before launching

```bash
python train_scripts/train_all_augmentation_matrix.py \
  --dataset-root /marimo/oacp_nc_seed44_dataset \
  --project /marimo/oacp_nc_seed44_runs \
  --datasets levir tinyperson \
  --models yolov8 yolov11 \
  --methods oacp copy_paste \
  --oacp-variants r2 \
  --copy-paste-variants negative_canvas_r2 negative_canvas_r4 \
  --tinyperson-mosaic-modes mosaic \
  --seeds 44 --split-seed 42 \
  --baseline-augmentation-parity --print-effective-config
```

Expected output: `job_count: 12`.

## Monitor

```python
from pathlib import Path
import subprocess

print(subprocess.run(
    ["bash", "-lc", "pgrep -af 'train_all_augmentation_matrix.py.*oacp_nc_seed44' || true"],
    capture_output=True, text=True,
).stdout)
print(Path("/marimo/oacp_nc_seed44_parity.log").read_text(encoding="utf-8")[-8000:])
```

The queue is resumable: verified remote prefixes are skipped by the matrix
runner. Reuse the same `project`, `dataset-root`, and HF repository when a
server restart is needed; do not change the prefixes or overwrite a different
run directory.
