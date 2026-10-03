# Universal Marimo DETR / RT-DETR setup

`tools/setup_marimo_detr_rtdetr.py` prepares multiple live Marimo servers for the native Hugging Face matrix used by the RT-DETR probe.

It currently registers:

- DETR: `facebook/detr-resnet-50`
- RT-DETR: `PekingU/rtdetr_r18vd`
- datasets: Varroa, LEVIR-Ship, TinyPerson, and VisDrone (`/marimo/VisDrone2019` on the supplied servers)
- fixed split seed: `42`
- training seeds: `42`, `43`
- runtime defaults: AdamW, `lr=1e-4`, backbone `lr=1e-5`, `weight_decay=1e-4`, image size `512`, batch `2`, workers `8`, epochs `100`, patience `15`

The script only performs setup and validation. It does not launch training. Long-running jobs must use the repository Marimo workflow and `python -m utils.marimo_ops launch`.

## Server file

Create a local file with owner-only permissions. Do not commit it:

```text
HOST=sb-example-1.sb.molab.run
ACCESS_TOKEN=...

HOST=sb-example-2.sb.molab.run
ACCESS_TOKEN=...
```

```bash
chmod 600 /path/to/marimo_servers.env
```

Duplicate hosts are ignored. Access tokens are used only for the live request and are not written to the remote setup report.

## Run setup

```bash
python tools/setup_marimo_detr_rtdetr.py \
  --servers-file /path/to/marimo_servers.env
```

For a quick parse-only check:

```bash
python tools/setup_marimo_detr_rtdetr.py \
  --servers-file /path/to/marimo_servers.env \
  --dry-run
```

The default remote workspace is `/marimo/hf_detr_rtdetr`. Each server receives:

- `detr_rtdetr_setup.json`, containing the model, dataset, and runtime contract
- `setup_report.json`, containing Python, Transformers, checkpoint, and mount checks

The setup uses `/tmp/uv-venv/bin/python` by default. Override it when a server uses another prepared interpreter:

```bash
python tools/setup_marimo_detr_rtdetr.py \
  --servers-file /path/to/marimo_servers.env \
  --python /marimo/mmdet-venv/bin/python
```

## Important gates

A setup report is not a successful training run. Before launching a job, verify:

1. the exact Git checkout and commit are present on the server
2. the runner and dataset adapter exist in that checkout
3. the declared dataset root and `train`, `val`, and `test` paths are valid
4. the task-specific Hugging Face repository is configured
5. both `val/AP50`, `val/mAP50-95`, `test/AP50`, and `test/mAP50-95` will be emitted

The setup script deliberately does not install packages, alter datasets, or silently substitute missing mounts.
