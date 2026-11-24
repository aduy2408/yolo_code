# Marimo connection

Use the repository workflow in `.agents/workflows/marimo-train.md` and the shared implementation in `utils/marimo_ops.py` for preflight, detached launch, status, artifact checks, and Hugging Face verification.

Keep the workflow CLI/client code in the submission branch:

- `utils/marimo_ops.py`: preflight, launch, status, artifact and upload gates.
- `utils/marimo_client.py`: authenticated live-kernel HTTP client.
- `utils/marimo_run.py`: compatibility CLI for `-c/--code`, `-f/--file`, and `--set-config`.
- `.agents/workflows/marimo-train.md`: required state machine and launch contract.

Typical live-kernel commands:

```bash
python -m utils.marimo_ops preflight --help
python -m utils.marimo_ops launch --help
python utils/marimo_run.py -c 'print("connected")'
```

The `train/` launchers only select canonical runners. They do not replace Marimo preflight. Expected training runtime is the configured Marimo Python, not an arbitrary local interpreter. `HF_TOKEN` is read from the live Marimo global kernel and is never printed or stored.
