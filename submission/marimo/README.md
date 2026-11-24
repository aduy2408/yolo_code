# Marimo connection

Use the repository workflow in `.agents/workflows/marimo-train.md` and the shared implementation in `utils/marimo_ops.py` for preflight, detached launch, status, artifact checks, and Hugging Face verification.

The `train/` launchers only select canonical runners. They do not replace Marimo preflight. Expected training runtime is the configured Marimo Python, not an arbitrary local interpreter. `HF_TOKEN` is read from the live Marimo global kernel and is never printed or stored.
