# Marimo watcher

`utils/marimo_watch.py` provides a visible, interval-based status monitor for detached Marimo supervisors.

Credentials are environment-only:

```bash
export MARIMO_WATCH_SERVER_1_URL=https://server-one
export MARIMO_WATCH_SERVER_1_TOKEN='...'
export MARIMO_WATCH_SERVER_2_URL=https://server-two
export MARIMO_WATCH_SERVER_2_TOKEN='...'
```

Run continuously:

```bash
python -m utils.marimo_watch \
  --repo /marimo/yolo_code_stw_oacp_4835e29_new \
  --interval 600
```

Run one check:

```bash
python -m utils.marimo_watch \
  --repo /marimo/yolo_code_stw_oacp_4835e29_new \
  --once
```

The monitor reports liveness, process identity, continuation state, upload verification, and required artifacts. It never restarts or mutates a run.
