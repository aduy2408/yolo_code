# Augmentation research

Canonical implementation remains under `project_ultralytics/`:

- OACP R2: `context_augment.py` with `OACP_PROFILE=r2`.
- Copy-Paste R2/R4: `copy_paste.py`, `negative_canvas_copy_paste.py`, `adaptive_negative_canvas.py`, and `stcp_copy_paste.py`.
- Protocol names: `negative_canvas_r2`, `negative_canvas_r4`, `all_canvas_r2`, `all_canvas_r4` in `copy_paste_protocol.py`.
- Three-dataset matrix: `train_scripts/train_all_augmentation_matrix.py`.

This folder intentionally contains documentation only; duplicating augmentation logic would create two sources of truth.
