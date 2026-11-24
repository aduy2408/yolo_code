# Marimo queue inventory

These scripts are tracked experiment supervisors in `misc/`, not runtime utilities
and not evidence that a job is currently running. They were written for historical
Marimo experiments and should only be migrated when the corresponding experiment
is still active.

## Current / likely reusable

- `marimo_oacp_queue.py`: current matched OACP queue. It covers LEVIR and TinyPerson
  budget/density runs and is the first migration target. OACP queueing is still part
  of the active workflow.

## Historical or one-off until explicitly reactivated

- `marimo_levir_nomosaic_queue.py`: older sequential no-Mosaic queue.
- `marimo_oacp_five_way_queue.py`: older R2 five-policy queue.
- `marimo_adaptive_oacp_queue.py`: older adaptive-policy matrix queue.
- `marimo_post_mosaic_oacp_queue.py`: older post-Mosaic placement/signal queue.
- `queue_m4_after_modulation.py`: one-off continuation script gated on an M3 marker.

## Migration rule

Do not run or modify a historical queue merely because it is tracked. If one is
reactivated, migrate it through the Marimo queue adapter first. The adapter must
create an immutable run contract, pass an explicit artifact root, launch only after
preflight, poll `status()`, and use `complete_verified()` rather than file presence
or a local upload marker as the completion decision.

The current OACP queue is intentionally migrated separately because its runners
have heterogeneous command schemas and artifact names. The generic Marimo launcher
must not infer model or dataset options from those commands.
