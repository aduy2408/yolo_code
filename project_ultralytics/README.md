# Project Ultralytics boundary

This directory is the landing zone for project-specific Ultralytics extensions.

## Current state

- `vendor/ultralytics_upstream` is the clean, pinned upstream snapshot.
- `models_related/ultralytics` remains the legacy compatibility fork for existing
  experiments. It is intentionally not modified by this layout change.
- New custom layers, losses, registries, and experiment glue should be migrated
  here instead of being added to the upstream tree.

## Migration rule

Keep upstream changes and project changes separate. A project extension may
depend on public Ultralytics APIs, but the upstream snapshot must remain
unchanged. Every experiment should record the upstream submodule commit and the
project source commit in its run manifest.

The legacy fork will be retired only after the custom modules and model parser
registry have been migrated and the training/evaluation smoke tests pass.
