# Repository instructions

This repo studies tiny-ship detection on LEVIR-Ship with custom YOLO variants.

## General agent defaults

- Be concise and prefer the smallest working change.
- Establish the repository root before inspecting files; do not assume the current directory or Git state.
- Read the nearest project instructions before non-trivial work.
- Use the narrowest matching tools: read/search for inspection and targeted edits for changes. Add specialized tools only when required.
- Do not load optional skills, extensions, subagents, web research, or broad tools unless the task needs them.
- If a specialized skill or extension is needed, use only that one and restore the lightweight profile afterward when practical.
- Never trade correctness for lower context on multi-file changes, debugging, security, persistence, concurrency, visual work, research, or uncertain tasks.
- Before reporting completion, verify changed files and run the narrowest meaningful check. Never claim an unrun check passed.
- Minimize resource usage during testing: run targeted tests, avoid unnecessary parallel jobs or watchers, cap worker counts when possible, and use timeouts.
- Do not install packages, change environments, delete data, or publish without approval.
- Match the user's language and requested level of detail. Ask only when a choice materially affects the result.

## Before non-trivial work

- Read `README_LEVIR.md` and the relevant report in `docs/reports/`.
- Inspect the current implementation and existing experiment results before proposing a new method.
- Keep experiments controlled; state the hypothesis, baseline, metric, and decision rule.

## Repository topology and Ultralytics boundary

- `vendor/ultralytics_upstream/` is a clean pinned upstream submodule. Never edit it directly.
- `project_ultralytics/` is the destination for new custom modules, losses, registries, adapters, and training glue.
- `models_related/ultralytics/` is the legacy compatibility fork and historical YAML oracle. Do not modify it casually because active experiments may depend on it.
- `models_related/models_config/` contains historical model and experiment YAMLs. A YAML loading failure is not automatically a YAML syntax error.
- Before reporting completion, distinguish YAML call errors, legacy runtime limitations, server-specific dependency differences, and unmigrated modules.

## Implementation

- Run commands from `/mnt/data/varroa/yolo_related`.
- New custom code belongs in `project_ultralytics`, not in either Ultralytics tree.
- Use `PYTHONPATH=models_related/ultralytics` only when intentionally reproducing a historical experiment or using the legacy parser oracle.
- Use the clean upstream submodule plus `PYTHONPATH=.` when testing project-owned code. The project registry is explicit and is not automatically wired into the upstream parser.
- For YAML changes, verify downstream indices and run model construction plus a dummy forward.
- Treat empty custom YAML args such as `ChannelAttention []` as potentially intentional because legacy parser branches may inject channels.
- Use a repository `.venv`/`venv` if present; otherwise use the configured `ml2` Python environment.


## Evaluation and safety

- Final validation, test, and selected-box inference use NMS IoU `0.5`; record `nms_iou: 0.5` in final artifacts.
- Distinguish candidate geometry, ranking, and final detection metrics.
- Do not launch long training, overwrite experiments, delete data, or commit without approval.
- Run a focused smoke test before expensive evaluation and report skipped checks honestly.
