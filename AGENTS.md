# Repository agent instructions

## Marimo training is workflow-gated

When a user asks to train, evaluate, or upload through Marimo, the agent MUST:

1. Read `.agents/workflows/marimo-train.md` before touching the live runtime.
2. Use `utils/marimo_ops.py` for preflight, detached launch, status, and artifact
   checks. Do not recreate those operations in scratchpad code.
3. Run the complete preflight before spawning any process. This includes the
   exact checkout, executable, dataset, runner import, requested epochs,
   patience, seed, variants, HF repository, and authentication.
4. Launch only through `python -m utils.marimo_ops launch`. Direct
   `subprocess.Popen`, `nohup`, or ad-hoc background launches are forbidden for
   training jobs.
5. Stop at the first failed gate. Never add `--no-upload`, change the dataset
   root, or substitute a repository to make a run proceed.
6. After each run, verify local artifacts and the remote upload before starting
   the next variant.

The policy is enforced at runtime too: upload-required runners must receive the
`MARIMO_TRAIN_WORKFLOW=1` context injected by the shared launch helper and must
have a valid `HF_TOKEN`. `AGENTS.md` is not a substitute for those fail-closed
checks.

## Local Python testing

When local smoke tests need PyTorch or Ultralytics dependencies, use the Conda environment `ml2`:

```bash
conda run -n ml2 python ...
```

Do not assume the system Python has the project ML dependencies.

## Repository topology and Ultralytics boundary

- `vendor/ultralytics_upstream/` is a clean pinned upstream submodule. Never edit it directly.
- `project_ultralytics/` is the destination for new custom modules, losses, registries, adapters, and training glue.
- `models_related/ultralytics/` is the legacy compatibility fork for historical experiments. Do not modify it casually.
- `models_related/models_config/` contains historical model and experiment YAMLs. A YAML load failure may be a legacy runtime or dependency issue, not a malformed YAML.
- Use `PYTHONPATH=models_related/ultralytics` only for historical reproduction. Use `PYTHONPATH=.` for project-owned tests.
- Keep unrelated dirty files and active training scripts untouched. Check Git status before staging or committing.
