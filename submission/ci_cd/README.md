# CI/CD workflow map

The submission branch keeps the repository's CI/CD orchestration in its canonical locations; this folder documents the entrypoints without duplicating them.

| Responsibility | Canonical source |
|---|---|
| Static checks and orchestration tests | `.github/workflows/ci.yml` |
| GitHub Actions remote train/evaluate/export dispatch | `.github/workflows/mlops.yml` |
| Local GitHub workflow dispatcher and monitor | `scripts/run_marimo_workflow.sh` |
| Jetson download/evaluate/export pipeline | `scripts/run_jetson_workflow.sh` |
| Marimo connection and detached lifecycle | `utils/marimo_client.py`, `utils/marimo_ops.py`, `utils/marimo_run.py`, `utils/marimo_watch.py` |
| CI remote operation adapter | `utils/marimo_ci.py` |
| Contract and experiment identity | `utils/marimo/contracts.py`, `research_loop.py` |
| Workflow contract | `.agents/workflows/marimo-train.md` |

The workflow requires explicit `MARIMO_URL`, Marimo token, and (for training/upload) `HF_TOKEN`. It validates the exact checkout revision before launch and records durable run state, artifacts, and remote upload verification. Do not replace it with ad-hoc `nohup`, `Popen`, or browser automation.

Typical local entrypoints:

```bash
python -m utils.marimo_ops preflight --help
python -m utils.marimo_ops launch --help
python utils/marimo_run.py -c 'print("connected")'
./scripts/run_marimo_workflow.sh --help
```
