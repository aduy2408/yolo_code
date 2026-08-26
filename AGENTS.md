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
