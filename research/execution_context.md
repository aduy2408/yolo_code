# Current Execution Context

Only one experiment may be active here at a time. The manifest is immutable;
this file records the current lifecycle state.

## Active experiment

- Hypothesis: none
- Experiment: none
- Run: none
- Status: idle
- Manifest: none
- Run directory: none
- Server: none
- Next action: create and validate a hypothesis manifest

## Operator notes

- Do not launch from this file alone; validate the manifest first.
- Use `utils.marimo_ops` for remote execution.
- A dead PID or checkpoint is not completion evidence.
