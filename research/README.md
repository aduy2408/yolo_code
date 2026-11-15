# Bounded Research Loop

1. Read `prior_evidence.md` and the relevant source reports before proposing a method.
2. Copy `hypotheses/HYPOTHESIS_TEMPLATE.md` to a new `H###` card.
3. Record prior evidence and define one falsifiable mechanism.
4. Copy `protocols/experiment_manifest.template.json` to a unique experiment directory.
5. Validate the manifest before launch:

   ```bash
   python -m utils.research_loop validate-manifest \
     --manifest research/runs/H001/E001/experiment_manifest.json \
     --ledger research/experiment_ledger.jsonl
   ```

6. Validate the implementation locally, then run a bounded smoke test.
7. Ask `experiment-operator` to execute only the approved manifest on the server.
8. Use `utils.marimo_ops status`, `artifacts`, and `complete_verified` before interpreting metrics.
9. Append one lifecycle record:

   ```bash
   python -m utils.research_loop append-record \
     --ledger research/experiment_ledger.jsonl \
     --record research/runs/H001/E001/ledger_record.json
   ```

10. Ask `failure-analyst` for a structured diagnosis.
11. Record one decision in `decisions/decision_log.md` and update `execution_context.md`.

## Prior evidence index

`prior_evidence.md` classifies historical methods as foundations, active leads,
diagnostics, auxiliary candidates, closed directions, or rejected methods.
Closed methods remain available as historical evidence but are excluded from
new proposals unless a new hypothesis explains why the old failure should not
repeat.

## State separation

- `hypotheses/`: durable scientific claims and falsification rules;
- `prior_evidence.md`: curated report-derived method status and reusable rules;
- `protocols/`: immutable manifest/schema templates;
- `execution_context.md`: the single currently active run;
- `experiment_ledger.jsonl`: append-only machine-readable lifecycle history;
- `decisions/decision_log.md`: human-readable scientific decisions.


## Role handoff

Use the roles in this order:

1. `research-advisor`: proposes at most three evidence-grounded ideas; no code,
   manifest, launch, or grid search.
2. `omp Advisor/Watchdog`: observes the main session and workers continuously;
   raises `nit`, `concern`, or `blocker`; never edits or authorizes a run.
3. `research-director`: selects one idea and enforces the research policy.
4. `experiment-designer`: converts the selected idea into one hypothesis card
   and one immutable manifest.
5. `research-reviewer`: gates the card, protocol, diagnostics, and manifest.
6. `experiment-operator`: runs only the approved manifest on the server.
7. `failure-analyst`: diagnoses the verified result and recommends one bounded
   decision.

The advisor is allowed to suggest ideas, not to authorize them. The reviewer is
allowed to block a run, not to expand it into a sweep. The director owns the
final hypothesis selection and the single terminal decision.

The manifest defines run identity and must not be edited after launch. The
ledger records mutable lifecycle state. Every retry receives a new run ID and
run directory.

The baseline is immutable. A failed run is evidence only after its protocol and
artifacts are validated. Keep `nms_iou` at `0.5` in final artifacts.
