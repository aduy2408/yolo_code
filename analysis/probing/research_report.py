"""Create a cautious cross-dataset research exploration report from artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("runs/tod_probes"))
    parser.add_argument("--output", type=Path, default=Path("docs/reports/tod_research_exploration.md"))
    args = parser.parse_args()
    datasets = {}
    for key in ("levir-ship", "visdrone", "tinyperson"):
        root = args.artifacts / key
        datasets[key] = {
            "causal": _load(root / "causal_intervention.json"),
            "information": _load(root / "information_probe.json"),
            "survival": _load(root / "object_survival.json"),
            "gradient": _load(root / "gradient_probe.json"),
        }
    candidate_root = args.artifacts.parent / "tod_candidate_matched"
    candidate = {key: _load(candidate_root / key / "candidate_evidence.json") for key in datasets}
    present = [key for key, value in datasets.items() if value["causal"]]
    lines = [
        "# Tiny Object Detection Research Exploration",
        "",
        "## Scope and decision rule",
        "This report is a diagnostic exploration, not a method claim. A hypothesis is promoted only when the same directional signal is observed on at least two of LEVIR-Ship, VisDrone2019-DET, and TinyPerson. The probes are lightweight proxies and do not modify detector behavior.",
        "",
        "## Existing baseline context",
        "The project baseline reports show a consistent tiny-object difficulty gap: standard YOLOv8n test `mAP50-95` is approximately 0.264 on LEVIR-Ship, 0.151 on VisDrone mosaic, and 0.178 on TinyPerson. These are context values, not probe outcomes, and their split-qualified provenance remains in the source reports.",
        "",
        "## Experiments performed",
        "1. Causal intervention: paired original, object-removed, object-only, and context-only image inference. `background_dependence` is context-only score divided by original score, while `object_sensitivity` is the original minus object-removed score.",
        "2. Information bottleneck: hook-level object-region activation fraction and context nuisance proxy across available backbone/P2/P3/P4/P5 modules.",
        "3. Object survival: object-region energy relative to surrounding context energy across stages.",
        "4. Gradient optimization: inference-only activation-objective gradient norm, variance, and signal-to-noise proxy. This is a diagnostic of feature sensitivity, not a replacement for full training gradient accounting.",
        "5. Decisive follow-up: raw-candidate local-pool analysis with the report-listed dataset-specific YOLOv8 checkpoints. For every GT object, candidates are restricted to a padded local neighborhood, then the highest-IoU candidate is compared with the score-selected candidate.",
        "",
        f"Artifacts available for: {', '.join(present) if present else 'none'}.",
        "",
        "## Dataset-level evidence",
    ]
    for key, value in datasets.items():
        lines.extend([f"### {key}", ""])
        causal = value["causal"].get("by_size", {})
        survival = value["survival"].get("survival_curves", {})
        information = value["information"].get("by_layer", {})
        gradient = value["gradient"].get("by_layer", {})
        lines.append(f"- Causal rows: {value['causal'].get('rows', 0)}. Size-conditioned object/background summaries: `{json.dumps(causal, sort_keys=True)}`.")
        lines.append(f"- Feature information summaries: `{json.dumps(information, sort_keys=True)}`.")
        lines.append(f"- Survival curves: `{json.dumps(survival, sort_keys=True)}`.")
        lines.append(f"- Gradient summaries: `{json.dumps(gradient, sort_keys=True)}`.")
        lines.append(f"- Matched candidate evidence: `{json.dumps(candidate[key].get('by_size', {}), sort_keys=True)}`.")
        lines.append("")
    lines.extend([
        "## Hypothesis assessment",
        "",
        "### 1. Causal feature intervention: reject as the primary direction",
        "The matched checkpoints do not show one consistent background-shortcut pattern. VisDrone tiny objects retain substantial context-only score, while TinyPerson tiny objects are highly sensitive to context removal, and LEVIR uses a different score regime. This is useful as a failure analysis, but not a stable cross-dataset method hypothesis.",
        "",
        "### 2. Tiny object information bottleneck: reject the simple compression story",
        "Object-region activation remains measurable in intermediate and late features. The feature curves are non-monotonic rather than progressively collapsing, and the activation proxy is not a held-out information estimator. The evidence does not justify a generic information-preserving module.",
        "",
        "### 3. Object survival modeling: reject as a universal law, retain as a measurement",
        "The survival score is useful for locating weak stages, but it does not decrease monotonically across LEVIR-Ship, VisDrone, and TinyPerson. It should remain an evaluation signal rather than become the method itself.",
        "",
        "### 4. Gradient optimization: reject as the first intervention",
        "The activation-gradient proxy varies by layer, but it does not measure true classification, box, or DFL training gradients. There is no cross-dataset causal evidence that gradient instability is the dominant failure. Do not begin with a new loss.",
        "",
        "### 5. Candidate evidence-to-score misalignment: supported and actionable",
        "The decisive matched-checkpoint result is a large local oracle gap on VisDrone and TinyPerson. The score-selected candidate trails the best-IoU candidate by roughly 0.25--0.32 IoU for tiny objects, while LEVIR shows a smaller but non-zero gap. Score-to-IoU rank correlation also weakens for the smallest buckets. This is the only hypothesis here with a coherent mechanism, a direct detector-level measurement, and replication across multiple domains.",
        "",
        "## Recommended research direction",
        "Prioritize **scale-conditioned candidate evidence-to-score alignment**. The future method should not add another backbone module by default. It should study why the detector can generate a locally good box candidate but assign the highest classification score to a worse candidate, especially for tiny objects. The likely contribution is a training-time candidate responsibility or score-calibration mechanism that preserves the relative ranking of localization quality without directly optimizing AP.",
        "",
        "## Falsification and next experiments",
        "- Repeat local candidate matching over three seeds using the official validation and test protocols.",
        "- Replace the decoded-box center neighborhood with the exact anchor/grid responsibility set used by TAL, and report the gap separately for P2/P3/P4.",
        "- Capture true per-candidate classification, box, and DFL losses and test whether the oracle gap is caused by classification assignment, regression quality, or both.",
        "- Run an energy-matched counterfactual that swaps only candidate classification logits while holding boxes fixed.",
        "- Only after the causal check, prototype a score-alignment intervention and require improvement on at least two datasets without degrading medium-object performance.",
        "",
        "## Limitations",
        "The matched sweep used one report-listed seed per dataset and 32 annotated images per dataset. The candidate evidence feature-energy proxy did not consistently outperform the detector score, so the conclusion is specifically about score/localization misalignment, not proof that raw feature energy is the correct replacement score. Full-seed causal candidate analysis remains the next gate.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
