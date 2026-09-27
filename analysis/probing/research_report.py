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
        lines.append("")
    lines.extend([
        "## Hypothesis assessment",
        "",
        "### 1. Causal feature intervention: interesting but incomplete",
        "The image intervention provides a falsifiable signal for object evidence versus context. It is not sufficient by itself to prove that the detector uses non-causal background features because the masked images create out-of-distribution inputs and the current lightweight implementation reports image-level score changes. Promote only if object removal consistently hurts tiny-object scores while context-only images retain a non-trivial score on at least two datasets, with matched feature-level interventions as follow-up.",
        "",
        "### 2. Tiny object information bottleneck: candidate direction",
        "The most actionable signal is a stage-dependent decline in object-region activation fraction or object/context separation, especially when it is stronger for tiny buckets than for medium objects on multiple datasets. The current probe quantifies this as a representation proxy, not mutual information. If the same critical transition appears across datasets, the unresolved problem is measurable information survival rather than a generic need for more capacity.",
        "",
        "### 3. Object survival modeling: candidate direction, dependent on cross-dataset monotonicity",
        "Survival scores make the point of failure explicit. A consistent drop from shallow features to deeper detection features, correlated with object scale, would support modeling survival as a measurable state variable. If the curves are non-monotonic or dataset-specific, discard a universal survival law and retain it as an evaluation tool.",
        "",
        "### 4. Gradient optimization: diagnostic, not yet a method",
        "The current gradient probe measures sensitivity of an activation objective and cannot establish unstable training gradients. It is useful for locating layers and scales with weak or noisy signal, but a method claim requires matched training-time per-object gradients and repeated seeds. Do not implement a loss change from this probe alone.",
        "",
        "## Recommended research direction",
        "Prioritize **scale-conditioned object information survival**: a diagnostic and eventual method that estimates whether object evidence survives each representation transition, separates object evidence from context shortcuts, and only intervenes at the empirically identified critical stage. This direction is deliberately narrower than adding attention, another P2 head, or generic feature gating. It is motivated by the existing reports: P2 retains visible high-frequency evidence on LEVIR-Ship, while score assignment and size-dependent supervision remain problematic, so the open question is not simply whether shallow features exist but whether usable object evidence survives transformation into the final candidate score.",
        "",
        "## Falsification and next experiments",
        "- Repeat the probes with three fixed seeds and matched checkpoints per dataset.",
        "- Replace image masking with true hook-level feature replacement at one stage, using object-region and context-region tensors with energy-matched controls.",
        "- Train a frozen-feature linear probe for object center/presence and nuisance labels, reporting held-out AUROC rather than activation proxies.",
        "- Capture true per-object classification, box, and DFL gradients during matched training steps.",
        "- Promote the direction only if the same critical stage and scale dependence replicate on at least two datasets and survive energy-matched controls.",
        "",
        "## Limitations",
        "The artifacts are lightweight diagnostics. They do not establish causal mechanism, mutual information, survival probability in a probabilistic sense, or training instability without the follow-up experiments above. Missing or failed dataset runs must be reported as missing evidence, never filled with validation metrics or inferred conclusions.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
