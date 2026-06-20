from __future__ import annotations

from pathlib import Path

from ..utils import read_json


def render_report(artifact_dir: str | Path) -> str:
    artifact_dir = Path(artifact_dir)
    summary = read_json(artifact_dir / "summary.json")
    interventions = read_json(artifact_dir / "interventions.json")
    features = read_json(artifact_dir / "features.json")
    baseline = summary["baseline_metrics"]
    selected = summary["validation_selected_sae_steering"]
    strongest_ablation = summary["strongest_causal_ablation"]
    ablation = interventions["causal_tests"]["sae_feature_ablation"]
    patch = interventions["causal_tests"]["sae_activation_patch"]

    if summary["evidence_class"] == "synthetic_fixture":
        evidence_note = (
            "This is a deterministic synthetic fixture run. It verifies the pipeline but is not "
            "evidence about a real language model."
        )
    elif "fixture" in summary["evidence_class"]:
        evidence_note = (
            "This is a real-model smoke-fixture measurement. It validates the empirical pipeline "
            "but is too small for a publication-grade claim."
        )
    else:
        evidence_note = "This report contains measurements from the configured open model."
    lines = [
        "# Glassbox Audit Report",
        "",
        f"**Evidence class:** `{summary['evidence_class']}`",
        "",
        evidence_note,
        "",
        "## Experiment",
        "",
        f"- Model: `{summary['model']}`",
        f"- Behavior: `{summary['behavior']}`",
        f"- Located layer: `{summary['target_layer']}`",
        f"- Test records: `{summary['test_records']}`",
        f"- Refusal threshold: `{summary['behavior_threshold']:.4f}` "
        f"(source: `{summary['threshold_source']}`)",
        "",
        "## Baseline Behavior",
        "",
        f"- Harmful refusal rate: {baseline['harmful_refusal_rate']:.3f}",
        f"- Benign over-refusal rate: {baseline['benign_overrefusal_rate']:.3f}",
        f"- Behavior gap: {baseline['behavior_gap']:.3f}",
        f"- Capability perplexity: {baseline['capability_perplexity']:.3f}",
        "",
        "## Sparse Feature Result",
        "",
        f"- Selected features: `{summary['selected_features']}`",
        f"- SAE variance explained: {summary['sae_metrics']['variance_explained']:.3f}",
        f"- SAE mean L0: {summary['sae_metrics']['mean_l0']:.2f}",
        f"- Largest feature effect size: {features[0]['effect_size']:.3f}",
        "",
        "## Causal Tests",
        "",
        f"- Validation-selected SAE steering scale: {selected['scale']:+.2f}",
        f"- Held-out harmful score delta: {selected['delta_mean_harmful_score_ci']['estimate']:+.3f} "
        f"[{selected['delta_mean_harmful_score_ci']['low']:+.3f}, "
        f"{selected['delta_mean_harmful_score_ci']['high']:+.3f}]",
        f"- Feature ablation harmful score delta: "
        f"{ablation['delta_mean_harmful_score_ci']['estimate']:+.3f}",
        f"- Benign-mean activation patch harmful score delta: "
        f"{patch['delta_mean_harmful_score_ci']['estimate']:+.3f}",
        f"- Strongest direction/feature ablation: `{strongest_ablation['method']}` "
        f"({strongest_ablation['delta_mean_harmful_score_ci']['estimate']:+.3f})",
        "",
        "## Baseline Comparison",
        "",
        "| Method | Scale | Harmful score delta | Benign score delta | Capability NLL delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in interventions["steering_sweeps"]:
        if row["scale"] == interventions["validation_selected_scales"][row["method"]]:
            lines.append(
                f"| {row['method']} | {row['scale']:+.2f} | "
                f"{row['delta_mean_harmful_score_ci']['estimate']:+.3f} | "
                f"{row['delta_mean_benign_score_ci']['estimate']:+.3f} | "
                f"{row['delta_capability_nll_ci']['estimate']:+.3f} |"
            )
    lines.extend(
        [
            "",
            "Random controls and all sweep points are stored in `interventions.json`. Confidence "
            "intervals are paired bootstrap intervals over prompts.",
            "",
            "## Interpretation Guardrails",
            "",
            "- A direction that steers behavior is not automatically a complete mechanism explanation.",
            "- Probe accuracy is correlational; ablation and patching provide the stronger causal tests.",
            "- Compare effect size and capability cost, not just whether an intervention moves the metric.",
            "- Real-model conclusions require multiple seeds, held-out prompt families, and replication.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir) / "REPORT.md"
    path.write_text(render_report(artifact_dir))
    return path
