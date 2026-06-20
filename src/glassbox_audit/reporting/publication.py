from __future__ import annotations

import os
from pathlib import Path

from ..analysis import refresh_failure_analysis
from ..utils import read_json


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def _ci(result: dict[str, object], key: str) -> str:
    value = result[key]
    return f"{value['estimate']:+.3f} [{value['low']:+.3f}, {value['high']:+.3f}]"


def _selected_steering(interventions: dict[str, object]) -> list[dict[str, object]]:
    scales = interventions["validation_selected_scales"]
    return [
        row
        for row in interventions["steering_sweeps"]
        if row["scale"] == scales[row["method"]]
    ]


def _generate_figures(artifact_dir: Path, figures_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/glassbox-matplotlib")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    scan = read_json(artifact_dir / "layer_scan.json")
    layer_controls = read_json(artifact_dir / "layerwise_controls.json")
    interventions = read_json(artifact_dir / "interventions.json")
    failure = read_json(artifact_dir / "failure_analysis.json")
    sae = read_json(artifact_dir / "sae_metrics.json")
    summary = read_json(artifact_dir / "summary.json")

    plt.style.use("dark_background")
    colors = {"sae_features": "#7dd3fc", "mean_difference": "#fbbf24", "linear_probe": "#c084fc"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        [row["layer"] for row in scan],
        [row["localization_score"] for row in scan],
        marker="o",
        color="#7dd3fc",
    )
    axes[0].axvline(summary["target_layer"], color="#fbbf24", linestyle="--", label="selected")
    axes[0].set(title="Train-only refusal localization", xlabel="Layer", ylabel="Localization score")
    axes[0].legend(frameon=False)
    axes[1].bar(
        [str(row["layer"]) for row in layer_controls],
        [row["delta_mean_harmful_score_ci"]["estimate"] for row in layer_controls],
        color=["#fbbf24" if row["is_selected_layer"] else "#475569" for row in layer_controls],
    )
    axes[1].axhline(0, color="white", linewidth=0.7)
    axes[1].set(
        title="Held-out mean-difference ablation by layer",
        xlabel="Layer",
        ylabel="Harmful refusal-score delta",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "layer_localization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    causal = interventions["causal_tests"]
    names = list(causal)
    estimates = [causal[name]["delta_mean_harmful_score_ci"]["estimate"] for name in names]
    lows = [causal[name]["delta_mean_harmful_score_ci"]["low"] for name in names]
    highs = [causal[name]["delta_mean_harmful_score_ci"]["high"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(names)))
    ax.errorbar(
        estimates,
        positions,
        xerr=[
            [estimate - low for estimate, low in zip(estimates, lows, strict=True)],
            [high - estimate for estimate, high in zip(estimates, highs, strict=True)],
        ],
        fmt="o",
        color="#7dd3fc",
        ecolor="#94a3b8",
        capsize=4,
    )
    ax.axvline(0, color="white", linewidth=0.7)
    ax.set_yticks(positions, [name.replace("_", " ") for name in names])
    ax.set(title="Held-out causal interventions with paired bootstrap intervals", xlabel="Harmful score delta")
    fig.tight_layout()
    fig.savefig(figures_dir / "causal_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in interventions["steering_sweeps"]:
        ax.scatter(
            row["delta_mean_harmful_score_ci"]["estimate"],
            row["delta_capability_nll_ci"]["estimate"],
            color=colors[row["method"]],
            s=45,
            alpha=0.8,
            label=row["method"],
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    ax.axvline(0, color="white", linewidth=0.5)
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set(
        title="Steering trade-off across frozen test sweeps",
        xlabel="Harmful refusal-score delta (lower is stronger)",
        ylabel="Capability NLL delta (lower is better)",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "steering_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    history = sae["history"]
    diagnostics = failure["sae_diagnostics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([row["epoch"] for row in history], [row["mse"] for row in history], color="#7dd3fc")
    axes[0].set(title="SAE reconstruction training", xlabel="Epoch", ylabel="MSE")
    labels = ["Active", "Dead"]
    values = [1 - diagnostics["dead_feature_fraction"], diagnostics["dead_feature_fraction"]]
    axes[1].bar(labels, values, color=["#34d399", "#fb7185"])
    axes[1].set_ylim(0, 1)
    axes[1].set(title="SAE dictionary utilization", ylabel="Fraction of features")
    fig.tight_layout()
    fig.savefig(figures_dir / "sae_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    categories = failure["category_analysis"]
    category_names = list(categories)
    positions = list(range(len(category_names)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [position - 0.2 for position in positions],
        [categories[name]["sae_feature_ablation_harmful_delta"] for name in category_names],
        height=0.38,
        label="SAE feature ablation",
        color="#7dd3fc",
    )
    ax.barh(
        [position + 0.2 for position in positions],
        [categories[name]["mean_difference_ablation_harmful_delta"] for name in category_names],
        height=0.38,
        label="Mean-difference ablation",
        color="#fbbf24",
    )
    ax.set_yticks(positions, category_names)
    ax.axvline(0, color="white", linewidth=0.7)
    ax.legend(frameon=False)
    ax.set(title="Category-level held-out causal effects", xlabel="Harmful refusal-score delta")
    fig.tight_layout()
    fig.savefig(figures_dir / "category_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    baseline = summary["baseline_metrics"]
    fig = plt.figure(figsize=(13, 7), facecolor="#07111f")
    fig.text(0.05, 0.92, "GLASSBOX · REFUSAL AUDIT WORKBENCH", color="#e2e8f0", fontsize=20, weight="bold")
    fig.text(
        0.05,
        0.875,
        f"{summary['model']} · layer {summary['target_layer']} · {summary['evidence_class']}",
        color="#94a3b8",
        fontsize=11,
    )
    cards = [
        ("Harmful refusal", f"{baseline['harmful_refusal_rate']:.1%}", "#7dd3fc"),
        ("Benign over-refusal", f"{baseline['benign_overrefusal_rate']:.1%}", "#fbbf24"),
        ("Behavior gap", f"{baseline['behavior_gap']:.3f}", "#c084fc"),
        ("SAE dead features", f"{summary['sae_metrics']['dead_feature_fraction']:.1%}", "#fb7185"),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.05 + index * 0.235
        box = plt.Rectangle((x, 0.67), 0.205, 0.14, color="#132238", transform=fig.transFigure)
        fig.patches.append(box)
        fig.text(x + 0.018, 0.76, label, color="#94a3b8", fontsize=9)
        fig.text(x + 0.018, 0.70, value, color=color, fontsize=22, weight="bold")
    ax = fig.add_axes((0.07, 0.12, 0.55, 0.42), facecolor="#0b1729")
    selected = _selected_steering(interventions)
    ax.barh(
        [row["method"].replace("_", " ") for row in selected],
        [row["delta_mean_harmful_score_ci"]["estimate"] for row in selected],
        color=[colors[row["method"]] for row in selected],
    )
    ax.axvline(0, color="white", linewidth=0.5)
    ax.set_title("Validation-selected steering on held-out test", loc="left", color="#e2e8f0")
    ax.set_xlabel("Harmful score delta")
    negative = failure["negative_findings"][:4]
    fig.text(0.68, 0.54, "FAILURE ANALYSIS", color="#e2e8f0", fontsize=12, weight="bold")
    for index, item in enumerate(negative):
        fig.text(0.68, 0.49 - index * 0.09, f"• {item}", color="#cbd5e1", fontsize=9, wrap=True)
    fig.savefig(figures_dir / "workbench_overview.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_publication_bundle(artifact_dir: str | Path, docs_dir: str | Path) -> dict[str, str]:
    artifact_dir, docs_dir = Path(artifact_dir), Path(docs_dir)
    refresh_failure_analysis(artifact_dir)
    figures_dir, tables_dir = docs_dir / "figures", docs_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(artifact_dir / "summary.json")
    manifest = read_json(artifact_dir / "manifest.json")
    protocol = read_json(artifact_dir / "protocol.json")
    interventions = read_json(artifact_dir / "interventions.json")
    failure = read_json(artifact_dir / "failure_analysis.json")
    layer_controls = read_json(artifact_dir / "layerwise_controls.json")
    comparisons = read_json(artifact_dir / "method_comparisons.json")
    component_artifact = (
        read_json(artifact_dir / "component_localization.json")
        if (artifact_dir / "component_localization.json").exists()
        else None
    )
    _generate_figures(artifact_dir, figures_dir)

    baseline = summary["baseline_metrics"]
    causal = interventions["causal_tests"]
    selected = _selected_steering(interventions)
    random_sae_protocol = interventions.get("random_sae_control_protocol")
    if random_sae_protocol:
        random_sae_setup = (
            "The untrained-SAE control uses the trained SAE architecture, trained activation "
            "center, zeroed encoder bias, decoder steering, and literal encode -> zero selected "
            "latent -> decode feature ablation. No feature, layer, threshold, or scale was retuned "
            "for this refresh."
        )
        random_sae_limitation = (
            "- The random untrained-SAE control is now train-centered and zero-bias matched, but "
            "it remains a seed-limited control distribution rather than a full random-SAE seed sweep."
        )
    else:
        random_sae_setup = (
            "The untrained-SAE control predates the stricter train-centered zero-bias refresh; "
            "interpret its latent controls as native untrained-SAE controls."
        )
        random_sae_limitation = (
            "- Random untrained-SAE latent controls use native untrained preprocessing; a "
            "train-centered, zero-bias matched-control rerun remains future work."
        )
    baseline_table = [
        "# Baseline Metrics",
        "",
        "| Metric | Held-out test value |",
        "|---|---:|",
    ]
    for key, value in baseline.items():
        baseline_table.append(f"| {key.replace('_', ' ')} | {value:.4f} |")
    _write(tables_dir / "baseline_metrics.md", "\n".join(baseline_table))

    causal_table = [
        "# Causal Results",
        "",
        "| Intervention | Harmful score delta (95% CI) | Benign score delta | Capability NLL delta |",
        "|---|---:|---:|---:|",
    ]
    for name, row in causal.items():
        causal_table.append(
            f"| {name.replace('_', ' ')} | {_ci(row, 'delta_mean_harmful_score_ci')} | "
            f"{_ci(row, 'delta_mean_benign_score_ci')} | {_ci(row, 'delta_capability_nll_ci')} |"
        )
    _write(tables_dir / "causal_results.md", "\n".join(causal_table))

    layer_table = [
        "# Layerwise Ablation Controls",
        "",
        "| Layer | Selected layer? | Harmful score delta | Benign score delta | Capability NLL delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in layer_controls:
        layer_table.append(
            f"| {row['layer']} | {row['is_selected_layer']} | "
            f"{_ci(row, 'delta_mean_harmful_score_ci')} | "
            f"{_ci(row, 'delta_mean_benign_score_ci')} | "
            f"{_ci(row, 'delta_capability_nll_ci')} |"
        )
    _write(tables_dir / "layerwise_controls.md", "\n".join(layer_table))

    comparison_table = [
        "# Direct Paired Method Comparisons",
        "",
        "Negative harmful-score differences mean the left method suppresses more than the right.",
        "",
        "| Setting | Left | Right | Harmful left-minus-right (95% CI) |",
        "|---|---|---|---:|",
    ]
    for setting, rows in comparisons.items():
        for row in rows:
            comparison_table.append(
                f"| {setting.replace('_', ' ')} | {row['left_method']} | {row['right_method']} | "
                f"{_ci(row, 'harmful_score_left_minus_right_ci')} |"
            )
    _write(tables_dir / "method_comparisons.md", "\n".join(comparison_table))

    category_table = [
        "# Category-Level Failure Analysis",
        "",
        "| Category | Baseline gap | SAE feature-ablation harmful delta | Mean-difference ablation harmful delta |",
        "|---|---:|---:|---:|",
    ]
    for category, row in failure["category_analysis"].items():
        category_table.append(
            f"| {category} | {row['baseline_behavior_gap']:+.3f} | "
            f"{row['sae_feature_ablation_harmful_delta']:+.3f} | "
            f"{row['mean_difference_ablation_harmful_delta']:+.3f} |"
        )
    _write(tables_dir / "category_analysis.md", "\n".join(category_table))

    component_section = ""
    if component_artifact is not None:
        component_table = [
            "# Component Localization",
            "",
            "Frozen component-output ablations on the held-out test split.",
            "",
            "| Layer | Component | Direction | Harmful score delta | Benign score delta | Capability NLL delta |",
            "|---:|---|---|---:|---:|---:|",
        ]
        for row in component_artifact["components"]:
            if row["status"] != "completed":
                component_table.append(
                    f"| {row['layer']} | {row['component']} | {row['direction']} | "
                    f"{row['status']} |  |  |"
                )
                continue
            component_table.append(
                f"| {row['layer']} | {row['component']} | {row['direction']} | "
                f"{_ci(row, 'delta_mean_harmful_score_ci')} | "
                f"{_ci(row, 'delta_mean_benign_score_ci')} | "
                f"{_ci(row, 'delta_capability_nll_ci')} |"
            )
        _write(tables_dir / "component_localization.md", "\n".join(component_table))
        component_section = f"""
## Component-Output Localization

The optional component analysis uses the frozen `{component_artifact['direction']}` direction at
layer `{component_artifact['target_layer']}`. It is `{component_artifact['evidence_scope']}`.
Residual-stream ablation is strong, while direct attention-output and MLP-output ablations at the
same layer are much smaller. See [component analysis](COMPONENT_ANALYSIS.md) and
[component localization table](tables/component_localization.md).
"""

    negatives = "\n".join(f"- {item}" for item in failure["negative_findings"])
    mean_vs_sae = next(
        row
        for row in comparisons["causal_ablation"]
        if row["left_method"] == "mean_difference_direction_ablation"
        and row["right_method"] == "sae_feature_ablation"
    )
    _write(
        docs_dir / "NEGATIVE_RESULTS.md",
        f"""# Negative Results

**Evidence class:** `{summary['evidence_class']}`

Negative results are generated from the same artifact contract as positive findings.

{negatives}

## SAE Versus Simpler Baselines

- SAE specificity-adjusted utility: `{failure['baseline_comparison']['sae_specificity_adjusted_utility']:.4f}`
- Best simple baseline: `{failure['baseline_comparison']['best_simple_method']}`
- Best simple utility: `{failure['baseline_comparison']['best_simple_specificity_adjusted_utility']:.4f}`
- SAE utility flag: `{failure['baseline_comparison']['sae_outperforms_simple_baselines']}`

## Interpretation

These diagnostics prevent a steerable SAE direction from being mistaken for a uniquely useful or
stable mechanism-level explanation. See `failure_analysis.json` for all machine-readable diagnostics.
""",
    )

    dataset = manifest["dataset"]
    selected_rows = "\n".join(
        f"| {row['method']} | {row['scale']:+.2f} | "
        f"{_ci(row, 'delta_mean_harmful_score_ci')} | "
        f"{_ci(row, 'delta_mean_benign_score_ci')} | "
        f"{_ci(row, 'delta_capability_nll_ci')} |"
        for row in selected
    )
    results_text = f"""# Results: Real-Model Refusal Audit

**Evidence class:** `{summary['evidence_class']}`

This document is generated from `{artifact_dir}`. Results are reported on the held-out test split
after layer/feature discovery on train and threshold/scale selection on validation.

## Headline Results

- Model: `{summary['model']}`
- Selected layer: `{summary['target_layer']}`
- Dataset: `{dataset['n_pairs']}` matched pairs across `{dataset['n_families']}` family-disjoint families
- Test harmful refusal: `{baseline['harmful_refusal_rate']:.1%}`
- Test benign over-refusal: `{baseline['benign_overrefusal_rate']:.1%}`
- SAE dead features: `{summary['sae_metrics']['dead_feature_fraction']:.1%}`
- SAE utility flag: `{summary['sae_outperforms_simple_baselines']}`

## Frozen Steering Comparison

| Method | Validation-selected scale | Harmful score delta | Benign score delta | Capability NLL delta |
|---|---:|---:|---:|---:|
{selected_rows}

## Strongest Causal Evidence

- SAE feature ablation: `{_ci(causal['sae_feature_ablation'], 'delta_mean_harmful_score_ci')}`
- SAE activation patch: `{_ci(causal['sae_activation_patch'], 'delta_mean_harmful_score_ci')}`
- Strongest ablation: `{summary['strongest_causal_ablation']['method']}`
- Strongest layerwise control: layer `{summary['strongest_layerwise_control']['layer']}`
- Direct mean-difference-minus-SAE ablation effect: `{_ci(mean_vs_sae, 'harmful_score_left_minus_right_ci')}`
- SAE feature-ablation random-control empirical p: `{failure['random_ablation_control_comparison']['empirical_one_sided_p']:.3f}`
- Random-SAE control protocol: `{random_sae_protocol['status'] if random_sae_protocol else 'native_untrained_preprocessing'}`

See [Negative Results](NEGATIVE_RESULTS.md), [causal results](tables/causal_results.md), and
[direct paired comparisons](tables/method_comparisons.md).
"""
    _write(docs_dir / "RESULTS_REAL_AUDIT.md", results_text)

    report = f"""# Glassbox: A Held-Out Causal Audit of Refusal in Qwen2.5-1.5B-Instruct

## Abstract

Glassbox tests whether sparse autoencoder features provide a more useful causal account of refusal
behavior than simple activation-space baselines. We use a controlled paired corpus with
family-disjoint train, validation, and test splits. Layer and feature discovery use train only;
behavior thresholds and intervention scales use validation only; all headline effects are frozen
held-out test measurements. The audit includes steering, feature ablation, activation patching,
direct paired method comparisons, random controls, layerwise controls, capability cost, and
failure analysis. The current run is labeled `{summary['evidence_class']}` and should be interpreted
at that evidence level.

## Motivation

Sparse features are often presented as explanations once they correlate with behavior or support
steering. A stronger standard asks whether they survive held-out causal tests and outperform simpler
directions at comparable collateral cost. Glassbox makes that comparison the central result.

## Dataset

The audit uses `{dataset['n_pairs']}` harmful/benign matched pairs from
`{next(iter(dataset['sources']))}`, spanning `{len(dataset['categories'])}` safety categories and
`{dataset['n_families']}` scenario families. Families never cross splits. Benign prompts retain the
sensitive topic while asking for prevention, detection, or safety guidance.

## Model

The default model is `{summary['model']}`. Refusal is operationalized as a contrast between the
teacher-forced log probabilities of a refusal prefix and a compliance prefix. The binary threshold
is selected on validation; causal effects use the continuous score.

## Experimental Protocol

{protocol['train_records_used_for_discovery_and_sae_only']} train records localize refusal across
configured residual-stream layers. A top-k SAE is trained only on train-token activations
(`{protocol['sae_training']['activation_samples']}` samples, up to
`{protocol['sae_training']['tokens_per_prompt']}` tokens per prompt). Final-token train activations
rank features. Mean-difference and logistic-probe directions use the same train data.

## SAE and Baselines

The SAE dictionary has `{summary['sae_metrics']['n_features']}` features with top-k
`{manifest['config']['sae']['top_k']}` sparsity. The audit compares its selected decoder-feature
direction and latent ablation against a train mean-difference direction, a train logistic probe,
isotropic random directions, and an architecture-matched untrained SAE. Random directions receive
steering and direction ablation; untrained-SAE controls receive decoder steering and literal latent
feature ablation.

{random_sae_setup}

## Causal Intervention Protocol

Validation selects each method's steering scale. Held-out test evaluates steering, SAE feature
ablation, direction ablation, activation patching, random directions, random untrained-SAE
directions, and mean-difference ablation at every scanned layer. Paired bootstrap intervals quantify
uncertainty.

## Results

![Layer localization](figures/layer_localization.png)

![Causal effects](figures/causal_effects.png)

![Steering trade-off](figures/steering_tradeoff.png)

The held-out behavior gap is `{baseline['behavior_gap']:.3f}`. The selected layer is
`{summary['target_layer']}`. SAE feature ablation changes harmful refusal score by
`{_ci(causal['sae_feature_ablation'], 'delta_mean_harmful_score_ci')}`. The SAE utility flag under
the pre-specified specificity/capability metric is
`{summary['sae_outperforms_simple_baselines']}`.

Directly comparing the strongest simple causal baseline with SAE feature ablation, the
mean-difference-minus-SAE harmful-score effect is
`{_ci(mean_vs_sae, 'harmful_score_left_minus_right_ci')}`. The SAE feature-ablation effect is
stronger than the observed random-ablation range, and with
`{failure['random_ablation_control_comparison']['n_controls']}` controls its empirical one-sided
p-value is `{failure['random_ablation_control_comparison']['empirical_one_sided_p']:.3f}`.

{component_section}

## Failure Analysis

![SAE diagnostics](figures/sae_diagnostics.png)

![Category effects](figures/category_effects.png)

{negatives}

## Limitations

- The controlled corpus is lexically matched and family-disjoint, but it is not a naturalistic
  refusal benchmark.
- Prefix-probability scoring is deterministic and intervention-friendly but not a semantic grader.
- One model and one primary SAE seed cannot establish mechanism stability.
- Residual-stream interventions may move activations off distribution.
- Statistical intervals quantify prompt uncertainty, not model- or SAE-seed uncertainty.
{random_sae_limitation}

## Reproducibility

```bash
glassbox build-real-audit-data --output data/refusal_controlled_v1.jsonl
CUDA_VISIBLE_DEVICES=0 glassbox run --config configs/qwen2.5-1.5b-real-audit.yaml --output artifacts/qwen-real-audit
CUDA_VISIBLE_DEVICES=0 glassbox refresh-controls --artifacts artifacts/qwen-real-audit --control random-sae --device cuda:0
CUDA_VISIBLE_DEVICES=0 glassbox component-localize --artifacts artifacts/qwen-real-audit --components residual,attention,mlp --direction mean_difference --device cuda:0
glassbox paper --artifacts artifacts/qwen-real-audit --docs docs
```

The exact configuration, runtime, split counts, selection protocol, outputs, and confidence
intervals are stored in the run artifact directory.
"""
    _write(docs_dir / "REPORT.md", report)
    return {
        "report": str(docs_dir / "REPORT.md"),
        "results": str(docs_dir / "RESULTS_REAL_AUDIT.md"),
        "negative_results": str(docs_dir / "NEGATIVE_RESULTS.md"),
        "figures": str(figures_dir),
        "tables": str(tables_dir),
    }
