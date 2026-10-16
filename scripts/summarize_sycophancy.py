#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish aggregate A7 sycophancy evidence")
    parser.add_argument("--artifacts", default="artifacts/qwen-sycophancy-1000")
    parser.add_argument("--output-json", default="results/sycophancy/qwen25-15b.json")
    parser.add_argument("--output-table", default="docs/tables/sycophancy_audit.md")
    parser.add_argument("--output-figure", default="docs/figures/sycophancy_audit.png")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    summary = read_json(artifacts / "summary.json")
    interventions = read_json(artifacts / "interventions.json")
    baseline = summary["baseline_metrics"]
    causal = {
        method: row["delta_mean_harmful_score_ci"]
        for method, row in interventions["causal_tests"].items()
    }
    baseline_signal_observed = bool(
        baseline["behavior_gap"] > 0 and baseline["balanced_accuracy"] > 0.5
    )
    result = {
        "schema_version": "1.0",
        "experiment": "A7_sycophancy",
        "source_artifact": str(artifacts),
        "model": summary["model"],
        "preregistration_hash": summary.get("preregistration_hash"),
        "test_pairs": summary["test_pairs"],
        "target_layer": summary["target_layer"],
        "baseline": baseline,
        "baseline_signal_observed": baseline_signal_observed,
        "causal_interpretation_status": (
            "eligible" if baseline_signal_observed else "not_supported_by_baseline_separation"
        ),
        "causal_harmful_score_deltas": causal,
        "sae_outperforms_simple_baselines": summary["sae_outperforms_simple_baselines"],
        "equivalence_tests_passed": summary["equivalence_tests_passed"],
        "equivalence_analysis_design": summary.get(
            "equivalence_analysis_design", "post_hoc_sensitivity"
        ),
    }
    write_json(Path(args.output_json), result)

    table = [
        "# A7 Sycophancy Audit",
        "",
        f"Model: `{summary['model']}`; held-out pairs: `{summary['test_pairs']}`; target layer: `{summary['target_layer']}`.",
        "",
        "| Baseline metric | Value |",
        "|---|---:|",
        f"| Pressure mean score | {baseline['mean_harmful_score']:.4f} |",
        f"| Neutral mean score | {baseline['mean_benign_score']:.4f} |",
        f"| Behavior gap | {baseline['behavior_gap']:+.4f} |",
        f"| Balanced accuracy | {baseline['balanced_accuracy']:.3f} |",
        "",
        "| Intervention | Harmful-score delta (95% CI) |",
        "|---|---:|",
    ]
    for method, ci in sorted(causal.items()):
        table.append(f"| {method} | {ci['estimate']:+.4f} [{ci['low']:+.4f}, {ci['high']:+.4f}] |")
    table.extend(
        [
            "",
            "The pressure/neutral baseline did not separate, so intervention rankings are descriptive and do not establish a sycophancy mechanism.",
        ]
    )
    table_path = Path(args.output_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(table) + "\n")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods = sorted(causal)
    estimates = [causal[method]["estimate"] for method in methods]
    lower = [estimates[i] - causal[method]["low"] for i, method in enumerate(methods)]
    upper = [causal[method]["high"] - estimates[i] for i, method in enumerate(methods)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(["Pressure", "Neutral"], [baseline["mean_harmful_score"], baseline["mean_benign_score"]])
    axes[0].set_ylabel("Agree-vs-disagree score")
    axes[0].set_title(f"Baseline gap {baseline['behavior_gap']:+.4f}")
    axes[1].errorbar(estimates, range(len(methods)), xerr=[lower, upper], fmt="o")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(range(len(methods)), [method.replace("_", " ") for method in methods])
    axes[1].set_xlabel("Held-out harmful-score delta")
    axes[1].set_title("Descriptive intervention effects")
    fig.text(0.01, 0.01, f"Source: {args.output_json}", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figure_path = Path(args.output_figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
