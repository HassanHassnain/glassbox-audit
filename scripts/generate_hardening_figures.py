#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#17202A"
BLUE = "#2463A6"
BLUE_DARK = "#17436F"
GOLD = "#C58B16"
GOLD_DARK = "#79530B"
GREY = "#A7ADB5"
GRID = "#E3E7EB"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def feature_budget(data: dict, output: Path) -> None:
    rows = data["feature_budget_frontier"]
    budgets = np.array([row["budget"] for row in rows])
    selected = np.array([row["selected"]["delta_harmful_score_ci"]["estimate"] for row in rows])
    low = np.array([row["selected"]["delta_harmful_score_ci"]["low"] for row in rows])
    high = np.array([row["selected"]["delta_harmful_score_ci"]["high"] for row in rows])
    controls = [
        [item["delta_harmful_score_ci"]["estimate"] for item in row["matched_random_summaries"]]
        for row in rows
    ]
    control_low = np.array([min(values) for values in controls])
    control_high = np.array([max(values) for values in controls])
    mean_effect = data["intervention_comparability"]["mean_direction_ablation"][
        "delta_harmful_score_ci"
    ]["estimate"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.19)
    ax.fill_between(
        budgets,
        control_low,
        control_high,
        color=GREY,
        alpha=0.3,
        label="8 same-dictionary random sets (range)",
    )
    ax.errorbar(
        budgets,
        selected,
        yerr=[selected - low, high - selected],
        color=BLUE,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2,
        capsize=3,
        label="Train-ranked SAE features (95% pair bootstrap CI)",
    )
    ax.axhline(
        mean_effect,
        color=GOLD_DARK,
        linestyle="--",
        linewidth=1.8,
        label=f"Mean-direction ablation ({mean_effect:.3f})",
    )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(budgets, labels=[str(value) for value in budgets])
    ax.set_xlabel("Number of ablated SAE features (train-ranked)")
    ax.set_ylabel("Held-out harmful refusal-score delta")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_title("SAE feature-budget frontier", loc="left", fontweight="bold", pad=32)
    fig.text(
        0.13,
        0.845,
        "Qwen2.5-1.5B-Instruct · frozen test split · n=250 harmful prompts · lower is stronger suppression",
        fontsize=9,
        color="#56616B",
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(
        0.97,
        0.04,
        "Source: results/extensions/paper-hardening/qwen15b.json · post-hoc sensitivity analysis",
        ha="right",
        fontsize=7.5,
        color="#6B747C",
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def comparability(data: dict, output: Path) -> None:
    display = {
        "mean_direction_ablation": ("Mean direction", "o", BLUE_DARK),
        "linear_probe_ablation": ("Linear probe", "s", BLUE),
        "selected_sae_residual_ablation": ("SAE top-5 residual", "^", BLUE),
        "matched_random_sae_residual_ablation": ("Same-dict. random", "D", GREY),
        "sae_reconstruction_only": ("SAE reconstruction only", "P", GOLD),
        "selected_sae_substitution": ("SAE top-5 substitution", "X", GOLD_DARK),
        "weighted_sparse_sae_ablation": ("Weighted SAE", "v", BLUE),
    }
    offsets = {
        "mean_direction_ablation": (6, 4),
        "linear_probe_ablation": (8, -7),
        "selected_sae_residual_ablation": (8, 10),
        "matched_random_sae_residual_ablation": (6, 5),
        "sae_reconstruction_only": (-145, -24),
        "selected_sae_substitution": (-150, 12),
        "weighted_sparse_sae_ablation": (6, -13),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.19)
    for name, row in data["intervention_comparability"].items():
        label, marker, color = display[name]
        x = row["hidden_l2_mean"]
        y = -row["delta_harmful_score_ci"]["estimate"]
        positive_nll = max(0.0, row["delta_capability_nll_ci"]["estimate"])
        size = 65 + 650 * min(positive_nll / 0.20, 1.0)
        ax.scatter(
            x,
            y,
            s=size,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        dx, dy = offsets[name]
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.2)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean hidden-state perturbation L2 (log scale)")
    ax.set_ylabel("Held-out harmful refusal suppression (−Δ score)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_title("Intervention comparability", loc="left", fontweight="bold", pad=32)
    fig.text(
        0.13,
        0.845,
        "Qwen2.5-1.5B-Instruct · n=250 harmful prompts · marker area encodes positive capability-NLL cost",
        fontsize=9,
        color="#56616B",
    )
    fig.text(
        0.97,
        0.04,
        "Source: results/extensions/paper-hardening/qwen15b.json · post-hoc sensitivity analysis",
        ha="right",
        fontsize=7.5,
        color="#6B747C",
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default="results/extensions/paper-hardening/qwen15b.json"
    )
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()
    data = json.loads(Path(args.result).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    feature_budget(data, output / "hardening_feature_budget")
    comparability(data, output / "hardening_intervention_comparability")


if __name__ == "__main__":
    main()
