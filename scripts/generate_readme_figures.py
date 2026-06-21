from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"

INK = "#172033"
MUTED = "#5E6E87"
GRID = "#DCE5EF"
BLUE = "#275DF1"
TEAL = "#0E9F98"
PURPLE = "#6750D8"
CORAL = "#D94D3D"
GOLD = "#B7791F"
SLATE = "#6B7A90"

METHOD_COLORS = {
    "Mean direction": BLUE,
    "SAE features": TEAL,
    "Linear probe": PURPLE,
}


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C8D3E0",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=240, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def method_label(method: str) -> str:
    return {
        "mean_difference_direction_ablation": "Mean direction",
        "mean_difference": "Mean direction",
        "sae_feature_ablation": "SAE features",
        "sae_features": "SAE features",
        "linear_probe_direction_ablation": "Linear probe",
        "linear_probe": "Linear probe",
    }.get(method, method.replace("_", " ").title())


def harmful_tests(family: str) -> list[dict]:
    stats = read_json("results/final/statistical_tests.json")
    tests = stats[family][0]["tests"]
    wanted = {
        "mean_difference_direction_ablation",
        "sae_feature_ablation",
        "linear_probe_direction_ablation",
    }
    rows = [row for row in tests if row["metric"] == "harmful_score" and row["method"] in wanted]
    order = {
        "mean_difference_direction_ablation": 0,
        "sae_feature_ablation": 1,
        "linear_probe_direction_ablation": 2,
    }
    return sorted(rows, key=lambda row: order[row["method"]])


def add_round_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = "#D4DEE9",
    radius: float = 0.03,
    linewidth: float = 1.2,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    return box


def axis_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    color: str = INK,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
) -> None:
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, transform=ax.transAxes)


def draw_stage(ax: plt.Axes, index: int, x: float, title: str, subtitle: str) -> None:
    ax.add_patch(Circle((x, 0.64), 0.035, facecolor="#EDF3FF", edgecolor=BLUE, linewidth=1.8, transform=ax.transAxes))
    axis_text(ax, x, 0.642, str(index), size=15, color=BLUE, weight="bold", ha="center")
    axis_text(ax, x, 0.565, title, size=12.2, color=INK, weight="bold", ha="center")
    axis_text(ax, x, 0.528, subtitle, size=9.8, color=MUTED, ha="center")


def draw_lane(
    ax: plt.Axes,
    *,
    y: float,
    label: str,
    color: str,
    stops: list[float],
    outcome: str,
    dashed_after: int | None = None,
) -> None:
    axis_text(ax, 0.18, y, label, size=13.8, color=color, weight="bold", ha="right")
    for i in range(len(stops) - 1):
        style = "--" if dashed_after is not None and i >= dashed_after else "-"
        arrow = FancyArrowPatch(
            (stops[i] + 0.03, y),
            (stops[i + 1] - 0.03, y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=2.2,
            linestyle=style,
            color=color,
            alpha=0.92 if style == "-" else 0.55,
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
    for i, x in enumerate(stops):
        ax.scatter([x], [y], s=210, color=color, edgecolor="white", linewidth=1.5, transform=ax.transAxes, zorder=4)
    axis_text(ax, stops[-1] + 0.04, y, outcome, size=12.2, color=color, weight="bold")


def blend(hex_a: str, hex_b: str, t: float) -> str:
    a = tuple(int(hex_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i : i + 2], 16) for i in (1, 3, 5))
    c = tuple(round(a[i] * (1 - t) + b[i] * t) for i in range(3))
    return "#" + "".join(f"{v:02x}" for v in c)


def generate_overview() -> None:
    audit = read_json("results/final/qwen2_5_1_5b_audit.json")
    component = read_json("results/component-path/component_path_summary.json")["artifacts"][0]
    residual_by_layer = {
        row["layer"]: abs(row["harmful_delta"])
        for row in component["component_outputs"]
        if row["component"] == "residual"
    }

    fig = plt.figure(figsize=(16, 7.0), constrained_layout=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    axis_text(ax, 0.055, 0.91, "GLASSBOX", size=17, color=BLUE, weight="bold")
    axis_text(ax, 0.055, 0.825, "Layer-27 residual stream audit", size=34, color=INK, weight="bold")
    axis_text(
        ax,
        0.055,
        0.762,
        "The supported result is a late residual direction; the unsupported result is SAE superiority or a full circuit claim.",
        size=16.2,
        color=MUTED,
    )

    add_round_box(ax, (0.055, 0.455), 0.89, 0.245, facecolor="#FBFCFE")
    axis_text(ax, 0.085, 0.666, "Qwen2.5-1.5B transformer layers", size=13.5, color=INK, weight="bold")
    axis_text(ax, 0.085, 0.632, "late residual stream selected before held-out testing", size=11.0, color=MUTED)

    x0, x1 = 0.085, 0.735
    y = 0.545
    n_layers = 28
    block_gap = 0.004
    block_w = (x1 - x0 - block_gap * (n_layers - 1)) / n_layers
    max_residual = max(residual_by_layer.values())
    for layer in range(n_layers):
        value = residual_by_layer.get(layer, 0.004 + 0.002 * max(0, layer - 20))
        intensity = min(1.0, value / max_residual) if layer in residual_by_layer else min(0.20, value / max_residual)
        fill = blend("#ECF2FA", BLUE, intensity * 0.82)
        x = x0 + layer * (block_w + block_gap)
        edge = BLUE if layer == audit["target_layer"] else "#D3DEEA"
        lw = 2.6 if layer == audit["target_layer"] else 0.7
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                block_w,
                0.095,
                boxstyle="round,pad=0.003,rounding_size=0.008",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                transform=ax.transAxes,
            )
        )
        if layer in {0, 4, 8, 12, 16, 20, 24, 27}:
            axis_text(ax, x + block_w / 2, y - 0.035, str(layer), size=9.5, color=MUTED, ha="center")

    layer = audit["target_layer"]
    target_x = x0 + layer * (block_w + block_gap) + block_w / 2
    ax.add_patch(
        FancyArrowPatch(
            (target_x, 0.715),
            (target_x, 0.652),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=BLUE,
            transform=ax.transAxes,
        )
    )
    axis_text(ax, target_x, 0.735, "selected layer 27", size=12.5, color=BLUE, weight="bold", ha="center")

    add_round_box(ax, (0.765, 0.455), 0.18, 0.245, facecolor="white")
    axis_text(ax, 0.79, 0.658, "Claim boundary", size=15.0, color=INK, weight="bold")
    boundary_rows = [
        (BLUE, "supported", "residual direction"),
        (CORAL, "failed", "SAE > mean/probe"),
        (GOLD, "not claimed", "full circuit"),
    ]
    for i, (color, label, text) in enumerate(boundary_rows):
        row_y = 0.608 - i * 0.06
        ax.scatter([0.792], [row_y], s=92, color=color, transform=ax.transAxes)
        axis_text(ax, 0.807, row_y + 0.012, label, size=10.5, color=color, weight="bold")
        axis_text(ax, 0.807, row_y - 0.015, text, size=9.5, color=MUTED)

    add_round_box(ax, (0.055, 0.125), 0.89, 0.245, facecolor="white")
    axis_text(ax, 0.085, 0.327, "Held-out harmful-score suppression", size=14.5, color=INK, weight="bold")
    axis_text(ax, 0.085, 0.296, "lower is stronger; values are committed final summaries", size=10.5, color=MUTED)
    methods = [
        ("Mean direction", audit["mean_harmful_delta"], BLUE),
        ("SAE features", audit["sae_harmful_delta"], TEAL),
        ("Linear probe", audit["probe_harmful_delta"], PURPLE),
    ]
    bar_left, bar_zero = 0.345, 0.83
    domain_min = -0.15
    for i, (name, value, color) in enumerate(methods):
        row_y = 0.262 - i * 0.050
        axis_text(ax, 0.085, row_y, name, size=12.2, color=INK, weight="bold")
        width = abs(value / domain_min) * (bar_zero - bar_left)
        ax.plot([bar_zero - width, bar_zero], [row_y, row_y], color=color, linewidth=9, solid_capstyle="round", transform=ax.transAxes)
        ax.scatter([bar_zero - width], [row_y], s=120, color=color, edgecolor="white", linewidth=1.2, transform=ax.transAxes, zorder=3)
        axis_text(ax, 0.855, row_y, f"{value:.3f}", size=12.2, color=color, weight="bold")
    ax.plot([bar_zero, bar_zero], [0.142, 0.30], color="#B7C4D3", linewidth=1.2, transform=ax.transAxes)
    axis_text(ax, bar_zero, 0.102, "0", size=9.5, color=MUTED, ha="center")
    axis_text(ax, bar_left, 0.102, "-0.15", size=9.5, color=MUTED, ha="center")

    save(fig, "readme_audit_overview.png")

def generate_causal_effects() -> None:
    fig = plt.figure(figsize=(16, 6.2), constrained_layout=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    axis_text(ax, 0.055, 0.90, "CAUSAL EFFECTS", size=17, color=BLUE, weight="bold")
    axis_text(ax, 0.055, 0.81, "Simple residual directions dominate SAE features", size=31, color=INK, weight="bold")
    axis_text(
        ax,
        0.055,
        0.742,
        "Mean harmful-score deltas with bootstrap intervals. Lower is stronger refusal suppression.",
        size=15.8,
        color=MUTED,
    )

    panels = [
        ("Held-out Qwen2.5-1.5B audit", harmful_tests("internal"), 0.055, 0.500),
        ("OR-Bench external transfer", harmful_tests("external_causal"), 0.525, 0.500),
    ]
    domain_min, domain_max = -0.18, 0.01
    ticks = [-0.18, -0.12, -0.06, 0.0]
    row_offsets = [0.470, 0.360, 0.250]

    def map_x(value: float, left: float, width: float) -> float:
        return left + (value - domain_min) / (domain_max - domain_min) * width

    for title, rows, panel_x, panel_w in panels:
        add_round_box(ax, (panel_x, 0.145), panel_w - 0.025, 0.515, facecolor="#FBFCFE")
        axis_text(ax, panel_x + 0.035, 0.610, title, size=18.5, color=INK, weight="bold")
        plot_left = panel_x + 0.185
        plot_w = panel_w - 0.285
        zero_x = map_x(0.0, plot_left, plot_w)
        ax.plot([zero_x, zero_x], [0.215, 0.535], color="#AAB8CA", linewidth=1.2, transform=ax.transAxes)
        for tick in ticks:
            x = map_x(tick, plot_left, plot_w)
            ax.plot([x, x], [0.215, 0.535], color=GRID, linewidth=1.0, transform=ax.transAxes)
            axis_text(ax, x, 0.178, f"{tick:.2f}" if tick else "0", size=9.8, color=MUTED, ha="center")
        for row_y, row in zip(row_offsets, rows, strict=True):
            label = method_label(row["method"])
            color = METHOD_COLORS[label]
            low, high = row["ci"]
            value = row["delta"]
            low_x = map_x(low, plot_left, plot_w)
            high_x = map_x(high, plot_left, plot_w)
            value_x = map_x(value, plot_left, plot_w)
            axis_text(ax, panel_x + 0.035, row_y, label, size=13.2, color=INK, weight="bold")
            ax.plot(
                [low_x, high_x],
                [row_y, row_y],
                color=color,
                linewidth=5.5,
                solid_capstyle="round",
                transform=ax.transAxes,
            )
            ax.scatter(
                [value_x],
                [row_y],
                s=115,
                color=color,
                edgecolor="white",
                linewidth=1.2,
                transform=ax.transAxes,
                zorder=4,
            )
            axis_text(ax, panel_x + panel_w - 0.065, row_y, f"{value:.3f}", size=12.5, color=color, weight="bold", ha="right")
        axis_text(
            ax,
            plot_left + plot_w / 2,
            0.105,
            "harmful-score delta",
            size=11.2,
            color=MUTED,
            ha="center",
        )
    axis_text(
        ax,
        0.055,
        0.045,
        "Interpretation: SAE features are causal, but the simple mean-difference residual direction is much stronger.",
        size=12.0,
        color=SLATE,
    )
    save(fig, "readme_causal_effects.png")


def generate_stability_path() -> None:
    stability = read_json("results/sae-stability/stability_grid.json")
    component = read_json("results/component-path/component_path_summary.json")["artifacts"][0]
    cells = stability["cells"]

    fig = plt.figure(figsize=(16, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.12], left=0.07, right=0.98, top=0.75, bottom=0.14, wspace=0.30)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    labels = [f"seed {cell['seed']} / width x{cell['width']}" for cell in cells]
    y = list(range(len(cells)))[::-1]
    sae = [cell["sae_harmful_delta"] for cell in cells]
    mean_ref = sum(cell["mean_harmful_delta"] for cell in cells) / len(cells)

    ax_left.axvline(0, color="#B6C2D0", linewidth=1.5)
    ax_left.axvline(mean_ref, color=BLUE, linewidth=2.2, linestyle="--")
    for yi, value, cell in zip(y, sae, cells, strict=True):
        low, high = cell["sae_harmful_ci"]
        ax_left.plot([low, high], [yi, yi], color=TEAL, linewidth=4.5, solid_capstyle="round")
        ax_left.scatter(value, yi, s=110, color=TEAL, edgecolor="white", linewidth=1.2, zorder=3)
        ax_left.text(0.004, yi, f"{value:.3f}", va="center", ha="left", fontsize=12, color=MUTED)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels, fontsize=13, color=INK)
    ax_left.set_xlim(-0.17, 0.02)
    ax_left.set_xticks([-0.15, -0.10, -0.05, 0.0])
    ax_left.set_xlabel("SAE harmful-score delta", fontsize=13)
    ax_left.set_title("SAE stability grid: 6/6 complete, 0/6 H3 passes", loc="left", fontsize=19, fontweight="bold")
    ax_left.text(mean_ref - 0.002, -0.85, "mean-direction reference", ha="right", fontsize=11.5, color=BLUE)
    ax_left.grid(axis="x")
    ax_left.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax_left.spines[spine].set_visible(False)

    rows = component["component_outputs"]
    layers = component["layers"]
    components = ["residual", "attention", "mlp"]
    matrix = [
        [next(row for row in rows if row["layer"] == layer and row["component"] == comp)["harmful_delta"] for comp in components]
        for layer in layers
    ]
    delta_cmap = LinearSegmentedColormap.from_list("glassbox_delta", [BLUE, "#F8FAFC", CORAL])
    sns.heatmap(
        matrix,
        ax=ax_right,
        cmap=delta_cmap,
        center=0,
        vmin=-0.15,
        vmax=0.05,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 13, "color": INK},
        linewidths=3,
        linecolor="white",
        cbar_kws={"label": "harmful-score delta", "shrink": 0.78},
    )
    ax_right.set_xticklabels(["Residual", "Attention", "MLP"], rotation=0, fontsize=13, color=INK)
    ax_right.set_yticklabels([f"Layer {layer}" for layer in layers], rotation=0, fontsize=13, color=INK)
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")
    ax_right.set_title("Component/path analysis: residual stream dominates", loc="left", fontsize=19, fontweight="bold")
    ax_right.tick_params(length=0)

    fig.suptitle(
        "Stability and path evidence keep the claim bounded",
        fontsize=27,
        fontweight="bold",
        color=INK,
        x=0.055,
        ha="left",
        y=0.965,
    )
    fig.text(
        0.055,
        0.875,
        "Stable SAE effects are not enough for SAE superiority; component analysis supports a residual-stream direction, not a full circuit.",
        fontsize=16,
        color=MUTED,
    )
    save(fig, "readme_stability_path.png")


def main() -> None:
    configure_style()
    generate_overview()
    generate_causal_effects()
    generate_stability_path()


if __name__ == "__main__":
    main()
