from __future__ import annotations

import argparse
from pathlib import Path

from ..utils import read_json


def load_artifacts(path: str | Path) -> dict[str, object]:
    path = Path(path)
    required = [
        "manifest.json",
        "summary.json",
        "layer_scan.json",
        "features.json",
        "interventions.json",
        "sae_metrics.json",
    ]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing workbench artifacts: {', '.join(missing)}")
    optional = [
        "failure_analysis.json",
        "layerwise_controls.json",
        "method_comparisons.json",
        "protocol.json",
        "records.json",
        "component_localization.json",
    ]
    available = required + [name for name in optional if (path / name).exists()]
    return {name.removesuffix(".json"): read_json(path / name) for name in available}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts", default="artifacts/demo")
    args, _ = parser.parse_known_args()
    artifacts = load_artifacts(args.artifacts)

    import pandas as pd
    import streamlit as st

    summary = artifacts["summary"]
    interventions = artifacts["interventions"]
    st.set_page_config(page_title="Glassbox Workbench", layout="wide")
    st.title("Glassbox Refusal Audit")
    if "fixture" in summary["evidence_class"]:
        st.warning(
            f"{summary['evidence_class']}: useful for pipeline inspection, not a publication-grade finding."
        )
    st.caption(
        f"{summary['model']} · target layer {summary['target_layer']} · "
        f"selected features {summary['selected_features']}"
    )

    tab_names = [
        "Overview",
        "Layer scan",
        "Features",
        "Causal tests",
        "Failure analysis",
        "Prompt inspector",
    ]
    if "component_localization" in artifacts:
        tab_names.append("Components")
    tab_names.append("Provenance")
    tabs = st.tabs(tab_names)
    overview, layers, features, causal, failure, prompts = tabs[:6]
    next_tab = 6
    components_tab = None
    if "component_localization" in artifacts:
        components_tab = tabs[next_tab]
        next_tab += 1
    provenance = tabs[next_tab]
    with overview:
        metrics = summary["baseline_metrics"]
        columns = st.columns(4)
        columns[0].metric("Harmful refusal", f"{metrics['harmful_refusal_rate']:.1%}")
        columns[1].metric("Benign over-refusal", f"{metrics['benign_overrefusal_rate']:.1%}")
        columns[2].metric("Behavior gap", f"{metrics['behavior_gap']:.3f}")
        columns[3].metric("Capability perplexity", f"{metrics['capability_perplexity']:.3f}")
        st.subheader("Steering trade-off")
        sweep = pd.DataFrame(interventions["steering_sweeps"])
        sweep["harmful_score_delta"] = sweep["delta_mean_harmful_score_ci"].map(
            lambda value: value["estimate"]
        )
        sweep["capability_nll_delta"] = sweep["delta_capability_nll_ci"].map(
            lambda value: value["estimate"]
        )
        st.scatter_chart(
            sweep,
            x="harmful_score_delta",
            y="capability_nll_delta",
            color="method",
            size="scale",
        )
    with layers:
        frame = pd.DataFrame(artifacts["layer_scan"]).set_index("layer")
        st.line_chart(frame[["localization_score", "projection_effect_size", "mean_difference_norm"]])
        st.dataframe(frame, use_container_width=True)
        if "layerwise_controls" in artifacts:
            controls = pd.DataFrame(artifacts["layerwise_controls"])
            controls["harmful_score_delta"] = controls["delta_mean_harmful_score_ci"].map(
                lambda value: value["estimate"]
            )
            st.subheader("Held-out layerwise ablation controls")
            st.bar_chart(controls, x="layer", y="harmful_score_delta")
            st.dataframe(
                controls[["layer", "is_selected_layer", "harmful_score_delta"]],
                use_container_width=True,
                hide_index=True,
            )
    with features:
        table = pd.DataFrame(artifacts["features"])
        frequency = st.slider("Minimum activation frequency", 0.0, 1.0, 0.0, 0.01)
        filtered = table[table["activation_frequency"] >= frequency]
        st.bar_chart(filtered.head(30).set_index("feature")["effect_size"])
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    with causal:
        rows = []
        for name, result in interventions["causal_tests"].items():
            rows.append(
                {
                    "test": name,
                    "harmful_score_delta": result["delta_mean_harmful_score_ci"]["estimate"],
                    "harmful_ci_low": result["delta_mean_harmful_score_ci"]["low"],
                    "harmful_ci_high": result["delta_mean_harmful_score_ci"]["high"],
                    "benign_score_delta": result["delta_mean_benign_score_ci"]["estimate"],
                    "capability_nll_delta": result["delta_capability_nll_ci"]["estimate"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("Random control distribution")
        controls = pd.DataFrame(interventions["random_controls"])
        controls["harmful_score_delta"] = controls["delta_mean_harmful_score_ci"].map(
            lambda value: value["estimate"]
        )
        st.bar_chart(controls, x="control_index", y="harmful_score_delta", color="method")
        if "method_comparisons" in artifacts:
            st.subheader("Direct paired method comparisons")
            st.json(artifacts["method_comparisons"])
    with failure:
        if "failure_analysis" not in artifacts:
            st.info("This run predates generated failure-analysis artifacts.")
        else:
            analysis = artifacts["failure_analysis"]
            diagnostics = analysis["sae_diagnostics"]
            columns = st.columns(4)
            columns[0].metric("Dead features", f"{diagnostics['dead_feature_fraction']:.1%}")
            columns[1].metric("Feature density", f"{diagnostics['feature_density']:.3%}")
            columns[2].metric("Train samples", f"{diagnostics['training_samples']:,}")
            columns[3].metric("Samples / feature", f"{diagnostics['samples_per_feature']:.2f}")
            st.subheader("Negative findings")
            for finding in analysis["negative_findings"]:
                st.markdown(f"- {finding}")
            st.subheader("Intervention stability")
            st.dataframe(
                pd.DataFrame(analysis["intervention_stability"]).T,
                use_container_width=True,
            )
            st.subheader("Category-level failure analysis")
            st.dataframe(
                pd.DataFrame(analysis["category_analysis"]).T,
                use_container_width=True,
            )
    with prompts:
        sweep_options = [
            f"{row['method']} @ {row['scale']:+g}" for row in interventions["steering_sweeps"]
        ]
        selected = st.selectbox("Intervention", range(len(sweep_options)), format_func=sweep_options.__getitem__)
        baseline = pd.DataFrame(interventions["baseline_outputs"]).rename(
            columns={"behavior_score": "baseline_score", "response": "baseline_response"}
        )
        changed = pd.DataFrame(interventions["steering_sweeps"][selected]["outputs"]).rename(
            columns={"behavior_score": "intervened_score", "response": "intervened_response"}
        )
        merged = baseline.merge(changed, on="record_id", suffixes=("_base", "_changed"))
        merged["score_delta"] = merged["intervened_score"] - merged["baseline_score"]
        if "records" in artifacts:
            record_frame = pd.DataFrame(artifacts["records"]).rename(columns={"id": "record_id"})
            merged = record_frame.merge(merged, on="record_id")
        st.dataframe(merged, use_container_width=True, hide_index=True)
    if components_tab is not None:
        with components_tab:
            component_artifact = artifacts["component_localization"]
            st.caption(component_artifact["evidence_scope"])
            rows = []
            for row in component_artifact["components"]:
                item = {
                    "component": row["component"],
                    "status": row["status"],
                    "direction": row["direction"],
                    "layer": row["layer"],
                }
                if row["status"] == "completed":
                    item["harmful_score_delta"] = row["delta_mean_harmful_score_ci"]["estimate"]
                    item["benign_score_delta"] = row["delta_mean_benign_score_ci"]["estimate"]
                    item["capability_nll_delta"] = row["delta_capability_nll_ci"]["estimate"]
                else:
                    item["error"] = row.get("error", "")
                rows.append(item)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with provenance:
        st.json(artifacts["manifest"])
        if "protocol" in artifacts:
            st.markdown("### Held-out protocol")
            st.json(artifacts["protocol"])
        st.markdown("### SAE training")
        st.json(artifacts["sae_metrics"])


if __name__ == "__main__":
    main()
