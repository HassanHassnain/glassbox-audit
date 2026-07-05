from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

from .stats import (
    grouped_bootstrap_ci,
    mean,
    normal_approx_power,
    one_sample_mean_test,
    paired_delta_ci,
    required_n_for_power,
    sample_standard_deviation,
    tost_equivalence_test,
)
from .types import PromptRecord
from .utils import read_json, write_json


def _estimate(row: dict[str, object], key: str) -> float:
    return float(row[key]["estimate"])


def _method_utility(row: dict[str, object]) -> float:
    harmful_reduction = -_estimate(row, "delta_mean_harmful_score_ci")
    benign_cost = abs(_estimate(row, "delta_mean_benign_score_ci"))
    capability_cost = max(0.0, _estimate(row, "delta_capability_nll_ci"))
    return harmful_reduction - benign_cost - 0.2 * capability_cost


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 1e-12 else 0.0


def paired_method_comparisons(
    records: list[PromptRecord],
    rows: dict[str, dict[str, object]],
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, object]]:
    results = []
    pair_by_id = {record.id: record.pair_id for record in records}
    harmful_ids = [record.id for record in records if record.harmful]
    benign_ids = [record.id for record in records if not record.harmful]
    all_ids = [record.id for record in records]
    for pair_index, (left_name, right_name) in enumerate(combinations(sorted(rows), 2)):
        left = {row["record_id"]: row for row in rows[left_name]["outputs"]}
        right = {row["record_id"]: row for row in rows[right_name]["outputs"]}
        results.append(
            {
                "left_method": left_name,
                "right_method": right_name,
                "interpretation": "Negative score deltas mean the left method suppresses more.",
                "harmful_score_left_minus_right_ci": paired_delta_ci(
                    [right[row_id]["behavior_score"] for row_id in harmful_ids],
                    [left[row_id]["behavior_score"] for row_id in harmful_ids],
                    bootstrap_samples,
                    confidence,
                    seed + pair_index * 3,
                    group_ids=[pair_by_id[row_id] for row_id in harmful_ids],
                ),
                "benign_score_left_minus_right_ci": paired_delta_ci(
                    [right[row_id]["behavior_score"] for row_id in benign_ids],
                    [left[row_id]["behavior_score"] for row_id in benign_ids],
                    bootstrap_samples,
                    confidence,
                    seed + pair_index * 3 + 1,
                    group_ids=[pair_by_id[row_id] for row_id in benign_ids],
                ),
                "capability_nll_left_minus_right_ci": paired_delta_ci(
                    [right[row_id]["capability_nll"] for row_id in all_ids],
                    [left[row_id]["capability_nll"] for row_id in all_ids],
                    bootstrap_samples,
                    confidence,
                    seed + pair_index * 3 + 2,
                    group_ids=[pair_by_id[row_id] for row_id in all_ids],
                ),
            }
        )
    return results


def _method_output_deltas(
    records: list[PromptRecord],
    left: dict[str, object],
    right: dict[str, object],
    *,
    harmful_only: bool,
    output_key: str,
) -> tuple[list[float], list[str]]:
    ids = [
        record.id
        for record in records
        if not harmful_only or record.harmful
    ]
    pair_by_id = {record.id: record.pair_id for record in records}
    left_outputs = {row["record_id"]: row for row in left["outputs"]}
    right_outputs = {row["record_id"]: row for row in right["outputs"]}
    return (
        [
            float(left_outputs[row_id][output_key]) - float(right_outputs[row_id][output_key])
            for row_id in ids
        ],
        [pair_by_id[row_id] for row_id in ids],
    )


def _equivalence_row(
    *,
    name: str,
    comparator: str,
    deltas: list[float],
    group_ids: list[str],
    margin: float,
    alpha: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    test = tost_equivalence_test(
        deltas,
        -margin,
        margin,
        alpha=alpha,
        samples=bootstrap_samples,
        seed=seed,
        group_ids=group_ids,
    )
    return {
        "test": name,
        "comparison": f"sae_feature_ablation_minus_{comparator}",
        "metric": "harmful_behavior_score",
        "n_prompt_pairs": len(set(group_ids)),
        "delta_interpretation": (
            "Positive means SAE leaves a higher harmful refusal score than the comparator "
            "after intervention, so SAE suppresses less."
        ),
        "margin": margin,
        "standard_deviation": sample_standard_deviation(deltas),
        "tost": test,
        "passed": bool(test["equivalent"]),
    }


def _inferiority_row(
    *,
    name: str,
    comparator: str,
    deltas: list[float],
    group_ids: list[str],
    margin: float,
    alpha: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    test = one_sample_mean_test(deltas, margin, alternative="greater")
    ci = grouped_bootstrap_ci(
        deltas,
        group_ids,
        samples=bootstrap_samples,
        confidence=0.95,
        seed=seed,
    )
    passed = float(test["p_value"]) <= alpha and float(ci["low"]) > margin
    return {
        "test": name,
        "comparison": f"sae_feature_ablation_minus_{comparator}",
        "metric": "harmful_behavior_score",
        "n_prompt_pairs": len(set(group_ids)),
        "delta_interpretation": (
            "Positive means SAE leaves a higher harmful refusal score than the comparator "
            "after intervention, so SAE suppresses less."
        ),
        "margin": margin,
        "ci": ci,
        "one_sided_test": test,
        "passed": passed,
    }


def build_equivalence_analysis(
    records: list[PromptRecord],
    causal_rows: dict[str, dict[str, object]],
    *,
    bootstrap_samples: int,
    equivalence_margin: float,
    inferiority_margin: float,
    alpha: float,
    power_effect_size: float,
    power_assumed_pair_sd: float,
    power_target: float,
    power_target_n: int,
    seed: int,
) -> dict[str, object]:
    sae = causal_rows["sae_feature_ablation"]
    comparators = {
        "mean_difference_direction_ablation": causal_rows["mean_difference_direction_ablation"],
        "linear_probe_direction_ablation": causal_rows["linear_probe_direction_ablation"],
    }
    tests = []
    for index, (comparator_name, comparator_row) in enumerate(comparators.items()):
        deltas, group_ids = _method_output_deltas(
            records,
            sae,
            comparator_row,
            harmful_only=True,
            output_key="behavior_score",
        )
        short_name = comparator_name.replace("_direction_ablation", "")
        tests.append(
            _equivalence_row(
                name=f"sae_equivalent_to_{short_name}",
                comparator=comparator_name,
                deltas=deltas,
                group_ids=group_ids,
                margin=equivalence_margin,
                alpha=alpha,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index * 10,
            )
        )
        tests.append(
            _inferiority_row(
                name=f"sae_inferior_to_{short_name}",
                comparator=comparator_name,
                deltas=deltas,
                group_ids=group_ids,
                margin=inferiority_margin,
                alpha=alpha,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index * 10 + 1,
            )
        )

    current_pairs = len({record.pair_id for record in records})
    power = {
        "assumptions": {
            "minimum_effect_size": power_effect_size,
            "assumed_pair_delta_standard_deviation": power_assumed_pair_sd,
            "alpha": alpha,
            "target_power": power_target,
            "target_n_prompt_pairs": power_target_n,
            "effect_units": "harmful_behavior_score_delta",
        },
        "primary_one_sided": {
            "alternative": "greater",
            "required_n_prompt_pairs": required_n_for_power(
                effect=power_effect_size,
                standard_deviation=power_assumed_pair_sd,
                target_power=power_target,
                alpha=alpha,
                alternative="greater",
            ),
            "power_at_target_n": normal_approx_power(
                n=power_target_n,
                effect=power_effect_size,
                standard_deviation=power_assumed_pair_sd,
                alpha=alpha,
                alternative="greater",
            ),
            "power_at_current_n": normal_approx_power(
                n=current_pairs,
                effect=power_effect_size,
                standard_deviation=power_assumed_pair_sd,
                alpha=alpha,
                alternative="greater",
            ),
        },
        "two_sided_sensitivity": {
            "alternative": "two-sided",
            "required_n_prompt_pairs": required_n_for_power(
                effect=power_effect_size,
                standard_deviation=power_assumed_pair_sd,
                target_power=power_target,
                alpha=alpha,
                alternative="two-sided",
            ),
            "power_at_target_n": normal_approx_power(
                n=power_target_n,
                effect=power_effect_size,
                standard_deviation=power_assumed_pair_sd,
                alpha=alpha,
                alternative="two-sided",
            ),
            "power_at_current_n": normal_approx_power(
                n=current_pairs,
                effect=power_effect_size,
                standard_deviation=power_assumed_pair_sd,
                alpha=alpha,
                alternative="two-sided",
            ),
        },
    }

    return {
        "schema_version": "1.0",
        "analysis": "A6 equivalence tests and paired-power analysis",
        "bootstrap_unit": "pair_id",
        "test_family": "held_out_causal_ablation",
        "alpha": alpha,
        "preregistered_bounds": {
            "equivalence_margin": equivalence_margin,
            "inferiority_margin": inferiority_margin,
        },
        "tests": tests,
        "power_analysis": power,
        "interpretation": (
            "Equivalence tests use TOST on SAE-minus-baseline harmful scores. Inferiority tests "
            "ask whether SAE is worse by at least the preregistered margin. Lower harmful scores "
            "after intervention are stronger suppression, so positive SAE-minus-baseline deltas "
            "favor the simple baseline."
        ),
    }


def build_failure_analysis(
    sae_metrics: dict[str, float],
    features: list[dict[str, float | int]],
    baseline_outputs: list[dict[str, object]],
    validation_sweeps: list[dict[str, object]],
    test_sweeps: list[dict[str, object]],
    selected_scales: dict[str, float],
    causal_tests: dict[str, dict[str, object]],
    random_controls: list[dict[str, object]],
    records: list[PromptRecord],
) -> dict[str, object]:
    selected_test = {
        method: next(
            row for row in test_sweeps if row["method"] == method and row["scale"] == scale
        )
        for method, scale in selected_scales.items()
    }
    selected_validation = {
        method: next(
            row for row in validation_sweeps if row["method"] == method and row["scale"] == scale
        )
        for method, scale in selected_scales.items()
    }
    baseline = {row["record_id"]: row for row in baseline_outputs}
    harmful_ids = [record.id for record in records if record.harmful]
    stability = {}
    for method, selected in selected_test.items():
        outputs = {row["record_id"]: row for row in selected["outputs"]}
        scale_rows = sorted(
            [row for row in test_sweeps if row["method"] == method], key=lambda row: row["scale"]
        )
        stability[method] = {
            "selected_scale": selected_scales[method],
            "harmful_prompt_sign_consistency": sum(
                outputs[row_id]["behavior_score"] < baseline[row_id]["behavior_score"]
                for row_id in harmful_ids
            )
            / len(harmful_ids),
            "validation_to_test_harmful_effect_gap": _estimate(
                selected, "delta_mean_harmful_score_ci"
            )
            - _estimate(selected_validation[method], "delta_mean_harmful_score_ci"),
            "scale_effect_correlation": _pearson(
                [float(row["scale"]) for row in scale_rows],
                [_estimate(row, "delta_mean_harmful_score_ci") for row in scale_rows],
            ),
            "specificity_adjusted_utility": _method_utility(selected),
        }

    sae_utility = stability["sae_features"]["specificity_adjusted_utility"]
    simple_utilities = {
        method: stability[method]["specificity_adjusted_utility"]
        for method in ["mean_difference", "linear_probe"]
    }
    best_simple_method = max(simple_utilities, key=simple_utilities.get)
    feature_ablation = causal_tests["sae_feature_ablation"]
    simple_ablation_effects = {
        method: _estimate(causal_tests[f"{method}_direction_ablation"], "delta_mean_harmful_score_ci")
        for method in ["mean_difference", "linear_probe"]
    }
    best_simple_ablation = min(simple_ablation_effects, key=simple_ablation_effects.get)
    controls = [
        _estimate(row, "delta_mean_harmful_score_ci")
        for row in random_controls
        if row["intervention"] in {"ablation_control", "feature_ablation_control"}
    ]
    feature_ablation_effect = _estimate(feature_ablation, "delta_mean_harmful_score_ci")
    empirical_p = (
        (1 + sum(value <= feature_ablation_effect for value in controls)) / (len(controls) + 1)
        if controls
        else None
    )

    negatives = []
    if sae_metrics["dead_feature_fraction"] > 0.5:
        negatives.append(
            f"SAE dead-feature fraction is high ({sae_metrics['dead_feature_fraction']:.1%})."
        )
    if sae_utility <= simple_utilities[best_simple_method]:
        negatives.append(
            f"SAE steering does not beat the best simple baseline ({best_simple_method}) "
            "on specificity-adjusted utility."
        )
    if float(feature_ablation["delta_mean_harmful_score_ci"]["high"]) >= 0:
        negatives.append("SAE feature-ablation harmful-score interval includes zero.")
    if feature_ablation_effect >= simple_ablation_effects[best_simple_ablation]:
        negatives.append(
            f"SAE feature ablation is weaker than {best_simple_ablation} direction ablation "
            "on harmful-score suppression."
        )
    if _estimate(causal_tests["sae_activation_patch"], "delta_mean_harmful_score_ci") >= 0:
        negatives.append("SAE activation patching does not reduce harmful refusal score.")
    if _estimate(causal_tests["sae_activation_patch"], "delta_capability_nll_ci") > 0.05:
        negatives.append("SAE activation patching has material capability-NLL cost.")
    if empirical_p is not None and empirical_p > 0.05:
        negatives.append(
            "SAE feature ablation does not clear the configured empirical random-control "
            f"threshold (one-sided p={empirical_p:.3f})."
        )
    if sae_metrics["samples_per_feature"] < 1:
        negatives.append("SAE has fewer training activation samples than dictionary features.")
    if not negatives:
        negatives.append("No configured automatic negative-result trigger fired.")

    feature_outputs = {
        row["record_id"]: row for row in causal_tests["sae_feature_ablation"]["outputs"]
    }
    mean_outputs = {
        row["record_id"]: row for row in causal_tests["mean_difference_direction_ablation"]["outputs"]
    }
    categories = {}
    for category in sorted({record.category for record in records}):
        category_records = [record for record in records if record.category == category]
        harmful_ids_for_category = [record.id for record in category_records if record.harmful]
        benign_ids_for_category = [record.id for record in category_records if not record.harmful]
        categories[category] = {
            "n_records": len(category_records),
            "baseline_behavior_gap": mean(
                [baseline[row_id]["behavior_score"] for row_id in harmful_ids_for_category]
            )
            - mean([baseline[row_id]["behavior_score"] for row_id in benign_ids_for_category]),
            "sae_feature_ablation_harmful_delta": mean(
                [
                    feature_outputs[row_id]["behavior_score"] - baseline[row_id]["behavior_score"]
                    for row_id in harmful_ids_for_category
                ]
            ),
            "mean_difference_ablation_harmful_delta": mean(
                [
                    mean_outputs[row_id]["behavior_score"] - baseline[row_id]["behavior_score"]
                    for row_id in harmful_ids_for_category
                ]
            ),
        }

    return {
        "sae_diagnostics": {
            **sae_metrics,
            "mean_prompt_feature_frequency": mean(
                [float(row["activation_frequency"]) for row in features]
            ),
            "max_prompt_feature_frequency": max(
                float(row["activation_frequency"]) for row in features
            ),
        },
        "intervention_stability": stability,
        "baseline_comparison": {
            "sae_specificity_adjusted_utility": sae_utility,
            "best_simple_method": best_simple_method,
            "best_simple_specificity_adjusted_utility": simple_utilities[best_simple_method],
            "sae_outperforms_simple_baselines": sae_utility > simple_utilities[best_simple_method],
        },
        "random_ablation_control_comparison": {
            "sae_feature_ablation_harmful_effect": feature_ablation_effect,
            "n_controls": len(controls),
            "empirical_one_sided_p": empirical_p,
            "control_effect_min": min(controls) if controls else None,
            "control_effect_max": max(controls) if controls else None,
        },
        "category_analysis": categories,
        "negative_findings": negatives,
    }


def refresh_failure_analysis(artifact_dir: str | Path) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    interventions = read_json(artifact_dir / "interventions.json")
    existing_failure = (
        read_json(artifact_dir / "failure_analysis.json")
        if (artifact_dir / "failure_analysis.json").exists()
        else {}
    )
    records = [
        PromptRecord.from_dict(row)
        for row in read_json(artifact_dir / "records.json")
        if row["split"] == "test"
    ]
    failure = build_failure_analysis(
        read_json(artifact_dir / "sae_metrics.json")["metrics"],
        read_json(artifact_dir / "features.json"),
        interventions["baseline_outputs"],
        interventions["validation_sweeps"],
        interventions["steering_sweeps"],
        interventions["validation_selected_scales"],
        interventions["causal_tests"],
        interventions["random_controls"],
        records,
    )
    if "preregistration" in existing_failure:
        failure["preregistration"] = existing_failure["preregistration"]
        failure["preregistration_hash"] = existing_failure.get("preregistration_hash")
    write_json(artifact_dir / "failure_analysis.json", failure)
    summary = read_json(artifact_dir / "summary.json")
    summary["sae_outperforms_simple_baselines"] = failure["baseline_comparison"][
        "sae_outperforms_simple_baselines"
    ]
    summary["negative_findings"] = failure["negative_findings"]
    write_json(artifact_dir / "summary.json", summary)
    return failure
