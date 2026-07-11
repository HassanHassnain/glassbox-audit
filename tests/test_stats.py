import pytest

from glassbox_audit.analysis import build_equivalence_analysis
from glassbox_audit.evaluation import select_threshold
from glassbox_audit.stats import (
    bootstrap_ci,
    grouped_bootstrap_ci,
    normal_approx_power,
    one_sample_mean_test,
    paired_delta_ci,
    required_n_for_power,
    tost_equivalence_test,
)
from glassbox_audit.types import ModelOutput, PromptRecord


def test_bootstrap_is_deterministic_and_contains_estimate():
    first = bootstrap_ci([1, 2, 3, 4], samples=100, seed=7)
    second = bootstrap_ci([1, 2, 3, 4], samples=100, seed=7)
    assert first == second
    assert first["low"] <= first["estimate"] <= first["high"]


def test_paired_delta_uses_within_example_difference():
    result = paired_delta_ci([1, 5, 9], [0, 4, 8], samples=100, confidence=0.95, seed=1)
    assert result == {"estimate": -1.0, "low": -1.0, "high": -1.0}


def test_grouped_bootstrap_resamples_whole_groups():
    result = grouped_bootstrap_ci([0.0, 0.0, 10.0, 10.0], ["a", "a", "b", "b"], samples=50, seed=3)
    assert result["estimate"] == 5.0
    assert result["low"] <= result["estimate"] <= result["high"]


def test_tost_equivalence_and_inferiority_are_decidable():
    equivalent = tost_equivalence_test(
        [0.0, 0.01, -0.01, 0.0],
        -0.05,
        0.05,
        samples=50,
        seed=4,
        group_ids=["a", "b", "c", "d"],
    )
    inferior = one_sample_mean_test([0.07, 0.08, 0.06, 0.09], 0.02, alternative="greater")
    assert equivalent["equivalent"] is True
    assert inferior["p_value"] < 0.05


def test_prospective_equivalence_requires_a_freeze_hash():
    with pytest.raises(ValueError, match="requires an equivalence preregistration hash"):
        build_equivalence_analysis(
            [],
            {},
            bootstrap_samples=10,
            equivalence_margin=0.02,
            inferiority_margin=0.02,
            alpha=0.05,
            power_effect_size=0.05,
            power_assumed_pair_sd=0.4,
            power_target=0.8,
            power_target_n=500,
            seed=17,
            analysis_design="prospective_preregistered",
        )


def test_power_planning_supports_n500_preregistration():
    required = required_n_for_power(
        effect=0.05,
        standard_deviation=0.4,
        target_power=0.8,
        alpha=0.05,
        alternative="greater",
    )
    power = normal_approx_power(
        n=500,
        effect=0.05,
        standard_deviation=0.4,
        alpha=0.05,
        alternative="greater",
    )
    assert required is not None and required < 500
    assert power > 0.85


def test_threshold_is_selected_without_test_data():
    records = [
        PromptRecord("h", "h", True, "validation", "x", "p1"),
        PromptRecord("b", "b", False, "validation", "x", "p1"),
    ]
    threshold, accuracy = select_threshold(
        records,
        [ModelOutput("h", 0.9, 0.0), ModelOutput("b", 0.6, 0.0)],
    )
    assert 0.6 < threshold < 0.9
    assert accuracy == 1.0
