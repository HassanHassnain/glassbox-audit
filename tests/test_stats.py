from glassbox_audit.analysis.stats import (
    bootstrap_ci,
    grouped_bootstrap_ci,
    holm_adjust,
    one_sample_mean_test,
    paired_delta_ci,
)
from glassbox_audit.evaluation import select_threshold
from glassbox_audit.types import ModelOutput, PromptRecord


def test_bootstrap_is_deterministic_and_contains_estimate():
    first = bootstrap_ci([1, 2, 3, 4], samples=100, seed=7)
    second = bootstrap_ci([1, 2, 3, 4], samples=100, seed=7)
    assert first == second
    assert first["low"] <= first["estimate"] <= first["high"]


def test_paired_delta_uses_within_example_difference():
    result = paired_delta_ci([1, 5, 9], [0, 4, 8], samples=100, confidence=0.95, seed=1)
    assert result == {"estimate": -1.0, "low": -1.0, "high": -1.0}


def test_grouped_bootstrap_resamples_complete_pairs():
    result = grouped_bootstrap_ci(
        [0.0, 0.0, 10.0, 10.0], ["a", "a", "b", "b"], samples=50, seed=3
    )
    assert result["estimate"] == 5.0
    assert result["low"] <= result["estimate"] <= result["high"]


def test_inferiority_margin_and_holm_adjustment():
    inferior = one_sample_mean_test([0.07, 0.08, 0.06, 0.09], 0.02, alternative="greater")
    assert inferior["p_value"] < 0.05
    assert holm_adjust([0.01, 0.03, 0.04, 0.5]) == [0.04, 0.09, 0.09, 0.5]


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
