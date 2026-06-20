import csv
import json
from pathlib import Path

import pytest

from glassbox_audit.analysis.release_hardening import (
    write_data_leakage_audit,
    write_dose_response,
    write_scorer_robustness,
)
from glassbox_audit.analysis.stats import benjamini_hochberg, paired_permutation_test

ROOT = Path(__file__).parents[1]


def test_paired_permutation_detects_negative_shift():
    result = paired_permutation_test([1.0, 1.1, 1.2, 1.3], [0.5, 0.6, 0.7, 0.8], alternative="less", samples=200, seed=1)
    assert result["statistic"] == pytest.approx(-0.5)
    assert result["p_value"] < 0.1


def test_bh_adjustment_is_monotone_in_sorted_order():
    adjusted = benjamini_hochberg([0.01, 0.03, 0.2])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert all(0 <= value <= 1 for value in adjusted)


def test_leakage_audit_passes_controlled_fixture(tmp_path):
    output = tmp_path / "leakage.json"
    doc = tmp_path / "leakage.md"
    result = write_data_leakage_audit(
        controlled_path=ROOT / "data/fixtures/refusal_pairs_tiny.jsonl",
        external_paths=[ROOT / "data/fixtures/external_refusal_fixture.jsonl"],
        output_json=output,
        output_doc=doc,
    )
    assert result["leakage_pass"] is True
    assert output.exists()
    assert doc.exists()


def test_scorer_robustness_exports_disagreements(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    records = [
        {
            "id": "h1",
            "prompt": "harmful",
            "harmful": True,
            "split": "test",
            "category": "x",
            "pair_id": "p1",
            "family_id": "f1",
            "source": "fixture",
        },
        {
            "id": "b1",
            "prompt": "benign",
            "harmful": False,
            "split": "test",
            "category": "x",
            "pair_id": "p1",
            "family_id": "f1",
            "source": "fixture",
        },
    ]
    (artifact / "records.json").write_text(json.dumps(records))
    (artifact / "summary.json").write_text(
        json.dumps({"behavior_threshold": 0.5, "baseline_metrics": {}})
    )
    baseline = [
        {"record_id": "h1", "behavior_score": 0.55, "capability_nll": 1.0, "response": ""},
        {"record_id": "b1", "behavior_score": 0.45, "capability_nll": 1.0, "response": ""},
    ]
    changed = [
        {"record_id": "h1", "behavior_score": 0.35, "capability_nll": 1.0, "response": ""},
        {"record_id": "b1", "behavior_score": 0.40, "capability_nll": 1.0, "response": ""},
    ]
    row = {
        "outputs": changed,
        "delta_mean_harmful_score_ci": {"estimate": -0.2, "low": -0.2, "high": -0.2},
        "delta_mean_benign_score_ci": {"estimate": -0.05, "low": -0.05, "high": -0.05},
        "delta_capability_nll_ci": {"estimate": 0.0, "low": 0.0, "high": 0.0},
    }
    (artifact / "interventions.json").write_text(
        json.dumps(
            {
                "baseline_outputs": baseline,
                "causal_tests": {
                    "sae_feature_ablation": row,
                    "mean_difference_direction_ablation": row,
                    "linear_probe_direction_ablation": row,
                },
            }
        )
    )
    csv_path = tmp_path / "disagreements.csv"
    result = write_scorer_robustness(
        artifact_dir=artifact,
        output_json=tmp_path / "scorer.json",
        output_doc=tmp_path / "scorer.md",
        disagreement_csv=csv_path,
    )
    assert result["keyword_response_scorer_limitation"]
    with csv_path.open() as handle:
        assert list(csv.DictReader(handle))


def test_dose_response_summarizes_fixed_grid(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "summary.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {
                    "harmful_refusal_rate": 1.0,
                    "benign_overrefusal_rate": 0.5,
                }
            }
        )
    )
    rows = []
    for method in ["sae_features", "mean_difference", "linear_probe"]:
        rows.append(
            {
                "method": method,
                "scale": -1.0,
                "metrics": {"harmful_refusal_rate": 0.5, "benign_overrefusal_rate": 0.4},
                "delta_mean_harmful_score_ci": {"estimate": -0.1, "low": -0.1, "high": -0.1},
                "delta_mean_benign_score_ci": {"estimate": -0.02, "low": -0.02, "high": -0.02},
                "delta_capability_nll_ci": {"estimate": 0.01, "low": 0.01, "high": 0.01},
            }
        )
    (artifact / "interventions.json").write_text(json.dumps({"steering_sweeps": rows}))
    result = write_dose_response(
        artifact_dir=artifact,
        output_json=tmp_path / "dose.json",
        output_doc=tmp_path / "dose.md",
        figure_prefix=tmp_path / "dose_response",
    )
    assert len(result["rows"]) == 3
