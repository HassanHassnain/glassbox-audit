import json

from glassbox_audit.drift import compare_artifact_drift


def _write(path, payload):
    path.write_text(json.dumps(payload))


def _artifact(root, *, name, harmful_high, benign_estimate=0.01, nll_estimate=0.01, p=0.01):
    artifact = root / name
    artifact.mkdir()
    _write(
        artifact / "summary.json",
        {
            "model": name,
            "behavior": "refusal",
            "evidence_class": "test",
            "selected_features": [1],
            "sae_outperforms_simple_baselines": False,
            "preregistration_hash": "hash",
        },
    )
    _write(
        artifact / "manifest.json",
        {
            "dataset": {"n_records": 4},
            "preregistration_hash": "hash",
            "config": {"model": {"backend": "toy", "name": name}},
        },
    )
    _write(
        artifact / "protocol.json",
        {
            "train_records_used_for_discovery_and_sae_only": 2,
            "validation_records_used_for_threshold_and_scale_selection_only": 1,
            "test_records_used_for_final_reporting_only": 1,
            "test_pairs_used_for_final_reporting_only": 1,
            "test_access_policy": "final metrics only",
        },
    )
    _write(
        artifact / "interventions.json",
        {
            "causal_tests": {
                "sae_feature_ablation": {
                    "delta_mean_harmful_score_ci": {
                        "estimate": harmful_high - 0.1,
                        "low": harmful_high - 0.2,
                        "high": harmful_high,
                    },
                    "delta_mean_benign_score_ci": {
                        "estimate": benign_estimate,
                        "low": benign_estimate,
                        "high": benign_estimate,
                    },
                    "delta_capability_nll_ci": {
                        "estimate": nll_estimate,
                        "low": nll_estimate,
                        "high": nll_estimate,
                    },
                }
            },
        },
    )
    _write(
        artifact / "failure_analysis.json",
        {"random_ablation_control_comparison": {"empirical_one_sided_p": p, "n_controls": 4}},
    )
    _write(artifact / "records.json", [])
    return artifact


def test_compare_artifact_drift_reports_degradation(tmp_path):
    source = _artifact(tmp_path, name="source", harmful_high=-0.1)
    new = _artifact(tmp_path, name="new", harmful_high=0.1, p=0.5)

    report = compare_artifact_drift(
        source_artifact=source,
        new_artifact=new,
        output_json=tmp_path / "drift.json",
        output_doc=tmp_path / "drift.md",
    )

    assert report["drift_status"] == "degraded"
    assert report["mechanism_survives"] is False
    assert any(row["rung"] == "held_out_causal" and row["changed"] for row in report["rung_comparison"])
    assert (tmp_path / "drift.json").exists()
    assert (tmp_path / "drift.md").exists()
