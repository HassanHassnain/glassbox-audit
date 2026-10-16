import json

from glassbox_audit.certification import write_certification_card


def _write(path, payload):
    path.write_text(json.dumps(payload))


def test_certification_card_from_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write(
        artifact / "summary.json",
        {
            "model": "model",
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
            "config": {"behavior": "refusal"},
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
    causal = {
        "delta_mean_harmful_score_ci": {"estimate": -0.2, "low": -0.3, "high": -0.1},
        "delta_mean_benign_score_ci": {"estimate": 0.01, "low": 0.0, "high": 0.02},
        "delta_capability_nll_ci": {"estimate": 0.01, "low": 0.0, "high": 0.02},
    }
    _write(
        artifact / "interventions.json",
        {"causal_tests": {"sae_feature_ablation": causal}},
    )
    _write(
        artifact / "failure_analysis.json",
        {"random_ablation_control_comparison": {"empirical_one_sided_p": 0.01}},
    )
    _write(artifact / "records.json", [])
    candidate = tmp_path / "candidate.json"
    _write(candidate, {"type": "artifact_method", "method": "sae_feature_ablation"})
    external = tmp_path / "external.json"
    _write(
        external,
        {
            "causal_tests": {
                "sae_feature_ablation": {
                    "delta_mean_harmful_score_ci": {
                        "estimate": -0.1,
                        "low": -0.2,
                        "high": -0.01,
                    }
                }
            }
        },
    )

    card = write_certification_card(
        artifact,
        output_json=tmp_path / "card.json",
        output_html=tmp_path / "card.html",
        output_pdf=tmp_path / "card.pdf",
        candidate_intervention=candidate,
        external_artifacts=[external],
        sign=True,
        git_ref="test-ref",
    )

    assert card["overall_verdict"]["passed"] is True
    assert card["certified_evidence_method"] == "sae_feature_ablation"
    assert card["candidate_intervention"]["type"] == "artifact_method"
    assert card["candidate_intervention"]["spec_sha256"]
    assert card["corpus_hash"]
    assert card["rungs"]["capability_cost"]["passed"] is True
    assert card["rungs"]["transfer"]["passed"] is True
    assert card["signature"]["digest"]
    assert (tmp_path / "card.json").exists()
    assert (tmp_path / "card.html").exists()
    assert (tmp_path / "card.pdf").read_bytes().startswith(b"%PDF-")
