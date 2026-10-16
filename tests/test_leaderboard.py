import json

from glassbox_audit.leaderboard import build_leaderboard


def _write(path, payload):
    path.write_text(json.dumps(payload))


def test_build_leaderboard_from_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write(
        artifact / "summary.json",
        {
            "model": "model",
            "behavior": "refusal",
            "evidence_class": "test",
            "target_layer": 2,
            "test_pairs": 1,
            "sae_source": {"source": "trained", "n_features": 8, "top_k": 2},
            "preregistration_hash": "hash",
        },
    )
    _write(
        artifact / "manifest.json",
        {
            "schema_version": "1.0",
            "dataset": {"n_records": 4},
            "preregistration_hash": "hash",
            "config": {"model": {"backend": "toy", "name": "model"}},
        },
    )
    _write(
        artifact / "protocol.json",
        {
            "test_pairs_used_for_final_reporting_only": 1,
        },
    )
    row = {
        "delta_mean_harmful_score_ci": {"estimate": -0.2, "low": -0.3, "high": -0.1},
        "delta_mean_benign_score_ci": {"estimate": 0.01, "low": 0.0, "high": 0.02},
        "delta_capability_nll_ci": {"estimate": 0.01, "low": 0.0, "high": 0.02},
    }
    controls = [
        {
            "method": "random_sae",
            "intervention": "feature_ablation_control",
            "delta_mean_harmful_score_ci": {"estimate": 0.0, "low": 0.0, "high": 0.0},
        }
        for _ in range(20)
    ]
    _write(
        artifact / "interventions.json",
        {
            "causal_tests": {"sae_feature_ablation": row},
            "random_controls": controls,
        },
    )

    leaderboard = build_leaderboard(
        artifact_roots=[tmp_path],
        output_json=tmp_path / "leaderboard.json",
        output_html=tmp_path / "index.html",
        strict=True,
    )

    assert leaderboard["row_count"] == 1
    assert leaderboard["rows"][0]["verdict"] == "pass"
    assert leaderboard["rows"][0]["artifact_hashes"]["summary.json"]
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "index.html").exists()
