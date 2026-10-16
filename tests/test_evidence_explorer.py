import json

from glassbox_audit.evidence_explorer import build_evidence_explorer


def _write(path, payload):
    path.write_text(json.dumps(payload))


def test_build_evidence_explorer_from_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write(
        artifact / "summary.json",
        {
            "model": "model",
            "behavior": "refusal",
            "evidence_class": "test",
            "baseline_metrics": {
                "behavior_gap": 0.2,
                "harmful_refusal_rate": 0.9,
                "benign_overrefusal_rate": 0.1,
            },
        },
    )
    _write(artifact / "manifest.json", {"schema_version": "1.0", "config": {}})
    _write(artifact / "protocol.json", {})
    _write(
        artifact / "interventions.json",
        {
            "causal_tests": {
                "mean_difference_direction_ablation": {
                    "delta_mean_harmful_score_ci": {
                        "estimate": -0.2,
                        "low": -0.3,
                        "high": -0.1,
                    },
                    "delta_mean_benign_score_ci": {
                        "estimate": 0.01,
                        "low": 0.0,
                        "high": 0.02,
                    },
                    "delta_capability_nll_ci": {
                        "estimate": 0.01,
                        "low": 0.0,
                        "high": 0.02,
                    },
                }
            }
        },
    )
    bundle = build_evidence_explorer(
        artifact_roots=[tmp_path],
        output_json=tmp_path / "explorer.json",
        output_html=tmp_path / "index.html",
    )

    assert bundle["cell_count"] >= 5
    assert bundle["cells_by_level"]["1"] == 3
    assert any(cell["level"] == 2 and cell["verdict"] == "supported" for cell in bundle["cells"])
    assert (tmp_path / "explorer.json").exists()
    assert "Evidence Explorer" in (tmp_path / "index.html").read_text()
