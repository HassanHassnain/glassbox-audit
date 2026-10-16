import json
from pathlib import Path

import yaml

from glassbox_audit.developmental import (
    load_developmental_manifest,
    run_developmental_family,
    summarize_developmental_audit,
)

ROOT = Path(__file__).parents[1]


def _toy_manifest(path: Path) -> Path:
    manifest = {
        "schema_version": "1.0",
        "name": "toy_developmental_transfer",
        "seed": 17,
        "behavior": "refusal",
        "dataset_path": str(ROOT / "data/refusal_pairs.jsonl"),
        "evaluation": {
            "bootstrap_samples": 20,
            "confidence": 0.95,
            "random_controls": 1,
        },
        "success_criteria": {
            "source_validation_effect_size_gt": 0.5,
            "harmful_delta_lt": -0.01,
            "harmful_ci_high_lt": 0.0,
            "abs_benign_delta_lte": 1.0,
            "capability_nll_delta_lte": 1.0,
            "random_control_p_lte": 1.0,
        },
        "families": [
            {
                "slug": "toy-family",
                "source_model": "glassbox-toy-refusal",
                "source_backend": "toy",
                "target_model": "glassbox-toy-refusal",
                "target_backend": "toy",
                "target_revision": "main",
                "target_kind": "instruct",
                "dtype": "float32",
                "layers": [0, 1, 2, 3],
                "checkpoints": [
                    {"revision": "step10", "training_tokens_b": 1},
                    {"revision": "step20", "training_tokens_b": 2},
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def test_developmental_manifest_run_and_summary(tmp_path):
    manifest_path = _toy_manifest(tmp_path / "developmental.yaml")
    manifest = load_developmental_manifest(manifest_path)
    assert manifest["families"][0]["target_kind"] == "instruct"

    artifacts = tmp_path / "artifacts"
    result = run_developmental_family(
        manifest_path=manifest_path,
        family_slug="toy-family",
        output_root=artifacts,
        revisions=["step10"],
        source_device="cpu",
        target_device="cpu",
    )
    assert result["cells_completed"] == 1

    evidence_path = artifacts / "toy-family" / "step10" / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "completed"
    assert evidence["bootstrap_unit"] == "pair_id"
    assert evidence["claim_scope"] == "final_instruct_causal_transfer"
    assert isinstance(evidence["source_separation_passed"], bool)
    assert len(evidence["matched_random_direction_controls"]) == 1
    assert (artifacts / "toy-family" / "step10" / "direction.pt").exists()
    assert (artifacts / "toy-family" / "step10" / "outputs.json").exists()

    summary = summarize_developmental_audit(
        manifest_path=manifest_path,
        artifacts_root=artifacts,
        output_json=tmp_path / "results" / "a3_summary.json",
        output_table=tmp_path / "docs" / "developmental.md",
        output_figure=tmp_path / "docs" / "developmental.png",
    )
    assert summary["cells_predeclared"] == 2
    assert summary["cells_completed"] == 1
    assert summary["cells_unrun"] == 1
    assert summary["rows"][0]["status"] == "completed"
    assert summary["rows"][1]["status"] == "unrun"
    assert (tmp_path / "results" / "toy-family_step10.json").exists()
    assert (tmp_path / "results" / "toy-family_step20.json").exists()
