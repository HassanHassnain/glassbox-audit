import json
from pathlib import Path

import pytest
import torch
import yaml

from glassbox_audit.data import load_records
from glassbox_audit.modeling import ToyRefusalModel
from glassbox_audit.robustness import (
    _assert_aggregate_only,
    run_robustness_audit,
    summarize_robustness_audit,
)

ROOT = Path(__file__).parents[1]
SHIFT_FAMILIES = ["paraphrase", "persona", "multilingual", "jailbreak_format"]


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload))


def _source_artifact(path: Path) -> Path:
    path.mkdir()
    _write_json(
        path / "manifest.json",
        {
            "config": {
                "seed": 17,
                "behavior": "refusal",
                "model": {
                    "backend": "toy",
                    "name": "glassbox-toy-refusal",
                    "layers": [0, 1, 2, 3],
                    "dtype": "float32",
                    "device": "cpu",
                },
                "sae": {},
                "evaluation": {},
            }
        },
    )
    _write_json(
        path / "summary.json",
        {
            "model": "glassbox-toy-refusal",
            "behavior": "refusal",
            "behavior_threshold": 0.5,
            "target_layer": 1,
        },
    )
    _write_json(
        path / "interventions.json",
        {
            "causal_tests": {
                "mean_difference_direction_ablation": {
                    "delta_mean_harmful_score_ci": {"estimate": -0.5, "low": -0.6, "high": -0.4}
                }
            }
        },
    )
    model = ToyRefusalModel(seed=17)
    torch.save(
        {"directions": {"mean_difference": model.behavior_direction}},
        path / "discovery_tensors.pt",
    )
    return path


def _shifted_corpus(path: Path) -> Path:
    test_records = [record for record in load_records(ROOT / "data/refusal_pairs.jsonl") if record.split == "test"]
    pair_ids = sorted({record.pair_id for record in test_records})
    shift_by_pair = {
        pair_id: SHIFT_FAMILIES[index * len(SHIFT_FAMILIES) // len(pair_ids)]
        for index, pair_id in enumerate(pair_ids)
    }
    with path.open("w") as handle:
        for record in test_records:
            row = record.__dict__ | {"shift_family": shift_by_pair[record.pair_id]}
            handle.write(json.dumps(row) + "\n")
    return path


def test_robustness_audit_is_aggregate_only(tmp_path):
    source = _source_artifact(tmp_path / "source")
    corpus = _shifted_corpus(tmp_path / "shifted.jsonl")
    config = {
        "schema_version": "1.0",
        "name": "toy_robustness",
        "seed": 17,
        "aggregate_only": True,
        "source_artifacts": str(source),
        "shifted_corpus_path": str(corpus),
        "shift_families": SHIFT_FAMILIES,
        "evaluation": {"bootstrap_samples": 20, "confidence": 0.95, "random_controls": 1},
        "success_criteria": {
            "harmful_delta_lt": 1.0,
            "harmful_ci_high_lt": 1.0,
            "abs_benign_delta_lte": 1.0,
            "capability_nll_delta_lte": 1.0,
            "min_source_effect_retention": 0.0,
            "random_control_p_lte": 1.0,
        },
    }
    config_path = tmp_path / "robustness.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    evidence_dir = tmp_path / "results"

    result = run_robustness_audit(
        config_path=config_path,
        output_dir=evidence_dir,
        device="cpu",
    )
    assert result["aggregate_only"] is True
    assert len(result["shift_families"]) == 4
    serialized = json.dumps(result).lower()
    for forbidden in ['"prompt"', '"response"', '"record_id"', '"outputs"']:
        assert forbidden not in serialized

    public = summarize_robustness_audit(
        config_path=config_path,
        evidence_dir=evidence_dir,
        output_json=tmp_path / "public.json",
        output_table=tmp_path / "table.md",
        output_figure=tmp_path / "figure.png",
    )
    assert public["cells_completed"] == 4
    assert public["cells_unrun"] == 0
    assert (tmp_path / "figure.png").exists()


def test_aggregate_only_guard_rejects_nested_output_aliases():
    with pytest.raises(ValueError, match="forbidden key"):
        _assert_aggregate_only({"summary": {"baseline_outputs": []}})
