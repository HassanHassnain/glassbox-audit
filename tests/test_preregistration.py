import json

import pytest
import yaml

from glassbox_audit.preregistration import (
    freeze_preregistration,
    load_frozen_preregistration,
    preregistration_metadata,
)


def _valid_preregistration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "name": "fixture",
        "hypotheses": {"H1": "fixed hypothesis"},
        "run_order": ["cell_a", "cell_b"],
        "success_criteria": {"H1": {"delta_lt": -0.05}},
        "stopping_rules": ["do_not_retune"],
    }


def test_freeze_preregistration_hash_is_stable(tmp_path):
    source = tmp_path / "prereg.yaml"
    source.write_text(yaml.safe_dump(_valid_preregistration()))
    first = freeze_preregistration(source, tmp_path / "frozen-a.json")
    second = freeze_preregistration(source, tmp_path / "frozen-b.json")
    assert first["preregistration_hash"] == second["preregistration_hash"]
    assert load_frozen_preregistration(tmp_path / "frozen-a.json")["preregistration_hash"] == first[
        "preregistration_hash"
    ]


def test_freeze_preregistration_rejects_missing_required_field(tmp_path):
    source = tmp_path / "bad.yaml"
    payload = _valid_preregistration()
    payload.pop("run_order")
    source.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="run_order"):
        freeze_preregistration(source)


def test_preregistration_metadata_rejects_hash_mismatch(tmp_path):
    source = tmp_path / "prereg.yaml"
    output = tmp_path / "frozen.json"
    source.write_text(yaml.safe_dump(_valid_preregistration()))
    freeze_preregistration(source, output)
    with pytest.raises(ValueError, match="does not match"):
        preregistration_metadata(output, "0" * 64)


def test_load_frozen_preregistration_detects_tampering(tmp_path):
    source = tmp_path / "prereg.yaml"
    output = tmp_path / "frozen.json"
    source.write_text(yaml.safe_dump(_valid_preregistration()))
    frozen = freeze_preregistration(source, output)
    frozen["preregistration"]["run_order"].append("cell_c")
    output.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_preregistration(output)
