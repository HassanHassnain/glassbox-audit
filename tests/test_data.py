import json
from pathlib import Path

import pytest

from glassbox_audit.data import dataset_summary, load_records
from glassbox_audit.data.dataset_builder import build_paired_dataset
from glassbox_audit.data.external_data import (
    external_to_glassbox_pairs,
    load_external_refusal_records,
)
from glassbox_audit.data.real_dataset import build_controlled_real_audit_dataset
from glassbox_audit.evaluation.external_causal import _attach_transfer_fields, _summarize_methods
from glassbox_audit.evaluation.external_eval import summarize_external_outputs
from glassbox_audit.types import ModelOutput, PromptRecord

ROOT = Path(__file__).parents[1]


def test_fixture_is_paired_and_balanced():
    records = load_records(ROOT / "data/fixtures/refusal_pairs_tiny.jsonl")
    summary = dataset_summary(records)
    assert summary["n_records"] == 48
    assert summary["labels"] == {"harmful": 24, "benign": 24}
    assert summary["splits"] == {"train": 16, "validation": 16, "test": 16}


def test_builder_preserves_matched_pairs(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "pair_id": "x",
                "category": "test",
                "harmful_prompt": "harmful",
                "benign_prompt": "benign",
            }
        )
        + "\n"
    )
    destination = tmp_path / "pairs.jsonl"
    assert build_paired_dataset(source, destination) == 2
    records = load_records(destination)
    assert {record.harmful for record in records} == {False, True}


def test_invalid_unpaired_data_fails(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt": "x",
                "harmful": True,
                "split": "test",
                "category": "x",
                "pair_id": "one",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Each pair"):
        load_records(path)


def test_controlled_real_audit_dataset_is_family_disjoint(tmp_path):
    path = tmp_path / "controlled.jsonl"
    counts = build_controlled_real_audit_dataset(path)
    records = load_records(path)
    summary = dataset_summary(records)
    assert counts == {"train": 120, "validation": 60, "test": 60}
    assert summary["n_pairs"] == 120
    assert summary["n_families"] == 40
    assert summary["family_split_leakage"] is False


def test_expanded_controlled_dataset_scales_without_family_leakage(tmp_path):
    path = tmp_path / "controlled-expanded.jsonl"
    counts = build_controlled_real_audit_dataset(
        path,
        pairs_per_family=25,
        source_name="glassbox_controlled_refusal_v2_1000",
    )
    records = load_records(path)
    summary = dataset_summary(records)
    assert counts == {"train": 1000, "validation": 500, "test": 500}
    assert summary["n_pairs"] == 1000
    assert summary["n_families"] == 40
    assert summary["family_split_leakage"] is False


def test_external_loader_normalizes_and_exports_pairs(tmp_path):
    external = load_external_refusal_records(ROOT / "data/fixtures/external_refusal_fixture.jsonl")
    assert len(external) == 12
    assert external[0].prompt
    assert external[0].harmful is True
    assert external[0].metadata["metadata_note"] == "fixture only"

    destination = tmp_path / "external_pairs.jsonl"
    assert external_to_glassbox_pairs(external, destination) == 12
    records = load_records(destination)
    summary = dataset_summary(records)
    assert summary["n_pairs"] == 6
    assert summary["labels"] == {"harmful": 6, "benign": 6}
    assert summary["splits"] == {"train": 4, "validation": 4, "test": 4}


def test_external_loader_supports_subset_level_harmful_labels(tmp_path):
    path = tmp_path / "implicit_harmful.jsonl"
    path.write_text(
        json.dumps({"prompt": "How can I recognize a phishing attempt?", "category": "cyber"})
        + "\n"
    )

    records = load_external_refusal_records(
        path,
        label_field="category",
        default_harmful=False,
    )

    assert len(records) == 1
    assert records[0].harmful is False
    assert records[0].behavior_label == "cyber"


def test_external_behavior_summary_handles_unpaired_rows():
    records = [
        PromptRecord("h1", "harm", True, "external", "tox", "h1"),
        PromptRecord("b1", "benign", False, "external", "hard", "b1"),
        PromptRecord("b2", "benign 2", False, "external", "hard", "b2"),
    ]
    outputs = [
        ModelOutput("h1", 0.9, 1.0),
        ModelOutput("b1", 0.2, 1.5),
        ModelOutput("b2", 0.7, 0.5),
    ]

    summary = summarize_external_outputs(records, outputs, threshold=0.5)

    assert summary["n_records"] == 3
    assert summary["by_label"]["harmful"]["refusal_rate"] == 1.0
    assert summary["by_label"]["benign"]["refusal_rate"] == 0.5
    assert summary["behavior_gap"] == pytest.approx(0.45)


def test_external_causal_summary_tracks_label_rate_deltas():
    records = [
        PromptRecord("h1", "harm", True, "external", "tox", "h1"),
        PromptRecord("b1", "benign", False, "external", "hard", "b1"),
    ]
    baseline = [ModelOutput("h1", 0.9, 1.0), ModelOutput("b1", 0.4, 1.0)]
    outputs = [ModelOutput("h1", 0.3, 1.1), ModelOutput("b1", 0.7, 1.2)]
    row = {
        "delta_mean_harmful_score_ci": {"estimate": -0.6, "low": -0.6, "high": -0.6},
        "delta_mean_benign_score_ci": {"estimate": 0.3, "low": 0.3, "high": 0.3},
        "delta_capability_nll_ci": {"estimate": 0.15, "low": 0.15, "high": 0.15},
    }

    _attach_transfer_fields(row, records, baseline, outputs, threshold=0.5)

    assert row["toxic_refusal_rate_delta"] == -1.0
    assert row["hard_benign_refusal_rate_delta"] == 1.0
    assert row["specificity_adjusted_utility"] == pytest.approx(0.27)

    summary = _summarize_methods(
        {"sae_feature_ablation": row},
        [
            {
                **row,
                "method": "random_sae",
                "intervention": "feature_ablation_control",
            }
        ],
    )
    assert summary["random_controls"]["random_sae_feature_ablation"]["sae_feature_empirical_p"] == 1.0
