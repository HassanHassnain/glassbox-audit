from pathlib import Path

import pytest
import torch

from glassbox_audit.config import EvaluationConfig, ExperimentConfig, ModelConfig, SAEConfig
from glassbox_audit.interventions import AblateSAEFeatures
from glassbox_audit.pipeline import _random_sae_controls, run_audit
from glassbox_audit.reporting.workbench import load_artifacts
from glassbox_audit.utils import read_json

ROOT = Path(__file__).parents[1]


def test_toy_pipeline_runs_end_to_end(tmp_path):
    config = ExperimentConfig(
        seed=17,
        evidence_class="synthetic_fixture",
        dataset_path=str(ROOT / "data/fixtures/refusal_pairs_tiny.jsonl"),
        model=ModelConfig(backend="toy", layers=[0, 1, 2, 3], dtype="float32", device="cpu"),
        sae=SAEConfig(
            expansion_factor=2,
            top_k=4,
            epochs=15,
            batch_size=64,
            n_features_to_intervene=2,
            device="cpu",
        ),
        evaluation=EvaluationConfig(
            bootstrap_samples=30,
            steering_scales=[-3.0, 1.0],
            random_controls=2,
            control_scale=-2.0,
        ),
    )
    summary = run_audit(config, tmp_path)
    assert summary["evidence_class"] == "synthetic_fixture"
    assert summary["target_layer"] != 0
    assert (tmp_path / "REPORT.md").exists()
    artifacts = load_artifacts(tmp_path)
    interventions = artifacts["interventions"]
    assert len(interventions["steering_sweeps"]) == 6
    assert len(interventions["random_controls"]) == 8
    assert sum(
        row["method"] == "random_sae" and row["intervention"] == "feature_ablation_control"
        for row in interventions["random_controls"]
    ) == 2
    assert "mean_difference_direction_ablation" in interventions["causal_tests"]
    assert (tmp_path / "failure_analysis.json").exists()
    assert (tmp_path / "method_comparisons.json").exists()
    assert (tmp_path / "layerwise_controls.json").exists()
    assert (tmp_path / "protocol.json").exists()
    assert read_json(tmp_path / "manifest.json")["schema_version"] == "1.0"


def test_toy_model_has_measurable_baseline_behavior(tmp_path):
    config = ExperimentConfig(
        evidence_class="synthetic_fixture",
        dataset_path=str(ROOT / "data/fixtures/refusal_pairs_tiny.jsonl"),
        model=ModelConfig(backend="toy", layers=[1]),
        sae=SAEConfig(
            expansion_factor=2,
            top_k=2,
            epochs=5,
            n_features_to_intervene=1,
            device="cpu",
        ),
        evaluation=EvaluationConfig(
            bootstrap_samples=10,
            steering_scales=[-2.0],
            random_controls=1,
        ),
    )
    summary = run_audit(config, tmp_path)
    assert summary["baseline_metrics"]["behavior_gap"] > 0.2


def test_real_audit_mode_enforces_held_out_protocol(tmp_path):
    config = ExperimentConfig(
        mode="real_audit",
        evidence_class="empirical_bounded_real_audit",
        dataset_path=str(ROOT / "data/fixtures/refusal_pairs_tiny.jsonl"),
        model=ModelConfig(backend="toy", layers=[0, 1]),
    )
    with pytest.raises(ValueError, match="Invalid real-audit configuration"):
        run_audit(config, tmp_path)


def test_random_sae_controls_match_center_and_bias():
    center = torch.arange(4, dtype=torch.float32)
    sae, controls = _random_sae_controls(
        input_dim=4,
        expansion_factor=2,
        top_k=2,
        count=1,
        seed=3,
        activation_center=center,
    )
    feature_index, _direction = controls[0]
    assert torch.allclose(sae.activation_center, center)
    assert torch.count_nonzero(sae.encoder.bias).item() == 0

    centered = torch.empty(0)
    for _ in range(100):
        candidate = torch.randn(3, 4) * 3
        if (sae.encode(candidate)[:, feature_index] > 0).any():
            centered = candidate
            break
    assert centered.numel() > 0
    hidden = center + centered
    before = sae.decode(sae.encode(centered))
    after_hidden = AblateSAEFeatures(sae, [feature_index])(hidden)
    after = sae.decode(sae.encode(after_hidden - center))
    assert not torch.allclose(before, after)
