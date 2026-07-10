import json

import torch

from glassbox_audit.discovery import train_l1_feature_weights, weighted_feature_direction
from glassbox_audit.pipeline import _refresh_sae_zoo_league_table
from glassbox_audit.sae import SAELensAdapter, evaluate_sae_metrics, sae_state_dict, train_sae


class FakeSAELensSAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W_dec = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.W_enc = torch.nn.Parameter(self.W_dec.T.clone())

    def encode(self, x):
        return torch.relu(x @ self.W_enc)

    def decode(self, features):
        return features @ self.W_dec


def test_sae_zoo_league_table_preserves_failed_cells(tmp_path):
    failure = {
        "schema_version": "1.0",
        "status": "failed",
        "model": "gated/model",
        "behavior": "refusal",
        "target_layer": 3,
        "sae_source": {"release": "public-release", "sae_id": "layer_3"},
        "failure": {"error_type": "access_denied"},
    }
    (tmp_path / "failed.json").write_text(json.dumps(failure))
    _refresh_sae_zoo_league_table(tmp_path)

    league = json.loads((tmp_path / "league_table.json").read_text())
    assert league["rows"][0]["status"] == "failed"
    assert league["rows"][0]["failure"]["error_type"] == "access_denied"


def test_sae_is_sparse_and_reconstructs():
    torch.manual_seed(3)
    activations = torch.randn(64, 8)
    result = train_sae(
        activations,
        expansion_factor=2,
        top_k=3,
        epochs=20,
        batch_size=32,
        learning_rate=0.01,
        l1_coefficient=0.0001,
        seed=3,
        device="cpu",
    )
    assert result.metrics["mean_l0"] <= 3
    assert result.metrics["reconstruction_mse"] < 2
    assert result.model.feature_direction(0).shape == (8,)


def test_saelens_adapter_exposes_public_dictionary_metadata():
    adapter = SAELensAdapter(
        FakeSAELensSAE(),
        {
            "source": "saelens",
            "release": "test-release",
            "sae_id": "layer_1/test",
            "d_in": 2,
            "d_sae": 3,
            "hook_layer": 1,
            "top_k": 2,
        },
    )
    metrics = evaluate_sae_metrics(adapter, torch.randn(8, 2))
    state = sae_state_dict(adapter)

    assert adapter.input_dim == 2
    assert adapter.n_features == 3
    assert adapter.feature_direction(2).shape == (2,)
    assert metrics["n_features"] == 3
    assert state["source"] == "saelens"
    assert state["release"] == "test-release"
    assert "state_dict" not in state


def test_l1_feature_frontier_follows_sae_device_and_returns_cpu_direction():
    activations = torch.randn(12, 4)
    labels = torch.tensor([False, True] * 6)
    result = train_sae(
        activations,
        expansion_factor=2,
        top_k=2,
        epochs=2,
        batch_size=12,
        learning_rate=0.01,
        l1_coefficient=0.0001,
        seed=3,
        device="cpu",
    )
    weights, metrics = train_l1_feature_weights(
        result.model,
        activations,
        labels,
        seed=4,
        epochs=3,
    )
    direction = weighted_feature_direction(result.model, weights, [0, 1])
    assert weights.device.type == "cpu"
    assert direction.device.type == "cpu"
    assert metrics["epochs"] == 3
