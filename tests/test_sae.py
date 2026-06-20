import torch

from glassbox_audit.sae import train_sae


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
