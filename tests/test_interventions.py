import torch

from glassbox_audit.interventions import (
    AblateDirection,
    AddDirection,
    PatchProjection,
    ReconstructSAE,
    SubstituteSAEFeatures,
    WeightedAblateSAEFeatures,
)
from glassbox_audit.sae import TopKSparseAutoencoder


def test_direction_interventions_have_expected_geometry():
    hidden = torch.tensor([2.0, 3.0])
    direction = torch.tensor([1.0, 0.0])
    assert torch.allclose(AddDirection(direction, -1)(hidden), torch.tensor([1.0, 3.0]))
    assert torch.allclose(AblateDirection(direction)(hidden), torch.tensor([0.0, 3.0]))
    assert torch.allclose(PatchProjection(direction, 5)(hidden), torch.tensor([5.0, 3.0]))


def test_sae_reconstruction_substitution_and_weighted_ablation_are_distinct():
    sae = TopKSparseAutoencoder(2, 2, 2)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(2))
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(torch.eye(2))
    sae.register_buffer("activation_center", torch.zeros(2))
    hidden = torch.tensor([2.0, 3.0])

    assert torch.allclose(ReconstructSAE(sae)(hidden), hidden)
    assert torch.allclose(SubstituteSAEFeatures(sae, [0])(hidden), torch.tensor([0.0, 3.0]))
    assert torch.allclose(
        WeightedAblateSAEFeatures(sae, [0], [0.5])(hidden),
        torch.tensor([1.0, 3.0]),
    )
