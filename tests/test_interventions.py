import torch

from glassbox_audit.interventions import AblateDirection, AddDirection, PatchProjection


def test_direction_interventions_have_expected_geometry():
    hidden = torch.tensor([2.0, 3.0])
    direction = torch.tensor([1.0, 0.0])
    assert torch.allclose(AddDirection(direction, -1)(hidden), torch.tensor([1.0, 3.0]))
    assert torch.allclose(AblateDirection(direction)(hidden), torch.tensor([0.0, 3.0]))
    assert torch.allclose(PatchProjection(direction, 5)(hidden), torch.tensor([5.0, 3.0]))
