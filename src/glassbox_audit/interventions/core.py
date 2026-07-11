from __future__ import annotations

from dataclasses import dataclass

import torch

from ..sae import TopKSparseAutoencoder


def normalized(direction: torch.Tensor) -> torch.Tensor:
    return direction.detach().float() / direction.detach().float().norm().clamp_min(1e-8)


@dataclass
class AddDirection:
    direction: torch.Tensor
    scale: float

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        direction = normalized(self.direction).to(device=hidden.device, dtype=hidden.dtype)
        return hidden + self.scale * direction


@dataclass
class AblateDirection:
    direction: torch.Tensor

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        direction = normalized(self.direction).to(device=hidden.device, dtype=hidden.dtype)
        projection = torch.sum(hidden * direction, dim=-1, keepdim=True)
        return hidden - projection * direction


@dataclass
class PatchProjection:
    direction: torch.Tensor
    target_projection: float

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        direction = normalized(self.direction).to(device=hidden.device, dtype=hidden.dtype)
        current = torch.sum(hidden * direction, dim=-1, keepdim=True)
        return hidden + (self.target_projection - current) * direction


@dataclass
class AblateSAEFeatures:
    sae: TopKSparseAutoencoder
    feature_indices: list[int]

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        device, dtype = hidden.device, hidden.dtype
        sae = self.sae.to(device=device, dtype=torch.float32)
        flat = hidden.float()
        center = sae.activation_center.to(device)
        features = sae.encode(flat - center)
        ablated = features.clone()
        ablated[..., self.feature_indices] = 0
        delta = sae.decode(ablated) - sae.decode(features)
        return (flat + delta).to(dtype=dtype)


@dataclass
class ReconstructSAE:
    """Replace an activation by its SAE reconstruction without removing latents."""

    sae: TopKSparseAutoencoder

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        device, dtype = hidden.device, hidden.dtype
        sae = self.sae.to(device=device, dtype=torch.float32)
        flat = hidden.float()
        center = sae.activation_center.to(device)
        return (sae.decode(sae.encode(flat - center)) + center).to(dtype=dtype)


@dataclass
class SubstituteSAEFeatures:
    """Encode, zero selected latents, and substitute the decoded activation."""

    sae: TopKSparseAutoencoder
    feature_indices: list[int]

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        device, dtype = hidden.device, hidden.dtype
        sae = self.sae.to(device=device, dtype=torch.float32)
        flat = hidden.float()
        center = sae.activation_center.to(device)
        features = sae.encode(flat - center)
        ablated = features.clone()
        ablated[..., self.feature_indices] = 0
        return (sae.decode(ablated) + center).to(dtype=dtype)


@dataclass
class WeightedAblateSAEFeatures:
    """Subtract a train-fitted weighted subset while preserving SAE reconstruction residual."""

    sae: TopKSparseAutoencoder
    feature_indices: list[int]
    weights: list[float]

    def __post_init__(self) -> None:
        if len(self.feature_indices) != len(self.weights):
            raise ValueError("feature_indices and weights must have equal length")

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        device, dtype = hidden.device, hidden.dtype
        sae = self.sae.to(device=device, dtype=torch.float32)
        flat = hidden.float()
        center = sae.activation_center.to(device)
        features = sae.encode(flat - center)
        removed = torch.zeros_like(features)
        weights = torch.tensor(self.weights, device=device, dtype=features.dtype)
        removed[..., self.feature_indices] = features[..., self.feature_indices] * weights
        return (flat - sae.decode(removed)).to(dtype=dtype)
