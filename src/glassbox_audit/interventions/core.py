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
