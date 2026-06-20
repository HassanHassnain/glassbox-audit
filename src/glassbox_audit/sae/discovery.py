from __future__ import annotations

import torch
from torch.nn import functional as F

from ..analysis.stats import cohen_d
from .core import TopKSparseAutoencoder


def layer_scan(
    activations: dict[int, torch.Tensor],
    labels: torch.Tensor,
) -> list[dict[str, float | int]]:
    rows = []
    for layer, values in sorted(activations.items()):
        harmful = values[labels].mean(dim=0)
        benign = values[~labels].mean(dim=0)
        direction = F.normalize(harmful - benign, dim=0)
        projections = values @ direction
        difference_norm = (harmful - benign).norm().item()
        effect_size = cohen_d(projections[labels].tolist(), projections[~labels].tolist())
        rows.append(
            {
                "layer": layer,
                "mean_difference_norm": difference_norm,
                "projection_effect_size": effect_size,
                "localization_score": abs(effect_size) * difference_norm,
            }
        )
    return rows


def mean_difference_direction(activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.normalize(activations[labels].mean(dim=0) - activations[~labels].mean(dim=0), dim=0)


def train_probe_direction(
    activations: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    epochs: int = 400,
) -> tuple[torch.Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    x = activations.float()
    y = labels.float()
    weight = torch.zeros(x.shape[-1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05, weight_decay=0.01)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probability = torch.sigmoid(x @ weight + bias)
        accuracy = ((probability >= 0.5) == labels).float().mean().item()
    return F.normalize(weight.detach(), dim=0), {"train_accuracy": accuracy, "train_loss": loss.item()}


def feature_table(
    sae: TopKSparseAutoencoder,
    activations: torch.Tensor,
    labels: torch.Tensor,
    record_ids: list[str] | None = None,
    top_examples: int = 5,
) -> list[dict[str, object]]:
    with torch.no_grad():
        centered = activations.float() - sae.activation_center
        features = sae.encode(centered)
    record_ids = record_ids or [str(index) for index in range(features.shape[0])]
    rows = []
    for index in range(features.shape[-1]):
        positive = features[labels, index].tolist()
        negative = features[~labels, index].tolist()
        effect = cohen_d(positive, negative)
        values, indices = torch.topk(features[:, index], min(top_examples, features.shape[0]))
        rows.append(
            {
                "feature": index,
                "effect_size": effect,
                "harmful_mean": sum(positive) / len(positive),
                "benign_mean": sum(negative) / len(negative),
                "activation_frequency": (features[:, index] > 0).float().mean().item(),
                "decoder_norm": sae.feature_direction(index).norm().item(),
                "top_activating_record_ids": [
                    record_ids[int(position)] for value, position in zip(values, indices, strict=True)
                    if float(value) > 0
                ],
                "top_activation_values": [
                    float(value) for value in values if float(value) > 0
                ],
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["effect_size"])), reverse=True)


def combined_feature_direction(
    sae: TopKSparseAutoencoder,
    table: list[dict[str, float | int]],
    n_features: int,
) -> tuple[torch.Tensor, list[int]]:
    selected = [row for row in table if float(row["effect_size"]) > 0][:n_features]
    if not selected:
        selected = table[:n_features]
    indices = [int(row["feature"]) for row in selected]
    directions = []
    for row in selected:
        direction = sae.feature_direction(int(row["feature"]))
        directions.append(direction * float(row["effect_size"]))
    return F.normalize(torch.stack(directions).sum(dim=0), dim=0), indices


def random_directions(
    hidden_size: int,
    count: int,
    seed: int,
    orthogonal_to: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    directions = []
    for _ in range(count):
        direction = torch.randn(hidden_size, generator=generator)
        if orthogonal_to is not None:
            reference = F.normalize(orthogonal_to, dim=0)
            direction -= direction.dot(reference) * reference
        directions.append(F.normalize(direction, dim=0))
    return directions
