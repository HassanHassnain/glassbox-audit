from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, n_features: int, top_k: int):
        super().__init__()
        if not 0 < top_k <= n_features:
            raise ValueError("top_k must be in [1, n_features]")
        self.input_dim = input_dim
        self.n_features = n_features
        self.top_k = top_k
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dense = torch.relu(self.encoder(x))
        values, indices = torch.topk(dense, self.top_k, dim=-1)
        sparse = torch.zeros_like(dense)
        return sparse.scatter(-1, indices, values)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decode(features), features

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def feature_direction(self, feature_index: int) -> torch.Tensor:
        return self.decoder.weight[:, feature_index].detach().clone()


@dataclass
class SAETrainingResult:
    model: TopKSparseAutoencoder
    history: list[dict[str, float]]
    metrics: dict[str, float]


def train_sae(
    activations: torch.Tensor,
    expansion_factor: int,
    top_k: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l1_coefficient: float,
    seed: int,
    device: str = "auto",
) -> SAETrainingResult:
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train_device = torch.device(device)
    x = activations.detach().float().to(train_device)
    center = x.mean(dim=0, keepdim=True)
    x_centered = x - center
    n_features = x.shape[-1] * expansion_factor
    model = TopKSparseAutoencoder(x.shape[-1], n_features, min(top_k, n_features)).to(train_device)
    model.encoder.bias.data.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=train_device).manual_seed(seed)
    history = []

    for epoch in range(epochs):
        permutation = torch.randperm(len(x_centered), generator=generator, device=train_device)
        losses = []
        for start in range(0, len(x_centered), batch_size):
            batch = x_centered[permutation[start : start + batch_size]]
            reconstruction, features = model(batch)
            mse = torch.mean((reconstruction - batch) ** 2)
            sparsity = features.abs().mean()
            loss = mse + l1_coefficient * sparsity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            losses.append((loss.item(), mse.item(), sparsity.item()))
        if epoch in {0, epochs - 1} or (epoch + 1) % max(epochs // 10, 1) == 0:
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": sum(row[0] for row in losses) / len(losses),
                    "mse": sum(row[1] for row in losses) / len(losses),
                    "mean_feature_activation": sum(row[2] for row in losses) / len(losses),
                }
            )

    with torch.no_grad():
        reconstruction, features = model(x_centered)
        mse = torch.mean((reconstruction - x_centered) ** 2).item()
        variance = torch.var(x_centered).item()
        l0 = (features > 0).float().sum(dim=-1).mean().item()
        active_mask = features.sum(dim=0) > 0
        dead = (~active_mask).float().mean().item()
        active_features = int(active_mask.sum().item())
    model = model.cpu()
    model.register_buffer("activation_center", center.squeeze(0).cpu())
    return SAETrainingResult(
        model=model,
        history=history,
        metrics={
            "reconstruction_mse": mse,
            "variance_explained": 1 - mse / max(variance, 1e-12),
            "mean_l0": l0,
            "feature_density": l0 / n_features,
            "dead_feature_fraction": dead,
            "active_features": active_features,
            "n_features": n_features,
            "training_samples": len(x_centered),
            "samples_per_feature": len(x_centered) / n_features,
        },
    )


def sae_state_dict(model: TopKSparseAutoencoder) -> dict[str, object]:
    return {
        "input_dim": model.input_dim,
        "n_features": model.n_features,
        "top_k": model.top_k,
        "state_dict": model.state_dict(),
    }


def load_sae_from_state(path: str) -> TopKSparseAutoencoder:
    raw = torch.load(path, map_location="cpu")
    model = TopKSparseAutoencoder(
        int(raw["input_dim"]),
        int(raw["n_features"]),
        int(raw["top_k"]),
    )
    state_dict = dict(raw["state_dict"])
    activation_center = state_dict.pop("activation_center", None)
    model.load_state_dict(state_dict)
    if activation_center is None:
        activation_center = torch.zeros(model.input_dim)
    model.register_buffer("activation_center", activation_center.detach().float().cpu())
    return model
