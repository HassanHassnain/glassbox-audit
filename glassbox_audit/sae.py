from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


class SAELensAdapter(nn.Module):
    def __init__(self, inner: nn.Module, metadata: dict[str, Any]):
        super().__init__()
        self.inner = inner
        self.source_metadata = metadata
        decoder = self._decoder_weight()
        d_in = metadata.get("d_in")
        d_sae = metadata.get("d_sae")
        if d_in is not None and d_sae is not None:
            self.input_dim = int(d_in)
            self.n_features = int(d_sae)
        elif decoder.ndim == 2:
            if decoder.shape[0] >= decoder.shape[1]:
                self.n_features, self.input_dim = int(decoder.shape[0]), int(decoder.shape[1])
            else:
                self.input_dim, self.n_features = int(decoder.shape[0]), int(decoder.shape[1])
        else:
            raise ValueError("Could not infer SAELens decoder shape")
        self.top_k = int(metadata.get("top_k") or self.n_features)
        self.register_buffer("activation_center", torch.zeros(self.input_dim))

    def _decoder_weight(self) -> torch.Tensor:
        decoder = getattr(self.inner, "W_dec", None)
        if decoder is not None:
            return decoder.detach().float().cpu()
        decoder_module = getattr(self.inner, "decoder", None)
        if decoder_module is not None and hasattr(decoder_module, "weight"):
            return decoder_module.weight.detach().float().cpu()
        raise ValueError("SAELens SAE does not expose W_dec or decoder.weight")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.inner, "encode"):
            return self.inner.encode(x)
        if hasattr(self.inner, "encode_standard"):
            return self.inner.encode_standard(x)
        raise ValueError("SAELens SAE does not expose an encode method")

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if hasattr(self.inner, "decode"):
            return self.inner.decode(features)
        if hasattr(self.inner, "decode_standard"):
            return self.inner.decode_standard(features)
        raise ValueError("SAELens SAE does not expose a decode method")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decode(features), features

    def feature_direction(self, feature_index: int) -> torch.Tensor:
        decoder = self._decoder_weight()
        if decoder.shape[0] == self.n_features:
            return decoder[feature_index].detach().clone()
        return decoder[:, feature_index].detach().clone()


@dataclass
class SAETrainingResult:
    model: TopKSparseAutoencoder | SAELensAdapter
    history: list[dict[str, float]]
    metrics: dict[str, float]


def _cfg_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def load_saelens_sae(
    *,
    release: str,
    sae_id: str,
    device: str = "cpu",
) -> SAELensAdapter:
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise ImportError("SAELens support requires installing the `sae-lens` package") from exc

    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    inner = loaded[0] if isinstance(loaded, tuple) else loaded
    cfg = loaded[1] if isinstance(loaded, tuple) and len(loaded) > 1 else getattr(inner, "cfg", None)
    metadata = {
        "source": "saelens",
        "release": release,
        "sae_id": sae_id,
        "d_in": _cfg_value(cfg, "d_in"),
        "d_sae": _cfg_value(cfg, "d_sae"),
        "hook_layer": _cfg_value(cfg, "hook_layer"),
        "hook_name": _cfg_value(cfg, "hook_name"),
        "architecture": _cfg_value(cfg, "architecture"),
        "top_k": _cfg_value(cfg, "top_k"),
    }
    return SAELensAdapter(inner, metadata)


def _batched_sae_metrics(
    model: TopKSparseAutoencoder | SAELensAdapter,
    centered_activations: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(centered_activations) == 0:
        raise ValueError("SAE metrics require at least one activation")

    squared_error = 0.0
    nonzero_features = 0
    active_mask = torch.zeros(
        model.n_features,
        dtype=torch.bool,
        device=centered_activations.device,
    )
    with torch.no_grad():
        for batch in centered_activations.split(batch_size):
            reconstruction, features = model(batch)
            squared_error += torch.nn.functional.mse_loss(
                reconstruction,
                batch,
                reduction="sum",
            ).item()
            nonzero_features += int(torch.count_nonzero(features).item())
            active_mask |= torch.any(features != 0, dim=0)

    mse = squared_error / centered_activations.numel()
    variance = torch.var(centered_activations).item()
    l0 = nonzero_features / len(centered_activations)
    active_features = int(active_mask.sum().item())
    return {
        "reconstruction_mse": mse,
        "variance_explained": 1 - mse / max(variance, 1e-12),
        "mean_l0": l0,
        "feature_density": l0 / model.n_features,
        "dead_feature_fraction": 1 - active_features / model.n_features,
        "active_features": active_features,
    }


def evaluate_sae_metrics(
    model: TopKSparseAutoencoder | SAELensAdapter,
    activations: torch.Tensor,
    batch_size: int = 256,
) -> dict[str, float]:
    try:
        metric_device = next(model.parameters()).device
    except StopIteration:
        metric_device = model.activation_center.device
    with torch.no_grad():
        x = activations.detach().float().to(metric_device)
        center = model.activation_center.to(metric_device)
        centered = x - center
        metrics = _batched_sae_metrics(model, centered, batch_size)
    return {
        **metrics,
        "n_features": model.n_features,
        "training_samples": len(activations),
        "samples_per_feature": len(activations) / model.n_features,
    }


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

    metrics = _batched_sae_metrics(model, x_centered, batch_size)
    model = model.cpu()
    model.register_buffer("activation_center", center.squeeze(0).cpu())
    return SAETrainingResult(
        model=model,
        history=history,
        metrics={
            **metrics,
            "n_features": n_features,
            "training_samples": len(x_centered),
            "samples_per_feature": len(x_centered) / n_features,
        },
    )


def sae_state_dict(model: TopKSparseAutoencoder | SAELensAdapter) -> dict[str, object]:
    if isinstance(model, SAELensAdapter):
        return {
            "source": "saelens",
            "release": model.source_metadata["release"],
            "sae_id": model.source_metadata["sae_id"],
            "source_metadata": model.source_metadata,
            "input_dim": model.input_dim,
            "n_features": model.n_features,
            "top_k": model.top_k,
        }
    return {
        "source": "trained",
        "input_dim": model.input_dim,
        "n_features": model.n_features,
        "top_k": model.top_k,
        "state_dict": model.state_dict(),
    }


def load_sae_from_state(path: str, device: str = "cpu") -> TopKSparseAutoencoder | SAELensAdapter:
    raw = torch.load(path, map_location="cpu")
    if raw.get("source") == "saelens":
        metadata = raw.get("source_metadata", {})
        release = raw.get("release") or metadata.get("release")
        sae_id = raw.get("sae_id") or metadata.get("sae_id")
        if not release or not sae_id:
            raise ValueError(f"{path} is missing SAELens release or sae_id metadata")
        return load_saelens_sae(release=str(release), sae_id=str(sae_id), device=device)
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
