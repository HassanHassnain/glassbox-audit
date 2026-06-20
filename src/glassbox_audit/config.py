from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backend: str = "toy"
    name: str = "glassbox-toy-refusal"
    layers: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    dtype: str = "float16"
    device: str = "auto"
    trust_remote_code: bool = False
    cache_dir: str | None = None


@dataclass
class SAEConfig:
    expansion_factor: int = 4
    top_k: int = 8
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 0.003
    l1_coefficient: float = 0.0001
    n_features_to_intervene: int = 3
    device: str = "auto"
    tokens_per_prompt: int = 1
    max_activation_samples: int | None = None
    activation_subsample_seed: int | None = None


@dataclass
class EvaluationConfig:
    threshold: float | None = 0.5
    bootstrap_samples: int = 500
    confidence: float = 0.95
    steering_scales: list[float] = field(default_factory=lambda: [-4.0, -2.0, -1.0, 1.0])
    random_controls: int = 8
    control_scale: float = -2.0
    layerwise_controls: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 17
    mode: str = "standard"
    behavior: str = "refusal"
    evidence_class: str = "unspecified"
    dataset_path: str = "data/refusal_pairs.jsonl"
    target_layer: int | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _project_root_for_config(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "data").exists():
            return parent
    return path.parent.parent


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    root = _project_root_for_config(path)
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    model = ModelConfig(**raw.pop("model", {}))
    sae = SAEConfig(**raw.pop("sae", {}))
    evaluation = EvaluationConfig(**raw.pop("evaluation", {}))
    config = ExperimentConfig(model=model, sae=sae, evaluation=evaluation, **raw)
    if config.dataset_path and not Path(config.dataset_path).is_absolute():
        config.dataset_path = str((root / config.dataset_path).resolve())
    if config.model.cache_dir and not Path(config.model.cache_dir).is_absolute():
        config.model.cache_dir = str((root / config.model.cache_dir).resolve())
    if os.environ.get("GLASSBOX_MODEL_DEVICE"):
        config.model.device = os.environ["GLASSBOX_MODEL_DEVICE"]
    if os.environ.get("GLASSBOX_SAE_DEVICE"):
        config.sae.device = os.environ["GLASSBOX_SAE_DEVICE"]
    if os.environ.get("GLASSBOX_HF_CACHE_DIR"):
        config.model.cache_dir = os.environ["GLASSBOX_HF_CACHE_DIR"]
    return config
