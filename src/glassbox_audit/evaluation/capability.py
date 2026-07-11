from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..config import config_from_manifest
from ..interventions import AblateDirection, AblateSAEFeatures
from ..models import AuditModel, load_model
from ..sae import load_sae_from_state
from ..utils import read_json, seed_everything, write_json


def frozen_token_blocks(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    block_size: int,
    max_blocks: int,
    seed: int,
) -> tuple[list[list[int]], dict[str, object]]:
    """Build a deterministic model-tokenized subset from a benchmark text split."""

    if block_size < 8:
        raise ValueError("block_size must be at least 8")
    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    joined = "\n\n".join(str(text) for text in texts if str(text).strip())
    token_ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    window = block_size + 1
    available = len(token_ids) // window
    if available < 1:
        raise ValueError("Benchmark split does not contain enough tokens for one complete block")
    indices = list(range(available))
    random.Random(seed).shuffle(indices)
    selected_indices = sorted(indices[: min(max_blocks, available)])
    blocks = [token_ids[index * window : (index + 1) * window] for index in selected_indices]
    encoded = json.dumps(blocks, separators=(",", ":")).encode("utf-8")
    return blocks, {
        "selection_policy": "seeded_sample_without_replacement_of_nonoverlapping_token_blocks",
        "seed": seed,
        "block_size_scored_tokens": block_size,
        "available_blocks": available,
        "selected_blocks": len(blocks),
        "selected_block_indices": selected_indices,
        "subset_token_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _aggregate(rows: Sequence[dict[str, float | int]], indices: Sequence[int]) -> float:
    loss = sum(float(rows[index]["nll_sum"]) for index in indices)
    tokens = sum(int(rows[index]["token_count"]) for index in indices)
    return loss / tokens


def _interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    ordered = sorted(values)
    alpha = (1.0 - confidence) / 2.0

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return percentile(alpha), percentile(1.0 - alpha)


def summarize_lm_conditions(
    condition_rows: dict[str, list[dict[str, float | int]]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    """Compute token-weighted estimates and paired block-bootstrap intervals."""

    if "baseline" not in condition_rows:
        raise ValueError("A baseline condition is required")
    count = len(condition_rows["baseline"])
    if count < 1 or any(len(rows) != count for rows in condition_rows.values()):
        raise ValueError("All conditions must contain the same non-zero number of blocks")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    full = list(range(count))
    estimates = {name: _aggregate(rows, full) for name, rows in condition_rows.items()}
    draws: dict[str, list[float]] = {name: [] for name in condition_rows}
    delta_draws: dict[str, list[float]] = {
        name: [] for name in condition_rows if name != "baseline"
    }
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(count) for _ in range(count)]
        sampled = {name: _aggregate(rows, indices) for name, rows in condition_rows.items()}
        for name, value in sampled.items():
            draws[name].append(value)
        for name in delta_draws:
            delta_draws[name].append(sampled[name] - sampled["baseline"])

    result: dict[str, object] = {}
    baseline_nll = estimates["baseline"]
    for name, nll in estimates.items():
        nll_low, nll_high = _interval(draws[name], confidence)
        row: dict[str, object] = {
            "nll_ci": {"estimate": nll, "low": nll_low, "high": nll_high},
            "perplexity_ci": {
                "estimate": math.exp(nll),
                "low": math.exp(nll_low),
                "high": math.exp(nll_high),
            },
            "blocks": count,
            "scored_tokens": sum(int(item["token_count"]) for item in condition_rows[name]),
        }
        if name != "baseline":
            delta = nll - baseline_nll
            delta_low, delta_high = _interval(delta_draws[name], confidence)
            row["delta_nll_ci"] = {
                "estimate": delta,
                "low": delta_low,
                "high": delta_high,
            }
            row["perplexity_ratio_ci"] = {
                "estimate": math.exp(delta),
                "low": math.exp(delta_low),
                "high": math.exp(delta_high),
            }
        result[name] = row
    return result


def _load_wikitext(
    dataset_name: str,
    dataset_config: str,
    split: str,
    revision: str | None,
) -> tuple[list[str], dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'WikiText evaluation requires the data extra: pip install -e ".[data]"'
        ) from exc
    kwargs: dict[str, object] = {"split": split}
    if revision is not None:
        kwargs["revision"] = revision
    dataset = load_dataset(dataset_name, dataset_config, **kwargs)
    if "text" not in dataset.column_names:
        raise ValueError(f"{dataset_name}/{dataset_config} has no 'text' column")
    return list(dataset["text"]), {
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "revision": revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "source_rows": len(dataset),
    }


def run_wikitext_capability_eval(
    artifact_dir: str | Path,
    output_path: str | Path,
    *,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    revision: str | None = None,
    max_blocks: int = 128,
    block_size: int = 256,
    batch_size: int = 4,
    bootstrap_samples: int | None = None,
    confidence: float | None = None,
    seed: int | None = None,
    device: str | None = None,
    expected_subset_sha256: str | None = None,
    model: AuditModel | None = None,
    texts: Sequence[str] | None = None,
) -> dict[str, object]:
    """Evaluate frozen audit interventions on unrelated WikiText-2 language modeling."""

    artifact_dir = Path(artifact_dir)
    manifest = read_json(artifact_dir / "manifest.json")
    summary = read_json(artifact_dir / "summary.json")
    config = config_from_manifest(manifest)
    if device is not None:
        config.model.device = device
    eval_seed = config.seed if seed is None else seed
    samples = config.evaluation.bootstrap_samples if bootstrap_samples is None else bootstrap_samples
    level = config.evaluation.confidence if confidence is None else confidence
    seed_everything(eval_seed)
    model = model or load_model(config.model, config.seed)
    if texts is None:
        texts, source = _load_wikitext(dataset_name, dataset_config, split, revision)
    else:
        source = {
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "revision": revision,
            "dataset_fingerprint": None,
            "source_rows": len(texts),
        }
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("WikiText capability evaluation requires a model with a tokenizer")
    blocks, subset = frozen_token_blocks(
        texts, tokenizer, block_size=block_size, max_blocks=max_blocks, seed=eval_seed
    )
    if (
        expected_subset_sha256 is not None
        and subset["subset_token_sha256"] != expected_subset_sha256
    ):
        raise ValueError(
            "Frozen WikiText subset hash mismatch: "
            f"expected {expected_subset_sha256}, got {subset['subset_token_sha256']}"
        )

    discovery = torch.load(artifact_dir / "discovery_tensors.pt", map_location="cpu")
    directions = discovery["directions"]
    sae = load_sae_from_state(str(artifact_dir / "sae.pt"))
    selected_features = [int(index) for index in summary["selected_features"]]
    target_layer = int(summary["target_layer"])
    interventions = {
        "baseline": None,
        "mean_difference_direction_ablation": AblateDirection(directions["mean_difference"]),
        "sae_feature_ablation": AblateSAEFeatures(sae, selected_features),
    }
    condition_rows = {
        name: model.evaluate_lm_token_blocks(
            blocks, target_layer if intervention is not None else None, intervention, batch_size=batch_size
        )
        for name, intervention in interventions.items()
    }
    per_block = []
    selected_indices = subset["selected_block_indices"]
    for position, block in enumerate(blocks):
        block_results = {}
        for name, rows in condition_rows.items():
            token_count = int(rows[position]["token_count"])
            nll_sum = float(rows[position]["nll_sum"])
            block_results[name] = {
                "nll_sum": nll_sum,
                "mean_nll": nll_sum / token_count,
            }
        per_block.append(
            {
                "block_index": selected_indices[position],
                "token_count": int(condition_rows["baseline"][position]["token_count"]),
                "token_sha256": hashlib.sha256(
                    json.dumps(block, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "conditions": block_results,
            }
        )
    results = summarize_lm_conditions(
        condition_rows,
        bootstrap_samples=samples,
        confidence=level,
        seed=eval_seed + 9100,
    )
    artifact: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation": "wikitext_perplexity",
        "status": "completed",
        "model": model.name,
        "source_audit_artifacts": str(artifact_dir),
        "target_layer": target_layer,
        "selected_sae_features": selected_features,
        "benchmark": {**source, **subset},
        "statistics": {
            "unit": "nonoverlapping_token_block",
            "bootstrap": "paired_nonparametric_block_bootstrap",
            "bootstrap_samples": samples,
            "confidence": level,
            "nll_aggregation": "token_weighted",
        },
        "conditions": results,
        "per_block": per_block,
        "claim_scope": (
            "Unrelated general-language modeling on a frozen WikiText-2 subset; this does not "
            "establish preservation of every downstream capability."
        ),
    }
    write_json(output_path, artifact)
    return artifact
