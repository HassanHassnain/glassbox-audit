from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..analysis import refresh_failure_analysis
from ..config import config_from_manifest as _config_from_manifest
from ..evaluation import comparison
from ..models import load_model
from ..pipeline import _random_sae_controls, _serialize_outputs
from ..types import ModelOutput, PromptRecord
from ..utils import read_json, seed_everything, write_json
from . import AblateSAEFeatures, AddDirection


def _outputs_from_dict(rows: list[dict[str, Any]]) -> list[ModelOutput]:
    return [
        ModelOutput(
            record_id=str(row["record_id"]),
            behavior_score=float(row["behavior_score"]),
            capability_nll=float(row["capability_nll"]),
            response=str(row.get("response", "")),
        )
        for row in rows
    ]


def _sae_shape_and_center(path: Path) -> tuple[int, int, int, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    state_dict = raw["state_dict"]
    center = state_dict.get("activation_center")
    if center is None:
        raise ValueError(f"{path} does not contain a trained SAE activation_center buffer")
    return int(raw["input_dim"]), int(raw["n_features"]), int(raw["top_k"]), center


def refresh_random_sae_controls(
    artifact_dir: str | Path,
    *,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Rerun only the frozen random-SAE controls with stricter matched preprocessing.

    The function does not recompute layer localization, SAE training, feature selection,
    thresholds, steering scales, or any simple baseline. It reloads the frozen held-out test
    records and baseline outputs, evaluates architecture-matched untrained SAE controls with the
    trained activation center, and refreshes derived negative-result artifacts.
    """

    source_dir = Path(artifact_dir)
    target_dir = Path(output_dir) if output_dir is not None else source_dir
    if target_dir != source_dir:
        raise NotImplementedError("Writing refreshed controls to a separate directory is not yet supported")

    manifest = read_json(source_dir / "manifest.json")
    summary = read_json(source_dir / "summary.json")
    interventions = read_json(source_dir / "interventions.json")
    config = _config_from_manifest(manifest)
    config.evaluation.threshold = float(summary["behavior_threshold"])
    if device is not None:
        config.model.device = device
        config.sae.device = device

    seed_everything(config.seed)
    records = [
        PromptRecord.from_dict(row)
        for row in read_json(source_dir / "records.json")
        if row["split"] == "test"
    ]
    baseline = _outputs_from_dict(interventions["baseline_outputs"])
    input_dim, n_features, top_k, activation_center = _sae_shape_and_center(source_dir / "sae.pt")
    if n_features % input_dim != 0:
        raise ValueError("Random-SAE refresh currently expects n_features to equal input_dim * expansion_factor")
    random_sae, random_sae_controls = _random_sae_controls(
        input_dim=input_dim,
        expansion_factor=n_features // input_dim,
        top_k=top_k,
        count=config.evaluation.random_controls,
        seed=config.seed + 200,
        activation_center=activation_center,
    )

    model = load_model(config.model, config.seed)
    target_layer = int(summary["target_layer"])
    refreshed_rows = []
    for feature_index, direction in random_sae_controls:
        steering_outputs = model.evaluate(
            records,
            target_layer,
            AddDirection(direction, config.evaluation.control_scale),
        )
        steering = comparison(
            records,
            baseline,
            steering_outputs,
            config.evaluation.threshold,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed,
        )
        steering.update(
            {
                "method": "random_sae",
                "control_index": feature_index,
                "scale": config.evaluation.control_scale,
                "intervention": "steering_control",
                "outputs": _serialize_outputs(steering_outputs),
            }
        )
        refreshed_rows.append(steering)

        ablation_outputs = model.evaluate(
            records,
            target_layer,
            AblateSAEFeatures(random_sae, [feature_index]),
        )
        ablation = comparison(
            records,
            baseline,
            ablation_outputs,
            config.evaluation.threshold,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed,
        )
        ablation.update(
            {
                "method": "random_sae",
                "control_index": feature_index,
                "scale": None,
                "intervention": "feature_ablation_control",
                "outputs": _serialize_outputs(ablation_outputs),
            }
        )
        refreshed_rows.append(ablation)

    interventions["random_controls"] = [
        row for row in interventions["random_controls"] if row["method"] != "random_sae"
    ] + refreshed_rows
    interventions["random_sae_control_protocol"] = {
        "status": "matched_center_zero_encoder_bias",
        "architecture": {
            "input_dim": input_dim,
            "n_features": n_features,
            "top_k": top_k,
        },
        "activation_center": "trained_sae_activation_center",
        "encoder_bias": "zeroed",
        "latent_intervention": "literal_encode_zero_selected_latent_decode",
        "selection_policy": "frozen; no feature, layer, threshold, or scale retuning",
    }
    write_json(source_dir / "interventions.json", interventions)

    failure = refresh_failure_analysis(source_dir)
    summary = read_json(source_dir / "summary.json")
    summary["random_sae_control_protocol"] = interventions["random_sae_control_protocol"]
    write_json(source_dir / "summary.json", summary)
    return {
        "artifact_dir": str(source_dir),
        "refreshed_random_sae_rows": len(refreshed_rows),
        "random_sae_control_protocol": interventions["random_sae_control_protocol"],
        "random_ablation_control_comparison": failure["random_ablation_control_comparison"],
        "negative_findings": failure["negative_findings"],
    }
