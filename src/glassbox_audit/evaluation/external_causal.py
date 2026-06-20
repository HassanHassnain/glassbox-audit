from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..analysis import _method_utility
from ..interventions import AblateDirection, AblateSAEFeatures, AddDirection
from ..interventions.control_refresh import _config_from_manifest
from ..models import load_model
from ..pipeline import _random_sae_controls, _serialize_outputs
from ..sae import load_sae_from_state
from ..sae.discovery import random_directions
from ..types import ModelOutput, PromptRecord
from ..utils import read_json, seed_everything, write_json
from .core import calculate_metrics, comparison
from .external_eval import bounded_external_sample, load_external_eval_records


def _estimate(row: dict[str, Any], key: str) -> float:
    return float(row[key]["estimate"])


def _outputs_by_id(outputs: list[ModelOutput]) -> dict[str, ModelOutput]:
    return {output.record_id: output for output in outputs}


def _rate_delta(
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    intervened: list[ModelOutput],
    threshold: float,
    *,
    harmful: bool,
) -> float:
    before = _outputs_by_id(baseline)
    after = _outputs_by_id(intervened)
    ids = [record.id for record in records if record.harmful is harmful]
    if not ids:
        return 0.0
    before_rate = sum(before[row_id].behavior_score >= threshold for row_id in ids) / len(ids)
    after_rate = sum(after[row_id].behavior_score >= threshold for row_id in ids) / len(ids)
    return after_rate - before_rate


def _attach_transfer_fields(
    row: dict[str, Any],
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    outputs: list[ModelOutput],
    threshold: float,
) -> None:
    row["toxic_refusal_rate_delta"] = _rate_delta(
        records, baseline, outputs, threshold, harmful=True
    )
    row["hard_benign_refusal_rate_delta"] = _rate_delta(
        records, baseline, outputs, threshold, harmful=False
    )
    row["specificity_adjusted_utility"] = _method_utility(row)


def _compare(
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    outputs: list[ModelOutput],
    threshold: float,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    row = comparison(records, baseline, outputs, threshold, bootstrap_samples, confidence, seed)
    row["outputs"] = _serialize_outputs(outputs)
    _attach_transfer_fields(row, records, baseline, outputs, threshold)
    return row


def _control_p_value(effect: float, controls: list[float]) -> float | None:
    if not controls:
        return None
    return (1 + sum(value <= effect for value in controls)) / (len(controls) + 1)


def _summarize_methods(
    causal_tests: dict[str, dict[str, Any]],
    random_controls: list[dict[str, Any]],
) -> dict[str, Any]:
    random_direction_effects = [
        _estimate(row, "delta_mean_harmful_score_ci")
        for row in random_controls
        if row["method"] == "random_direction" and row["intervention"] == "ablation_control"
    ]
    random_sae_effects = [
        _estimate(row, "delta_mean_harmful_score_ci")
        for row in random_controls
        if row["method"] == "random_sae" and row["intervention"] == "feature_ablation_control"
    ]
    main = {}
    for name, row in causal_tests.items():
        main[name] = {
            "harmful_score_delta": _estimate(row, "delta_mean_harmful_score_ci"),
            "harmful_score_ci": [
                float(row["delta_mean_harmful_score_ci"]["low"]),
                float(row["delta_mean_harmful_score_ci"]["high"]),
            ],
            "benign_score_delta": _estimate(row, "delta_mean_benign_score_ci"),
            "capability_nll_delta": _estimate(row, "delta_capability_nll_ci"),
            "toxic_refusal_rate_delta": row["toxic_refusal_rate_delta"],
            "hard_benign_refusal_rate_delta": row["hard_benign_refusal_rate_delta"],
            "specificity_adjusted_utility": row["specificity_adjusted_utility"],
        }
    sae_effect = main["sae_feature_ablation"]["harmful_score_delta"]
    return {
        "main_methods": main,
        "random_controls": {
            "random_direction_ablation": {
                "n": len(random_direction_effects),
                "min": min(random_direction_effects) if random_direction_effects else None,
                "max": max(random_direction_effects) if random_direction_effects else None,
                "sae_feature_empirical_p": _control_p_value(sae_effect, random_direction_effects),
            },
            "random_sae_feature_ablation": {
                "n": len(random_sae_effects),
                "min": min(random_sae_effects) if random_sae_effects else None,
                "max": max(random_sae_effects) if random_sae_effects else None,
                "sae_feature_empirical_p": _control_p_value(sae_effect, random_sae_effects),
            },
        },
    }


def run_external_causal_transfer(
    *,
    artifact_dir: str | Path,
    normalized_paths: list[str | Path],
    output_path: str | Path,
    max_records_per_label: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    random_controls: int | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    manifest = read_json(artifact_dir / "manifest.json")
    summary = read_json(artifact_dir / "summary.json")
    source_interventions = read_json(artifact_dir / "interventions.json")
    config = _config_from_manifest(manifest)
    if seed is not None:
        config.seed = seed
    config.evaluation.threshold = float(summary["behavior_threshold"])
    if random_controls is not None:
        config.evaluation.random_controls = random_controls
    if device is not None:
        config.model.device = device
        config.sae.device = device
    seed_everything(config.seed)

    records = bounded_external_sample(
        load_external_eval_records(normalized_paths),
        max_records_per_label=max_records_per_label,
        seed=config.seed,
    )
    if not any(record.harmful for record in records) or not any(not record.harmful for record in records):
        raise ValueError("External causal transfer requires both harmful and benign records")

    model = load_model(config.model, config.seed)
    target_layer = int(summary["target_layer"])
    tensors = torch.load(artifact_dir / "discovery_tensors.pt", map_location="cpu")
    directions = {
        name: tensor.detach().float()
        for name, tensor in tensors["directions"].items()
        if name in {"sae_features", "mean_difference", "linear_probe"}
    }
    sae = load_sae_from_state(str(artifact_dir / "sae.pt"))
    selected_features = [int(index) for index in summary["selected_features"]]
    selected_scales = source_interventions["validation_selected_scales"]
    baseline = model.evaluate(records)
    baseline_metrics = calculate_metrics(records, baseline, config.evaluation.threshold)

    causal_tests: dict[str, dict[str, Any]] = {}
    causal_tests["sae_feature_ablation"] = _compare(
        records,
        baseline,
        model.evaluate(records, target_layer, AblateSAEFeatures(sae, selected_features)),
        config.evaluation.threshold,
        config.evaluation.bootstrap_samples,
        config.evaluation.confidence,
        config.seed + 1000,
    )
    for method in ["sae_features", "mean_difference", "linear_probe"]:
        direction = directions[method]
        steering_outputs = model.evaluate(
            records,
            target_layer,
            AddDirection(direction, float(selected_scales[method])),
        )
        causal_tests[f"{method}_steering"] = _compare(
            records,
            baseline,
            steering_outputs,
            config.evaluation.threshold,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed + 1010 + len(causal_tests),
        )
        ablation_outputs = model.evaluate(records, target_layer, AblateDirection(direction))
        causal_tests[f"{method}_direction_ablation"] = _compare(
            records,
            baseline,
            ablation_outputs,
            config.evaluation.threshold,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed + 1020 + len(causal_tests),
        )

    random_control_rows: list[dict[str, Any]] = []
    for index, direction in enumerate(
        random_directions(model.hidden_size, config.evaluation.random_controls, config.seed + 1100)
    ):
        for intervention_name, intervention, scale in [
            ("steering_control", AddDirection(direction, config.evaluation.control_scale), config.evaluation.control_scale),
            ("ablation_control", AblateDirection(direction), None),
        ]:
            outputs = model.evaluate(records, target_layer, intervention)
            row = _compare(
                records,
                baseline,
                outputs,
                config.evaluation.threshold,
                config.evaluation.bootstrap_samples,
                config.evaluation.confidence,
                config.seed + 1200 + len(random_control_rows),
            )
            row.update(
                {
                    "method": "random_direction",
                    "control_index": index,
                    "intervention": intervention_name,
                    "scale": scale,
                }
            )
            random_control_rows.append(row)

    random_sae, random_sae_controls = _random_sae_controls(
        model.hidden_size,
        config.sae.expansion_factor,
        min(config.sae.top_k, model.hidden_size * config.sae.expansion_factor),
        config.evaluation.random_controls,
        config.seed + 1300,
        sae.activation_center,
    )
    for feature_index, direction in random_sae_controls:
        for intervention_name, intervention, scale in [
            ("steering_control", AddDirection(direction, config.evaluation.control_scale), config.evaluation.control_scale),
            ("feature_ablation_control", AblateSAEFeatures(random_sae, [feature_index]), None),
        ]:
            outputs = model.evaluate(records, target_layer, intervention)
            row = _compare(
                records,
                baseline,
                outputs,
                config.evaluation.threshold,
                config.evaluation.bootstrap_samples,
                config.evaluation.confidence,
                config.seed + 1400 + len(random_control_rows),
            )
            row.update(
                {
                    "method": "random_sae",
                    "control_index": feature_index,
                    "intervention": intervention_name,
                    "scale": scale,
                }
            )
            random_control_rows.append(row)

    method_summary = _summarize_methods(causal_tests, random_control_rows)
    artifact = {
        "schema_version": "1.0",
        "artifact_notice": (
            "External causal-transfer evaluation from frozen controlled-corpus artifacts. "
            "No external rediscovery or retuning was performed. OR-Bench rows are unpaired, so "
            "this is reported separately from paired controlled-corpus causal results."
        ),
        "source_artifact": str(artifact_dir),
        "model": summary["model"],
        "target_layer": target_layer,
        "selected_features": selected_features,
        "threshold": config.evaluation.threshold,
        "selected_scales_source": str(artifact_dir / "interventions.json"),
        "normalized_inputs": [str(path) for path in normalized_paths],
        "max_records_per_label": max_records_per_label,
        "baseline_metrics": baseline_metrics,
        "baseline_outputs": _serialize_outputs(baseline),
        "causal_tests": causal_tests,
        "random_controls": random_control_rows,
        "summary": method_summary,
        "random_sae_control_protocol": {
            "status": "matched_center_zero_encoder_bias",
            "activation_center": "trained_sae_activation_center",
            "encoder_bias": "zeroed",
            "latent_intervention": "literal_encode_zero_selected_latent_decode",
            "selection_policy": "frozen controlled-corpus artifact; no external rediscovery",
        },
    }
    write_json(output_path, artifact)
    snapshot = {
        key: value
        for key, value in artifact.items()
        if key not in {"baseline_outputs", "causal_tests", "random_controls"}
    }
    snapshot["records_artifact"] = str(output_path)
    snapshot_path = Path("results/external-causal") / (Path(output_path).stem + "_summary.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    return artifact
