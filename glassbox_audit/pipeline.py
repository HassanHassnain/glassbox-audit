from __future__ import annotations

import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .analysis import (
    build_equivalence_analysis,
    build_failure_analysis,
    paired_method_comparisons,
)
from .config import ExperimentConfig, load_config
from .data import dataset_summary, load_records, split_records
from .discovery import (
    combined_feature_direction,
    feature_table,
    layer_scan,
    mean_difference_direction,
    random_directions,
    train_l1_feature_weights,
    train_probe_direction,
    weighted_feature_direction,
)
from .evaluation import calculate_metrics, comparison, select_threshold
from .interventions import (
    AblateDirection,
    AblateSAEFeatures,
    AddDirection,
    PatchProjection,
    normalized,
)
from .modeling import AuditModel, load_model
from .preregistration import preregistration_metadata
from .report import write_report
from .sae import (
    SAETrainingResult,
    TopKSparseAutoencoder,
    evaluate_sae_metrics,
    load_saelens_sae,
    sae_state_dict,
    train_sae,
)
from .types import ModelOutput, PromptRecord
from .utils import read_json, seed_everything, write_json


def _labels(records: list[PromptRecord]) -> torch.Tensor:
    return torch.tensor([record.harmful for record in records], dtype=torch.bool)


def _orient(direction: torch.Tensor, activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    direction = normalized(direction)
    positive = (activations[labels] @ direction).mean()
    negative = (activations[~labels] @ direction).mean()
    return direction if positive >= negative else -direction


def _serialize_outputs(outputs: list[ModelOutput]) -> list[dict[str, object]]:
    return [output.to_dict() for output in outputs]


def _stamp_preregistration(
    artifact: dict[str, object],
    preregistration: dict[str, object] | None,
) -> dict[str, object]:
    if preregistration is None:
        return artifact
    return {
        **artifact,
        "preregistration": preregistration,
        "preregistration_hash": preregistration["hash"],
    }


def _run_comparison(
    config: ExperimentConfig,
    model: AuditModel,
    test_records: list[PromptRecord],
    baseline: list[ModelOutput],
    target_layer: int,
    intervention,
) -> dict[str, object]:
    intervened = model.evaluate(test_records, target_layer, intervention)
    result = comparison(
        test_records,
        baseline,
        intervened,
        config.evaluation.threshold,
        config.evaluation.bootstrap_samples,
        config.evaluation.confidence,
        config.seed,
    )
    result["outputs"] = _serialize_outputs(intervened)
    return result


def _control_checkpoint_name(control_name: str, index: int, intervention_name: str) -> str:
    return f"{control_name}_{index}_{intervention_name}.json".replace("/", "_")


def _random_sae_controls(
    input_dim: int,
    expansion_factor: int,
    top_k: int,
    count: int,
    seed: int,
    activation_center: torch.Tensor | None = None,
) -> tuple[TopKSparseAutoencoder, list[tuple[int, torch.Tensor]]]:
    torch.manual_seed(seed)
    random_sae = TopKSparseAutoencoder(input_dim, input_dim * expansion_factor, top_k)
    random_sae.encoder.bias.data.zero_()
    indices = torch.randperm(random_sae.n_features)[:count].tolist()
    center = (
        activation_center.detach().float().cpu().clone()
        if activation_center is not None
        else torch.zeros(input_dim)
    )
    random_sae.register_buffer("activation_center", center)
    return random_sae, [
        (index, normalized(random_sae.feature_direction(index))) for index in indices
    ]


def _public_sae_controls(
    sae,
    count: int,
    seed: int,
) -> list[tuple[int, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(sae.n_features, generator=generator)[:count].tolist()
    return [(int(index), normalized(sae.feature_direction(int(index)))) for index in indices]


def _sae_source_details(config: ExperimentConfig, sae) -> dict[str, Any]:
    metadata = dict(getattr(sae, "source_metadata", {}) or {})
    source = str(metadata.get("source") or config.sae.source)
    details: dict[str, Any] = {
        "source": source,
        "input_dim": int(sae.input_dim),
        "n_features": int(sae.n_features),
        "top_k": int(sae.top_k),
    }
    if source == "saelens":
        hook_layer = metadata.get("hook_layer")
        details.update(
            {
                "release": metadata.get("release") or config.sae.saelens_release,
                "sae_id": metadata.get("sae_id") or config.sae.saelens_sae_id,
                "hook_layer": hook_layer if hook_layer is not None else config.sae.saelens_layer,
                "hook_name": metadata.get("hook_name"),
                "architecture": metadata.get("architecture"),
            }
        )
    else:
        details.update(
            {
                "expansion_factor": config.sae.expansion_factor,
                "training": "local_train_split_topk_sae",
            }
        )
    return details


def _random_sae_control_protocol(
    *,
    sae_source: dict[str, Any],
    config: ExperimentConfig,
    hidden_size: int,
) -> dict[str, Any]:
    if sae_source["source"] == "saelens":
        return {
            "status": "matched_public_dictionary_random_latents",
            "source": sae_source,
            "architecture": {
                "input_dim": sae_source["input_dim"],
                "n_features": sae_source["n_features"],
                "top_k": sae_source["top_k"],
            },
            "latent_source": "same SAELens dictionary as the target intervention",
            "latent_intervention": "literal_encode_zero_selected_latent_decode",
            "selection_policy": "train-only target features; controls are seeded random public latents",
        }
    return {
        "status": "matched_center_zero_encoder_bias",
        "architecture": {
            "input_dim": hidden_size,
            "n_features": hidden_size * config.sae.expansion_factor,
            "top_k": min(config.sae.top_k, hidden_size * config.sae.expansion_factor),
        },
        "activation_center": "trained_sae_activation_center",
        "encoder_bias": "zeroed",
        "latent_intervention": "literal_encode_zero_selected_latent_decode",
        "selection_policy": "frozen; no feature, layer, threshold, or scale retuning",
    }


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return slug or "unknown"


def _effect_ci(row: dict[str, Any], key: str = "delta_mean_harmful_score_ci") -> dict[str, float]:
    value = row[key]
    return {
        "estimate": float(value["estimate"]),
        "low": float(value["low"]),
        "high": float(value["high"]),
    }


def _control_distribution(
    rows: list[dict[str, Any]],
    *,
    method: str,
    intervention: str,
) -> dict[str, Any]:
    values = [
        float(row["delta_mean_harmful_score_ci"]["estimate"])
        for row in rows
        if row.get("method") == method and row.get("intervention") == intervention
    ]
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def _refresh_sae_zoo_league_table(directory: Path) -> None:
    artifacts = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "league_table.json":
            continue
        try:
            artifact = read_json(path)
        except Exception:
            continue
        if artifact.get("schema_version") != "1.0":
            continue
        if "held_out" not in artifact or "sae_source" not in artifact:
            continue
        artifacts.append(artifact)
    rows = []
    for artifact in artifacts:
        held_out = artifact["held_out"]
        comparisons = artifact["comparisons"]
        controls = artifact["matched_controls"]["random_sae_feature_ablation"]
        rows.append(
            {
                "model": artifact["model"],
                "release": artifact["sae_source"].get("release"),
                "sae_id": artifact["sae_source"].get("sae_id"),
                "layer": artifact["target_layer"],
                "behavior": artifact["behavior"],
                "sae_harmful_delta": held_out["sae_feature_ablation"]["estimate"],
                "mean_difference_harmful_delta": held_out["mean_difference_ablation"]["estimate"],
                "linear_probe_harmful_delta": held_out["linear_probe_ablation"]["estimate"],
                "beats_mean_difference": comparisons["beats_mean_difference"],
                "beats_linear_probe": comparisons["beats_linear_probe"],
                "random_sae_controls_n": controls["n"],
            }
        )
    rows.sort(key=lambda row: (str(row["model"]), str(row["release"]), str(row["sae_id"])))
    write_json(directory / "league_table.json", {"schema_version": "1.0", "rows": rows})
    table = [
        "# SAE Zoo League Table",
        "",
        "| Model | Release | SAE id | Layer | SAE delta | Mean delta | Probe delta | Beats mean? | Beats probe? | Controls |",
        "|---|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['model']} | {row['release']} | {row['sae_id']} | {row['layer']} | "
            f"{row['sae_harmful_delta']:+.4f} | {row['mean_difference_harmful_delta']:+.4f} | "
            f"{row['linear_probe_harmful_delta']:+.4f} | {row['beats_mean_difference']} | "
            f"{row['beats_linear_probe']} | {row['random_sae_controls_n']} |"
        )
    (directory / "league_table.md").write_text("\n".join(table) + "\n")


def _write_sae_zoo_result(
    *,
    config: ExperimentConfig,
    model: AuditModel,
    target_layer: int,
    selected_features: list[int],
    sae_source: dict[str, Any],
    sae_metrics: dict[str, float],
    causal_tests: dict[str, dict[str, Any]],
    random_control_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any] | None:
    if sae_source["source"] != "saelens":
        return None
    sae_effect = _effect_ci(causal_tests["sae_feature_ablation"])
    mean_effect = _effect_ci(causal_tests["mean_difference_direction_ablation"])
    probe_effect = _effect_ci(causal_tests["linear_probe_direction_ablation"])
    artifact = {
        "schema_version": "1.0",
        "artifact_notice": (
            "A1 public SAE-zoo evaluation. Feature ranking used train activations only; "
            "controls draw random latents from the same public SAE dictionary."
        ),
        "model": model.name,
        "behavior": config.behavior,
        "target_layer": target_layer,
        "selected_features": selected_features,
        "sae_source": sae_source,
        "sae_metrics": sae_metrics,
        "held_out": {
            "sae_feature_ablation": sae_effect,
            "mean_difference_ablation": mean_effect,
            "linear_probe_ablation": probe_effect,
        },
        "matched_controls": {
            "random_direction_ablation": _control_distribution(
                random_control_rows,
                method="random_direction",
                intervention="ablation_control",
            ),
            "random_sae_feature_ablation": _control_distribution(
                random_control_rows,
                method="random_sae",
                intervention="feature_ablation_control",
            ),
        },
        "comparisons": {
            "beats_mean_difference": sae_effect["estimate"] < mean_effect["estimate"],
            "beats_linear_probe": sae_effect["estimate"] < probe_effect["estimate"],
        },
        "source_artifact_dir": str(output_dir),
    }
    zoo_dir = Path("results") / "sae-zoo"
    zoo_dir.mkdir(parents=True, exist_ok=True)
    path = zoo_dir / (
        f"{_safe_slug(model.name)}_{_safe_slug(sae_source.get('release'))}_"
        f"{_safe_slug(sae_source.get('sae_id'))}.json"
    )
    write_json(path, artifact)
    _refresh_sae_zoo_league_table(zoo_dir)
    return {"path": str(path), "league_table": str(zoo_dir / "league_table.json")}


def _selection_score(result: dict[str, object]) -> float:
    harmful = float(result["delta_mean_harmful_score_ci"]["estimate"])
    benign = abs(float(result["delta_mean_benign_score_ci"]["estimate"]))
    capability = max(0.0, float(result["delta_capability_nll_ci"]["estimate"]))
    return harmful + benign + 0.2 * capability


def _validate_real_audit_config(config: ExperimentConfig, records: list[PromptRecord]) -> list[str]:
    checks = []
    if config.mode != "real_audit":
        return checks
    requirements = [
        (config.sae.source in {"trained", "saelens"}, "sae.source must be trained or saelens"),
        (
            config.target_layer is None or config.sae.source == "saelens",
            "target_layer must be null for trained SAEs so layer selection uses train only",
        ),
        (
            config.sae.source != "saelens"
            or (
                config.sae.saelens_release is not None
                and config.sae.saelens_sae_id is not None
                and config.sae.saelens_layer is not None
            ),
            "SAELens runs must predeclare sae.saelens_release, sae.saelens_sae_id, and sae.saelens_layer",
        ),
        (
            config.sae.source != "saelens" or config.sae.saelens_layer in config.model.layers,
            "SAELens layer must be included in model.layers",
        ),
        (len(config.model.layers) >= 3, "at least three layers must be scanned"),
        (config.evaluation.threshold is None, "threshold must be selected on validation"),
        (config.evaluation.layerwise_controls, "layerwise controls must be enabled"),
        (config.evaluation.random_controls >= 4, "at least four random controls are required"),
        (config.evaluation.bootstrap_samples >= 200, "at least 200 bootstrap samples are required"),
        (config.evaluation.equivalence_margin > 0, "equivalence margin must be positive"),
        (config.evaluation.inferiority_margin > 0, "inferiority margin must be positive"),
        (0 < config.evaluation.test_alpha < 0.5, "test alpha must be between 0 and 0.5"),
        (config.evaluation.power_effect_size > 0, "power effect size must be positive"),
        (config.evaluation.power_assumed_pair_sd > 0, "power assumed pair SD must be positive"),
        (config.evaluation.power_target >= 0.8, "power target must be at least 0.8"),
        (config.evaluation.power_target_n >= 500, "power target n must be at least 500 pairs"),
        (
            bool(config.evaluation.feature_budgets)
            and all(budget > 0 for budget in config.evaluation.feature_budgets),
            "feature budgets must be positive",
        ),
        (
            config.evaluation.feature_budgets
            == sorted(set(config.evaluation.feature_budgets)),
            "feature budgets must be sorted and unique",
        ),
        (
            config.evaluation.feature_budget_l1_coefficient >= 0,
            "feature-budget L1 coefficient must be nonnegative",
        ),
        (
            config.evaluation.feature_budget_l1_epochs > 0,
            "feature-budget L1 epochs must be positive",
        ),
        (
            config.sae.samples_per_feature is None or config.sae.samples_per_feature > 0,
            "SAE samples_per_feature must be positive when configured",
        ),
        (config.sae.tokens_per_prompt > 1, "SAE must train on multiple train-prompt token activations"),
        (
            all(len(split_records(records, split)) >= 20 for split in ["train", "validation", "test"]),
            "each split must contain at least 20 records",
        ),
    ]
    failed = [message for passed, message in requirements if not passed]
    if failed:
        raise ValueError("Invalid real-audit configuration: " + "; ".join(failed))
    return [message for _passed, message in requirements]


def run_audit(
    config: ExperimentConfig | str | Path,
    output_dir: str | Path,
    model: AuditModel | None = None,
) -> dict[str, object]:
    if not isinstance(config, ExperimentConfig):
        config = load_config(config)
    preregistration = preregistration_metadata(
        config.preregistration_path,
        config.preregistration_hash,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    random_checkpoint_dir = checkpoint_dir / "random_controls"
    random_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_timings: dict[str, dict[str, object]] = {}
    stage_starts: dict[str, float] = {}

    def start_stage(name: str) -> None:
        stage_starts[name] = time.time()
        write_json(
            checkpoint_dir / "current_stage.json",
            {"stage": name, "status": "running", "started_at": stage_starts[name]},
        )

    def finish_stage(name: str, **extra: object) -> None:
        now = time.time()
        start = stage_starts.get(name, now)
        row: dict[str, object] = {
            "stage": name,
            "status": "completed",
            "started_at": start,
            "finished_at": now,
            "seconds": now - start,
            **extra,
        }
        stage_timings[name] = row
        write_json(checkpoint_dir / f"{name}.complete.json", row)
        write_json(checkpoint_dir / "stage_timings.json", stage_timings)
        with (checkpoint_dir / "progress.log").open("a") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} completed {name}\n")

    if config.sae.source not in {"trained", "saelens"}:
        raise ValueError("sae.source must be either 'trained' or 'saelens'")
    if config.sae.source == "saelens" and (
        not config.sae.saelens_release
        or not config.sae.saelens_sae_id
        or config.sae.saelens_layer is None
    ):
        raise ValueError(
            "sae.source=saelens requires sae.saelens_release, sae.saelens_sae_id, "
            "and sae.saelens_layer"
        )

    seed_everything(config.seed)

    start_stage("data_load_split_validation")
    records = load_records(config.dataset_path)
    real_audit_checks = _validate_real_audit_config(config, records)
    train_records = split_records(records, "train")
    validation_records = split_records(records, "validation")
    test_records = split_records(records, "test")
    discovery_records = train_records
    labels = _labels(discovery_records)
    finish_stage(
        "data_load_split_validation",
        train_records=len(train_records),
        validation_records=len(validation_records),
        test_records=len(test_records),
    )

    start_stage("model_load")
    model = model or load_model(config.model, config.seed)
    finish_stage("model_load", model=model.name, hidden_size=model.hidden_size)
    scaled_activation_sample_cap = None
    if config.sae.source == "trained" and config.sae.samples_per_feature is not None:
        scaled_activation_sample_cap = int(
            round(model.hidden_size * config.sae.expansion_factor * config.sae.samples_per_feature)
        )
        if scaled_activation_sample_cap <= 0:
            raise ValueError("samples_per_feature produced a non-positive activation sample cap")

    start_stage("layer_scan")
    activations = model.collect(discovery_records, config.model.layers)
    scan = layer_scan(activations, labels)
    target_layer = config.target_layer
    if config.sae.source == "saelens":
        if target_layer is not None and target_layer != config.sae.saelens_layer:
            raise ValueError("config.target_layer must match sae.saelens_layer for SAELens runs")
        target_layer = config.sae.saelens_layer
    elif target_layer is None:
        target_layer = int(max(scan, key=lambda row: float(row["localization_score"]))["layer"])
    if target_layer not in activations:
        raise ValueError(f"target layer {target_layer} was not collected in model.layers")
    target_activations = activations[target_layer]
    write_json(output_dir / "layer_scan.json", scan)
    finish_stage("layer_scan", target_layer=target_layer, sae_source=config.sae.source)

    start_stage("sae_training")
    sae_training_activations = model.collect_sae_training(
        train_records, target_layer, config.sae.tokens_per_prompt
    )
    collected_sae_training_samples = len(sae_training_activations)
    effective_max_activation_samples = config.sae.max_activation_samples
    if scaled_activation_sample_cap is not None:
        effective_max_activation_samples = (
            min(effective_max_activation_samples, scaled_activation_sample_cap)
            if effective_max_activation_samples is not None
            else scaled_activation_sample_cap
        )
    if (
        effective_max_activation_samples is not None
        and len(sae_training_activations) > effective_max_activation_samples
    ):
        generator = torch.Generator().manual_seed(
            config.sae.activation_subsample_seed
            if config.sae.activation_subsample_seed is not None
            else config.seed
        )
        indices = torch.randperm(len(sae_training_activations), generator=generator)[
            :effective_max_activation_samples
        ]
        sae_training_activations = sae_training_activations[indices]

    if config.sae.source == "saelens":
        sae_device = "cpu" if config.sae.device == "auto" else config.sae.device
        public_sae = load_saelens_sae(
            release=str(config.sae.saelens_release),
            sae_id=str(config.sae.saelens_sae_id),
            device=sae_device,
        )
        public_sae = public_sae.to(device=torch.device(sae_device), dtype=torch.float32)
        if public_sae.input_dim != target_activations.shape[-1]:
            raise ValueError(
                "SAELens SAE input dimension does not match the target layer hidden size: "
                f"{public_sae.input_dim} != {target_activations.shape[-1]}"
            )
        metadata_layer = public_sae.source_metadata.get("hook_layer")
        if metadata_layer is not None:
            try:
                metadata_layer_int = int(metadata_layer)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"SAELens hook_layer is not an integer: {metadata_layer}") from exc
            if metadata_layer_int != target_layer:
                raise ValueError(
                    f"SAELens hook_layer {metadata_layer} does not match target layer {target_layer}"
                )
        public_sae_metrics = evaluate_sae_metrics(public_sae, sae_training_activations)
        public_sae = public_sae.cpu()
        sae_result = SAETrainingResult(
            model=public_sae,
            history=[],
            metrics=public_sae_metrics,
        )
    else:
        sae_result = train_sae(
            sae_training_activations,
            expansion_factor=config.sae.expansion_factor,
            top_k=config.sae.top_k,
            epochs=config.sae.epochs,
            batch_size=config.sae.batch_size,
            learning_rate=config.sae.learning_rate,
            l1_coefficient=config.sae.l1_coefficient,
            seed=config.seed,
            device=config.sae.device,
        )
    sae_source = _sae_source_details(config, sae_result.model)
    sae_metrics_artifact = {
        "source": sae_source,
        "metrics": sae_result.metrics,
        "history": sae_result.history,
    }
    write_json(
        output_dir / "sae_metrics.json",
        _stamp_preregistration(sae_metrics_artifact, preregistration),
    )
    torch.save(sae_state_dict(sae_result.model), output_dir / "sae.pt")
    finish_stage(
        "sae_training",
        sae_source=config.sae.source,
        collected_activation_samples=collected_sae_training_samples,
        activation_samples=len(sae_training_activations),
        scaled_activation_sample_cap=scaled_activation_sample_cap,
        effective_max_activation_samples=effective_max_activation_samples,
    )

    start_stage("feature_discovery")
    features = feature_table(
        sae_result.model,
        target_activations,
        labels,
        [record.id for record in discovery_records],
    )
    sae_direction, selected_features = combined_feature_direction(
        sae_result.model, features, config.sae.n_features_to_intervene
    )
    sae_direction = _orient(sae_direction, target_activations, labels)
    mean_direction = _orient(
        mean_difference_direction(target_activations, labels), target_activations, labels
    )
    probe_direction, probe_metrics = train_probe_direction(target_activations, labels, config.seed)
    probe_direction = _orient(probe_direction, target_activations, labels)
    directions = {
        "sae_features": sae_direction,
        "mean_difference": mean_direction,
        "linear_probe": probe_direction,
    }
    write_json(output_dir / "features.json", features)
    torch.save(
        {
            "record_ids": [record.id for record in discovery_records],
            "labels": labels,
            "activations": activations,
            "directions": directions,
        },
        output_dir / "discovery_tensors.pt",
    )
    finish_stage("feature_discovery", selected_features=selected_features)

    start_stage("validation_and_baseline_scoring")
    validation_baseline = model.evaluate(validation_records)
    threshold_source = "config"
    validation_threshold_accuracy = None
    if config.evaluation.threshold is None:
        threshold, validation_threshold_accuracy = select_threshold(
            validation_records, validation_baseline
        )
        config.evaluation.threshold = threshold
        threshold_source = "validation"
    baseline = model.evaluate(test_records)
    baseline_metrics = calculate_metrics(test_records, baseline, config.evaluation.threshold)
    write_json(checkpoint_dir / "baseline_outputs.json", _serialize_outputs(baseline))
    validation_sweeps = []
    selected_scales = {}
    for method, direction in directions.items():
        method_rows = []
        for scale in config.evaluation.steering_scales:
            result = _run_comparison(
                config,
                model,
                validation_records,
                validation_baseline,
                target_layer,
                AddDirection(direction, scale),
            )
            result.update({"method": method, "scale": scale, "intervention": "steering"})
            result.pop("outputs")
            result["selection_score"] = _selection_score(result)
            method_rows.append(result)
        validation_sweeps.extend(method_rows)
        selected_scales[method] = min(method_rows, key=lambda row: row["selection_score"])["scale"]
    write_json(
        checkpoint_dir / "validation_selection.json",
        {
            "validation_sweeps": validation_sweeps,
            "validation_selected_scales": selected_scales,
            "threshold": config.evaluation.threshold,
            "threshold_source": threshold_source,
            "validation_threshold_balanced_accuracy": validation_threshold_accuracy,
        },
    )
    finish_stage("validation_and_baseline_scoring", selected_scales=selected_scales)

    start_stage("held_out_steering_sweeps")
    steering_sweeps = []
    for method, direction in directions.items():
        for scale in config.evaluation.steering_scales:
            result = _run_comparison(
                config,
                model,
                test_records,
                baseline,
                target_layer,
                AddDirection(direction, scale),
            )
            result.update({"method": method, "scale": scale, "intervention": "steering"})
            steering_sweeps.append(result)
    write_json(checkpoint_dir / "steering_sweeps.json", steering_sweeps)
    finish_stage("held_out_steering_sweeps", rows=len(steering_sweeps))

    start_stage("held_out_causal_tests")
    benign_projection = (
        target_activations[~labels] @ normalized(sae_direction)
    ).mean().item()
    causal_tests = {
        "sae_feature_ablation": _run_comparison(
            config,
            model,
            test_records,
            baseline,
            target_layer,
            AblateSAEFeatures(sae_result.model, selected_features),
        ),
        "sae_direction_ablation": _run_comparison(
            config,
            model,
            test_records,
            baseline,
            target_layer,
            AblateDirection(sae_direction),
        ),
        "sae_activation_patch": _run_comparison(
            config,
            model,
            test_records,
            baseline,
            target_layer,
            PatchProjection(sae_direction, benign_projection),
        ),
    }
    for method in ["mean_difference", "linear_probe"]:
        direction = directions[method]
        causal_tests[f"{method}_direction_ablation"] = _run_comparison(
            config,
            model,
            test_records,
            baseline,
            target_layer,
            AblateDirection(direction),
        )
    write_json(checkpoint_dir / "causal_tests.json", causal_tests)
    finish_stage("held_out_causal_tests", rows=len(causal_tests))

    start_stage("feature_budget_frontier")
    positive_ranked_features = [
        int(row["feature"]) for row in features if float(row["effect_size"]) > 0
    ]
    fallback_ranked_features = [int(row["feature"]) for row in features]
    ranked_features = positive_ranked_features + [
        feature for feature in fallback_ranked_features if feature not in positive_ranked_features
    ]
    budgets = [
        min(int(budget), len(ranked_features))
        for budget in config.evaluation.feature_budgets
        if int(budget) > 0
    ]
    budgets = sorted(set(budget for budget in budgets if budget > 0))
    l1_weights, l1_metrics = train_l1_feature_weights(
        sae_result.model,
        target_activations,
        labels,
        config.seed + 700,
        l1_coefficient=config.evaluation.feature_budget_l1_coefficient,
        epochs=config.evaluation.feature_budget_l1_epochs,
    )
    l1_ranked_features = sorted(
        range(len(l1_weights)),
        key=lambda index: abs(float(l1_weights[index])),
        reverse=True,
    )
    frontier_rows = []
    for budget in budgets:
        selected_budget_features = ranked_features[:budget]
        topk_result = _run_comparison(
            config,
            model,
            test_records,
            baseline,
            target_layer,
            AblateSAEFeatures(sae_result.model, selected_budget_features),
        )
        topk_result.update(
            {
                "method": "sae_topk_feature_ablation",
                "budget": budget,
                "requested_budget": budget,
                "selected_features": selected_budget_features,
                "selection_policy": "train feature effect-size rank; positive effects first",
            }
        )
        frontier_rows.append(topk_result)

        selected_l1_features = l1_ranked_features[:budget]
        l1_direction = _orient(
            weighted_feature_direction(sae_result.model, l1_weights, selected_l1_features),
            target_activations,
            labels,
        )
        l1_result = _run_comparison(
            config,
            model,
            test_records,
            baseline,
            target_layer,
            AblateDirection(l1_direction),
        )
        l1_result.update(
            {
                "method": "sae_l1_weighted_direction_ablation",
                "budget": budget,
                "requested_budget": budget,
                "selected_features": selected_l1_features,
                "selection_policy": "train L1 logistic weights ranked by absolute value",
            }
        )
        frontier_rows.append(l1_result)
    feature_budget_frontier = {
        "schema_version": "1.0",
        "artifact_notice": (
            "A4 feature-budget frontier. Feature ranking and L1 weights are train-only; "
            "all rows are held-out test evaluations."
        ),
        "budgets": budgets,
        "configured_budgets": config.evaluation.feature_budgets,
        "target_layer": target_layer,
        "l1_weight_metrics": l1_metrics,
        "mean_difference_reference": {
            key: value
            for key, value in causal_tests["mean_difference_direction_ablation"].items()
            if key != "outputs"
        },
        "rows": frontier_rows,
    }
    write_json(
        output_dir / "feature_budget_frontier.json",
        _stamp_preregistration(feature_budget_frontier, preregistration),
    )
    finish_stage("feature_budget_frontier", rows=len(frontier_rows), budgets=budgets)

    start_stage("random_direction_and_matched_sae_controls")
    random_control_rows = []
    if sae_source["source"] == "saelens":
        random_sae = sae_result.model
        random_sae_controls = _public_sae_controls(
            sae_result.model,
            config.evaluation.random_controls,
            config.seed + 200,
        )
    else:
        random_sae, random_sae_controls = _random_sae_controls(
            model.hidden_size,
            config.sae.expansion_factor,
            min(config.sae.top_k, model.hidden_size * config.sae.expansion_factor),
            config.evaluation.random_controls,
            config.seed + 200,
            sae_result.model.activation_center,
        )
    random_sae_protocol = _random_sae_control_protocol(
        sae_source=sae_source,
        config=config,
        hidden_size=model.hidden_size,
    )
    control_groups = {
        "random_direction": [
            (index, direction, AblateDirection(direction), "ablation_control")
            for index, direction in enumerate(
                random_directions(
                    model.hidden_size,
                    config.evaluation.random_controls,
                    config.seed + 100,
                )
            )
        ],
        "random_sae": [
            (
                feature_index,
                direction,
                AblateSAEFeatures(random_sae, [feature_index]),
                "feature_ablation_control",
            )
            for feature_index, direction in random_sae_controls
        ],
    }
    for control_name, controls in control_groups.items():
        for index, direction, ablation, ablation_name in controls:
            steering_path = random_checkpoint_dir / _control_checkpoint_name(
                control_name, index, "steering_control"
            )
            if steering_path.exists():
                steering_result = read_json(steering_path)
            else:
                steering_result = _run_comparison(
                    config,
                    model,
                    test_records,
                    baseline,
                    target_layer,
                    AddDirection(direction, config.evaluation.control_scale),
                )
                steering_result.update(
                    {
                        "method": control_name,
                        "control_index": index,
                        "scale": config.evaluation.control_scale,
                        "intervention": "steering_control",
                        "control_protocol": random_sae_protocol["status"]
                        if control_name == "random_sae"
                        else "isotropic_random_direction",
                    }
                )
                write_json(steering_path, steering_result)
            random_control_rows.append(steering_result)
            ablation_path = random_checkpoint_dir / _control_checkpoint_name(
                control_name, index, ablation_name
            )
            if ablation_path.exists():
                ablation_result = read_json(ablation_path)
            else:
                ablation_result = _run_comparison(
                    config,
                    model,
                    test_records,
                    baseline,
                    target_layer,
                    ablation,
                )
                ablation_result.update(
                    {
                        "method": control_name,
                        "control_index": index,
                        "scale": None,
                        "intervention": ablation_name,
                        "control_protocol": random_sae_protocol["status"]
                        if control_name == "random_sae"
                        else "isotropic_random_direction",
                    }
                )
                write_json(ablation_path, ablation_result)
            random_control_rows.append(ablation_result)
    finish_stage("random_direction_and_matched_sae_controls", rows=len(random_control_rows))

    start_stage("layerwise_controls")
    layerwise_controls = []
    if config.evaluation.layerwise_controls:
        for layer, layer_activations in sorted(activations.items()):
            direction = _orient(
                mean_difference_direction(layer_activations, labels), layer_activations, labels
            )
            result = _run_comparison(
                config,
                model,
                test_records,
                baseline,
                layer,
                AblateDirection(direction),
            )
            result.update(
                {
                    "layer": layer,
                    "method": "mean_difference_direction_ablation",
                    "is_selected_layer": layer == target_layer,
                }
            )
            layerwise_controls.append(result)
    write_json(output_dir / "layerwise_controls.json", layerwise_controls)
    finish_stage("layerwise_controls", rows=len(layerwise_controls))

    start_stage("failure_analysis_and_final_artifacts")
    selected_sae = next(
        row
        for row in steering_sweeps
        if row["method"] == "sae_features" and row["scale"] == selected_scales["sae_features"]
    )
    strongest_ablation_name, strongest_ablation = min(
        (
            (name, result)
            for name, result in causal_tests.items()
            if name.endswith("_ablation")
        ),
        key=lambda item: float(item[1]["delta_mean_harmful_score_ci"]["estimate"]),
    )
    selected_steering_rows = {
        method: next(
            row
            for row in steering_sweeps
            if row["method"] == method and row["scale"] == selected_scales[method]
        )
        for method in directions
    }
    selected_ablation_rows = {
        method: causal_tests[method]
        for method in [
            "sae_feature_ablation",
            "mean_difference_direction_ablation",
            "linear_probe_direction_ablation",
        ]
    }
    method_comparisons = {
        "selected_steering": paired_method_comparisons(
            test_records,
            selected_steering_rows,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed + 300,
        ),
        "causal_ablation": paired_method_comparisons(
            test_records,
            selected_ablation_rows,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed + 600,
        ),
    }
    equivalence_analysis = build_equivalence_analysis(
        test_records,
        selected_ablation_rows,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        equivalence_margin=config.evaluation.equivalence_margin,
        inferiority_margin=config.evaluation.inferiority_margin,
        alpha=config.evaluation.test_alpha,
        power_effect_size=config.evaluation.power_effect_size,
        power_assumed_pair_sd=config.evaluation.power_assumed_pair_sd,
        power_target=config.evaluation.power_target,
        power_target_n=config.evaluation.power_target_n,
        seed=config.seed + 900,
    )
    failure_analysis = build_failure_analysis(
        sae_result.metrics,
        features,
        _serialize_outputs(baseline),
        validation_sweeps,
        steering_sweeps,
        selected_scales,
        causal_tests,
        random_control_rows,
        test_records,
    )
    strongest_layer_control = (
        min(
            layerwise_controls,
            key=lambda row: float(row["delta_mean_harmful_score_ci"]["estimate"]),
        )
        if layerwise_controls
        else None
    )
    best_feature_budget_row = min(
        frontier_rows,
        key=lambda row: float(row["delta_mean_harmful_score_ci"]["estimate"]),
    )
    sae_zoo_result = _write_sae_zoo_result(
        config=config,
        model=model,
        target_layer=target_layer,
        selected_features=selected_features,
        sae_source=sae_source,
        sae_metrics=sae_result.metrics,
        causal_tests=causal_tests,
        random_control_rows=random_control_rows,
        output_dir=output_dir,
    )
    summary = {
        "evidence_class": config.evidence_class,
        "model": model.name,
        "behavior": config.behavior,
        "target_layer": target_layer,
        "selected_features": selected_features,
        "sae_source": sae_source,
        "sae_zoo_result": sae_zoo_result,
        "test_records": len(test_records),
        "test_pairs": len({record.pair_id for record in test_records}),
        "behavior_threshold": config.evaluation.threshold,
        "threshold_source": threshold_source,
        "validation_threshold_balanced_accuracy": validation_threshold_accuracy,
        "baseline_metrics": baseline_metrics,
        "sae_metrics": sae_result.metrics,
        "probe_metrics": probe_metrics,
        "validation_selected_sae_steering": {
            key: value for key, value in selected_sae.items() if key != "outputs"
        },
        "strongest_causal_ablation": {
            "method": strongest_ablation_name,
            **{key: value for key, value in strongest_ablation.items() if key != "outputs"},
        },
        "strongest_layerwise_control": (
            {
                key: value
                for key, value in strongest_layer_control.items()
                if key != "outputs"
            }
            if strongest_layer_control
            else None
        ),
        "best_feature_budget_frontier": {
            key: value for key, value in best_feature_budget_row.items() if key != "outputs"
        },
        "sae_outperforms_simple_baselines": failure_analysis["baseline_comparison"][
            "sae_outperforms_simple_baselines"
        ],
        "equivalence_tests_passed": [
            row["test"] for row in equivalence_analysis["tests"] if row["passed"]
        ],
        "negative_findings": failure_analysis["negative_findings"],
    }
    if preregistration is not None:
        summary["preregistration"] = preregistration
        summary["preregistration_hash"] = preregistration["hash"]
    protocol = {
        "mode": config.mode,
        "train_records_used_for_discovery_and_sae_only": len(train_records),
        "validation_records_used_for_threshold_and_scale_selection_only": len(validation_records),
        "test_records_used_for_final_reporting_only": len(test_records),
        "test_pairs_used_for_final_reporting_only": len(
            {record.pair_id for record in test_records}
        ),
        "family_disjoint_splits": True,
        "layer_selection": (
            "predeclared SAELens hook layer"
            if sae_source["source"] == "saelens"
            else "train localization_score only"
        ),
        "feature_selection": (
            "train final-token public SAE latent effect size only"
            if sae_source["source"] == "saelens"
            else "train final-token feature effect size only"
        ),
        "sae_source": sae_source,
        "sae_training": {
            "split": "train",
            "tokens_per_prompt": config.sae.tokens_per_prompt,
            "collected_activation_samples": collected_sae_training_samples,
            "activation_samples": len(sae_training_activations),
            "max_activation_samples": config.sae.max_activation_samples,
            "samples_per_feature": config.sae.samples_per_feature,
            "scaled_activation_sample_cap": scaled_activation_sample_cap,
            "effective_max_activation_samples": effective_max_activation_samples,
            "source": sae_source["source"],
        },
        "behavior_threshold_selection": threshold_source,
        "intervention_scale_selection": "validation specificity/capability-adjusted score",
        "test_access_policy": "final metrics and paired comparisons only",
        "statistical_protocol": {
            "bootstrap_unit": "pair_id",
            "equivalence_margin": config.evaluation.equivalence_margin,
            "inferiority_margin": config.evaluation.inferiority_margin,
            "alpha": config.evaluation.test_alpha,
            "power_effect_size": config.evaluation.power_effect_size,
            "power_assumed_pair_sd": config.evaluation.power_assumed_pair_sd,
            "power_target": config.evaluation.power_target,
            "power_target_n": config.evaluation.power_target_n,
        },
        "real_audit_requirements_passed": real_audit_checks,
        "checkpointing": {
            "stage_markers_dir": str(checkpoint_dir),
            "random_controls_dir": str(random_checkpoint_dir),
            "random_controls_resumable": True,
            "final_summary_written_after_all_expected_artifacts": True,
        },
    }
    if preregistration is not None:
        protocol["preregistration"] = preregistration
        protocol["preregistration_hash"] = preregistration["hash"]
    manifest = {
        "schema_version": "1.0",
        "config": config.to_dict(),
        "sae_source": sae_source,
        "dataset": dataset_summary(records),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "artifact_notice": (
            "Synthetic fixture results are pipeline validation only."
            if config.model.backend == "toy"
            else "Empirical results from the configured model; inspect config and seed before reuse."
        ),
    }
    if preregistration is not None:
        manifest["preregistration"] = preregistration
        manifest["preregistration_hash"] = preregistration["hash"]
    intervention_artifact = {
        "baseline_outputs": _serialize_outputs(baseline),
        "validation_sweeps": validation_sweeps,
        "validation_selected_scales": selected_scales,
        "steering_sweeps": steering_sweeps,
        "causal_tests": causal_tests,
        "random_controls": random_control_rows,
        "random_sae_control_protocol": random_sae_protocol,
    }

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "protocol.json", protocol)
    write_json(
        output_dir / "records.json",
        [
            {
                "id": record.id,
                "prompt": record.prompt,
                "harmful": record.harmful,
                "split": record.split,
                "category": record.category,
                "pair_id": record.pair_id,
                "family_id": record.family_id,
                "source": record.source,
            }
            for record in records
        ],
    )
    write_json(output_dir / "layer_scan.json", scan)
    write_json(output_dir / "layerwise_controls.json", layerwise_controls)
    write_json(output_dir / "features.json", features)
    write_json(
        output_dir / "sae_metrics.json",
        _stamp_preregistration(sae_metrics_artifact, preregistration),
    )
    write_json(
        output_dir / "interventions.json",
        _stamp_preregistration(intervention_artifact, preregistration),
    )
    write_json(
        output_dir / "method_comparisons.json",
        _stamp_preregistration(method_comparisons, preregistration),
    )
    write_json(
        output_dir / "equivalence_analysis.json",
        _stamp_preregistration(equivalence_analysis, preregistration),
    )
    write_json(
        output_dir / "feature_budget_frontier.json",
        _stamp_preregistration(feature_budget_frontier, preregistration),
    )
    write_json(
        output_dir / "failure_analysis.json",
        _stamp_preregistration(failure_analysis, preregistration),
    )
    write_json(output_dir / "summary.json", summary)
    torch.save(sae_state_dict(sae_result.model), output_dir / "sae.pt")
    torch.save(
        {
            "record_ids": [record.id for record in discovery_records],
            "labels": labels,
            "activations": activations,
            "directions": directions,
        },
        output_dir / "discovery_tensors.pt",
    )
    write_report(output_dir)
    finish_stage("failure_analysis_and_final_artifacts", summary_written=True)
    return summary
