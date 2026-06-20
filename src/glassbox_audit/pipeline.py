from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import torch

from .analysis import build_failure_analysis, paired_method_comparisons
from .config import ExperimentConfig, load_config
from .data import dataset_summary, load_records, split_records
from .evaluation import calculate_metrics, comparison, select_threshold
from .interventions import (
    AblateDirection,
    AblateSAEFeatures,
    AddDirection,
    PatchProjection,
    normalized,
)
from .models import AuditModel, load_model
from .reporting.report import write_report
from .sae import TopKSparseAutoencoder, sae_state_dict, train_sae
from .sae.discovery import (
    combined_feature_direction,
    feature_table,
    layer_scan,
    mean_difference_direction,
    random_directions,
    train_probe_direction,
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
        (config.target_layer is None, "target_layer must be null so layer selection uses train only"),
        (len(config.model.layers) >= 3, "at least three layers must be scanned"),
        (config.evaluation.threshold is None, "threshold must be selected on validation"),
        (config.evaluation.layerwise_controls, "layerwise controls must be enabled"),
        (config.evaluation.random_controls >= 4, "at least four random controls are required"),
        (config.evaluation.bootstrap_samples >= 200, "at least 200 bootstrap samples are required"),
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

    start_stage("layer_scan")
    activations = model.collect(discovery_records, config.model.layers)
    scan = layer_scan(activations, labels)
    target_layer = config.target_layer
    if target_layer is None:
        target_layer = int(max(scan, key=lambda row: float(row["localization_score"]))["layer"])
    target_activations = activations[target_layer]
    write_json(output_dir / "layer_scan.json", scan)
    finish_stage("layer_scan", target_layer=target_layer)

    start_stage("sae_training")
    sae_training_activations = model.collect_sae_training(
        train_records, target_layer, config.sae.tokens_per_prompt
    )
    collected_sae_training_samples = len(sae_training_activations)
    if (
        config.sae.max_activation_samples is not None
        and len(sae_training_activations) > config.sae.max_activation_samples
    ):
        generator = torch.Generator().manual_seed(
            config.sae.activation_subsample_seed
            if config.sae.activation_subsample_seed is not None
            else config.seed
        )
        indices = torch.randperm(len(sae_training_activations), generator=generator)[
            : config.sae.max_activation_samples
        ]
        sae_training_activations = sae_training_activations[indices]

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
    write_json(
        output_dir / "sae_metrics.json",
        {"metrics": sae_result.metrics, "history": sae_result.history},
    )
    torch.save(sae_state_dict(sae_result.model), output_dir / "sae.pt")
    finish_stage(
        "sae_training",
        collected_activation_samples=collected_sae_training_samples,
        activation_samples=len(sae_training_activations),
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

    start_stage("random_direction_and_matched_sae_controls")
    random_control_rows = []
    random_sae, random_sae_controls = _random_sae_controls(
        model.hidden_size,
        config.sae.expansion_factor,
        min(config.sae.top_k, model.hidden_size * config.sae.expansion_factor),
        config.evaluation.random_controls,
        config.seed + 200,
        sae_result.model.activation_center,
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
    summary = {
        "evidence_class": config.evidence_class,
        "model": model.name,
        "behavior": config.behavior,
        "target_layer": target_layer,
        "selected_features": selected_features,
        "test_records": len(test_records),
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
        "sae_outperforms_simple_baselines": failure_analysis["baseline_comparison"][
            "sae_outperforms_simple_baselines"
        ],
        "negative_findings": failure_analysis["negative_findings"],
    }
    protocol = {
        "mode": config.mode,
        "train_records_used_for_discovery_and_sae_only": len(train_records),
        "validation_records_used_for_threshold_and_scale_selection_only": len(validation_records),
        "test_records_used_for_final_reporting_only": len(test_records),
        "family_disjoint_splits": True,
        "layer_selection": "train localization_score only",
        "feature_selection": "train final-token feature effect size only",
        "sae_training": {
            "split": "train",
            "tokens_per_prompt": config.sae.tokens_per_prompt,
            "collected_activation_samples": collected_sae_training_samples,
            "activation_samples": len(sae_training_activations),
            "max_activation_samples": config.sae.max_activation_samples,
        },
        "behavior_threshold_selection": threshold_source,
        "intervention_scale_selection": "validation specificity/capability-adjusted score",
        "test_access_policy": "final metrics and paired comparisons only",
        "real_audit_requirements_passed": real_audit_checks,
        "checkpointing": {
            "stage_markers_dir": str(checkpoint_dir),
            "random_controls_dir": str(random_checkpoint_dir),
            "random_controls_resumable": True,
            "final_summary_written_after_all_expected_artifacts": True,
        },
    }
    manifest = {
        "schema_version": "1.0",
        "config": config.to_dict(),
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
    intervention_artifact = {
        "baseline_outputs": _serialize_outputs(baseline),
        "validation_sweeps": validation_sweeps,
        "validation_selected_scales": selected_scales,
        "steering_sweeps": steering_sweeps,
        "causal_tests": causal_tests,
        "random_controls": random_control_rows,
        "random_sae_control_protocol": {
            "status": "matched_center_zero_encoder_bias",
            "architecture": {
                "input_dim": model.hidden_size,
                "n_features": model.hidden_size * config.sae.expansion_factor,
                "top_k": min(config.sae.top_k, model.hidden_size * config.sae.expansion_factor),
            },
            "activation_center": "trained_sae_activation_center",
            "encoder_bias": "zeroed",
            "latent_intervention": "literal_encode_zero_selected_latent_decode",
            "selection_policy": "frozen; no feature, layer, threshold, or scale retuning",
        },
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
        {"metrics": sae_result.metrics, "history": sae_result.history},
    )
    write_json(output_dir / "interventions.json", intervention_artifact)
    write_json(output_dir / "method_comparisons.json", method_comparisons)
    write_json(output_dir / "failure_analysis.json", failure_analysis)
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
