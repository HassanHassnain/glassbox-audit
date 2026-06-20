from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..evaluation import comparison
from ..interventions import AblateDirection, PatchProjection, normalized
from ..interventions.control_refresh import _config_from_manifest, _outputs_from_dict
from ..models import load_model
from ..pipeline import _serialize_outputs
from ..sae.discovery import mean_difference_direction
from ..types import ModelOutput, PromptRecord
from ..utils import read_json, seed_everything, write_json
from .stats import paired_delta_ci


def _subset_outputs(outputs: list[dict[str, Any]], records: list[PromptRecord]) -> list[ModelOutput]:
    by_id = {row["record_id"]: row for row in outputs}
    return _outputs_from_dict([by_id[record.id] for record in records])


def _projection_patch_summary(
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    outputs: list[ModelOutput],
    threshold: float,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    before = {row.record_id: row for row in baseline}
    after = {row.record_id: row for row in outputs}
    ids = [record.id for record in records]
    before_refusal = sum(before[row_id].behavior_score >= threshold for row_id in ids) / len(ids)
    after_refusal = sum(after[row_id].behavior_score >= threshold for row_id in ids) / len(ids)
    return {
        "n_records": len(records),
        "refusal_rate_before": before_refusal,
        "refusal_rate_after": after_refusal,
        "refusal_rate_delta": after_refusal - before_refusal,
        "behavior_score_delta_ci": paired_delta_ci(
            [before[row_id].behavior_score for row_id in ids],
            [after[row_id].behavior_score for row_id in ids],
            bootstrap_samples,
            confidence,
            seed,
        ),
        "capability_nll_delta_ci": paired_delta_ci(
            [before[row_id].capability_nll for row_id in ids],
            [after[row_id].capability_nll for row_id in ids],
            bootstrap_samples,
            confidence,
            seed + 1,
        ),
        "outputs": _serialize_outputs(outputs),
    }


def run_path_component_analysis(
    artifact_dir: str | Path,
    *,
    layers: list[int] | None = None,
    components: list[str] | None = None,
    device: str | None = None,
    max_records: int | None = None,
    output_name: str = "path_component_analysis.json",
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    manifest = read_json(artifact_dir / "manifest.json")
    summary = read_json(artifact_dir / "summary.json")
    source_interventions = read_json(artifact_dir / "interventions.json")
    config = _config_from_manifest(manifest)
    config.evaluation.threshold = float(summary["behavior_threshold"])
    if device:
        config.model.device = device
        config.sae.device = device
    seed_everything(config.seed)

    all_test_records = [
        PromptRecord.from_dict(row)
        for row in read_json(artifact_dir / "records.json")
        if row["split"] == "test"
    ]
    records = all_test_records[:max_records] if max_records is not None else all_test_records
    harmful_records = [record for record in records if record.harmful]
    benign_records = [record for record in records if not record.harmful]
    baseline_all = _subset_outputs(source_interventions["baseline_outputs"], records)
    baseline_harmful = _subset_outputs(source_interventions["baseline_outputs"], harmful_records)
    baseline_benign = _subset_outputs(source_interventions["baseline_outputs"], benign_records)

    tensors = torch.load(artifact_dir / "discovery_tensors.pt", map_location="cpu")
    labels = tensors["labels"].bool()
    train_activations: dict[int, torch.Tensor] = tensors["activations"]
    target_layer = int(summary["target_layer"])
    layers = layers or [target_layer]
    components = components or ["residual", "attention", "mlp"]
    model = load_model(config.model, config.seed)

    residual_projection_patches = []
    residual_ablation_rows = []
    directions_by_layer: dict[int, torch.Tensor] = {}
    for index, layer in enumerate(layers):
        if layer not in train_activations:
            residual_projection_patches.append(
                {"layer": layer, "status": "unsupported", "error": "missing train activations"}
            )
            continue
        activations = train_activations[layer]
        direction = normalized(mean_difference_direction(activations, labels))
        directions_by_layer[layer] = direction
        harmful_projection = float((activations[labels] @ direction).mean().item())
        benign_projection = float((activations[~labels] @ direction).mean().item())

        ablation_outputs = model.evaluate(records, layer, AblateDirection(direction))
        ablation = comparison(
            records,
            baseline_all,
            ablation_outputs,
            config.evaluation.threshold,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed + 1600 + index,
        )
        ablation.update(
            {
                "layer": layer,
                "component": "residual",
                "intervention": "residual_direction_ablation",
                "outputs": _serialize_outputs(ablation_outputs),
            }
        )
        residual_ablation_rows.append(ablation)

        harmful_to_benign = model.evaluate(
            harmful_records,
            layer,
            PatchProjection(direction, benign_projection),
        )
        benign_to_harmful = model.evaluate(
            benign_records,
            layer,
            PatchProjection(direction, harmful_projection),
        )
        residual_projection_patches.append(
            {
                "layer": layer,
                "direction": "train_mean_difference",
                "train_harmful_projection": harmful_projection,
                "train_benign_projection": benign_projection,
                "necessity_like_harmful_to_benign_projection": _projection_patch_summary(
                    harmful_records,
                    baseline_harmful,
                    harmful_to_benign,
                    config.evaluation.threshold,
                    config.evaluation.bootstrap_samples,
                    config.evaluation.confidence,
                    config.seed + 1700 + index * 2,
                ),
                "sufficiency_like_benign_to_harmful_projection": _projection_patch_summary(
                    benign_records,
                    baseline_benign,
                    benign_to_harmful,
                    config.evaluation.threshold,
                    config.evaluation.bootstrap_samples,
                    config.evaluation.confidence,
                    config.seed + 1701 + index * 2,
                ),
            }
        )

    component_rows = []
    for layer_index, layer in enumerate(layers):
        direction = directions_by_layer.get(layer)
        if direction is None:
            for component in components:
                component_rows.append(
                    {
                        "layer": layer,
                        "component": component,
                        "direction": "mean_difference",
                        "intervention": "component_direction_ablation",
                        "status": "unsupported",
                        "error": "missing train activations",
                    }
                )
            continue
        for component_index, component in enumerate(components):
            try:
                outputs = model.evaluate_component(
                    records,
                    layer,
                    component,
                    AblateDirection(direction),
                )
                row = comparison(
                    records,
                    baseline_all,
                    outputs,
                    config.evaluation.threshold,
                    config.evaluation.bootstrap_samples,
                    config.evaluation.confidence,
                    config.seed + 1800 + layer_index * 30 + component_index,
                )
                row.update(
                    {
                        "layer": layer,
                        "component": component,
                        "direction": "mean_difference",
                        "intervention": "component_direction_ablation",
                        "status": "completed",
                        "outputs": _serialize_outputs(outputs),
                    }
                )
            except Exception as exc:
                row = {
                    "layer": layer,
                    "component": component,
                    "direction": "mean_difference",
                    "intervention": "component_direction_ablation",
                    "status": "unsupported",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            component_rows.append(row)
    artifact = {
        "schema_version": "1.0",
        "evidence_scope": (
            "component-output ablation plus residual projection patching approximation; "
            "not head-level or full path-patching evidence"
        ),
        "selection_policy": "frozen artifact layers/directions/threshold; no retuning",
        "model": summary["model"],
        "source_artifact": str(artifact_dir),
        "target_layer": target_layer,
        "layers": layers,
        "components": components,
        "n_records": len(records),
        "residual_ablation": residual_ablation_rows,
        "residual_projection_patching": residual_projection_patches,
        "component_outputs": component_rows,
        "claim_status": {
            "confirmed_circuit": False,
            "reason": (
                "This artifact can support or weaken residual/component localization. It does not "
                "establish necessity, sufficiency, specificity, random path controls, and "
                "replicated path mediation required for a confirmed circuit claim."
            ),
        },
    }
    write_json(artifact_dir / output_name, artifact)
    snapshot = {
        key: value
        for key, value in artifact.items()
        if key not in {"component_outputs", "residual_ablation", "residual_projection_patching"}
    }
    snapshot["records_artifact"] = str(artifact_dir / output_name)
    snapshot_dir = Path("results/component-path")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    write_json(snapshot_dir / f"{artifact_dir.name}_{Path(output_name).stem}_summary.json", snapshot)
    return artifact
