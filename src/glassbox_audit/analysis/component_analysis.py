from __future__ import annotations

from pathlib import Path

import torch

from ..evaluation import comparison
from ..interventions import AblateDirection
from ..interventions.control_refresh import _config_from_manifest, _outputs_from_dict
from ..models import load_model
from ..pipeline import _serialize_outputs
from ..sae.discovery import mean_difference_direction
from ..types import PromptRecord
from ..utils import read_json, seed_everything, write_json


def run_component_localization(
    artifact_dir: str | Path,
    *,
    components: list[str] | None = None,
    layers: list[int] | None = None,
    direction_name: str = "mean_difference",
    device: str | None = None,
    max_records: int | None = None,
    output_name: str = "component_localization.json",
) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    components = components or ["residual", "attention", "mlp"]
    manifest = read_json(artifact_dir / "manifest.json")
    summary = read_json(artifact_dir / "summary.json")
    interventions = read_json(artifact_dir / "interventions.json")
    config = _config_from_manifest(manifest)
    config.evaluation.threshold = float(summary["behavior_threshold"])
    if device is not None:
        config.model.device = device
        config.sae.device = device
    seed_everything(config.seed)

    records = [
        PromptRecord.from_dict(row)
        for row in read_json(artifact_dir / "records.json")
        if row["split"] == "test"
    ]
    if max_records is not None:
        records = records[:max_records]
    baseline_by_id = {row["record_id"]: row for row in interventions["baseline_outputs"]}
    baseline = _outputs_from_dict([baseline_by_id[record.id] for record in records])
    tensors = torch.load(artifact_dir / "discovery_tensors.pt", map_location="cpu")
    if direction_name != "mean_difference" and direction_name not in tensors["directions"]:
        raise ValueError(f"Unknown direction {direction_name!r}; available: {sorted(tensors['directions'])}")
    labels = tensors["labels"].bool()
    model = load_model(config.model, config.seed)
    target_layer = int(summary["target_layer"])
    layers = layers or [target_layer]

    rows = []
    for layer_index, layer in enumerate(layers):
        if direction_name == "mean_difference":
            if layer not in tensors["activations"]:
                raise ValueError(f"Layer {layer} is missing from discovery_tensors.pt")
            direction = mean_difference_direction(tensors["activations"][layer], labels)
        else:
            if layer != target_layer:
                raise ValueError(f"Direction {direction_name!r} is only defined for target layer {target_layer}")
            direction = tensors["directions"][direction_name]
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
                    baseline,
                    outputs,
                    config.evaluation.threshold,
                    config.evaluation.bootstrap_samples,
                    config.evaluation.confidence,
                    config.seed + 900 + layer_index * 30 + component_index,
                )
                row.update(
                    {
                        "layer": layer,
                        "component": component,
                        "direction": direction_name,
                        "intervention": "component_direction_ablation",
                        "status": "completed",
                        "outputs": _serialize_outputs(outputs),
                    }
                )
            except Exception as exc:
                row = {
                    "layer": layer,
                    "component": component,
                    "direction": direction_name,
                    "intervention": "component_direction_ablation",
                    "status": "unsupported",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)

    artifact = {
        "schema_version": "1.0",
        "evidence_scope": "component-output activation intervention; not a full path-patching mechanism claim",
        "selection_policy": "frozen artifact direction/layer/threshold; no retuning",
        "model": summary["model"],
        "target_layer": target_layer,
        "direction": direction_name,
        "n_records": len(records),
        "layers": layers,
        "components": rows,
    }
    write_json(artifact_dir / output_name, artifact)
    return artifact
