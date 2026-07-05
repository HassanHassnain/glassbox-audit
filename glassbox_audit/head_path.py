from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .control_refresh import _config_from_manifest, _outputs_from_dict
from .discovery import mean_difference_direction
from .evaluation import comparison
from .interventions import AblateDirection, PatchProjection, normalized
from .modeling import load_model
from .pipeline import _serialize_outputs
from .stats import benjamini_hochberg, cohen_d, one_sample_mean_test
from .types import ModelOutput, PromptRecord
from .utils import read_json, seed_everything, write_json


def _subset_outputs(outputs: list[dict[str, Any]], records: list[PromptRecord]) -> list[ModelOutput]:
    by_id = {row["record_id"]: row for row in outputs}
    return _outputs_from_dict([by_id[record.id] for record in records])


def _head_slice(direction: torch.Tensor, head_index: int, num_heads: int) -> torch.Tensor:
    if direction.shape[-1] % num_heads != 0:
        raise ValueError("direction dimension must divide evenly across heads")
    head_dim = direction.shape[-1] // num_heads
    start = head_index * head_dim
    return direction[start : start + head_dim].detach().float()


def _shortlist_heads(
    model,
    records: list[PromptRecord],
    labels: torch.Tensor,
    layer: int,
    direction: torch.Tensor,
) -> list[dict[str, Any]]:
    outputs = model.collect_attention_head_outputs(records, layer)
    num_heads = int(outputs.shape[1])
    rows = []
    for head_index in range(num_heads):
        head_direction = _head_slice(direction, head_index, num_heads)
        projections = outputs[:, head_index, :] @ head_direction
        harmful = projections[labels].tolist()
        benign = projections[~labels].tolist()
        effect = cohen_d(harmful, benign)
        mean_gap = (
            (projections[labels].mean() - projections[~labels].mean()).item()
            if harmful and benign
            else 0.0
        )
        rows.append(
            {
                "layer": layer,
                "head": head_index,
                "score": abs(effect) * head_direction.norm().item(),
                "projection_effect_size": effect,
                "projection_mean_gap": mean_gap,
                "direction_norm": head_direction.norm().item(),
                "train_harmful_projection": sum(harmful) / len(harmful) if harmful else None,
                "train_benign_projection": sum(benign) / len(benign) if benign else None,
            }
        )
    return sorted(rows, key=lambda row: float(row["score"]), reverse=True)


def _effect_p_value(
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    outputs: list[ModelOutput],
) -> float:
    before = {row.record_id: row for row in baseline}
    after = {row.record_id: row for row in outputs}
    deltas = [
        after[record.id].behavior_score - before[record.id].behavior_score
        for record in records
    ]
    return float(one_sample_mean_test(deltas, 0.0, alternative="less")["p_value"])


def _comparison_with_outputs(
    *,
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    outputs: list[ModelOutput],
    threshold: float,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    row = comparison(
        records,
        baseline,
        outputs,
        threshold,
        bootstrap_samples,
        confidence,
        seed,
    )
    row["one_sided_behavior_p_value"] = _effect_p_value(records, baseline, outputs)
    row["outputs"] = _serialize_outputs(outputs)
    return row


def run_head_path_analysis(
    artifact_dir: str | Path,
    *,
    layers: list[int] | None = None,
    top_k: int = 20,
    device: str | None = None,
    max_records: int | None = None,
    output_name: str = "head_path_analysis.json",
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

    records_all = [PromptRecord.from_dict(row) for row in read_json(artifact_dir / "records.json")]
    train_records = [record for record in records_all if record.split == "train"]
    test_records = [record for record in records_all if record.split == "test"]
    records = test_records[:max_records] if max_records is not None else test_records
    baseline = _subset_outputs(source_interventions["baseline_outputs"], records)

    tensors = torch.load(artifact_dir / "discovery_tensors.pt", map_location="cpu")
    labels = tensors["labels"].bool()
    train_activations: dict[int, torch.Tensor] = tensors["activations"]
    target_layer = int(summary["target_layer"])
    layers = layers or [target_layer]
    model = load_model(config.model, config.seed, behavior=config.behavior)

    shortlist_candidates = []
    unsupported = []
    directions_by_layer: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer not in train_activations:
            unsupported.append(
                {"layer": layer, "status": "unsupported", "error": "missing train activations"}
            )
            continue
        direction = normalized(mean_difference_direction(train_activations[layer], labels))
        directions_by_layer[layer] = direction
        try:
            shortlist_candidates.extend(
                _shortlist_heads(model, train_records, labels, layer, direction)
            )
        except Exception as exc:
            unsupported.append(
                {
                    "layer": layer,
                    "status": "unsupported",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    selected_candidates = sorted(
        shortlist_candidates,
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:top_k]

    verified = []
    p_values = []
    for index, candidate in enumerate(selected_candidates):
        layer = int(candidate["layer"])
        head = int(candidate["head"])
        direction = directions_by_layer[layer]
        num_heads = model.num_attention_heads()
        head_direction = normalized(_head_slice(direction, head, num_heads))
        train_benign_projection = candidate["train_benign_projection"]
        try:
            ablation_outputs = model.evaluate_attention_head(
                records,
                layer,
                head,
                AblateDirection(head_direction),
            )
            ablation = _comparison_with_outputs(
                records=records,
                baseline=baseline,
                outputs=ablation_outputs,
                threshold=config.evaluation.threshold,
                bootstrap_samples=config.evaluation.bootstrap_samples,
                confidence=config.evaluation.confidence,
                seed=config.seed + 2400 + index * 2,
            )
            patch_outputs = model.evaluate_attention_head(
                records,
                layer,
                head,
                PatchProjection(head_direction, float(train_benign_projection or 0.0)),
            )
            patch = _comparison_with_outputs(
                records=records,
                baseline=baseline,
                outputs=patch_outputs,
                threshold=config.evaluation.threshold,
                bootstrap_samples=config.evaluation.bootstrap_samples,
                confidence=config.evaluation.confidence,
                seed=config.seed + 2401 + index * 2,
            )
            p_values.append(float(ablation["one_sided_behavior_p_value"]))
            verified.append(
                {
                    **candidate,
                    "status": "completed",
                    "interventions": {
                        "head_direction_ablation": ablation,
                        "head_projection_patch_to_train_benign_mean": patch,
                    },
                }
            )
        except Exception as exc:
            verified.append(
                {
                    **candidate,
                    "status": "unsupported",
                    "error": f"{type(exc).__name__}: {exc}",
                    "interventions": {},
                }
            )

    adjusted = benjamini_hochberg(p_values)
    adjusted_index = 0
    for row in verified:
        if row["status"] != "completed":
            continue
        row["bh_adjusted_behavior_p_value"] = adjusted[adjusted_index]
        adjusted_index += 1

    completed = [row for row in verified if row["status"] == "completed"]
    significant = [
        row
        for row in completed
        if float(row.get("bh_adjusted_behavior_p_value", 1.0)) <= config.evaluation.test_alpha
        and float(
            row["interventions"]["head_direction_ablation"]["delta_mean_harmful_score_ci"]["estimate"]
        )
        < 0
    ]
    artifact = {
        "schema_version": "1.0",
        "evidence_scope": (
            "attention-output head-slice shortlist and held-out verification; generic fallback, "
            "not architecture-specific pre-output path patching"
        ),
        "selection_policy": "train-only head shortlist; held-out verification only",
        "model": summary["model"],
        "source_artifact": str(artifact_dir),
        "target_layer": target_layer,
        "layers": layers,
        "top_k": top_k,
        "n_train_records_for_shortlist": len(train_records),
        "n_test_records_for_verification": len(records),
        "unsupported_layers": unsupported,
        "shortlist": selected_candidates,
        "verified_heads": verified,
        "multiple_testing": {
            "method": "Benjamini-Hochberg",
            "alpha": config.evaluation.test_alpha,
            "family": "verified head_direction_ablation one-sided harmful-score tests",
        },
        "claim_status": {
            "confirmed_head_path": bool(significant),
            "significant_heads": [
                {"layer": row["layer"], "head": row["head"]} for row in significant
            ],
            "reason": (
                "A confirmed circuit claim still requires architecture-specific path patching, "
                "necessity, sufficiency, specificity, and replication. This artifact settles the "
                "generic attention-output head-slice test for the configured artifact."
            ),
        },
    }
    if "preregistration" in manifest:
        artifact["preregistration"] = manifest["preregistration"]
        artifact["preregistration_hash"] = manifest.get("preregistration_hash")
    write_json(artifact_dir / output_name, artifact)
    snapshot = {
        key: value
        for key, value in artifact.items()
        if key not in {"verified_heads"}
    }
    snapshot["records_artifact"] = str(artifact_dir / output_name)
    snapshot_dir = Path("results/head-path")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    write_json(snapshot_dir / f"{artifact_dir.name}_{Path(output_name).stem}_summary.json", snapshot)
    return artifact


def write_head_path_table(artifact: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Head Path Analysis",
        "",
        "| Layer | Head | Status | Score | Ablation harmful delta | BH p |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in artifact["verified_heads"]:
        ablation = row.get("interventions", {}).get("head_direction_ablation", {})
        delta = ablation.get("delta_mean_harmful_score_ci", {}).get("estimate")
        lines.append(
            "| {layer} | {head} | {status} | {score:.4f} | {delta} | {p_value} |".format(
                layer=row["layer"],
                head=row["head"],
                status=row["status"],
                score=float(row["score"]),
                delta="" if delta is None else f"{float(delta):+.4f}",
                p_value=(
                    ""
                    if row.get("bh_adjusted_behavior_p_value") is None
                    else f"{float(row['bh_adjusted_behavior_p_value']):.4f}"
                ),
            )
        )
    output_path.write_text("\n".join(lines) + "\n")
    return output_path
