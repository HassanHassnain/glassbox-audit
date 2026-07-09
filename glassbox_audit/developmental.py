from __future__ import annotations

import gc
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from .config import ModelConfig
from .data import load_records, split_records
from .discovery import layer_scan, mean_difference_direction, random_directions
from .evaluation import comparison, select_threshold
from .interventions import AblateDirection
from .modeling import AuditModel, load_model
from .preregistration import load_frozen_preregistration, preregistration_metadata
from .stats import benjamini_hochberg, cohen_d
from .types import ModelOutput, PromptRecord
from .utils import seed_everything, write_json

SCHEMA_VERSION = "1.0"
ModelLoader = Callable[[ModelConfig, int, str], AuditModel]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direction_hash(direction: torch.Tensor) -> str:
    values = direction.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _read_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def load_developmental_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError("Developmental manifest must contain a top-level mapping")
    required = ["schema_version", "name", "dataset_path", "families", "evaluation"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Developmental manifest is missing: {', '.join(missing)}")
    if str(manifest["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Developmental manifest schema_version must be {SCHEMA_VERSION}")
    families = manifest["families"]
    if not isinstance(families, list) or not families:
        raise ValueError("Developmental manifest families must be a non-empty list")
    family_slugs: set[str] = set()
    for family in families:
        for key in ["slug", "source_model", "target_model", "target_kind", "layers", "checkpoints"]:
            if key not in family:
                raise ValueError(f"Developmental family is missing {key}")
        slug = str(family["slug"])
        if slug in family_slugs:
            raise ValueError(f"Duplicate developmental family slug: {slug}")
        family_slugs.add(slug)
        if family["target_kind"] not in {"base", "instruct"}:
            raise ValueError(f"Unsupported target_kind for {slug}: {family['target_kind']}")
        if len(family["layers"]) < 3:
            raise ValueError(f"Developmental family {slug} must scan at least three layers")
        revisions = [str(row["revision"]) for row in family["checkpoints"]]
        if len(revisions) != len(set(revisions)):
            raise ValueError(f"Developmental family {slug} has duplicate revisions")
    evaluation = manifest["evaluation"]
    if int(evaluation.get("bootstrap_samples", 0)) < 20:
        raise ValueError("Developmental bootstrap_samples must be at least 20")
    if int(evaluation.get("random_controls", 0)) < 1:
        raise ValueError("Developmental random_controls must be at least one")
    return manifest


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent.parent / path).resolve()


def _family(manifest: dict[str, Any], slug: str) -> dict[str, Any]:
    for family in manifest["families"]:
        if str(family["slug"]) == slug:
            return family
    raise ValueError(f"Unknown developmental family: {slug}")


def _validate_frozen_design(manifest: dict[str, Any], prereg_path: Path | None) -> None:
    if prereg_path is None:
        return
    frozen = load_frozen_preregistration(prereg_path)["preregistration"]
    frozen_transfer = frozen["success_criteria"]["final_endpoint_transfer"]
    executable_transfer = manifest.get("success_criteria", {})
    for key, frozen_value in frozen_transfer.items():
        if executable_transfer.get(key) != frozen_value:
            raise ValueError(
                f"Developmental manifest criterion {key} does not match frozen preregistration"
            )
    frozen_separation = frozen["success_criteria"]["source_validation_separation"]
    separation_value = executable_transfer.get("source_validation_effect_size_gt")
    if separation_value != frozen_separation.get("projection_effect_size_gt"):
        raise ValueError(
            "Developmental manifest source separation criterion does not match frozen preregistration"
        )
    frozen_thresholds = frozen["thresholds"]
    executable_evaluation = manifest["evaluation"]
    threshold_pairs = {
        "bootstrap_samples": "bootstrap_samples",
        "random_direction_controls": "random_controls",
    }
    for frozen_key, executable_key in threshold_pairs.items():
        if executable_evaluation.get(executable_key) != frozen_thresholds.get(frozen_key):
            raise ValueError(
                f"Developmental manifest {executable_key} does not match frozen preregistration"
            )


def _labels(records: list[PromptRecord]) -> torch.Tensor:
    return torch.tensor([record.harmful for record in records], dtype=torch.bool)


def _oriented_direction(activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    direction = mean_difference_direction(activations, labels)
    projections = activations @ direction
    if projections[labels].mean() < projections[~labels].mean():
        direction = -direction
    return F.normalize(direction.float().cpu(), dim=0)


def _separation(
    activations: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, float]:
    projections = activations.float() @ direction.float()
    positive = projections[labels].tolist()
    negative = projections[~labels].tolist()
    return {
        "projection_effect_size": cohen_d(positive, negative),
        "projection_mean_gap": sum(positive) / len(positive) - sum(negative) / len(negative),
        "positive_mean": sum(positive) / len(positive),
        "negative_mean": sum(negative) / len(negative),
    }


def _model_config(
    family: dict[str, Any],
    *,
    source: bool,
    revision: str | None,
    device: str,
    cache_dir: str | None,
) -> ModelConfig:
    prefix = "source" if source else "target"
    return ModelConfig(
        backend=str(family.get(f"{prefix}_backend", "huggingface")),
        name=str(family[f"{prefix}_model"]),
        revision=revision,
        layers=[int(layer) for layer in family["layers"]],
        dtype=str(family.get("dtype", "float16")),
        device=device,
        trust_remote_code=bool(family.get("trust_remote_code", False)),
        cache_dir=cache_dir,
    )


def _default_model_loader(config: ModelConfig, seed: int, behavior: str) -> AuditModel:
    return load_model(config, seed, behavior=behavior)


def _clear_model_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _outputs(rows: list[ModelOutput]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]


def _effect(result: dict[str, Any]) -> float:
    return float(result["delta_mean_harmful_score_ci"]["estimate"])


def _cell_path(root: Path, family_slug: str, revision: str) -> Path:
    return root / family_slug / _slug(revision)


def run_developmental_family(
    *,
    manifest_path: str | Path,
    family_slug: str,
    output_root: str | Path = "artifacts/developmental",
    revisions: list[str] | None = None,
    source_device: str = "cuda:0",
    target_device: str | None = None,
    cache_dir: str | None = None,
    model_loader: ModelLoader = _default_model_loader,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_developmental_manifest(manifest_path)
    family = _family(manifest, family_slug)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target_device = target_device or source_device
    seed = int(manifest.get("seed", 17))
    behavior = str(manifest.get("behavior", "refusal"))
    evaluation = manifest["evaluation"]
    bootstrap_samples = int(evaluation["bootstrap_samples"])
    confidence = float(evaluation.get("confidence", 0.95))
    random_control_count = int(evaluation["random_controls"])
    criteria = manifest.get("success_criteria", {})
    target_revision = family.get("target_revision")
    dataset_path = _resolve_path(manifest_path, str(manifest["dataset_path"]))
    records = load_records(dataset_path)
    train_records = split_records(records, "train")
    validation_records = split_records(records, "validation")
    test_records = split_records(records, "test")
    train_labels = _labels(train_records)
    validation_labels = _labels(validation_records)
    prereg_path_value = manifest.get("preregistration_path")
    prereg_path = (
        _resolve_path(manifest_path, str(prereg_path_value)) if prereg_path_value else None
    )
    preregistration = preregistration_metadata(
        prereg_path,
        manifest.get("preregistration_hash"),
    )
    _validate_frozen_design(manifest, prereg_path)
    checkpoint_rows = [
        row
        for row in family["checkpoints"]
        if revisions is None or str(row["revision"]) in set(revisions)
    ]
    if not checkpoint_rows:
        raise ValueError(f"No requested revisions found for {family_slug}")

    seed_everything(seed)
    discovered: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(checkpoint_rows):
        revision = str(checkpoint["revision"])
        cell_dir = _cell_path(output_root, family_slug, revision)
        cell_dir.mkdir(parents=True, exist_ok=True)
        source_config = _model_config(
            family,
            source=True,
            revision=revision,
            device=source_device,
            cache_dir=cache_dir,
        )
        source_model = model_loader(source_config, seed + checkpoint_index, behavior)
        train_activations = source_model.collect(train_records, source_config.layers)
        scan = layer_scan(train_activations, train_labels)
        selected = max(scan, key=lambda row: float(row["localization_score"]))
        layer = int(selected["layer"])
        direction = _oriented_direction(train_activations[layer], train_labels)
        validation_activations = source_model.collect(validation_records, [layer])[layer]
        validation_separation = _separation(validation_activations, validation_labels, direction)
        direction_path = cell_dir / "direction.pt"
        direction_digest = _direction_hash(direction)
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "family": family_slug,
                "source_model": source_config.name,
                "source_revision": revision,
                "layer": layer,
                "direction": direction,
                "direction_sha256": direction_digest,
                "train_record_ids": [record.id for record in train_records],
            },
            direction_path,
        )
        write_json(cell_dir / "source_layer_scan.json", scan)
        discovered.append(
            {
                "checkpoint": checkpoint,
                "source_config": source_config,
                "layer": layer,
                "direction": direction,
                "direction_path": direction_path,
                "direction_sha256": direction_digest,
                "train_separation": _separation(
                    train_activations[layer], train_labels, direction
                ),
                "validation_separation": validation_separation,
                "selected_layer_scan": selected,
            }
        )
        del source_model
        _clear_model_cache()

    target_config = _model_config(
        family,
        source=False,
        revision=None if target_revision in {None, "main"} else str(target_revision),
        device=target_device,
        cache_dir=cache_dir,
    )
    target_model = model_loader(target_config, seed, behavior)
    if any(row["direction"].numel() != target_model.hidden_size for row in discovered):
        raise ValueError(
            f"Source and target hidden sizes do not match for developmental family {family_slug}"
        )
    target_layers = sorted({int(row["layer"]) for row in discovered})
    target_train_activations = target_model.collect(train_records, target_layers)
    target_native = {
        layer: _oriented_direction(target_train_activations[layer], train_labels)
        for layer in target_layers
    }
    validation_baseline = target_model.evaluate(validation_records)
    threshold, threshold_accuracy = select_threshold(validation_records, validation_baseline)
    test_baseline = target_model.evaluate(test_records)

    evidence_rows = []
    for checkpoint_index, row in enumerate(discovered):
        revision = str(row["checkpoint"]["revision"])
        cell_dir = _cell_path(output_root, family_slug, revision)
        layer = int(row["layer"])
        direction = row["direction"]
        native_direction = target_native[layer]
        transferred_outputs = target_model.evaluate(
            test_records,
            layer,
            AblateDirection(direction),
        )
        transferred = comparison(
            test_records,
            test_baseline,
            transferred_outputs,
            threshold,
            bootstrap_samples,
            confidence,
            seed + checkpoint_index * 100,
        )
        native_outputs = target_model.evaluate(
            test_records,
            layer,
            AblateDirection(native_direction),
        )
        native = comparison(
            test_records,
            test_baseline,
            native_outputs,
            threshold,
            bootstrap_samples,
            confidence,
            seed + checkpoint_index * 100 + 10,
        )
        control_rows = []
        control_outputs = []
        for control_index, control_direction in enumerate(
            random_directions(
                target_model.hidden_size,
                random_control_count,
                seed + checkpoint_index * 1000,
                orthogonal_to=direction,
            )
        ):
            outputs = target_model.evaluate(
                test_records,
                layer,
                AblateDirection(control_direction),
            )
            result = comparison(
                test_records,
                test_baseline,
                outputs,
                threshold,
                bootstrap_samples,
                confidence,
                seed + checkpoint_index * 1000 + control_index,
            )
            control_rows.append(
                {
                    "control_index": control_index,
                    "direction_sha256": _direction_hash(control_direction),
                    "comparison": result,
                }
            )
            control_outputs.append(
                {
                    "control_index": control_index,
                    "outputs": _outputs(outputs),
                }
            )
        observed_effect = _effect(transferred)
        control_effects = [_effect(item["comparison"]) for item in control_rows]
        empirical_p = (1 + sum(effect <= observed_effect for effect in control_effects)) / (
            len(control_effects) + 1
        )
        harmful_ci = transferred["delta_mean_harmful_score_ci"]
        benign_ci = transferred["delta_mean_benign_score_ci"]
        capability_ci = transferred["delta_capability_nll_ci"]
        transfer_passed = (
            float(harmful_ci["estimate"])
            < float(criteria.get("harmful_delta_lt", -0.03))
            and float(harmful_ci["high"])
            < float(criteria.get("harmful_ci_high_lt", 0.0))
            and abs(float(benign_ci["estimate"]))
            <= float(criteria.get("abs_benign_delta_lte", 0.03))
            and float(capability_ci["estimate"])
            <= float(criteria.get("capability_nll_delta_lte", 0.05))
            and empirical_p <= float(criteria.get("random_control_p_lte", 0.05))
        )
        outputs_path = cell_dir / "outputs.json"
        write_json(
            outputs_path,
            {
                "schema_version": SCHEMA_VERSION,
                "baseline": _outputs(test_baseline),
                "transferred_direction": _outputs(transferred_outputs),
                "target_native_direction": _outputs(native_outputs),
                "random_direction_controls": control_outputs,
            },
        )
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "A3_developmental_checkpoint_transfer",
            "status": "completed",
            "family": family_slug,
            "source_model": row["source_config"].name,
            "source_revision": revision,
            "training_tokens_b": row["checkpoint"].get("training_tokens_b"),
            "target_model": target_config.name,
            "target_revision": target_revision or "main",
            "target_kind": family["target_kind"],
            "claim_scope": (
                "final_instruct_causal_transfer"
                if family["target_kind"] == "instruct"
                else "final_base_representation_transfer"
            ),
            "behavior": behavior,
            "layer": layer,
            "layer_depth_fraction": (layer + 1) / (max(family["layers"]) + 1),
            "hidden_size": target_model.hidden_size,
            "direction_path": str(row["direction_path"]),
            "direction_sha256": row["direction_sha256"],
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256_file(dataset_path),
            "records": {
                "train": len(train_records),
                "validation": len(validation_records),
                "test": len(test_records),
            },
            "selection_wall": "layer and direction selected from source-checkpoint train records only",
            "threshold": threshold,
            "threshold_source": "target_endpoint_validation",
            "validation_threshold_balanced_accuracy": threshold_accuracy,
            "source_train_separation": row["train_separation"],
            "source_validation_separation": row["validation_separation"],
            "source_separation_passed": float(
                row["validation_separation"]["projection_effect_size"]
            )
            > float(criteria.get("source_validation_effect_size_gt", 0.5)),
            "selected_layer_scan": row["selected_layer_scan"],
            "target_direction_cosine": float(F.cosine_similarity(direction, native_direction, dim=0)),
            "transferred_direction_ablation": transferred,
            "target_native_direction_ablation": native,
            "matched_random_direction_controls": control_rows,
            "random_control_empirical_p": empirical_p,
            "transfer_passed": transfer_passed,
            "bootstrap_unit": "pair_id",
            "bootstrap_samples": bootstrap_samples,
            "outputs_path": str(outputs_path),
            "preregistration": preregistration,
        }
        write_json(cell_dir / "evidence.json", evidence)
        evidence_rows.append(evidence)

    del target_model
    _clear_model_cache()
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family_slug,
        "cells_completed": len(evidence_rows),
        "output_root": str(output_root),
        "target_kind": family["target_kind"],
        "preregistration_hash": preregistration["hash"] if preregistration else None,
    }


def _unrun_row(manifest: dict[str, Any], family: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "A3_developmental_checkpoint_transfer",
        "status": "unrun",
        "family": family["slug"],
        "source_model": family["source_model"],
        "source_revision": checkpoint["revision"],
        "training_tokens_b": checkpoint.get("training_tokens_b"),
        "target_model": family["target_model"],
        "target_revision": family.get("target_revision", "main"),
        "target_kind": family["target_kind"],
        "claim_scope": (
            "final_instruct_causal_transfer"
            if family["target_kind"] == "instruct"
            else "final_base_representation_transfer"
        ),
        "layer": None,
        "source_validation_effect_size": None,
        "source_separation_passed": None,
        "target_harmful_delta": None,
        "target_harmful_ci_low": None,
        "target_harmful_ci_high": None,
        "target_direction_cosine": None,
        "random_control_empirical_p": None,
        "transfer_passed": None,
        "preregistration_hash": manifest.get("preregistration_hash"),
    }


def _summary_row(evidence: dict[str, Any]) -> dict[str, Any]:
    harmful = evidence["transferred_direction_ablation"]["delta_mean_harmful_score_ci"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": evidence["experiment"],
        "status": "completed",
        "family": evidence["family"],
        "source_model": evidence["source_model"],
        "source_revision": evidence["source_revision"],
        "training_tokens_b": evidence.get("training_tokens_b"),
        "target_model": evidence["target_model"],
        "target_revision": evidence["target_revision"],
        "target_kind": evidence["target_kind"],
        "claim_scope": evidence["claim_scope"],
        "layer": evidence["layer"],
        "source_validation_effect_size": evidence["source_validation_separation"][
            "projection_effect_size"
        ],
        "source_separation_passed": evidence["source_separation_passed"],
        "target_harmful_delta": harmful["estimate"],
        "target_harmful_ci_low": harmful["low"],
        "target_harmful_ci_high": harmful["high"],
        "target_direction_cosine": evidence["target_direction_cosine"],
        "random_control_empirical_p": evidence["random_control_empirical_p"],
        "transfer_passed": evidence["transfer_passed"],
        "preregistration_hash": (
            evidence.get("preregistration") or {}
        ).get("hash"),
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A3 Developmental Checkpoint Audit",
        "",
        "| Status | Family | Revision | Tokens B | Target | Layer | Source val d | Sep pass | Target delta [CI] | Cosine | Control p | Transfer pass |",
        "|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "completed":
            interval = (
                f"{row['target_harmful_delta']:+.4f} "
                f"[{row['target_harmful_ci_low']:+.4f}, {row['target_harmful_ci_high']:+.4f}]"
            )
            effect = f"{row['source_validation_effect_size']:.3f}"
            separation_passed = "yes" if row["source_separation_passed"] else "no"
            cosine = f"{row['target_direction_cosine']:.3f}"
            control_p = f"{row['random_control_empirical_p']:.4f}"
            passed = "yes" if row["transfer_passed"] else "no"
        else:
            interval = effect = separation_passed = cosine = control_p = passed = ""
        lines.append(
            f"| {row['status']} | {row['family']} | {row['source_revision']} | "
            f"{row.get('training_tokens_b', '')} | {row['target_kind']} | "
            f"{'' if row.get('layer') is None else row['layer']} | {effect} | "
            f"{separation_passed} | {interval} | "
            f"{cosine} | {control_p} | {passed} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> str | None:
    completed = [row for row in rows if row["status"] == "completed"]
    if not completed:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    families = sorted({str(row["family"]) for row in completed})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for family in families:
        values = sorted(
            [row for row in completed if row["family"] == family],
            key=lambda row: float(row["training_tokens_b"]),
        )
        x = [row["training_tokens_b"] for row in values]
        axes[0].plot(x, [row["source_validation_effect_size"] for row in values], marker="o", label=family)
        axes[1].plot(x, [row["target_harmful_delta"] for row in values], marker="o", label=family)
        axes[2].plot(x, [row["target_direction_cosine"] for row in values], marker="o", label=family)
    axes[0].set_title("Source checkpoint separation")
    axes[0].set_ylabel("Validation projection effect size")
    axes[1].set_title("Frozen direction transfer")
    axes[1].set_ylabel("Held-out harmful-score delta")
    axes[2].set_title("Checkpoint-to-endpoint alignment")
    axes[2].set_ylabel("Direction cosine")
    for axis in axes:
        axis.set_xlabel("Training tokens (B)")
        axis.set_xscale("log")
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def summarize_developmental_audit(
    *,
    manifest_path: str | Path,
    artifacts_root: str | Path = "artifacts/developmental",
    output_json: str | Path = "results/developmental/a3_summary.json",
    output_table: str | Path = "docs/tables/developmental_audit.md",
    output_figure: str | Path = "docs/figures/developmental_audit.png",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_developmental_manifest(manifest_path)
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    for evidence_path in Path(artifacts_root).glob("*/*/evidence.json"):
        evidence = _read_json(evidence_path)
        completed[(str(evidence["family"]), str(evidence["source_revision"]))] = _summary_row(
            evidence
        )
    rows = []
    for family in manifest["families"]:
        for checkpoint in family["checkpoints"]:
            key = (str(family["slug"]), str(checkpoint["revision"]))
            rows.append(completed.get(key) or _unrun_row(manifest, family, checkpoint))
    completed_indices = [index for index, row in enumerate(rows) if row["status"] == "completed"]
    if completed_indices:
        adjusted = benjamini_hochberg(
            [float(rows[index]["random_control_empirical_p"]) for index in completed_indices]
        )
        for index, adjusted_p in zip(completed_indices, adjusted, strict=True):
            rows[index]["random_control_bh_p"] = adjusted_p
    output_json = Path(output_json)
    output_table = Path(output_table)
    output_figure = Path(output_figure)
    for row in rows:
        write_json(
            output_json.parent / f"{_slug(row['family'])}_{_slug(row['source_revision'])}.json",
            row,
        )
    _write_table(output_table, rows)
    figure = _write_figure(output_figure, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "A3_developmental_checkpoint_transfer",
        "manifest": str(manifest_path),
        "preregistration_hash": manifest.get("preregistration_hash"),
        "cells_predeclared": len(rows),
        "cells_completed": len(completed_indices),
        "cells_unrun": len(rows) - len(completed_indices),
        "rows": rows,
        "table": str(output_table),
        "figure": figure,
    }
    write_json(output_json, summary)
    return summary
