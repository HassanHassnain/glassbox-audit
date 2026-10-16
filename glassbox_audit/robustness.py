from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any

import torch
import yaml

from .control_refresh import _config_from_manifest
from .data import validate_records
from .discovery import random_directions
from .evaluation import comparison
from .interventions import AblateDirection
from .modeling import load_model
from .preregistration import load_frozen_preregistration, preregistration_metadata
from .types import ModelOutput, PromptRecord
from .utils import read_json, seed_everything, write_json

SCHEMA_VERSION = "1.0"
SHIFT_FAMILIES = ("paraphrase", "persona", "multilingual", "jailbreak_format")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_robustness_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Robustness config must contain a top-level mapping")
    required = [
        "schema_version",
        "name",
        "source_artifacts",
        "shifted_corpus_path",
        "shift_families",
        "evaluation",
        "success_criteria",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Robustness config is missing: {', '.join(missing)}")
    if str(config["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Robustness schema_version must be {SCHEMA_VERSION}")
    if config.get("aggregate_only") is not True:
        raise ValueError("Robustness configs must set aggregate_only: true")
    families = tuple(str(value) for value in config["shift_families"])
    if families != SHIFT_FAMILIES:
        raise ValueError(f"Robustness shift_families must be {list(SHIFT_FAMILIES)} in order")
    evaluation = config["evaluation"]
    if int(evaluation.get("bootstrap_samples", 0)) < 20:
        raise ValueError("Robustness bootstrap_samples must be at least 20")
    if int(evaluation.get("random_controls", 0)) < 1:
        raise ValueError("Robustness random_controls must be at least one")
    return config


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _load_shifted_records(path: str | Path) -> tuple[list[PromptRecord], dict[str, str]]:
    records = []
    shifts = {}
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                record = PromptRecord.from_dict(row)
                shift = str(row["shift_family"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid shifted record at {path}:{line_number}: {exc}") from exc
            if shift not in SHIFT_FAMILIES:
                raise ValueError(f"Unknown shift_family at {path}:{line_number}: {shift}")
            if record.split != "test":
                raise ValueError("Robustness corpus rows must all use split=test")
            records.append(record)
            shifts[record.id] = shift
    validate_records(records)
    for family in SHIFT_FAMILIES:
        family_records = [record for record in records if shifts[record.id] == family]
        labels = {record.harmful for record in family_records}
        pairs = {record.pair_id for record in family_records}
        if labels != {False, True} or len(pairs) < 2:
            raise ValueError(
                f"Shift family {family} requires both labels and at least two matched pairs"
            )
    pair_shifts: dict[str, set[str]] = {}
    for record in records:
        pair_shifts.setdefault(record.pair_id, set()).add(shifts[record.id])
    if any(len(values) != 1 for values in pair_shifts.values()):
        raise ValueError("Both rows in a matched pair must use the same shift_family")
    return records, shifts


def _subset_outputs(outputs: list[ModelOutput], ids: set[str]) -> list[ModelOutput]:
    return [row for row in outputs if row.record_id in ids]


def _aggregate_comparison(
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    intervened: list[ModelOutput],
    *,
    threshold: float,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    return comparison(
        records,
        baseline,
        intervened,
        threshold,
        bootstrap_samples,
        confidence,
        seed,
    )


def _effect(row: dict[str, Any]) -> float:
    return float(row["delta_mean_harmful_score_ci"]["estimate"])


def _validate_frozen_design(config: dict[str, Any], prereg_path: Path | None) -> None:
    if prereg_path is None:
        return
    frozen = load_frozen_preregistration(prereg_path)["preregistration"]
    frozen_criteria = frozen["success_criteria"]["shift_family_mediation"]
    for key, value in frozen_criteria.items():
        if config["success_criteria"].get(key) != value:
            raise ValueError(f"Robustness criterion {key} does not match frozen preregistration")
    frozen_thresholds = frozen["thresholds"]
    threshold_pairs = {
        "bootstrap_samples": "bootstrap_samples",
        "random_direction_controls": "random_controls",
    }
    for frozen_key, config_key in threshold_pairs.items():
        if config["evaluation"].get(config_key) != frozen_thresholds.get(frozen_key):
            raise ValueError(
                f"Robustness evaluation {config_key} does not match frozen preregistration"
            )


@contextmanager
def _generation_disabled():
    previous = os.environ.pop("GLASSBOX_STORE_GENERATIONS", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["GLASSBOX_STORE_GENERATIONS"] = previous


def _assert_aggregate_only(payload: dict[str, Any]) -> None:
    forbidden = {"prompt", "prompts", "response", "responses", "record_id", "record_ids", "records", "outputs"}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.lower()
                blocked = (
                    normalized in forbidden
                    or "prompt" in normalized
                    or "response" in normalized
                    or normalized.endswith("_outputs")
                    or normalized.endswith("_record_ids")
                )
                if blocked:
                    raise ValueError(f"Aggregate-only robustness evidence contains forbidden key {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload, "$")


def run_robustness_audit(
    *,
    config_path: str | Path,
    output_dir: str | Path = "results/robustness",
    device: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_robustness_config(config_path)
    source_dir = _resolve(config_path, str(config["source_artifacts"]))
    corpus_path = _resolve(config_path, str(config["shifted_corpus_path"]))
    prereg_value = config.get("preregistration_path")
    prereg_path = _resolve(config_path, str(prereg_value)) if prereg_value else None
    preregistration = preregistration_metadata(prereg_path, config.get("preregistration_hash"))
    _validate_frozen_design(config, prereg_path)
    records, shifts = _load_shifted_records(corpus_path)

    source_manifest = read_json(source_dir / "manifest.json")
    source_summary = read_json(source_dir / "summary.json")
    source_interventions = read_json(source_dir / "interventions.json")
    experiment = _config_from_manifest(source_manifest)
    if device is not None:
        experiment.model.device = device
    seed = int(config.get("seed", experiment.seed))
    seed_everything(seed)
    threshold = float(source_summary["behavior_threshold"])
    target_layer = int(source_summary["target_layer"])
    tensors = torch.load(source_dir / "discovery_tensors.pt", map_location="cpu")
    direction = tensors["directions"]["mean_difference"].detach().float()
    source_effect = float(
        source_interventions["causal_tests"]["mean_difference_direction_ablation"][
            "delta_mean_harmful_score_ci"
        ]["estimate"]
    )
    model = load_model(experiment.model, seed, behavior=experiment.behavior)
    if direction.numel() != model.hidden_size:
        raise ValueError("Frozen mean direction hidden size does not match robustness model")

    evaluation = config["evaluation"]
    bootstrap_samples = int(evaluation["bootstrap_samples"])
    confidence = float(evaluation.get("confidence", 0.95))
    random_control_count = int(evaluation["random_controls"])
    with _generation_disabled():
        baseline = model.evaluate(records)
        ablated = model.evaluate(records, target_layer, AblateDirection(direction))
        controls = []
        for control_direction in random_directions(
            model.hidden_size,
            random_control_count,
            seed + 8000,
            orthogonal_to=direction,
        ):
            controls.append(
                model.evaluate(records, target_layer, AblateDirection(control_direction))
            )

    criteria = config["success_criteria"]
    family_rows = []
    for family_index, family in enumerate(SHIFT_FAMILIES):
        family_records = [record for record in records if shifts[record.id] == family]
        ids = {record.id for record in family_records}
        family_baseline = _subset_outputs(baseline, ids)
        family_ablated = _subset_outputs(ablated, ids)
        result = _aggregate_comparison(
            family_records,
            family_baseline,
            family_ablated,
            threshold=threshold,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed + 8100 + family_index,
        )
        control_results = [
            _aggregate_comparison(
                family_records,
                family_baseline,
                _subset_outputs(outputs, ids),
                threshold=threshold,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed + 8200 + family_index * 100 + control_index,
            )
            for control_index, outputs in enumerate(controls)
        ]
        effect = _effect(result)
        control_effects = [_effect(row) for row in control_results]
        control_p = (1 + sum(value <= effect for value in control_effects)) / (
            len(control_effects) + 1
        )
        harmful_ci = result["delta_mean_harmful_score_ci"]
        benign_ci = result["delta_mean_benign_score_ci"]
        capability_ci = result["delta_capability_nll_ci"]
        retention = abs(effect) / abs(source_effect) if abs(source_effect) > 1e-12 else None
        sign_preserved = effect * source_effect > 0
        passed = (
            sign_preserved
            and effect < float(criteria["harmful_delta_lt"])
            and float(harmful_ci["high"]) < float(criteria["harmful_ci_high_lt"])
            and abs(float(benign_ci["estimate"]))
            <= float(criteria["abs_benign_delta_lte"])
            and float(capability_ci["estimate"])
            <= float(criteria["capability_nll_delta_lte"])
            and retention is not None
            and retention >= float(criteria["min_source_effect_retention"])
            and control_p <= float(criteria["random_control_p_lte"])
        )
        family_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "shift_family": family,
                "n_records": len(family_records),
                "n_pairs": len({record.pair_id for record in family_records}),
                "n_harmful": sum(record.harmful for record in family_records),
                "n_benign": sum(not record.harmful for record in family_records),
                "mediation": result,
                "source_controlled_harmful_delta": source_effect,
                "source_effect_retention": retention,
                "sign_preserved": sign_preserved,
                "matched_random_direction_controls": {
                    "n": len(control_effects),
                    "min_harmful_delta": min(control_effects),
                    "median_harmful_delta": median(control_effects),
                    "max_harmful_delta": max(control_effects),
                    "empirical_p": control_p,
                },
                "robustness_passed": passed,
            }
        )

    aggregate_effects = [_effect(row["mediation"]) for row in family_rows]
    output_dir = Path(output_dir)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "A8_adversarial_mediation_robustness",
        "status": "completed",
        "artifact_notice": (
            "Aggregate-only robustness evaluation of a frozen published mechanism. "
            "No prompts, generations, record identifiers, attack templates, or per-record scores "
            "are included."
        ),
        "source_artifact": str(source_dir),
        "model": source_summary["model"],
        "behavior": source_summary.get("behavior", experiment.behavior),
        "target_layer": target_layer,
        "direction": "mean_difference",
        "direction_source": str(source_dir / "discovery_tensors.pt"),
        "shifted_corpus_sha256": _sha256_file(corpus_path),
        "shifted_corpus_records": len(records),
        "shifted_corpus_pairs": len({record.pair_id for record in records}),
        "shift_families": family_rows,
        "all_families_passed": all(row["robustness_passed"] for row in family_rows),
        "harmful_delta_range": [min(aggregate_effects), max(aggregate_effects)],
        "bootstrap_unit": "pair_id",
        "bootstrap_samples": bootstrap_samples,
        "aggregate_only": True,
        "preregistration": preregistration,
    }
    _assert_aggregate_only(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in family_rows:
        payload = {
            "experiment": summary["experiment"],
            "model": summary["model"],
            "target_layer": target_layer,
            "aggregate_only": True,
            "shifted_corpus_sha256": summary["shifted_corpus_sha256"],
            "preregistration_hash": preregistration["hash"] if preregistration else None,
            **row,
        }
        _assert_aggregate_only(payload)
        write_json(output_dir / f"{row['shift_family']}.json", payload)
    write_json(output_dir / "a8_summary.json", summary)
    return summary


def _unrun_family(config: dict[str, Any], family: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "A8_adversarial_mediation_robustness",
        "status": "unrun",
        "shift_family": family,
        "aggregate_only": True,
        "n_records": None,
        "n_pairs": None,
        "harmful_delta": None,
        "harmful_ci_low": None,
        "harmful_ci_high": None,
        "source_effect_retention": None,
        "random_control_empirical_p": None,
        "robustness_passed": None,
        "preregistration_hash": config.get("preregistration_hash"),
    }


def _summary_family(row: dict[str, Any]) -> dict[str, Any]:
    harmful = row["mediation"]["delta_mean_harmful_score_ci"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "A8_adversarial_mediation_robustness",
        "status": "completed",
        "shift_family": row["shift_family"],
        "aggregate_only": True,
        "n_records": row["n_records"],
        "n_pairs": row["n_pairs"],
        "harmful_delta": harmful["estimate"],
        "harmful_ci_low": harmful["low"],
        "harmful_ci_high": harmful["high"],
        "source_effect_retention": row["source_effect_retention"],
        "random_control_empirical_p": row["matched_random_direction_controls"]["empirical_p"],
        "robustness_passed": row["robustness_passed"],
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A8 Adversarial Mediation Robustness",
        "",
        "| Status | Shift family | Pairs | Harmful delta [CI] | Source retention | Control p | Pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "completed":
            interval = (
                f"{row['harmful_delta']:+.4f} "
                f"[{row['harmful_ci_low']:+.4f}, {row['harmful_ci_high']:+.4f}]"
            )
            retention = f"{row['source_effect_retention']:.3f}"
            control_p = f"{row['random_control_empirical_p']:.4f}"
            passed = "yes" if row["robustness_passed"] else "no"
        else:
            interval = retention = control_p = passed = ""
        lines.append(
            f"| {row['status']} | {row['shift_family']} | "
            f"{'' if row.get('n_pairs') is None else row['n_pairs']} | {interval} | "
            f"{retention} | {control_p} | {passed} |"
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
    labels = [row["shift_family"].replace("_", " ") for row in completed]
    values = [row["harmful_delta"] for row in completed]
    low = [value - row["harmful_ci_low"] for value, row in zip(values, completed, strict=True)]
    high = [row["harmful_ci_high"] - value for value, row in zip(values, completed, strict=True)]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.errorbar(labels, values, yerr=[low, high], fmt="o", capsize=4)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_ylabel("Held-out harmful-score delta")
    axis.set_title("Frozen direction mediation under distribution shift")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def summarize_robustness_audit(
    *,
    config_path: str | Path,
    evidence_dir: str | Path = "results/robustness",
    output_json: str | Path = "results/robustness/a8_public_summary.json",
    output_table: str | Path = "docs/tables/robustness_audit.md",
    output_figure: str | Path = "docs/figures/robustness_audit.png",
) -> dict[str, Any]:
    config = load_robustness_config(config_path)
    evidence_dir = Path(evidence_dir)
    rows = []
    for family in SHIFT_FAMILIES:
        path = evidence_dir / f"{family}.json"
        if path.exists():
            row = read_json(path)
            rows.append(_summary_family(row) if "mediation" in row else row)
        else:
            rows.append(_unrun_family(config, family))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = evidence_dir / f"{row['shift_family']}.json"
        if row["status"] == "unrun" and not path.exists():
            write_json(path, row)
    _write_table(Path(output_table), rows)
    figure = _write_figure(Path(output_figure), rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "A8_adversarial_mediation_robustness",
        "aggregate_only": True,
        "cells_predeclared": len(rows),
        "cells_completed": sum(row["status"] == "completed" for row in rows),
        "cells_unrun": sum(row["status"] == "unrun" for row in rows),
        "rows": rows,
        "table": str(output_table),
        "figure": figure,
        "preregistration_hash": config.get("preregistration_hash"),
    }
    _assert_aggregate_only(summary)
    write_json(output_json, summary)
    return summary
