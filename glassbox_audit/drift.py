from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .certification import build_certification_card

DRIFT_SCHEMA_VERSION = "1.0"


def _read_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _local_training_manifest(model_name: str) -> dict[str, Any] | None:
    model_path = Path(model_name)
    for candidate in [model_path / "training_manifest.json", model_path.parent / "training_manifest.json"]:
        if candidate.exists():
            return _read_json(candidate)
    return None


def _portable_model_config(config: Any) -> dict[str, Any]:
    payload = dict(config.__dict__)
    cache_dir = payload.get("cache_dir")
    if cache_dir and Path(cache_dir).is_absolute():
        try:
            payload["cache_dir"] = Path(cache_dir).relative_to(Path.cwd()).as_posix()
        except ValueError:
            payload["cache_dir"] = None
    return payload


def _estimate(row: dict[str, Any], key: str) -> float:
    return float(row[key]["estimate"])


def _ci_high_lt(row: dict[str, Any], key: str, threshold: float) -> bool:
    return float(row[key]["high"]) < threshold


def _ci_abs_estimate_lte(row: dict[str, Any], key: str, threshold: float) -> bool:
    return abs(float(row[key]["estimate"])) <= threshold


def _control_family(method_name: str) -> tuple[str, str] | None:
    if "sae" in method_name:
        return ("random_sae", "feature_ablation_control")
    if "direction" in method_name or "vector" in method_name:
        return ("random_direction", "ablation_control")
    return None


def _control_p_value(
    random_controls: list[dict[str, Any]],
    *,
    method_name: str,
    candidate_effect: float,
) -> dict[str, Any]:
    family = _control_family(method_name)
    if family is None:
        return {
            "control_family": None,
            "n_controls": 0,
            "empirical_one_sided_p": None,
        }
    control_method, control_intervention = family
    effects = [
        _estimate(row, "delta_mean_harmful_score_ci")
        for row in random_controls
        if row.get("method") == control_method and row.get("intervention") == control_intervention
    ]
    if not effects:
        return {
            "control_family": f"{control_method}:{control_intervention}",
            "n_controls": 0,
            "empirical_one_sided_p": None,
        }
    return {
        "control_family": f"{control_method}:{control_intervention}",
        "n_controls": len(effects),
        "empirical_one_sided_p": (1 + sum(value <= candidate_effect for value in effects))
        / (len(effects) + 1),
    }


def _rungs_from_evaluation(
    *,
    selected_features: list[int],
    causal_tests: dict[str, dict[str, Any]],
    random_controls: list[dict[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    if method_name not in causal_tests:
        raise ValueError(f"Drift evidence method {method_name!r} missing from causal tests")
    causal = causal_tests[method_name]
    control = _control_p_value(
        random_controls,
        method_name=method_name,
        candidate_effect=_estimate(causal, "delta_mean_harmful_score_ci"),
    )
    random_p = control["empirical_one_sided_p"]
    return {
        "separation": {
            "passed": bool(selected_features),
            "evidence": "frozen source selected_features are present",
        },
        "held_out_causal": {
            "passed": _ci_high_lt(causal, "delta_mean_harmful_score_ci", 0.0),
            "method": method_name,
            "harmful_delta_ci": causal["delta_mean_harmful_score_ci"],
        },
        "specificity": {
            "passed": _ci_abs_estimate_lte(causal, "delta_mean_benign_score_ci", 0.03),
            "threshold_abs_estimate": 0.03,
            "benign_delta_ci": causal["delta_mean_benign_score_ci"],
        },
        "capability_cost": {
            "passed": float(causal["delta_capability_nll_ci"]["estimate"]) <= 0.05,
            "threshold_estimate": 0.05,
            "capability_delta_ci": causal["delta_capability_nll_ci"],
        },
        "matched_controls": {
            "passed": random_p is not None and float(random_p) <= 0.05,
            **control,
        },
    }


def _rung_passed(rungs: dict[str, Any], name: str) -> bool | None:
    row = rungs.get(name)
    if row is None:
        return None
    return row.get("passed")


def _compare_rungs(source_rungs: dict[str, Any], new_rungs: dict[str, Any]) -> list[dict[str, Any]]:
    names = sorted(set(source_rungs) | set(new_rungs))
    rows = []
    for name in names:
        source_passed = _rung_passed(source_rungs, name)
        new_passed = _rung_passed(new_rungs, name)
        rows.append(
            {
                "rung": name,
                "source_passed": source_passed,
                "new_passed": new_passed,
                "changed": (
                    source_passed is not None
                    and new_passed is not None
                    and source_passed != new_passed
                ),
            }
        )
    return rows


def _status(source_passed: bool, new_passed: bool, rung_comparison: list[dict[str, Any]]) -> str:
    if source_passed and not new_passed:
        return "degraded"
    if not source_passed and new_passed:
        return "improved"
    if any(row["changed"] for row in rung_comparison):
        return "changed"
    return "stable"


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_dir": card["artifact_dir"],
        "model": card["model"],
        "behavior": card["behavior"],
        "evidence_class": card["evidence_class"],
        "certified_evidence_method": card["certified_evidence_method"],
        "overall_verdict": card["overall_verdict"],
        "rungs": card["rungs"],
        "corpus_hash": card.get("corpus_hash"),
        "preregistration_hash": card.get("preregistration_hash"),
    }


def write_drift_doc(report: dict[str, Any], output_doc: str | Path) -> Path:
    lines = [
        "# Mechanism Drift Report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Status: `{report['drift_status']}`",
        f"- Source model: `{report['source']['model']}`",
        f"- New model: `{report['new']['model']}`",
        f"- Certified method: `{report['certified_evidence_method']}`",
        "",
        "| Rung | Source | New | Changed |",
        "|---|---|---|---|",
    ]
    for row in report["rung_comparison"]:
        lines.append(
            f"| {row['rung']} | {row['source_passed']} | {row['new_passed']} | {row['changed']} |"
        )
    lines.extend(
        [
            "",
            "The JSON report contains the full rung evidence, confidence intervals, hashes, and",
            "baseline/intervention outputs when produced by live model-swap mode.",
        ]
    )
    output_doc = Path(output_doc)
    output_doc.parent.mkdir(parents=True, exist_ok=True)
    output_doc.write_text("\n".join(lines) + "\n")
    return output_doc


def compare_artifact_drift(
    *,
    source_artifact: str | Path,
    new_artifact: str | Path,
    output_json: str | Path,
    output_doc: str | Path | None = None,
    candidate_intervention: str | Path | None = None,
) -> dict[str, Any]:
    source_card = build_certification_card(
        source_artifact,
        candidate_intervention=candidate_intervention,
    )
    new_card = build_certification_card(
        new_artifact,
        candidate_intervention=candidate_intervention,
    )
    rung_comparison = _compare_rungs(source_card["rungs"], new_card["rungs"])
    report = {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "mode": "artifact_comparison",
        "artifact_notice": (
            "Mechanism-drift comparison between two completed Glassbox artifacts. This mode "
            "does not reapply source directions to the new model; use model-swap mode for a "
            "frozen-handle compatibility check."
        ),
        "source": _compact_card(source_card),
        "new": _compact_card(new_card),
        "certified_evidence_method": source_card["certified_evidence_method"],
        "mechanism_survives": bool(new_card["overall_verdict"]["passed"]),
        "drift_status": _status(
            bool(source_card["overall_verdict"]["passed"]),
            bool(new_card["overall_verdict"]["passed"]),
            rung_comparison,
        ),
        "rung_comparison": rung_comparison,
    }
    if output_doc is not None:
        report["output_doc"] = str(write_drift_doc(report, output_doc))
    _write_json(output_json, report)
    return report


def run_model_swap_drift(
    *,
    source_artifact: str | Path,
    output_json: str | Path,
    new_config: str | Path | None = None,
    new_model: str | None = None,
    device: str | None = None,
    max_records: int | None = None,
    random_controls: int | None = None,
    candidate_intervention: str | Path | None = None,
    output_doc: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    from .certification import load_intervention_spec
    from .config import load_config
    from .control_refresh import _config_from_manifest
    from .discovery import random_directions
    from .evaluation import calculate_metrics
    from .external_causal import _compare
    from .interventions import AblateDirection, AblateSAEFeatures, AddDirection
    from .modeling import load_model
    from .pipeline import (
        _public_sae_controls,
        _random_sae_control_protocol,
        _random_sae_controls,
        _sae_source_details,
        _serialize_outputs,
    )
    from .sae import load_sae_from_state
    from .types import PromptRecord
    from .utils import seed_everything

    source_artifact = Path(source_artifact)
    manifest = _read_json(source_artifact / "manifest.json")
    summary = _read_json(source_artifact / "summary.json")
    source_interventions = _read_json(source_artifact / "interventions.json")
    source_card = build_certification_card(
        source_artifact,
        candidate_intervention=candidate_intervention,
    )
    candidate = load_intervention_spec(candidate_intervention)
    method_name = source_card["certified_evidence_method"]
    config = _config_from_manifest(manifest)
    if new_config is not None:
        replacement = load_config(new_config)
        config.model = replacement.model
        config.behavior = replacement.behavior
    if new_model is not None:
        config.model.name = new_model
    if device is not None:
        config.model.device = device
        config.sae.device = device
    if random_controls is not None:
        config.evaluation.random_controls = random_controls
    config.evaluation.threshold = float(summary["behavior_threshold"])
    seed_everything(config.seed)

    records = [
        PromptRecord.from_dict(row)
        for row in _read_json(source_artifact / "records.json")
        if row["split"] == "test"
    ]
    if max_records is not None:
        records = records[:max_records]
    model = load_model(config.model, config.seed, behavior=config.behavior)
    target_layer = int(summary["target_layer"])
    tensors = torch.load(source_artifact / "discovery_tensors.pt", map_location="cpu")
    directions = {
        name: tensor.detach().float()
        for name, tensor in tensors["directions"].items()
        if name in {"sae_features", "mean_difference", "linear_probe"}
    }
    incompatible = [
        name
        for name, direction in directions.items()
        if int(direction.shape[-1]) != model.hidden_size
    ]
    if incompatible:
        raise ValueError(
            "Frozen direction hidden size does not match the new model for: "
            + ", ".join(incompatible)
        )
    sae_device = "cpu" if config.sae.device == "auto" else config.sae.device
    sae = load_sae_from_state(str(source_artifact / "sae.pt"), device=sae_device)
    if int(sae.input_dim) != model.hidden_size:
        raise ValueError(
            f"Frozen SAE input dimension {sae.input_dim} does not match new model "
            f"hidden size {model.hidden_size}"
        )

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
        config.seed + 2100,
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
            config.seed + 2110 + len(causal_tests),
        )
        ablation_outputs = model.evaluate(records, target_layer, AblateDirection(direction))
        causal_tests[f"{method}_direction_ablation"] = _compare(
            records,
            baseline,
            ablation_outputs,
            config.evaluation.threshold,
            config.evaluation.bootstrap_samples,
            config.evaluation.confidence,
            config.seed + 2120 + len(causal_tests),
        )

    random_control_rows: list[dict[str, Any]] = []
    for index, direction in enumerate(
        random_directions(model.hidden_size, config.evaluation.random_controls, config.seed + 2200)
    ):
        for intervention_name, intervention, scale in [
            (
                "steering_control",
                AddDirection(direction, config.evaluation.control_scale),
                config.evaluation.control_scale,
            ),
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
                config.seed + 2300 + len(random_control_rows),
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

    sae_source = _sae_source_details(config, sae)
    if sae_source["source"] == "saelens":
        random_sae = sae
        random_sae_controls = _public_sae_controls(
            sae,
            config.evaluation.random_controls,
            config.seed + 2400,
        )
    else:
        random_sae, random_sae_controls = _random_sae_controls(
            model.hidden_size,
            config.sae.expansion_factor,
            min(config.sae.top_k, model.hidden_size * config.sae.expansion_factor),
            config.evaluation.random_controls,
            config.seed + 2400,
            sae.activation_center,
        )
    random_sae_protocol = _random_sae_control_protocol(
        sae_source=sae_source,
        config=config,
        hidden_size=model.hidden_size,
    )
    for feature_index, direction in random_sae_controls:
        for intervention_name, intervention, scale in [
            (
                "steering_control",
                AddDirection(direction, config.evaluation.control_scale),
                config.evaluation.control_scale,
            ),
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
                config.seed + 2500 + len(random_control_rows),
            )
            row.update(
                {
                    "method": "random_sae",
                    "control_index": feature_index,
                    "intervention": intervention_name,
                    "scale": scale,
                    "control_protocol": random_sae_protocol["status"],
                }
            )
            random_control_rows.append(row)

    new_rungs = _rungs_from_evaluation(
        selected_features=selected_features,
        causal_tests=causal_tests,
        random_controls=random_control_rows,
        method_name=method_name,
    )
    rung_comparison = _compare_rungs(source_card["rungs"], new_rungs)
    required = [
        "separation",
        "held_out_causal",
        "specificity",
        "capability_cost",
        "matched_controls",
    ]
    new_passed = all(bool(new_rungs[name]["passed"]) for name in required)
    source_passed = bool(source_card["overall_verdict"]["passed"])
    report = {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "mode": "model_swap",
        "artifact_notice": (
            "Mechanism-drift model-swap run. Frozen source artifact directions, SAE latents, "
            "threshold, selected features, and validation-selected scales were evaluated on the "
            "new model without rediscovery or retuning."
        ),
        "source": _compact_card(source_card),
        "new": {
            "model": model.name,
            "model_config": _portable_model_config(config.model),
            "training_manifest": _local_training_manifest(config.model.name),
            "behavior": config.behavior,
            "target_layer": target_layer,
            "baseline_metrics": baseline_metrics,
            "rungs": new_rungs,
            "overall_verdict": {"passed": new_passed, "required_rungs": required},
        },
        "candidate_intervention": candidate,
        "certified_evidence_method": method_name,
        "mechanism_survives": new_passed,
        "drift_status": _status(source_passed, new_passed, rung_comparison),
        "rung_comparison": rung_comparison,
        "records": [record.__dict__ for record in records],
        "baseline_outputs": _serialize_outputs(baseline),
        "causal_tests": causal_tests,
        "random_controls": random_control_rows,
        "random_sae_control_protocol": random_sae_protocol,
    }
    if output_doc is not None:
        report["output_doc"] = str(write_drift_doc(report, output_doc))
    _write_json(output_json, report)
    return report
