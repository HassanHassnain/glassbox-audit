from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

A2_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "slug": "qwen25-05b-instruct",
        "family": "qwen2.5",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.5,
        "layers": [3, 6, 9, 12, 15, 18, 21],
    },
    {
        "slug": "qwen25-15b-instruct",
        "family": "qwen2.5",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "layers": [4, 8, 12, 16, 20, 24, 27],
    },
    {
        "slug": "qwen25-3b-instruct",
        "family": "qwen2.5",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.0,
        "layers": [6, 12, 18, 24, 30, 35],
    },
    {
        "slug": "qwen25-7b-instruct",
        "family": "qwen2.5",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7.0,
        "layers": [4, 8, 12, 16, 20, 24, 27],
    },
    {
        "slug": "qwen25-14b-instruct",
        "family": "qwen2.5",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "params_b": 14.0,
        "layers": [8, 16, 24, 32, 40, 47],
    },
    {
        "slug": "llama32-1b-instruct",
        "family": "llama",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "params_b": 1.0,
        "layers": [2, 4, 6, 8, 12, 15],
    },
    {
        "slug": "llama32-3b-instruct",
        "family": "llama",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "params_b": 3.0,
        "layers": [4, 8, 12, 16, 20, 24, 27],
    },
    {
        "slug": "llama31-8b-instruct",
        "family": "llama",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "params_b": 8.0,
        "layers": [4, 8, 12, 16, 20, 24, 28, 31],
    },
    {
        "slug": "gemma2-2b-it",
        "family": "gemma",
        "model": "google/gemma-2-2b-it",
        "params_b": 2.0,
        "layers": [4, 8, 12, 16, 20, 25],
    },
    {
        "slug": "gemma2-9b-it",
        "family": "gemma",
        "model": "google/gemma-2-9b-it",
        "params_b": 9.0,
        "layers": [7, 14, 21, 28, 35, 41],
    },
    {
        "slug": "mistral-7b-instruct-v03",
        "family": "mistral",
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "params_b": 7.0,
        "layers": [4, 8, 12, 16, 20, 24, 28, 31],
    },
]


def _read_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def _spec_by_model() -> dict[str, dict[str, Any]]:
    return {str(spec["model"]): spec for spec in A2_MODEL_SPECS}


def _spec_by_slug() -> dict[str, dict[str, Any]]:
    return {str(spec["slug"]): spec for spec in A2_MODEL_SPECS}


def _preregistration_hash(path: str | Path | None, configured_hash: str | None) -> str | None:
    if configured_hash:
        return configured_hash
    if path is None:
        return None
    frozen = _read_json(path)
    return str(frozen["preregistration_hash"])


def _config_payload(
    spec: dict[str, Any],
    *,
    dataset_path: str,
    preregistration_path: str | None,
    preregistration_hash: str | None,
    device: str,
    sae_device: str,
    cache_dir: str | None,
    seed: int,
    tokens_per_prompt: int,
    samples_per_feature: float,
    epochs: int,
    bootstrap_samples: int,
    random_controls: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "mode": "real_audit",
        "behavior": "refusal",
        "evidence_class": "empirical_cross_model_sweep_preregistered",
        "dataset_path": dataset_path,
        "preregistration_path": preregistration_path,
        "preregistration_hash": preregistration_hash,
        "target_layer": None,
        "model": {
            "backend": "huggingface",
            "name": spec["model"],
            "layers": spec["layers"],
            "dtype": "float16",
            "device": device,
            "trust_remote_code": False,
            "cache_dir": cache_dir,
        },
        "sae": {
            "source": "trained",
            "expansion_factor": 2,
            "top_k": 32,
            "epochs": epochs,
            "batch_size": 512,
            "learning_rate": 0.0005,
            "l1_coefficient": 0.0001,
            "n_features_to_intervene": 5,
            "device": sae_device,
            "tokens_per_prompt": tokens_per_prompt,
            "max_activation_samples": None,
            "samples_per_feature": samples_per_feature,
            "activation_subsample_seed": seed,
        },
        "evaluation": {
            "threshold": None,
            "bootstrap_samples": bootstrap_samples,
            "confidence": 0.95,
            "steering_scales": [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0],
            "random_controls": random_controls,
            "control_scale": -4.0,
            "layerwise_controls": True,
        },
    }


def write_model_sweep_configs(
    *,
    output_dir: str | Path,
    dataset_path: str = "data/refusal_controlled_v2_1000.jsonl",
    preregistration_path: str | None = "docs/preregistration_model_sweep.frozen.json",
    preregistration_hash: str | None = None,
    device: str = "cuda:0",
    sae_device: str | None = None,
    cache_dir: str | None = None,
    seed: int = 17,
    tokens_per_prompt: int = 64,
    samples_per_feature: float = 8.0,
    epochs: int = 220,
    bootstrap_samples: int = 2500,
    random_controls: int = 24,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sae_device = sae_device or device
    frozen_hash = _preregistration_hash(preregistration_path, preregistration_hash)
    rows = []
    for spec in A2_MODEL_SPECS:
        payload = _config_payload(
            spec,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
            preregistration_hash=frozen_hash,
            device=device,
            sae_device=sae_device,
            cache_dir=cache_dir,
            seed=seed,
            tokens_per_prompt=tokens_per_prompt,
            samples_per_feature=samples_per_feature,
            epochs=epochs,
            bootstrap_samples=bootstrap_samples,
            random_controls=random_controls,
        )
        path = output_dir / f"{spec['slug']}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        rows.append(
            {
                "slug": spec["slug"],
                "model": spec["model"],
                "params_b": spec["params_b"],
                "config": str(path),
            }
        )
    return {
        "schema_version": "1.0",
        "sweep": "A2_cross_model_refusal",
        "configs": rows,
        "config_dir": str(output_dir),
        "preregistration_path": preregistration_path,
        "preregistration_hash": frozen_hash,
        "samples_per_feature": samples_per_feature,
        "tokens_per_prompt": tokens_per_prompt,
    }


def _estimate(row: dict[str, Any], key: str = "delta_mean_harmful_score_ci") -> float | None:
    value = row.get(key)
    if not isinstance(value, dict) or value.get("estimate") is None:
        return None
    return float(value["estimate"])


def _config_row(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    model_name = str(raw.get("model", {}).get("name", path.stem))
    spec = _spec_by_model().get(model_name, {})
    return {
        "schema_version": "1.0",
        "status": "unrun",
        "slug": str(spec.get("slug") or path.stem),
        "model": model_name,
        "family": spec.get("family"),
        "params_b": spec.get("params_b"),
        "config": str(path),
        "artifact_dir": None,
        "target_layer": None,
        "target_layer_depth_fraction": None,
        "refusal_direction_effect_size": None,
        "sae_harmful_delta": None,
        "mean_difference_harmful_delta": None,
        "linear_probe_harmful_delta": None,
        "sae_minus_mean_gap": None,
        "sae_minus_probe_gap": None,
        "test_records": None,
        "test_pairs": None,
    }


def write_model_sweep_failure(
    *,
    config_path: str | Path,
    artifact_dir: str | Path,
    failure_stage: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    config_path = Path(config_path)
    row = _config_row(config_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "schema_version": "1.0",
        "status": "failed",
        "slug": row["slug"],
        "model": row["model"],
        "config": str(config_path),
        "artifact_dir": str(artifact_dir),
        "failure_stage": failure_stage,
        "error_type": error_type,
        "error_message": error_message,
    }
    _write_json(artifact_dir / "failure.json", failure)
    return failure


def _failed_row(artifact_dir: Path) -> dict[str, Any]:
    failure = _read_json(artifact_dir / "failure.json")
    model_name = str(failure["model"])
    spec = _spec_by_model().get(model_name, {})
    return {
        "schema_version": "1.0",
        "status": "failed",
        "slug": str(failure.get("slug") or spec.get("slug") or _slug(model_name)),
        "model": model_name,
        "family": spec.get("family"),
        "params_b": spec.get("params_b"),
        "config": failure.get("config"),
        "artifact_dir": str(artifact_dir),
        "target_layer": None,
        "target_layer_depth_fraction": None,
        "refusal_direction_effect_size": None,
        "sae_harmful_delta": None,
        "mean_difference_harmful_delta": None,
        "linear_probe_harmful_delta": None,
        "sae_minus_mean_gap": None,
        "sae_minus_probe_gap": None,
        "test_records": None,
        "test_pairs": None,
        "failure_stage": failure.get("failure_stage"),
        "error_type": failure.get("error_type"),
        "error_message": failure.get("error_message"),
    }


def _completed_row(artifact_dir: Path) -> dict[str, Any]:
    summary = _read_json(artifact_dir / "summary.json")
    manifest = _read_json(artifact_dir / "manifest.json")
    interventions = _read_json(artifact_dir / "interventions.json")
    layer_scan = _read_json(artifact_dir / "layer_scan.json")
    model_name = str(summary["model"])
    spec = _spec_by_model().get(model_name, {})
    config = manifest.get("config", {})
    layers = config.get("model", {}).get("layers", [])
    target_layer = int(summary["target_layer"])
    selected_layer = next(
        (row for row in layer_scan if int(row["layer"]) == target_layer),
        {},
    )
    causal = interventions["causal_tests"]
    sae_delta = _estimate(causal["sae_feature_ablation"])
    mean_delta = _estimate(causal["mean_difference_direction_ablation"])
    probe_delta = _estimate(causal["linear_probe_direction_ablation"])
    max_layer = max([int(layer) for layer in layers], default=target_layer)
    depth_fraction = (target_layer + 1) / (max_layer + 1)
    return {
        "schema_version": "1.0",
        "status": "completed",
        "slug": str(spec.get("slug") or _slug(model_name)),
        "model": model_name,
        "family": spec.get("family"),
        "params_b": spec.get("params_b"),
        "config": None,
        "artifact_dir": str(artifact_dir),
        "target_layer": target_layer,
        "target_layer_depth_fraction": depth_fraction,
        "refusal_direction_effect_size": selected_layer.get("projection_effect_size"),
        "sae_harmful_delta": sae_delta,
        "mean_difference_harmful_delta": mean_delta,
        "linear_probe_harmful_delta": probe_delta,
        "sae_minus_mean_gap": (
            sae_delta - mean_delta if sae_delta is not None and mean_delta is not None else None
        ),
        "sae_minus_probe_gap": (
            sae_delta - probe_delta if sae_delta is not None and probe_delta is not None else None
        ),
        "test_records": summary.get("test_records"),
        "test_pairs": summary.get("test_pairs"),
        "preregistration_hash": summary.get("preregistration_hash"),
        "sae_outperforms_simple_baselines": summary.get("sae_outperforms_simple_baselines"),
        "equivalence_tests_passed": summary.get("equivalence_tests_passed"),
        "equivalence_analysis_design": summary.get(
            "equivalence_analysis_design", "post_hoc_sensitivity"
        ),
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A2 Model Sweep",
        "",
        "| Status | Model | Params B | Target layer | Depth frac | Refusal d | SAE delta | SAE-mean gap | Test pairs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {status} | {model} | {params_b} | {layer} | {depth} | {effect} | {sae} | {gap} | {pairs} |".format(
                status=row["status"],
                model=row["model"],
                params_b="" if row.get("params_b") is None else f"{row['params_b']:.1f}",
                layer="" if row.get("target_layer") is None else row["target_layer"],
                depth=(
                    ""
                    if row.get("target_layer_depth_fraction") is None
                    else f"{row['target_layer_depth_fraction']:.3f}"
                ),
                effect=(
                    ""
                    if row.get("refusal_direction_effect_size") is None
                    else f"{row['refusal_direction_effect_size']:.3f}"
                ),
                sae=(
                    ""
                    if row.get("sae_harmful_delta") is None
                    else f"{row['sae_harmful_delta']:+.4f}"
                ),
                gap=(
                    ""
                    if row.get("sae_minus_mean_gap") is None
                    else f"{row['sae_minus_mean_gap']:+.4f}"
                ),
                pairs="" if row.get("test_pairs") is None else row["test_pairs"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> str | None:
    completed = [
        row
        for row in rows
        if row["status"] == "completed" and row.get("params_b") is not None
    ]
    if not completed:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    completed = sorted(completed, key=lambda row: float(row["params_b"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    series = [
        ("refusal_direction_effect_size", "Refusal direction d"),
        ("sae_minus_mean_gap", "SAE minus mean delta"),
        ("target_layer_depth_fraction", "Target depth fraction"),
    ]
    for ax, (key, title) in zip(axes, series, strict=True):
        series_rows = [row for row in completed if row.get(key) is not None]
        ax.plot(
            [row["params_b"] for row in series_rows],
            [row[key] for row in series_rows],
            marker="o",
        )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Parameters (B)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def summarize_model_sweep(
    *,
    configs_dir: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    artifact_dirs: list[str | Path] | None = None,
    output_json: str | Path = "results/model-sweep/a2_summary.json",
    output_table: str | Path = "docs/tables/model_sweep.md",
    output_figure: str | Path = "docs/figures/model_sweep.png",
) -> dict[str, Any]:
    rows_by_slug: dict[str, dict[str, Any]] = {}
    if configs_dir is not None:
        for path in sorted(Path(configs_dir).glob("*.yaml")):
            row = _config_row(path)
            rows_by_slug[row["slug"]] = row

    candidate_artifacts = []
    if artifacts_root is not None:
        root = Path(artifacts_root)
        if root.exists():
            candidate_artifacts.extend(path for path in sorted(root.iterdir()) if path.is_dir())
    if artifact_dirs:
        candidate_artifacts.extend(Path(path) for path in artifact_dirs)

    for artifact_dir in candidate_artifacts:
        if (artifact_dir / "summary.json").exists():
            row = _completed_row(artifact_dir)
        elif (artifact_dir / "failure.json").exists():
            row = _failed_row(artifact_dir)
        else:
            continue
        existing = rows_by_slug.get(row["slug"])
        if existing is not None and existing.get("config"):
            row["config"] = existing["config"]
        rows_by_slug[row["slug"]] = row

    for spec in A2_MODEL_SPECS:
        rows_by_slug.setdefault(
            str(spec["slug"]),
            {
                "schema_version": "1.0",
                "status": "unrun",
                "slug": spec["slug"],
                "model": spec["model"],
                "family": spec["family"],
                "params_b": spec["params_b"],
                "config": None,
                "artifact_dir": None,
                "target_layer": None,
                "target_layer_depth_fraction": None,
                "refusal_direction_effect_size": None,
                "sae_harmful_delta": None,
                "mean_difference_harmful_delta": None,
                "linear_probe_harmful_delta": None,
                "sae_minus_mean_gap": None,
                "sae_minus_probe_gap": None,
                "test_records": None,
                "test_pairs": None,
            },
        )

    rows = sorted(
        rows_by_slug.values(),
        key=lambda row: (float(row["params_b"] or 0), str(row["model"])),
    )
    output_json = Path(output_json)
    model_dir = output_json.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        _write_json(model_dir / f"{row['slug']}.json", row)
    figure_path = _write_figure(Path(output_figure), rows)
    _write_table(Path(output_table), rows)
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] == "failed"]
    artifact = {
        "schema_version": "1.0",
        "sweep": "A2_cross_model_refusal",
        "artifact_notice": (
            "A2 cross-model sweep summary. Completed rows are read from frozen audit artifacts; "
            "failed and unrun rows remain explicit so the preregistered grid is visible."
        ),
        "models_predeclared": len(A2_MODEL_SPECS),
        "models_completed": len(completed),
        "models_failed": len(failed),
        "rows": rows,
        "per_model_evidence_dir": str(model_dir),
        "table": str(output_table),
        "figure": figure_path,
    }
    _write_json(output_json, artifact)
    return artifact
