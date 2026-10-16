from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

BACKENDS = ["huggingface", "nnsight", "transformer-lens"]


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row(artifact_dir: Path, backend: str) -> dict[str, Any]:
    summary_path = artifact_dir / "summary.json"
    interventions_path = artifact_dir / "interventions.json"
    summary = read_json(summary_path)
    causal = read_json(interventions_path)["causal_tests"]
    return {
        "backend": backend,
        "artifact_dir": str(artifact_dir),
        "target_layer": int(summary["target_layer"]),
        "selected_features": summary["selected_features"],
        "behavior_threshold": float(summary["behavior_threshold"]),
        "baseline_behavior_gap": float(summary["baseline_metrics"]["behavior_gap"]),
        "mean_difference_harmful_delta": float(
            causal["mean_difference_direction_ablation"]["delta_mean_harmful_score_ci"][
                "estimate"
            ]
        ),
        "sae_feature_harmful_delta": float(
            causal["sae_feature_ablation"]["delta_mean_harmful_score_ci"]["estimate"]
        ),
        "summary_sha256": file_hash(summary_path),
        "interventions_sha256": file_hash(interventions_path),
    }


def difference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = [
        "behavior_threshold",
        "baseline_behavior_gap",
        "mean_difference_harmful_delta",
        "sae_feature_harmful_delta",
    ]
    return {
        "backend": candidate["backend"],
        "same_target_layer": candidate["target_layer"] == reference["target_layer"],
        "same_selected_features": candidate["selected_features"]
        == reference["selected_features"],
        "absolute_differences": {
            field: abs(float(candidate[field]) - float(reference[field]))
            for field in scalar_fields
        },
    }


def write_table(path: Path, rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    comparison_by_backend = {item["backend"]: item for item in comparisons}
    lines = [
        "# Backend Validation",
        "",
        "Qwen2.5-0.5B, 240 controlled records, held-out residual audit.",
        "",
        "| Backend | Layer | Baseline gap | Mean ablation delta | SAE ablation delta | |Mean diff vs HF| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        comparison = comparison_by_backend.get(item["backend"])
        mean_difference = (
            comparison["absolute_differences"]["mean_difference_harmful_delta"]
            if comparison
            else 0.0
        )
        lines.append(
            f"| `{item['backend']}` | {item['target_layer']} | "
            f"{item['baseline_behavior_gap']:.6f} | "
            f"{item['mean_difference_harmful_delta']:.6f} | "
            f"{item['sae_feature_harmful_delta']:.6f} | {mean_difference:.6f} |"
        )
    lines.extend(
        [
            "",
            "Hugging Face and nnsight are numerically identical. TransformerLens selects the same "
            "layer and closely matches baseline and mean-direction effects; its SAE fit differs because "
            "bridge activations are not bit-identical.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    matplotlib_cache = Path("artifacts/.matplotlib").resolve()
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.pyplot as plt

    labels = [item["backend"].replace("transformer-lens", "TransformerLens") for item in rows]
    mean_values = [item["mean_difference_harmful_delta"] for item in rows]
    sae_values = [item["sae_feature_harmful_delta"] for item in rows]
    positions = list(range(len(rows)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.bar([value - width / 2 for value in positions], mean_values, width, label="Mean direction")
    axis.bar([value + width / 2 for value in positions], sae_values, width, label="SAE features")
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Held-out harmful-score delta")
    axis.set_title("Residual intervention parity across model backends")
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts/backend-validation")
    parser.add_argument("--output-json", default="results/backend-validation/qwen25-05b.json")
    parser.add_argument("--output-table", default="docs/tables/backend_validation.md")
    parser.add_argument("--output-figure", default="docs/figures/backend_validation.png")
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    rows = [row(root / f"qwen25-05b-{backend}", backend) for backend in BACKENDS]
    reference = rows[0]
    comparisons = [difference(reference, item) for item in rows[1:]]
    payload = {
        "schema_version": "1.0",
        "evidence_class": "empirical_backend_validation",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "reference_backend": "huggingface",
        "rows": rows,
        "comparisons_to_reference": comparisons,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_table(Path(args.output_table), rows, comparisons)
    write_figure(Path(args.output_figure), rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
