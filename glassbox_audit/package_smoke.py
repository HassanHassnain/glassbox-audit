from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

from .config import EvaluationConfig, ExperimentConfig, ModelConfig, SAEConfig
from .pipeline import run_audit


def run_installed_smoke(output_dir: str | Path) -> dict[str, object]:
    """Run the complete toy audit using only resources shipped in the wheel."""
    fixture = files("glassbox_audit.resources").joinpath("toy_refusal_pairs.jsonl")
    with as_file(fixture) as dataset_path:
        config = ExperimentConfig(
            seed=17,
            evidence_class="synthetic_fixture",
            dataset_path=str(dataset_path),
            model=ModelConfig(
                backend="toy",
                name="glassbox-toy-refusal",
                layers=[0, 1, 2, 3],
                dtype="float32",
                device="cpu",
            ),
            sae=SAEConfig(
                expansion_factor=2,
                top_k=4,
                epochs=12,
                batch_size=32,
                n_features_to_intervene=2,
                device="cpu",
            ),
            evaluation=EvaluationConfig(
                bootstrap_samples=20,
                feature_budgets=[1, 2, 5],
                feature_budget_l1_epochs=12,
                steering_scales=[-3.0, 1.0],
                random_controls=1,
                control_scale=-2.0,
            ),
        )
        summary = run_audit(config, output_dir)

    required = [
        "manifest.json",
        "summary.json",
        "interventions.json",
        "sae_metrics.json",
        "REPORT.md",
    ]
    missing = [name for name in required if not (Path(output_dir) / name).is_file()]
    if missing:
        raise RuntimeError(f"Installed smoke did not produce required artifacts: {missing}")
    return {
        "schema_version": "1.0",
        "status": "passed",
        "output_dir": str(Path(output_dir).resolve()),
        "required_artifacts": required,
        "target_layer": summary["target_layer"],
        "test_pairs": summary["test_pairs"],
    }
