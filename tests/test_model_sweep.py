import json

from glassbox_audit.model_sweep import summarize_model_sweep, write_model_sweep_configs


def test_model_sweep_configs_and_unrun_summary(tmp_path):
    prereg = tmp_path / "model_sweep.frozen.json"
    prereg.write_text(json.dumps({"preregistration_hash": "abc123"}))
    configs = tmp_path / "configs"
    result = write_model_sweep_configs(
        output_dir=configs,
        dataset_path="data/refusal_controlled_v2_1000.jsonl",
        preregistration_path=str(prereg),
        device="cpu",
        sae_device="cpu",
    )

    assert result["preregistration_hash"] == "abc123"
    assert len(result["configs"]) == 11
    assert (configs / "qwen25-15b-instruct.yaml").exists()

    summary = summarize_model_sweep(
        configs_dir=configs,
        artifacts_root=tmp_path / "missing-artifacts",
        output_json=tmp_path / "results" / "a2_summary.json",
        output_table=tmp_path / "tables" / "model_sweep.md",
        output_figure=tmp_path / "figures" / "model_sweep.png",
    )

    assert summary["models_predeclared"] == 11
    assert summary["models_completed"] == 0
    assert all(row["status"] == "unrun" for row in summary["rows"])
    assert (tmp_path / "results" / "qwen25-15b-instruct.json").exists()
    assert (tmp_path / "tables" / "model_sweep.md").exists()
