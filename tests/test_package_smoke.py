from importlib.resources import files

from glassbox_audit.package_smoke import run_installed_smoke


def test_packaged_toy_fixture_runs_complete_audit(tmp_path):
    fixture = files("glassbox_audit.resources").joinpath("toy_refusal_pairs.jsonl")
    assert fixture.is_file()

    result = run_installed_smoke(tmp_path)
    assert result["status"] == "passed"
    assert result["test_pairs"] == 2
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "interventions.json").is_file()
