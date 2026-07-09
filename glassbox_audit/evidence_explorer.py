from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

EXPLORER_SCHEMA_VERSION = "1.0"


CLAIM_LEVELS = [
    {
        "level": 1,
        "claim": "Behavior-only score pattern",
        "required_evidence": "Frozen scoring on held-out or external prompts",
        "status": "supported",
    },
    {
        "level": 2,
        "claim": "Residual-stream causal direction",
        "required_evidence": "Held-out residual intervention with uncertainty and controls",
        "status": "supported",
    },
    {
        "level": 3,
        "claim": "Component-output localization",
        "required_evidence": "Attention, MLP, and residual component interventions with specificity",
        "status": "partial",
    },
    {
        "level": 4,
        "claim": "Candidate path/component mechanism",
        "required_evidence": "Necessity, sufficiency, specificity, controls, and replication",
        "status": "not_supported",
    },
    {
        "level": 5,
        "claim": "Confirmed circuit",
        "required_evidence": "Component mechanism with held-out and cross-model replication",
        "status": "not_supported",
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


def _sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_artifact_dirs(roots: list[str | Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates = [root] if (root / "summary.json").exists() else []
        candidates.extend(path.parent for path in root.rglob("summary.json"))
        for path in candidates:
            if (path / "manifest.json").exists() and (path / "interventions.json").exists():
                found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda path: str(path))


def _ci(row: dict[str, Any], key: str) -> dict[str, float] | None:
    value = row.get(key)
    if not isinstance(value, dict):
        return None
    return {
        "estimate": float(value["estimate"]),
        "low": float(value["low"]),
        "high": float(value["high"]),
    }


def _cell(
    *,
    level: int,
    source_file: str | Path,
    title: str,
    model: str | None,
    behavior: str | None,
    family: str,
    metric: str,
    value: float | None = None,
    ci: dict[str, float] | None = None,
    verdict: str = "informational",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "level": level,
        "source_file": str(source_file),
        "source_sha256": _sha256_file(source_file),
        "title": title,
        "model": model,
        "behavior": behavior,
        "family": family,
        "metric": metric,
        "value": value,
        "ci": ci,
        "verdict": verdict,
        "details": details or {},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return {"id": f"cell-{digest}", **payload}


def _artifact_cells(artifact_dir: Path) -> list[dict[str, Any]]:
    cells = []
    summary_path = artifact_dir / "summary.json"
    interventions_path = artifact_dir / "interventions.json"
    summary = _read_json(summary_path)
    interventions = _read_json(interventions_path)
    model = summary.get("model")
    behavior = summary.get("behavior")
    baseline = summary.get("baseline_metrics", {})
    for metric in ["behavior_gap", "harmful_refusal_rate", "benign_overrefusal_rate"]:
        if metric in baseline:
            cells.append(
                _cell(
                    level=1,
                    source_file=summary_path,
                    title=f"{artifact_dir.name} baseline {metric}",
                    model=model,
                    behavior=behavior,
                    family="baseline",
                    metric=metric,
                    value=float(baseline[metric]),
                    verdict="supported",
                    details={"artifact_dir": str(artifact_dir)},
                )
            )
    for method, row in sorted(interventions.get("causal_tests", {}).items()):
        harmful = _ci(row, "delta_mean_harmful_score_ci")
        benign = _ci(row, "delta_mean_benign_score_ci")
        nll = _ci(row, "delta_capability_nll_ci")
        if harmful is not None:
            cells.append(
                _cell(
                    level=2,
                    source_file=interventions_path,
                    title=f"{artifact_dir.name} {method}",
                    model=model,
                    behavior=behavior,
                    family="held_out_causal",
                    metric="harmful_score_delta",
                    value=harmful["estimate"],
                    ci=harmful,
                    verdict="supported" if harmful["high"] < 0 else "failed",
                    details={"method": method, "benign_delta": benign, "capability_delta": nll},
                )
            )
    component_path = artifact_dir / "component_localization.json"
    if not component_path.exists():
        component_path = artifact_dir / "component_localization_layerwise.json"
    if component_path.exists():
        component = _read_json(component_path)
        for row in component.get("components", []):
            harmful = _ci(row, "delta_mean_harmful_score_ci")
            if harmful is None:
                continue
            cells.append(
                _cell(
                    level=3,
                    source_file=component_path,
                    title=f"{artifact_dir.name} {row.get('component')} layer {row.get('layer')}",
                    model=model,
                    behavior=behavior,
                    family="component_localization",
                    metric="harmful_score_delta",
                    value=harmful["estimate"],
                    ci=harmful,
                    verdict="partial" if row.get("component") == "residual" else "inconclusive",
                    details={
                        "component": row.get("component"),
                        "layer": row.get("layer"),
                        "intervention": row.get("intervention"),
                    },
                )
            )
    return cells


def _external_causal_cells(root: Path) -> list[dict[str, Any]]:
    cells = []
    for path in sorted((root / "external-causal").glob("*.json")):
        artifact = _read_json(path)
        model = artifact.get("model")
        for method, row in artifact.get("summary", {}).get("main_methods", {}).items():
            ci = row.get("harmful_score_ci")
            if not isinstance(ci, list) or len(ci) != 2:
                continue
            estimate = float(row["harmful_score_delta"])
            cells.append(
                _cell(
                    level=2,
                    source_file=path,
                    title=f"{path.stem} {method}",
                    model=model,
                    behavior="refusal",
                    family="external_causal_transfer",
                    metric="harmful_score_delta",
                    value=estimate,
                    ci={"estimate": estimate, "low": float(ci[0]), "high": float(ci[1])},
                    verdict="supported" if float(ci[1]) < 0 else "failed",
                    details={"method": method, "source_artifact": artifact.get("source_artifact")},
                )
            )
    return cells


def _component_path_cells(root: Path) -> list[dict[str, Any]]:
    cells = []
    for path in sorted((root / "component-path").glob("*.json")):
        artifact = _read_json(path)
        for entry in artifact.get("artifacts", []):
            model = entry.get("model")
            for row in entry.get("component_outputs", []):
                ci = row.get("harmful_ci")
                if not isinstance(ci, list) or len(ci) != 2:
                    continue
                estimate = float(row["harmful_delta"])
                cells.append(
                    _cell(
                        level=3,
                        source_file=path,
                        title=f"{Path(entry.get('artifact', path)).stem} {row.get('component')}",
                        model=model,
                        behavior="refusal",
                        family="component_path",
                        metric="harmful_score_delta",
                        value=estimate,
                        ci={"estimate": estimate, "low": float(ci[0]), "high": float(ci[1])},
                        verdict="partial"
                        if row.get("component") == "residual" and float(ci[1]) < 0
                        else "inconclusive",
                        details={
                            "component": row.get("component"),
                            "layer": row.get("layer"),
                            "nll_delta": row.get("nll_delta"),
                        },
                    )
                )
            cells.append(
                _cell(
                    level=4,
                    source_file=path,
                    title=f"{entry.get('source_artifact')} confirmed circuit status",
                    model=model,
                    behavior="refusal",
                    family="path_mechanism",
                    metric="confirmed_circuit",
                    value=1.0 if entry.get("confirmed_circuit") else 0.0,
                    verdict="supported" if entry.get("confirmed_circuit") else "failed",
                    details={"interpretation": entry.get("interpretation")},
                )
            )
    return cells


def build_evidence_explorer(
    *,
    artifact_roots: list[str | Path],
    output_json: str | Path,
    output_html: str | Path | None = None,
) -> dict[str, Any]:
    cells = []
    artifact_dirs = _discover_artifact_dirs(artifact_roots)
    for artifact_dir in artifact_dirs:
        cells.extend(_artifact_cells(artifact_dir))
    for root in [Path(path) for path in artifact_roots if Path(path).exists()]:
        cells.extend(_external_causal_cells(root))
        cells.extend(_component_path_cells(root))
    cells.append(
        _cell(
            level=5,
            source_file="docs/CLAIM_LADDER.md",
            title="Confirmed circuit claim",
            model=None,
            behavior=None,
            family="claim_boundary",
            metric="confirmed_circuit_supported",
            value=0.0,
            verdict="failed",
            details={"reason": "No component/path mechanism currently passes Level 5."},
        )
    )
    cells.sort(key=lambda row: (row["level"], row["family"], row["title"]))
    bundle = {
        "schema_version": EXPLORER_SCHEMA_VERSION,
        "artifact_notice": (
            "Static interactive evidence explorer. Cells are derived from frozen Glassbox artifacts "
            "and grouped by conservative claim-ladder level."
        ),
        "artifact_roots": [str(path) for path in artifact_roots],
        "levels": CLAIM_LEVELS,
        "cells": cells,
        "cell_count": len(cells),
        "cells_by_level": {
            str(level["level"]): sum(cell["level"] == level["level"] for cell in cells)
            for level in CLAIM_LEVELS
        },
    }
    _write_json(output_json, bundle)
    if output_html is not None:
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(render_evidence_explorer_html(bundle))
        bundle["output_html"] = str(output_html)
        _write_json(output_json, bundle)
    return bundle


def render_evidence_explorer_html(bundle: dict[str, Any]) -> str:
    payload = json.dumps(bundle, sort_keys=True).replace("</", "<\\/")
    buttons = "".join(
        f"<button type=\"button\" data-level=\"{level['level']}\">"
        f"Level {level['level']}: {html.escape(level['claim'])}</button>"
        for level in bundle["levels"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Glassbox Evidence Explorer</title>
  <style>
    :root {{ color-scheme: light dark; --line: #8a8f9855; --accent: #0f766e; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; line-height: 1.45; }}
    header, main {{ max-width: 1180px; margin: 0 auto; padding: 1rem; }}
    header {{ border-bottom: 1px solid var(--line); }}
    .levels {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }}
    button {{ border: 1px solid var(--line); background: Canvas; color: CanvasText; padding: 0.5rem 0.7rem; cursor: pointer; }}
    button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 700; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }}
    .stat {{ border: 1px solid var(--line); padding: 0.75rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 0.45rem; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: Canvas; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <header>
    <h1>Glassbox Evidence Explorer</h1>
    <div class="levels">{buttons}</div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <table>
      <thead><tr><th>Verdict</th><th>Family</th><th>Title</th><th>Metric</th><th>Value</th><th>CI</th><th>Source</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script type="application/json" id="evidence-data">{payload}</script>
  <script>
    const data = JSON.parse(document.getElementById('evidence-data').textContent);
    let active = 1;
    const esc = value => String(value ?? '').replace(/[&<>"']/g, char => (
      {{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}}[char]
    ));
    const fmt = value => value === null || value === undefined ? '' : Number(value).toFixed(4);
    const ci = value => value ? `${{fmt(value.estimate)}} [${{fmt(value.low)}}, ${{fmt(value.high)}}]` : '';
    function render() {{
      document.querySelectorAll('button[data-level]').forEach(button => {{
        button.classList.toggle('active', Number(button.dataset.level) === active);
      }});
      const level = data.levels.find(item => item.level === active);
      const cells = data.cells.filter(cell => cell.level === active);
      document.getElementById('summary').innerHTML = `
        <div class="stat"><strong>Claim</strong><br>${{esc(level.claim)}}</div>
        <div class="stat"><strong>Status</strong><br>${{esc(level.status)}}</div>
        <div class="stat"><strong>Evidence cells</strong><br>${{cells.length}}</div>
      `;
      document.getElementById('rows').innerHTML = cells.map(cell => `
        <tr>
          <td>${{esc(cell.verdict)}}</td>
          <td>${{esc(cell.family)}}</td>
          <td>${{esc(cell.title)}}</td>
          <td>${{esc(cell.metric)}}</td>
          <td>${{fmt(cell.value)}}</td>
          <td>${{ci(cell.ci)}}</td>
          <td><code>${{esc(cell.source_file)}}</code></td>
        </tr>
      `).join('');
    }}
    document.querySelectorAll('button[data-level]').forEach(button => {{
      button.addEventListener('click', () => {{ active = Number(button.dataset.level); render(); }});
    }});
    render();
  </script>
</body>
</html>
"""
