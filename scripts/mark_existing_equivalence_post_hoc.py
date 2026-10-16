#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

POST_HOC_INTERPRETATION = (
    "This is a post-hoc sensitivity analysis. Related Qwen results were examined before the "
    "equivalence and inferiority procedure was added, so passing tests are not confirmatory "
    "preregistered evidence."
)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def mark_artifact(artifact_dir: Path) -> dict:
    equivalence_path = artifact_dir / "equivalence_analysis.json"
    if not equivalence_path.exists():
        return {"artifact": str(artifact_dir), "status": "missing_equivalence_artifact"}
    equivalence = read_json(equivalence_path)
    legacy_bounds = equivalence.pop("preregistered_bounds", None)
    if "bounds" not in equivalence and legacy_bounds is not None:
        equivalence["bounds"] = legacy_bounds
    equivalence["analysis_design"] = "post_hoc_sensitivity"
    equivalence["equivalence_preregistration_hash"] = None
    equivalence["interpretation"] = POST_HOC_INTERPRETATION
    equivalence["effect_direction"] = (
        "Lower harmful scores after intervention are stronger suppression, so positive "
        "SAE-minus-baseline deltas favor the simple baseline."
    )
    write_json(equivalence_path, equivalence)

    summary_path = artifact_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        summary["equivalence_analysis_design"] = "post_hoc_sensitivity"
        write_json(summary_path, summary)

    protocol_path = artifact_dir / "protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        statistical = protocol.setdefault("statistical_protocol", {})
        statistical["equivalence_analysis_design"] = "post_hoc_sensitivity"
        statistical["equivalence_preregistration_hash"] = None
        write_json(protocol_path, protocol)
    return {"artifact": str(artifact_dir), "status": "marked_post_hoc"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark existing equivalence outputs as post-hoc sensitivity analyses"
    )
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help="Recursively find equivalence_analysis.json files below this root.",
    )
    args = parser.parse_args()
    artifact_dirs = {path.resolve(): path for path in args.artifacts}
    for root in args.root:
        for path in root.rglob("equivalence_analysis.json"):
            artifact_dirs[path.parent.resolve()] = path.parent
    if not artifact_dirs:
        parser.error("provide at least one artifact directory or --root")
    ordered = [artifact_dirs[key] for key in sorted(artifact_dirs, key=str)]
    print(json.dumps([mark_artifact(path) for path in ordered], indent=2))


if __name__ == "__main__":
    main()
