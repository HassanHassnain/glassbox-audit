#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox_audit.evidence_explorer import build_evidence_explorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static Glassbox evidence explorer")
    parser.add_argument(
        "--artifacts-root",
        action="append",
        default=None,
        help="Artifact/result root to scan. May be passed more than once.",
    )
    parser.add_argument("--output-json", default="results/evidence_explorer/explorer.json")
    parser.add_argument("--output-html", default="docs/evidence/index.html")
    args = parser.parse_args()
    artifact_roots = args.artifacts_root or ["results"]
    bundle = build_evidence_explorer(
        artifact_roots=artifact_roots,
        output_json=args.output_json,
        output_html=args.output_html,
    )
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "output_html": args.output_html,
                "cell_count": bundle["cell_count"],
                "cells_by_level": bundle["cells_by_level"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
