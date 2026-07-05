#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox_audit.leaderboard import build_leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static Glassbox leaderboard")
    parser.add_argument(
        "--artifacts-root",
        action="append",
        default=None,
        help="Artifact/result root to scan. May be passed more than once.",
    )
    parser.add_argument(
        "--output-json",
        default="results/leaderboard/leaderboard.json",
    )
    parser.add_argument(
        "--output-html",
        default="docs/leaderboard/index.html",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    artifact_roots = args.artifacts_root or ["results"]
    leaderboard = build_leaderboard(
        artifact_roots=artifact_roots,
        output_json=args.output_json,
        output_html=args.output_html,
        strict=args.strict,
    )
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "output_html": args.output_html,
                "artifact_count": leaderboard["artifact_count"],
                "row_count": leaderboard["row_count"],
                "artifact_errors": len(leaderboard["artifact_errors"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
