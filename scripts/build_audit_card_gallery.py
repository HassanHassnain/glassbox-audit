#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox_audit.certification import write_certification_card  # noqa: E402

CARDS = [
    {
        "id": "qwen-real-audit",
        "title": "Qwen2.5-1.5B main audit",
        "artifact_dir": "artifacts/qwen-real-audit",
        "expected": "pass",
    },
    {
        "id": "qwen2.5-3b-replication",
        "title": "Qwen2.5-3B replication",
        "artifact_dir": "artifacts/qwen2.5-3b-replication-preregistered",
        "expected": "fail",
    },
]


def main() -> None:
    gallery_rows = []
    for item in CARDS:
        output_json = Path("results/certification") / f"{item['id']}.card.json"
        output_html = Path("docs/audit-cards") / f"{item['id']}.card.html"
        output_pdf = Path("results/certification") / f"{item['id']}.card.pdf"
        card = write_certification_card(
            item["artifact_dir"],
            output_json=output_json,
            output_html=output_html,
            output_pdf=output_pdf,
            sign=True,
            git_ref="audit-card-gallery-v1",
        )
        gallery_rows.append(
            {
                "id": item["id"],
                "title": item["title"],
                "artifact_dir": item["artifact_dir"],
                "expected": item["expected"],
                "passed": card["overall_verdict"]["passed"],
                "failed_required_rungs": [
                    rung
                    for rung in card["overall_verdict"]["required_rungs"]
                    if not card["rungs"][rung]["passed"]
                ],
                "json": str(output_json),
                "html": str(output_html),
                "pdf": str(output_pdf),
            }
        )
    output = {
        "schema_version": "1.0",
        "artifact_notice": "README audit-card gallery generated from frozen B1 certification cards.",
        "cards": gallery_rows,
    }
    output_path = Path("results/certification/gallery.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "cards": len(gallery_rows),
                "passed": sum(row["passed"] for row in gallery_rows),
                "failed": sum(not row["passed"] for row in gallery_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
