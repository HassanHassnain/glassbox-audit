#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:-src}" python3 -m glassbox_audit.cli validate-data --data data/fixtures/refusal_pairs_tiny.jsonl
PYTHONPATH="${PYTHONPATH:-src}" python3 -m glassbox_audit.cli run --config configs/toy.yaml --output artifacts/demo
PYTHONPATH="${PYTHONPATH:-src}" python3 -m pytest
