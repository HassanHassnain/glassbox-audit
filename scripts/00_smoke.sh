#!/usr/bin/env bash
set -euo pipefail

python -m glassbox_audit.cli validate-data --data data/refusal_pairs.jsonl
python -m glassbox_audit.cli run --config configs/toy.yaml --output artifacts/demo
python -m pytest
