#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
ARTIFACTS="${ARTIFACTS:-artifacts/demo}"
python3 -m glassbox_audit.cli report --artifacts "${ARTIFACTS}"
