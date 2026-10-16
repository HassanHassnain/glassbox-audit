#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m glassbox_audit.cli run \
  --config configs/qwen2.5-1.5b-refusal-smoke.yaml \
  --output artifacts/qwen-smoke
