#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m glassbox_audit.cli run \
  --config configs/qwen2.5-1.5b-refusal.yaml \
  --output artifacts/qwen-full-fixture
