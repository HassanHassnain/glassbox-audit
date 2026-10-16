#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -m glassbox_audit.cli build-real-audit-data \
  --output artifacts/phase6-cleanroom/refusal_controlled_v2_1000.rebuilt.jsonl \
  --pairs-per-family 25 \
  --source-name glassbox_controlled_refusal_v2_1000

python -m glassbox_audit.cli run \
  --config configs/qwen2.5-1.5b-expanded-audit.yaml \
  --output artifacts/phase6-cleanroom/qwen15b-1000
