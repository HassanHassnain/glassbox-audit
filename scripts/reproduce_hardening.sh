#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PYTHONPATH:-src}"

ARTIFACTS="${ARTIFACTS:-artifacts/qwen-expanded-1000-reference}"
DATASET="${DATASET:-data/refusal_controlled_v2_1000.jsonl}"
OUTPUT="${OUTPUT:-artifacts/paper-hardening/qwen15b}"
RESULT="${RESULT:-results/extensions/paper-hardening/qwen15b.json}"
BATCH_SIZE="${BATCH_SIZE:-8}"

args=(
  -m glassbox_audit.cli paper-hardening
  --artifacts "${ARTIFACTS}"
  --dataset "${DATASET}"
  --config configs/qwen2.5-1.5b-expanded-audit.yaml
  --output-dir "${OUTPUT}"
  --result "${RESULT}"
  --device cuda:0
  --batch-size "${BATCH_SIZE}"
)
if [[ -n "${HF_HOME:-}" ]]; then
  args+=(--cache-dir "${HF_HOME}")
fi

python3 "${args[@]}"
