#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PYTHONPATH:-src}"

ARTIFACTS="${ARTIFACTS:-artifacts/qwen2.5-1.5b-expanded-audit}"
TOXIC="${TOXIC:-data/or_bench_toxic.normalized.jsonl}"
HARD_BENIGN="${HARD_BENIGN:-data/or_bench_hard1k.normalized.jsonl}"

python3 -m glassbox_audit.cli external-causal \
  --artifacts "${ARTIFACTS}" \
  --inputs "${TOXIC}" "${HARD_BENIGN}" \
  --output artifacts/external-causal/or_bench_qwen15b_1000.json \
  --max-records-per-label 100 \
  --device cuda:0

python3 -m glassbox_audit.cli path-component \
  --artifacts "${ARTIFACTS}" \
  --layers 27 \
  --components residual,attention,mlp \
  --device cuda:0 \
  --output-name path_component.json
