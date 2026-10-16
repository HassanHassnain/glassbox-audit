#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ARTIFACTS="${ARTIFACTS:-artifacts/phase5-cleanroom/qwen15b-1000}"

python -m glassbox_audit.cli external-causal \
  --artifacts "${ARTIFACTS}" \
  --inputs data/or_bench_toxic.normalized.jsonl data/or_bench_hard1k.normalized.jsonl \
  --output artifacts/phase5-cleanroom/or_bench_qwen15b_1000_external_causal.json \
  --max-records-per-label 100 \
  --device cuda:0

python -m glassbox_audit.cli path-component \
  --artifacts "${ARTIFACTS}" \
  --layers 27 \
  --components residual,attention,mlp \
  --device cuda:0 \
  --output-name path_component_phase5.json
