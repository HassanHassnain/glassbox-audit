#!/usr/bin/env bash
set -euo pipefail

PHASE5_ARTIFACTS="${PHASE5_ARTIFACTS:-artifacts/phase5-cleanroom/qwen15b-1000}"
PHASE5_EXTERNAL="${PHASE5_EXTERNAL:-artifacts/phase5-cleanroom/or_bench_qwen15b_1000_external_causal.json}"
PHASE5_COMPONENT="${PHASE5_COMPONENT:-artifacts/phase5-cleanroom/qwen15b-1000/path_component_phase5.json}"

if [[ -f "${PHASE5_ARTIFACTS}/summary.json" && -f "${PHASE5_EXTERNAL}" && -f "${PHASE5_COMPONENT}" ]]; then
  python -m glassbox_audit.cli reproducibility-diff \
    --phase4-artifacts artifacts/qwen-expanded-1000-phase4 \
    --phase5-artifacts "${PHASE5_ARTIFACTS}" \
    --phase4-external artifacts/external-causal/or_bench_qwen15b_1000_phase4.json \
    --phase5-external "${PHASE5_EXTERNAL}" \
    --phase4-component artifacts/qwen-expanded-1000-phase4/path_component_phase4.json \
    --phase5-component "${PHASE5_COMPONENT}"
fi

if [[ -f "${PHASE5_ARTIFACTS}/summary.json" ]]; then
  python -m glassbox_audit.cli statistical-tests \
    --phase4-artifacts artifacts/qwen-expanded-1000-phase4 \
    --phase5-artifacts "${PHASE5_ARTIFACTS}" \
    --external-artifacts artifacts/external-causal/or_bench_qwen15b_1000_phase4.json \
    --component-artifacts artifacts/qwen-expanded-1000-phase4/path_component_phase4.json
else
  python -m glassbox_audit.cli statistical-tests \
    --phase4-artifacts artifacts/qwen-expanded-1000-phase4 \
    --external-artifacts artifacts/external-causal/or_bench_qwen15b_1000_phase4.json \
    --component-artifacts artifacts/qwen-expanded-1000-phase4/path_component_phase4.json
fi

python -m glassbox_audit.cli scorer-robustness \
  --artifacts artifacts/qwen-expanded-1000-phase4

python -m glassbox_audit.cli leakage-audit \
  --controlled data/refusal_controlled_v2_1000.jsonl \
  --external data/or_bench_toxic.normalized.jsonl data/or_bench_hard1k.normalized.jsonl

python -m glassbox_audit.cli dose-response \
  --artifacts artifacts/qwen-expanded-1000-phase4
