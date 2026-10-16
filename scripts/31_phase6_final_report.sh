#!/usr/bin/env bash
set -euo pipefail

if [[ -f artifacts/phase6-cleanroom/qwen15b-1000/summary.json ]]; then
  python -m glassbox_audit.cli statistical-tests \
    --phase4-artifacts artifacts/qwen-expanded-1000-phase4 \
    --phase5-artifacts artifacts/phase6-cleanroom/qwen15b-1000 \
    --external-artifacts artifacts/external-causal/or_bench_qwen15b_1000_phase4.json \
    --component-artifacts artifacts/qwen-expanded-1000-phase4/path_component_phase4.json
else
  python -m glassbox_audit.cli statistical-tests \
    --phase4-artifacts artifacts/qwen-expanded-1000-phase4 \
    --external-artifacts artifacts/external-causal/or_bench_qwen15b_1000_phase4.json \
    --component-artifacts artifacts/qwen-expanded-1000-phase4/path_component_phase4.json
fi

python -m glassbox_audit.cli scorer-robustness \
  --artifacts artifacts/qwen-expanded-1000-phase4 \
  --output-json results/final/scorer_robustness.json \
  --output-doc docs/SCORER_ROBUSTNESS.md

python -m glassbox_audit.cli leakage-audit \
  --controlled data/refusal_controlled_v2_1000.jsonl \
  --external data/or_bench_toxic.normalized.jsonl data/or_bench_hard1k.normalized.jsonl

python -m glassbox_audit.cli dose-response \
  --artifacts artifacts/qwen-expanded-1000-phase4
