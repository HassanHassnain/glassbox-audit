# Results: Expanded Qwen Refusal Audit

**Status:** 520-pair medium expanded audit and 1,000-pair Phase 4 audit completed.

Expanded controlled corpora are available at `data/refusal_controlled_v2_520.jsonl` and
`data/refusal_controlled_v2_1000.jsonl`.

| Quantity | Value |
|---|---:|
| Matched pairs | 1,000 |
| Records | 2,000 |
| Families | 40 |
| Train / validation / test records | 1,000 / 500 / 500 |
| Categories | 10 |
| Family split leakage | false |

## Run Matrix

| Scale | Dataset | Pairs | Records | Config | Status |
|---|---|---:|---:|---|---|
| bounded completed | `data/refusal_controlled_v1.jsonl` | 120 | 240 | `configs/qwen2.5-1.5b-real-audit.yaml` | completed |
| medium expanded | `data/refusal_controlled_v2_520.jsonl` | 520 | 1,040 | `configs/qwen2.5-1.5b-expanded-medium.yaml` | completed |
| full expanded | `data/refusal_controlled_v2_1000.jsonl` | 1,000 | 2,000 | `configs/qwen2.5-1.5b-expanded-audit.yaml` | completed in Phase 4 |

Runnable full config:

```bash
CUDA_VISIBLE_DEVICES=1 GLASSBOX_MODEL_DEVICE=cuda:0 GLASSBOX_SAE_DEVICE=cuda:0 \
  glassbox run \
  --config configs/qwen2.5-1.5b-expanded-audit.yaml \
  --output artifacts/qwen-expanded-audit
```

The config keeps the same discipline as the bounded audit:

- train-only layer localization, SAE training, feature discovery, mean-difference direction, and
  logistic probe direction;
- validation-only behavior threshold and steering-scale selection;
- held-out test-only final reporting;
- family-disjoint splits, with all prompt variants for a scenario assigned to one split.

## Completed Medium Result

Artifact: `artifacts/qwen-expanded-medium-preregistered`

- selected layer: 27;
- held-out test records: 260;
- baseline harmful refusal / benign over-refusal: 66.9% / 17.7%;
- mean-difference ablation harmful-score delta: `-0.141 [-0.155, -0.128]`;
- SAE feature-ablation harmful-score delta: `-0.035 [-0.039, -0.032]`;
- matched random-SAE empirical p: `0.020`;
- SAE beats simple baselines under preregistered H3: no.

See [NEXT_PHASE_REPORT.md](NEXT_PHASE_REPORT.md) for hypothesis-level interpretation.

## Completed 1,000-Pair Phase 4 Result

Artifact: `artifacts/qwen-expanded-1000-phase4`

- selected layer: 27;
- held-out test records: 500;
- baseline harmful refusal / benign over-refusal: 64.0% / 14.8%;
- mean-difference ablation harmful-score delta: `-0.136 [-0.143, -0.128]`;
- SAE feature-ablation harmful-score delta: `-0.036 [-0.039, -0.034]`;
- H1 residual robustness: passed;
- preregistered H3 SAE utility: failed because SAE remains weaker than mean-difference on direct
  harmful-score suppression.
