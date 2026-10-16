# External Causal Validation

**Status:** completed on OR-Bench using frozen controlled-corpus Qwen2.5-1.5B artifacts.

External causal validation applies frozen controlled-corpus interventions to normalized external
rows. It does not rediscover layers, features, thresholds, or scales on OR-Bench.

**Phase 6:** OR-Bench external causal validation was not rerun from the Phase 6 clean-room artifact.
The main clean-room rerun consumed the available validation budget. The frozen Phase 4 OR-Bench
artifact remains the reported external causal result.

Preferred artifact: `artifacts/external-causal/or_bench_qwen15b_1000_phase4.json`  
Fallback artifact: `artifacts/external-causal/or_bench_qwen15b_520_phase4.json`  
Snapshot: `results/external-causal/or_bench_qwen15b_1000_phase4_summary.json`

## Baseline

The preferred run used 100 OR-Bench toxic prompts and 100 OR-Bench hard-benign prompts.

| Baseline metric | Value |
|---|---:|
| Toxic refusal rate | 96.0% |
| Hard-benign over-refusal rate | 82.0% |
| Toxic mean refusal score | 0.946 |
| Hard-benign mean refusal score | 0.908 |

## Frozen Causal Transfer

| Method | Toxic refusal-rate delta | Hard-benign refusal-rate delta | Harmful-score delta | Benign-score delta | NLL delta | Utility |
|---|---:|---:|---:|---:|---:|---:|
| SAE feature ablation | -0.12 | -0.09 | -0.046 | -0.040 | -0.000 | +0.0062 |
| Mean-difference ablation | -0.68 | -0.55 | -0.163 | -0.157 | -0.021 | +0.0060 |
| Linear-probe ablation | +0.00 | +0.00 | -0.004 | -0.003 | -0.007 | +0.0003 |

SAE feature ablation beat random controls:

- random direction ablations: empirical `p=0.030`;
- matched random-SAE feature ablations: empirical `p=0.030`.

## Interpretation

External causal transfer is present: frozen interventions learned from the controlled corpus
causally change OR-Bench refusal scores. However, this is not an SAE-superiority result.
Mean-difference ablation remains far stronger on direct toxic refusal-score suppression.

Both SAE and mean-difference reduce hard-benign over-refusal in this external setting. That is a
useful behavior effect, but it also means the intervention is a broad refusal-suppression direction
rather than a precise safety-category mechanism.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  glassbox external-causal \
  --artifacts artifacts/qwen-expanded-1000-phase4 \
  --inputs data/or_bench_toxic.normalized.jsonl data/or_bench_hard1k.normalized.jsonl \
  --output artifacts/external-causal/or_bench_qwen15b_1000_phase4.json \
  --max-records-per-label 100 \
  --device cuda:0
```
