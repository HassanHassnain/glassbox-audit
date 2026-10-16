# Results: Real-Model Refusal Audit

**Evidence class:** `empirical_cross_model_replication_preregistered`

This document is generated from `artifacts/qwen2.5-3b-replication-preregistered`. Results are reported on the held-out test split
after layer/feature discovery on train and threshold/scale selection on validation.

## Headline Results

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Selected layer: `35`
- Dataset: `120` matched pairs across `40` family-disjoint families
- Test harmful refusal: `100.0%`
- Test benign over-refusal: `3.3%`
- SAE dead features: `45.7%`
- Legacy utility-only SAE flag: `True`
- Preregistered H3 SAE-win criterion: `False`

## Frozen Steering Comparison

| Method | Validation-selected scale | Harmful score delta | Benign score delta | Capability NLL delta |
|---|---:|---:|---:|---:|
| sae_features | -1.00 | -0.000 [-0.000, -0.000] | -0.004 [-0.005, -0.004] | -0.003 [-0.003, -0.002] |
| mean_difference | -1.00 | -0.000 [-0.000, -0.000] | -0.006 [-0.007, -0.005] | -0.001 [-0.001, +0.000] |
| linear_probe | -1.00 | -0.000 [-0.000, -0.000] | -0.007 [-0.008, -0.006] | -0.002 [-0.003, -0.002] |

## Strongest Causal Evidence

- SAE feature ablation: `-0.006 [-0.007, -0.005]`
- SAE activation patch: `-0.000 [-0.000, -0.000]`
- Strongest ablation: `mean_difference_direction_ablation`
- Strongest layerwise control: layer `35`
- Direct mean-difference-minus-SAE ablation effect: `-0.084 [-0.098, -0.070]`
- SAE feature-ablation random-control empirical p: `0.024`
- Random-SAE control protocol: `matched_center_zero_encoder_bias`

The legacy utility-only flag is favorable because the simple baselines have large benign/capability
costs in this 3B run. It is not counted as an SAE win under the preregistered H3 rule, which also
requires direct held-out harmful-score competitiveness against mean-difference/probe baselines.

See [Negative Results](NEGATIVE_RESULTS.md), [causal results](tables/causal_results.md), and
[direct paired comparisons](tables/method_comparisons.md).
