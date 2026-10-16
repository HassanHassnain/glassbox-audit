# Results: Real-Model Refusal Audit

**Evidence class:** `empirical_expanded_medium_preregistered`

This document is generated from `artifacts/qwen-expanded-medium-preregistered`. Results are reported on the held-out test split
after layer/feature discovery on train and threshold/scale selection on validation.

## Headline Results

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Selected layer: `27`
- Dataset: `520` matched pairs across `40` family-disjoint families
- Test harmful refusal: `66.9%`
- Test benign over-refusal: `17.7%`
- SAE dead features: `16.0%`
- SAE utility flag: `False`

## Frozen Steering Comparison

| Method | Validation-selected scale | Harmful score delta | Benign score delta | Capability NLL delta |
|---|---:|---:|---:|---:|
| sae_features | -1.00 | -0.002 [-0.002, -0.001] | -0.002 [-0.002, -0.002] | -0.001 [-0.001, -0.001] |
| mean_difference | -1.00 | -0.003 [-0.003, -0.002] | -0.003 [-0.003, -0.003] | -0.001 [-0.001, -0.000] |
| linear_probe | -4.00 | -0.005 [-0.005, -0.004] | -0.003 [-0.003, -0.002] | -0.007 [-0.008, -0.007] |

## Strongest Causal Evidence

- SAE feature ablation: `-0.035 [-0.039, -0.032]`
- SAE activation patch: `+0.053 [+0.045, +0.061]`
- Strongest ablation: `mean_difference_direction_ablation`
- Strongest layerwise control: layer `27`
- Direct mean-difference-minus-SAE ablation effect: `-0.106 [-0.117, -0.094]`
- SAE feature-ablation random-control empirical p: `0.020`
- Random-SAE control protocol: `matched_center_zero_encoder_bias`

See [Negative Results](NEGATIVE_RESULTS.md), [causal results](tables/causal_results.md), and
[direct paired comparisons](tables/method_comparisons.md).
