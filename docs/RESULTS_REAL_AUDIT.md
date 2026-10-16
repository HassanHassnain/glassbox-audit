# Results: Real-Model Refusal Audit

**Evidence class:** `empirical_bounded_real_audit`

This document is generated from `artifacts/qwen-real-audit`. Results are reported on the held-out test split
after layer/feature discovery on train and threshold/scale selection on validation.

## Headline Results

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Selected layer: `27`
- Dataset: `120` matched pairs across `40` family-disjoint families
- Test harmful refusal: `90.0%`
- Test benign over-refusal: `6.7%`
- SAE dead features: `34.1%`
- SAE utility flag: `False`

## Frozen Steering Comparison

| Method | Validation-selected scale | Harmful score delta | Benign score delta | Capability NLL delta |
|---|---:|---:|---:|---:|
| sae_features | -1.00 | -0.001 [-0.001, -0.000] | -0.001 [-0.001, -0.001] | -0.001 [-0.001, -0.000] |
| mean_difference | -1.00 | -0.001 [-0.002, -0.001] | -0.002 [-0.002, -0.002] | -0.001 [-0.002, -0.001] |
| linear_probe | -4.00 | -0.004 [-0.005, -0.004] | -0.003 [-0.004, -0.003] | -0.006 [-0.008, -0.005] |

## Strongest Causal Evidence

- SAE feature ablation: `-0.020 [-0.023, -0.017]`
- SAE activation patch: `+0.009 [+0.006, +0.012]`
- Strongest ablation: `mean_difference_direction_ablation`
- Strongest layerwise control: layer `27`
- Direct mean-difference-minus-SAE ablation effect: `-0.141 [-0.158, -0.124]`
- SAE feature-ablation random-control empirical p: `0.024`
- Random-SAE control protocol: `matched_center_zero_encoder_bias`

See [Negative Results](NEGATIVE_RESULTS.md), [causal results](tables/causal_results.md), and
[direct paired comparisons](tables/method_comparisons.md).
