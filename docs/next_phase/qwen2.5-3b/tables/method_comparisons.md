# Direct Paired Method Comparisons

Negative harmful-score differences mean the left method suppresses more than the right.

| Setting | Left | Right | Harmful left-minus-right (95% CI) |
|---|---|---|---:|
| causal ablation | linear_probe_direction_ablation | mean_difference_direction_ablation | +0.049 [+0.041, +0.056] |
| causal ablation | linear_probe_direction_ablation | sae_feature_ablation | -0.035 [-0.042, -0.029] |
| causal ablation | mean_difference_direction_ablation | sae_feature_ablation | -0.084 [-0.098, -0.070] |
| selected steering | linear_probe | mean_difference | -0.000 [-0.000, -0.000] |
| selected steering | linear_probe | sae_features | -0.000 [-0.000, -0.000] |
| selected steering | mean_difference | sae_features | -0.000 [-0.000, -0.000] |
