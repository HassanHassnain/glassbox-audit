# Direct Paired Method Comparisons

Negative harmful-score differences mean the left method suppresses more than the right.

| Setting | Left | Right | Harmful left-minus-right (95% CI) |
|---|---|---|---:|
| causal ablation | linear_probe_direction_ablation | mean_difference_direction_ablation | +0.132 [+0.120, +0.144] |
| causal ablation | linear_probe_direction_ablation | sae_feature_ablation | +0.026 [+0.024, +0.029] |
| causal ablation | mean_difference_direction_ablation | sae_feature_ablation | -0.106 [-0.117, -0.094] |
| selected steering | linear_probe | mean_difference | -0.002 [-0.002, -0.002] |
| selected steering | linear_probe | sae_features | -0.003 [-0.003, -0.003] |
| selected steering | mean_difference | sae_features | -0.001 [-0.001, -0.001] |
