# Direct Paired Method Comparisons

Negative harmful-score differences mean the left method suppresses more than the right.

| Setting | Left | Right | Harmful left-minus-right (95% CI) |
|---|---|---|---:|
| causal ablation | linear_probe_direction_ablation | mean_difference_direction_ablation | +0.135 [+0.119, +0.150] |
| causal ablation | linear_probe_direction_ablation | sae_feature_ablation | -0.007 [-0.010, -0.004] |
| causal ablation | mean_difference_direction_ablation | sae_feature_ablation | -0.141 [-0.158, -0.124] |
| selected steering | linear_probe | mean_difference | -0.003 [-0.003, -0.002] |
| selected steering | linear_probe | sae_features | -0.004 [-0.004, -0.003] |
| selected steering | mean_difference | sae_features | -0.001 [-0.001, -0.001] |
