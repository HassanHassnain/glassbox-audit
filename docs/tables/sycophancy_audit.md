# A7 Sycophancy Audit

Model: `Qwen/Qwen2.5-1.5B-Instruct`; held-out pairs: `250`; target layer: `27`.

| Baseline metric | Value |
|---|---:|
| Pressure mean score | 0.7048 |
| Neutral mean score | 0.7097 |
| Behavior gap | -0.0048 |
| Balanced accuracy | 0.480 |

| Intervention | Harmful-score delta (95% CI) |
|---|---:|
| linear_probe_direction_ablation | -0.0460 [-0.0470, -0.0451] |
| mean_difference_direction_ablation | -0.0546 [-0.0556, -0.0536] |
| sae_activation_patch | -0.0757 [-0.0767, -0.0746] |
| sae_direction_ablation | +0.0005 [-0.0001, +0.0011] |
| sae_feature_ablation | +0.0408 [+0.0389, +0.0426] |

The pressure/neutral baseline did not separate, so intervention rankings are descriptive and do not establish a sycophancy mechanism.
