| Family | Evidence set | Method | Metric | Delta | 95% CI | p | BH p |
|---|---|---|---|---:|---|---:|---:|
| internal | qwen2.5-1.5b-expanded-audit | sae_feature_ablation | harmful_score | -0.036339 | [-0.038848, -0.034106] | 0.0002 | 0.0002 |
| internal | qwen2.5-1.5b-expanded-audit | mean_difference_direction_ablation | harmful_score | -0.135550 | [-0.143527, -0.127879] | 0.0002 | 0.0002 |
| internal | qwen2.5-1.5b-expanded-audit | linear_probe_direction_ablation | harmful_score | -0.002456 | [-0.003124, -0.001749] | 0.0002 | 0.0002 |
| external | or-bench-qwen2.5-1.5b-transfer | sae_feature_ablation | harmful_score | -0.046219 | [-0.053366, -0.039587] | 0.0002 | 0.000225 |
| external | or-bench-qwen2.5-1.5b-transfer | mean_difference_direction_ablation | harmful_score | -0.162908 | [-0.175076, -0.150633] | 0.0002 | 0.000225 |
| external | or-bench-qwen2.5-1.5b-transfer | linear_probe_direction_ablation | harmful_score | -0.003611 | [-0.004167, -0.003107] | 0.0002 | 0.000225 |
| component | qwen2.5-1.5b-component-path | residual_direction_ablation | harmful_score | -0.135550 | [-0.143585, -0.127312] | 0.0002 | 0.0002181 |
| component | qwen2.5-1.5b-component-path | residual_output_ablation | harmful_score | -0.135550 | [-0.143602, -0.127760] | 0.0002 | 0.0002181 |
| component | qwen2.5-1.5b-component-path | attention_output_ablation | harmful_score | -0.001878 | [-0.002046, -0.001705] | 0.0002 | 0.0002181 |
| component | qwen2.5-1.5b-component-path | mlp_output_ablation | harmful_score | 0.007387 | [0.006273, 0.008493] | 1 | 1 |
