# Glassbox Audit Report

**Evidence class:** `synthetic_fixture`

This is a deterministic synthetic fixture run. It verifies the pipeline but is not evidence about a real language model.

## Experiment

- Model: `glassbox-toy-refusal`
- Behavior: `refusal`
- Located layer: `1`
- Test records: `16`
- Refusal threshold: `0.5000` (source: `config`)

## Baseline Behavior

- Harmful refusal rate: 1.000
- Benign over-refusal rate: 0.250
- Behavior gap: 0.500
- Capability perplexity: 2.024

## Sparse Feature Result

- Selected features: `[79, 120, 1]`
- SAE variance explained: 0.989
- SAE mean L0: 8.00
- Largest feature effect size: -1.284

## Causal Tests

- Validation-selected SAE steering scale: -4.00
- Held-out harmful score delta: -0.567 [-0.609, -0.501]
- Feature ablation harmful score delta: -0.000
- Benign-mean activation patch harmful score delta: -0.029
- Strongest direction/feature ablation: `mean_difference_direction_ablation` (-0.327)

## Baseline Comparison

| Method | Scale | Harmful score delta | Benign score delta | Capability NLL delta |
|---|---:|---:|---:|---:|
| sae_features | -4.00 | -0.567 | -0.141 | +0.481 |
| mean_difference | -4.00 | -0.744 | -0.245 | -0.012 |
| linear_probe | -4.00 | -0.634 | -0.175 | +0.091 |

## Equivalence And Power

- Pair-level bootstrap unit: `pair_id`
- Passed equivalence/inferiority tests: `['sae_inferior_to_mean_difference', 'sae_inferior_to_linear_probe']`
- One-sided power at target n=500: 0.875
- Required n for target power: `396` prompt pairs

## Feature-Budget Frontier

- Budgets: `[1, 2, 5, 10, 20, 50, 100, 128]`
- Best SAE budget row: `sae_l1_weighted_direction_ablation` at k=`100` (-0.055)
- Mean-difference reference: -0.327

Random controls and all sweep points are stored in `interventions.json`. Confidence intervals are paired bootstrap intervals over matched prompt pairs.

## Interpretation Guardrails

- A direction that steers behavior is not automatically a complete mechanism explanation.
- Probe accuracy is correlational; ablation and patching provide the stronger causal tests.
- Compare effect size and capability cost, not just whether an intervention moves the metric.
- Real-model conclusions require multiple seeds, held-out prompt families, and replication.
