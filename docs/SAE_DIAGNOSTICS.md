# SAE Diagnostics

The bounded Qwen audit records SAE diagnostics in `sae_metrics.json`, `features.json`, and
`failure_analysis.json`.

Current completed bounded-audit values:

| Diagnostic | Value |
|---|---:|
| Training activation vectors | 3,840 |
| Dictionary features | 3,072 |
| Samples per feature | 1.25 |
| Reconstruction MSE | 0.0584 |
| Variance explained | 0.9977 |
| Mean L0 | 32.0 |
| Feature density | 1.04% |
| Dead-feature fraction | 34.1% |

New runs also attach `top_activating_record_ids` and `top_activation_values` to each feature row
so the workbench can inspect what activates a feature without re-running discovery.

## Expanded Training Settings

`SAEConfig` now includes:

- `tokens_per_prompt`: collect more than the final token for SAE training;
- `max_activation_samples`: cap the collected activation matrix for memory-safe training;
- `activation_subsample_seed`: make activation subsampling reproducible.

The expanded Qwen config collects up to 48 token activations per prompt and caps the training set at
64,000 vectors if needed.

## Completed 520-Pair Expanded Run

The preregistered 520-pair Qwen2.5-1.5B run used 24,628 SAE training activation vectors.

| Diagnostic | 120-pair bounded | 520-pair expanded |
|---|---:|---:|
| Training activation vectors | 3,840 | 24,628 |
| Samples per feature | 1.25 | 8.02 |
| Reconstruction MSE | 0.0584 | 0.0414 |
| Variance explained | 0.9977 | 0.9984 |
| Dead-feature fraction | 34.1% | 16.0% |
| SAE feature-ablation harmful delta | -0.020 | -0.035 |

SAE quality improved with more activations, and the SAE feature-ablation effect increased. It still
did not beat the mean-difference direction on preregistered held-out comparisons.

## Stability Grid

`configs/sae_stability_grid.yaml` fixes the next seed/width grid before running:

- seeds: `17`, `23`, `42`;
- expansion factors: `2`, `4`;
- top-k: `32`;
- tokens per prompt: `48`;
- max activation samples: `32,000`.

Phase 4 completed the full fixed six-cell grid:

| Cell | SAE features | Dead features | Reconstruction MSE | SAE harmful delta | H3 SAE beats simple? |
|---|---:|---:|---:|---:|---|
| seed17-exp2 | 3,072 | 16.0% | 0.0414 | -0.035 | false |
| seed23-exp2 | 3,072 | 14.0% | 0.0372 | -0.061 | false |
| seed42-exp2 | 3,072 | 14.8% | 0.0472 | -0.045 | false |
| seed17-exp4 | 6,144 | 44.0% | 0.0936 | -0.016 | false |
| seed23-exp4 | 6,144 | 48.8% | 0.0984 | -0.033 | false |
| seed42-exp4 | 6,144 | 39.4% | 0.0857 | -0.031 | false |

The 2x seed reruns preserved real SAE feature-ablation effects but varied in magnitude. The 4x
width ablations worsened dead-feature fractions and reconstruction MSE. No cell passed
preregistered H3. See [SAE_STABILITY_RESULTS.md](SAE_STABILITY_RESULTS.md).

## Interpretation

The completed bounded run has good reconstruction and moderate dead features, but only 1.25 training
samples per dictionary feature. The expanded 2x runs improve this substantially, while the 4x width
run shows that larger dictionaries are not automatically better at this activation scale. Negative
SAE-vs-baseline results should be read as real preregistered outcomes, not as a cue to retune until
the result becomes favorable.
