# Negative Results

**Evidence class:** `empirical_expanded_medium_preregistered`

Negative results are generated from the same artifact contract as positive findings.

- SAE steering does not beat the best simple baseline (linear_probe) on specificity-adjusted utility.
- SAE feature ablation is weaker than mean_difference direction ablation on harmful-score suppression.
- SAE activation patching does not reduce harmful refusal score.
- SAE activation patching has material capability-NLL cost.

## SAE Versus Simpler Baselines

- SAE specificity-adjusted utility: `-0.0004`
- Best simple baseline: `linear_probe`
- Best simple utility: `0.0016`
- SAE utility flag: `False`

## Interpretation

These diagnostics prevent a steerable SAE direction from being mistaken for a uniquely useful or
stable mechanism-level explanation. See `failure_analysis.json` for all machine-readable diagnostics.
