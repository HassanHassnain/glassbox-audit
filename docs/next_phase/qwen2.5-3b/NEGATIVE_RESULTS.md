# Negative Results

**Evidence class:** `empirical_cross_model_replication_preregistered`

Negative results are generated from the same artifact contract as positive findings.

- SAE feature ablation is weaker than mean_difference direction ablation on harmful-score suppression.
- SAE activation patching has material capability-NLL cost.
- SAE has fewer training activation samples than dictionary features.

## SAE Versus Simpler Baselines

- SAE specificity-adjusted utility: `-0.0042`
- Best simple baseline: `mean_difference`
- Best simple utility: `-0.0060`
- Legacy utility-only SAE flag: `True`
- Preregistered H3 SAE-win criterion: `False`

## Interpretation

These diagnostics prevent a steerable SAE direction from being mistaken for a uniquely useful or
stable mechanism-level explanation. See `failure_analysis.json` for all machine-readable diagnostics.

The legacy utility-only flag is not treated as a positive H3 result. Under the preregistered rule,
SAE also had to compete with mean-difference/probe baselines on direct harmful-score suppression,
and it did not.
