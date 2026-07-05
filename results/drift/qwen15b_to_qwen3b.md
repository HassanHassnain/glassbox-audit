# Mechanism Drift Report

- Mode: `artifact_comparison`
- Status: `degraded`
- Source model: `Qwen/Qwen2.5-1.5B-Instruct`
- New model: `Qwen/Qwen2.5-3B-Instruct`
- Certified method: `sae_feature_ablation`

| Rung | Source | New | Changed |
|---|---|---|---|
| baseline_superiority | False | True | True |
| capability_cost | True | True | False |
| equivalence_or_inferiority | None | None | False |
| held_out_causal | True | True | False |
| matched_controls | True | True | False |
| separation | True | True | False |
| specificity | True | False | True |
| transfer | None | None | False |

The JSON report contains the full rung evidence, confidence intervals, hashes, and
baseline/intervention outputs when produced by live model-swap mode.
