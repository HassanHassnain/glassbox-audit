# Mechanism Drift Report

- Mode: `model_swap`
- Status: `degraded`
- Source model: `Qwen/Qwen2.5-1.5B-Instruct`
- New model: `artifacts/lora/qwen25-15b-refusal-shifted/merged`
- Certified method: `sae_feature_ablation`

| Rung | Source | New | Changed |
|---|---|---|---|
| baseline_superiority | False | None | False |
| capability_cost | True | True | False |
| equivalence_or_inferiority | None | None | False |
| held_out_causal | True | True | False |
| matched_controls | True | True | False |
| separation | True | True | False |
| specificity | True | False | True |
| transfer | None | None | False |

The JSON report contains the full rung evidence, confidence intervals, hashes, and
baseline/intervention outputs when produced by live model-swap mode.
