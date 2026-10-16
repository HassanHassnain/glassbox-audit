# Negative Results

**Evidence classes:** `empirical_bounded_real_audit`, `empirical_expanded_medium_preregistered`,
`empirical_sae_stability_preregistered`, and cached Qwen2.5-3B replication.

Negative results are generated from the same artifact contract as positive findings.

- SAE steering does not beat the best simple baseline (linear_probe) on specificity-adjusted utility.
- SAE feature ablation is weaker than mean_difference direction ablation on harmful-score suppression.
- SAE activation patching does not reduce harmful refusal score.
- SAE activation patching has material capability-NLL cost.

## SAE Versus Simpler Baselines

- SAE specificity-adjusted utility: `-0.0002`
- Best simple baseline: `linear_probe`
- Best simple utility: `0.0007`
- SAE utility flag: `False`

## Interpretation

These diagnostics prevent a steerable SAE direction from being mistaken for a uniquely useful or
stable mechanism-level explanation. See `failure_analysis.json` for all machine-readable diagnostics.

## Preregistered Next-Phase Failures

- H2 component specificity failed: residual-stream interventions were strong, but attention/MLP
  output scans did not isolate a clean component satisfying specificity and capability constraints.
- H3 SAE causal utility failed in the 520-pair Qwen2.5-1.5B audit.
- H3 SAE causal utility failed in the cached Qwen2.5-3B replication under the preregistered
  harmful-score criterion, despite one older utility-only flag being favorable.
- H3 failed in every completed SAE stability cell.
- Phase 4 completed all six fixed SAE stability cells; H3 failed in all six.
- The 1,000-pair Qwen2.5-1.5B audit passed residual robustness, but SAE feature ablation remained
  much weaker than mean-difference on direct harmful-score suppression.
- OR-Bench external causal transfer was present, but SAE did not beat mean-difference on direct
  toxic refusal-score suppression.
- Phase 4 component/path approximation did not confirm a circuit: attention/MLP outputs were small
  or non-specific, and residual projection patching was not full path mediation evidence.
- Qwen2.5-3B showed late-layer interventions, but the largest effects had large benign or
  capability shifts and therefore did not count as clean specificity-preserving replication.
- OR-Bench external behavior-only validation showed high refusal on toxic prompts but also 81%
  refusal on sampled hard-benign prompts, warning that the frozen refusal score/model combination
  over-refuses outside the controlled paired corpus.

## What This Rules Out

The current evidence does not support:

- a confirmed refusal circuit;
- a path-level or head-level mechanism claim;
- an SAE method beating mean-difference/probe baselines under the preregistered H3 criterion;
- non-Qwen replication;
- an external SAE-superiority result.
