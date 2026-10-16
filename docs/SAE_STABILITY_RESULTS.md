# SAE Stability Results

**Status:** full fixed six-cell grid completed in Phase 4.

The stability grid was fixed before seeing held-out stability results in
[`configs/sae_stability_grid.yaml`](../configs/sae_stability_grid.yaml). Phase 4 completed all
six cells; no failed seed or width was excluded.

Machine-readable summary:
[`results/sae-stability/phase4_full_grid_summary.json`](../results/sae-stability/phase4_full_grid_summary.json)  
Table: [`docs/tables/sae_stability_phase4.md`](tables/sae_stability_phase4.md)

| Cell | Width | Seed | SAE harmful delta | Mean harmful delta | Probe harmful delta | Random-SAE p | H3 pass | Dead features |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| seed17-exp2 | 2x | 17 | -0.035 | -0.141 | -0.009 | 0.020 | false | 16.0% |
| seed23-exp2 | 2x | 23 | -0.061 | -0.141 | -0.009 | 0.020 | false | 14.0% |
| seed42-exp2 | 2x | 42 | -0.045 | -0.141 | -0.009 | 0.020 | false | 14.8% |
| seed17-exp4 | 4x | 17 | -0.016 | -0.141 | -0.009 | 0.020 | false | 44.0% |
| seed23-exp4 | 4x | 23 | -0.033 | -0.141 | -0.009 | 0.020 | false | 48.8% |
| seed42-exp4 | 4x | 42 | -0.031 | -0.141 | -0.009 | 0.020 | false | 39.4% |

## Aggregate

- completed cells: 6 / 6;
- H3 passing cells: 0 / 6;
- SAE harmful-score delta mean: `-0.037`;
- SAE harmful-score delta standard deviation: `0.014`;
- SAE harmful-score delta range: `-0.061` to `-0.016`;
- dead-feature fraction range: `14.0%` to `48.8%`.

## Interpretation

SAE feature ablation had a held-out effect in every completed cell and remained stronger than the
matched random-SAE latent-control distribution. However, the effect was not stable enough and never
beat mean-difference on direct harmful-score suppression.

The 4x dictionaries did not improve the result at this activation scale. They had fewer activation
samples per feature, substantially higher dead-feature fractions, worse reconstruction MSE, and no
H3 passes.

Under preregistered H2/H3, **SAE stability and SAE utility fail**. This is a valid negative result,
not a cue to retune until the result becomes favorable.

## Optional Unrun Scale Configs

The repository includes larger activation-scale configs for future work:

- `configs/qwen2.5-1.5b-sae-scale-96tok-64k-seed17-exp2.yaml`
- `configs/qwen2.5-1.5b-sae-scale-96tok-64k-seed42-exp2.yaml`

These are configured but unrun and should not be cited as evidence.
