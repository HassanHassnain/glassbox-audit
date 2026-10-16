# Reproducibility Report

**Status:** Phase 6 clean-room Qwen2.5-1.5B 1,000-pair audit completed and reproduced Phase 4 within fixed tolerances.

- Phase 4 artifact: `artifacts/qwen-expanded-1000-phase4`
- Phase 6 artifact: `artifacts/phase6-cleanroom/qwen15b-1000`
- Status: `reproduced_within_tolerance`

## Fixed Tolerance Diff

| Metric | Phase 4 | Phase 6 | Tolerance | Pass |
|---|---:|---:|---:|---|
| target_layer | 27 | 27 | exact | True |
| baseline_harmful_refusal_rate | 0.64 | 0.64 | 0.005 | True |
| baseline_benign_overrefusal_rate | 0.148 | 0.148 | 0.005 | True |
| behavior_threshold | 0.8465142846107483 | 0.8465142846107483 | 1e-09 | True |
| mean_difference_harmful_delta | -0.1355500293970108 | -0.1355500293970108 | 0.002 | True |
| sae_feature_harmful_delta | -0.03633926343917847 | -0.03633926343917847 | 0.002 | True |
| probe_harmful_delta | -0.0024561622142791747 | -0.0024561622142791747 | 0.002 | True |
| matched_random_sae_empirical_p | 0.030303030303030304 | 0.030303030303030304 | 0.02 | True |
| legacy_sae_outperforms_simple_baselines | True | True | exact | True |
| h3_sae_beats_mean_probe_direct | False | False | exact | True |

## Interpretation

The clean-room rerun completed the full artifact set and all required random-control checkpoints. It does not change the claim: residual-stream localization reproduced; SAE still fails direct H3 against mean/probe. OR-Bench external causal and component/path analyses were not rerun in Phase 6.
