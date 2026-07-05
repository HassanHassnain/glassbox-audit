# Statistical Tests

Glassbox reports pair-clustered paired bootstrap confidence intervals and paired sign-flip permutation tests for headline causal deltas. Benjamini-Hochberg correction is applied within each artifact family.

A6 adds affirmative negative-claim tests: internal artifacts now include TOST equivalence checks for SAE-vs-mean/probe harmful-score effects, one-sided inferiority tests for SAE being worse by a preregistered margin, and paired power analysis for the configured minimum effect size.

The statistical tests support the same interpretation as Phase 4: late residual mean-difference interventions are robust, while SAE effects do not beat the mean/probe baselines under the preregistered H3 rule.

See [`results/final/statistical_tests.json`](../results/final/statistical_tests.json) and [`docs/tables/statistical_summary.md`](tables/statistical_summary.md).
