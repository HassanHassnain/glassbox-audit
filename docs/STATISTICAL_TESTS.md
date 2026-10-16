# Statistical Tests

Glassbox reports pair-clustered paired bootstrap confidence intervals and paired sign-flip permutation tests for headline causal deltas. Benjamini-Hochberg correction is applied within each artifact family.

A6 adds TOST equivalence checks, one-sided inferiority tests, and paired power analysis. For the existing Qwen artifacts these are post-hoc sensitivity analyses because their results were examined before A6 was specified; they are not confirmatory preregistered evidence. A future cell is labeled prospective only when its equivalence bounds and power assumptions were frozen first and the resulting artifact carries that preregistration hash.

The statistical tests support the same interpretation as Phase 4: late residual mean-difference interventions are robust, while SAE effects do not beat the mean/probe baselines under the preregistered H3 rule.

See [`results/final/statistical_tests.json`](../results/final/statistical_tests.json) and [`docs/tables/statistical_summary.md`](tables/statistical_summary.md).
