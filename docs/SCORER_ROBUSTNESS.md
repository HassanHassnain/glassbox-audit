# Scorer Robustness

Glassbox's real-model refusal score is a contrastive log-probability score comparing a refusal prefix against a compliance prefix. Phase 5 evaluated fixed threshold variants and exported disagreement rows for manual review.

- Headline survives fixed contrastive threshold variants: `True`
- Keyword response scoring is implemented but marked unavailable for current real-model artifacts because generated responses were not stored.
- Manual review sample: `data/manual_review/refusal_scorer_disagreements.csv`

The scorer audit does not create a new positive claim; it checks whether the Phase 4 ordering is an artifact of the exact binary threshold.
