# Reviewer Q&A

## Is this a circuit paper?

No. It is a causal activation-level audit. The evidence supports late residual-stream localization.
The component/path analysis did not satisfy the fixed criteria for a confirmed circuit.

## Did the authors tune until SAE looked good?

No. The Phase 4 and Phase 5 reports preserve the preregistered interpretation. SAE does not beat
mean/probe under H3. Phase 5 dose-response analysis is descriptive over a fixed grid and does not
retune the headline scale.

## Why include a negative SAE result?

Because the project is an adjudication of interpretability tooling. A sparse feature can be real
relative to matched random-SAE controls while still being worse than a mean-difference direction.
That is a useful result.

## Are OR-Bench results causal?

The OR-Bench causal-transfer artifact applies frozen controlled-corpus interventions to external
prompts, so it is causal with respect to those interventions. OR-Bench rows are unpaired, so those
results are reported separately from paired controlled-corpus results.

## Is the refusal scorer robust?

Partially. The headline ordering survives fixed contrastive threshold variants. Keyword response
scoring is implemented but unavailable for current real-model artifacts because generated responses
were not stored.

## Was data leakage checked?

Yes. Phase 5 checks exact duplicates, normalized duplicates, family/pair split overlap,
harmful/benign prompt leakage, near duplicates, and external exact normalized overlap. The controlled
1,000-pair corpus passes the formal leakage checks.

## What is the biggest limitation?

The project does not provide head-level or full path-patching evidence, and it does not include a
non-Qwen completed replication. The correct claim is residual-stream localization, not a confirmed
mechanistic circuit.
