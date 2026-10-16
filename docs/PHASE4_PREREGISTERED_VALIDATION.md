# Phase 4 Preregistered Reviewer-Grade Validation

**Date frozen:** 2026-06-19  
**Scope:** refusal behavior only  
**Policy:** do not retune layers, features, thresholds, scales, prompts, metrics, seeds, widths, or
criteria after seeing held-out Phase 4 results.

Phase 4 upgrades Glassbox from a portfolio audit to a reviewer-grade validation study. It is not an
outcome search. Negative, failed, and inconclusive results remain first-class outcomes.

## Fixed Questions

1. Does the late residual-stream refusal-relevant direction remain robust at 1,000 matched pairs?
2. Are SAE feature interventions stable across seeds and widths?
3. Do frozen controlled-corpus interventions transfer to external OR-Bench toxic and hard-benign
   prompts?
4. Does component/path evidence justify a mechanism-level claim, or only residual-stream
   localization?
5. Does the result replicate outside the completed Qwen2.5-1.5B/Qwen2.5-3B runs when non-Qwen
   weights are already locally available or accessible without gated-download work?

## Fixed Hypotheses

### H1: Internal Residual Robustness

Passes only if the 1,000-pair Qwen2.5-1.5B audit:

- selects a late residual layer from train-only localization;
- mean-difference direction ablation reduces held-out harmful refusal score by at least `0.08`;
- bootstrap 95% CI upper bound is below `0`;
- absolute benign-score delta is at most `0.05`;
- capability-NLL delta is at most `0.05`.

### H2: SAE Stability And Utility

Passes only if SAE feature ablation:

- has harmful-score CI upper bound below `0` in at least four of six fixed stability cells;
- beats matched random-SAE latent controls with empirical `p <= 0.05` in those cells;
- beats both mean-difference and logistic-probe baselines on direct harmful-score suppression and
  specificity/capability-adjusted utility;
- does not require excluding failed seeds or widths.

If SAE effects are real but weaker/less stable than mean/probe, H2 fails and the result is reported
as a negative SAE utility result.

### H3: External Causal Transfer

Passes only if frozen controlled-corpus interventions, without rediscovery on OR-Bench:

- reduce toxic refusal score/rate under the intended harmful-suppression intervention;
- do not materially increase hard-benign over-refusal;
- maintain capability-NLL cost at or below `0.05`;
- outperform random direction and matched random-SAE controls;
- show the same method ranking direction as the controlled corpus.

Because OR-Bench rows are unpaired, external transfer is reported separately from paired
controlled-corpus causal claims.

### H4: Component/Path Evidence

A component/path claim requires all of:

- necessity: ablating the component/path substantially reduces harmful refusal score;
- sufficiency: patching/steering the component/path substantially restores or induces the effect;
- specificity: benign-score and capability-NLL costs meet the same thresholds as H1;
- controls: random component/path controls do not match the effect;
- replication: held-out prompt replication and, where available, cross-model consistency.

If attention/MLP/head/path interventions are large but non-specific, Phase 4 may only claim
residual-stream localization.

### H5: Cross-Model Replication

Non-Qwen replication is attempted only if weights are already local or access works without gated
download troubleshooting. No unavailable-model result is inferred.

## Fixed Run Order And Stopping Rules

1. Write this preregistration and machine-readable snapshot.
2. Run the 1,000-pair Qwen2.5-1.5B audit if GPU budget permits.
3. Complete remaining fixed SAE stability cells:
   - seed23 / 4x;
   - seed42 / 2x;
   - seed42 / 4x.
4. Run OR-Bench external causal transfer from frozen Qwen2.5-1.5B expanded/1,000-pair artifacts.
   If the 1,000-pair artifact is unavailable, use the already completed 520-pair artifact.
5. Run component/path analyses supported cleanly by the current wrapper:
   - layerwise residual output ablation;
   - attention-output ablation;
   - MLP-output ablation;
   - clean/corrupt residual projection patching if artifact directions are available.
   Head-level patching is attempted only if hooks expose head outputs without fragile architecture
   surgery.
6. Attempt non-Qwen model replication only if cached/access-ready weights are found.
7. Generate final docs and snapshots.

If compute expires, stop after the current fixed job finishes and mark remaining jobs unrun. Do not
replace fixed jobs with exploratory alternatives.

## Fixed Metrics

- harmful refusal-score delta and bootstrap 95% CI;
- toxic external refusal-rate delta;
- hard-benign over-refusal-rate delta;
- mean behavior-score delta by label;
- capability-NLL delta;
- specificity-adjusted utility:
  `harmful_reduction - abs(benign_delta) - 0.2 * max(0, capability_nll_delta)`;
- empirical one-sided random-control p-value;
- SAE reconstruction MSE, variance explained, mean L0, feature density, dead-feature fraction,
  samples per feature;
- seed/width variance of SAE harmful-score deltas.

## Fixed Baselines

- mean-difference direction;
- logistic-probe direction;
- random isotropic directions;
- architecture-matched random untrained-SAE latent controls with trained activation centering and
  zeroed encoder bias;
- layerwise residual controls;
- component-output random controls where implemented.

## Claim Ladder

Allowed claim levels:

1. behavior-only external score pattern;
2. residual-stream causal direction;
3. component-output localization;
4. path/component candidate mechanism;
5. confirmed circuit.

The phrase "confirmed circuit" is allowed only at level 5 with necessity, sufficiency,
specificity, controls, and held-out replication. Current prior evidence is level 2 with some
component-output negative evidence.
