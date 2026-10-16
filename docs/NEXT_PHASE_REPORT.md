# Next-Phase Preregistered Audit Report

> Phase 4 supersedes the unrun portions of this historical next-phase report. The 1,000-pair audit
> and full six-cell SAE stability grid were later completed; see
> [FINAL_REVIEWER_REPORT.md](FINAL_REVIEWER_REPORT.md) and
> [PHASE4_PREREGISTERED_VALIDATION.md](PHASE4_PREREGISTERED_VALIDATION.md).

This report summarizes the preregistered next-phase audit defined in
[PREREGISTERED_NEXT_PHASE_AUDIT.md](PREREGISTERED_NEXT_PHASE_AUDIT.md). The run order, metrics,
baselines, and stopping rules were fixed before the expanded held-out results were generated.

## Completed Evidence

### Qwen2.5-1.5B, 520-Pair Expanded Audit

Artifact: `artifacts/qwen-expanded-medium-preregistered`  
Snapshot: `results/qwen-expanded-medium-preregistered`

| Metric | Value |
|---|---:|
| Matched pairs | 520 |
| Held-out test records | 260 |
| Train SAE activation vectors | 24,628 |
| Selected layer | 27 |
| Baseline harmful refusal | 66.9% |
| Baseline benign over-refusal | 17.7% |
| Baseline behavior gap | 0.129 |
| SAE dead features | 16.0% |
| SAE samples per feature | 8.02 |

The residual-stream result replicated at larger scale. Mean-difference direction ablation at layer
27 changed held-out harmful refusal score by `-0.141 [-0.155, -0.128]` with benign-score delta
`-0.013` and capability-NLL delta `-0.021`. This satisfies preregistered H1 for the expanded
Qwen2.5-1.5B run.

SAE feature ablation also had a real held-out effect: `-0.035 [-0.039, -0.032]`, and it beat the
matched random-SAE latent-control distribution (`p=0.020`). It did not beat the mean-difference
direction on harmful-score suppression and did not beat the best simple baseline on the fixed
specificity/capability utility. H3 therefore fails for this run.

### Qwen2.5-3B Cross-Model Replication

Artifact: `artifacts/qwen2.5-3b-replication-preregistered`  
Snapshot: `results/qwen2.5-3b-replication-preregistered`

| Metric | Value |
|---|---:|
| Matched pairs | 120 |
| Held-out test records | 60 |
| Selected layer | 35 |
| Baseline harmful refusal | 100.0% |
| Baseline benign over-refusal | 3.3% |
| Baseline behavior gap | 0.679 |
| SAE dead features | 45.7% |
| SAE samples per feature | 0.94 |

The 3B model shows a train-selected late residual signal, but it is not a clean specificity
replication. Mean-difference ablation changed harmful score by `-0.090 [-0.103, -0.076]`, but also
increased benign score by `+0.390` and capability NLL by `+0.164`. Under the preregistered rule,
this fails H1 specificity even though harmful refusal is causally affected.

The 3B SAE feature ablation effect was small: `-0.006 [-0.007, -0.005]`. It beat matched random-SAE
latent controls (`p=0.024`) and had lower collateral cost than the mean-difference/probe ablations,
but it did not beat them on direct harmful-score suppression. H3 therefore fails under the
preregistered SAE-win criterion, despite the older utility-only flag being true.

## Component-Level Results

Layerwise component scans used frozen train-derived mean-difference directions, fixed layer lists,
and fixed components (`residual`, `attention`, `mlp`).

### Qwen2.5-1.5B Expanded

The strongest component-level effects were:

| Layer | Component | Harmful score delta | Benign delta | Capability NLL delta |
|---:|---|---:|---:|---:|
| 27 | residual | -0.141 | -0.013 | -0.021 |
| 20 | residual | -0.101 | +0.020 | +0.106 |
| 16 | residual | -0.071 | -0.004 | +0.107 |
| 16 | mlp | -0.023 | -0.010 | -0.003 |
| 16 | attention | -0.014 | +0.000 | +0.036 |

No attention or MLP component met preregistered H2. The layer-16 MLP/attention outputs were real but
too small relative to matched residual ablation to count as component-specific evidence.

### Qwen2.5-3B

The strongest component-level effects were:

| Layer | Component | Harmful score delta | Benign delta | Capability NLL delta |
|---:|---|---:|---:|---:|
| 35 | mlp | -0.640 | -0.299 | +0.313 |
| 35 | residual | -0.090 | +0.390 | +0.164 |
| 30 | residual | -0.067 | +0.338 | +0.667 |
| 24 | residual | -0.060 | +0.266 | -0.027 |

The layer-35 MLP intervention is large, but it fails H2 because it is not specific: it substantially
changes benign scores and capability NLL. This is evidence that the intervention touches a powerful
late component, not evidence for a clean refusal circuit.

## Failed Hypotheses

- H2 component specificity failed for both completed models.
- H3 SAE causal utility failed under the preregistered definition for both completed models and all
  completed SAE stability cells.
- The 3B cross-model residual result affected harmful refusal but failed the specificity criterion.

## SAE Stability Results

The fixed SAE stability subset completed three of six configured cells:

| Cell | Width | Seed | SAE harmful delta | Mean harmful delta | H3 SAE beats simple? |
|---|---:|---:|---:|---:|---|
| seed17-exp2 | 2x | 17 | -0.035 | -0.141 | false |
| seed23-exp2 | 2x | 23 | -0.061 | -0.141 | false |
| seed17-exp4 | 4x | 17 | -0.016 | -0.141 | false |

The SAE feature effect was real in every completed cell but varied substantially across seed/width.
The 4x dictionary had worse diagnostics: 44.0% dead features and higher reconstruction MSE. This
does not support an SAE-beats-mean conclusion. See
[SAE_STABILITY_RESULTS.md](SAE_STABILITY_RESULTS.md).

## External Behavior Validation

OR-Bench hard and toxic subsets were downloaded and normalized. They are unpaired, so they were
evaluated only as behavior scoring, not as a causal/intervention audit. A bounded Qwen2.5-1.5B
check sampled 100 toxic and 100 hard-benign prompts using the frozen expanded-audit threshold:

| External subset | Refusal rate |
|---|---:|
| OR-Bench toxic | 95.0% |
| OR-Bench hard benign | 81.0% |

This supports an over-refusal warning on naturalistic hard-benign prompts. It does not alter the
controlled-corpus causal claims. See [EXTERNAL_VALIDATION.md](EXTERNAL_VALIDATION.md).

## Inconclusive Results

- The 1,000-pair Qwen2.5-1.5B audit was configured but not run in this phase.
- The full six-cell SAE stability grid remains partially unrun; three fixed cells completed.
- No head-level or path-patching result is reported; the wrapper exposes attention/MLP outputs
  cleanly but not per-head path patching as a stable artifact contract.
- OR-Bench rows are not paired, so no external SAE/intervention conclusion is reported.

## Unsupported Claims

The completed evidence does not support the phrase "the refusal circuit." The supported claim is:

> Qwen2.5-1.5B has a replicated late residual-stream refusal-relevant direction on the controlled
> corpus. Component-output scans did not isolate a clean attention/MLP component satisfying the
> preregistered specificity rule.

For Qwen2.5-3B, the supported claim is weaker:

> The model has late residual/MLP interventions that change refusal scores, but the effects are
> broad and not specificity-preserving.

## Future Work

- Run the 1,000-pair expanded audit.
- Complete the remaining preregistered SAE stability cells.
- Add clean per-head hooks and path-patching contracts before making path-level claims.
- Repeat on naturalistic external refusal datasets with genuine matched pairs.
- Add semantic generation grading alongside prefix-probability scoring.
