# Preregistered Next-Phase Audit

**Registered before new expanded/cross-model held-out results.**

This phase is not an outcome-driven search for a circuit or for an SAE win. The goal is to test
fixed hypotheses with fixed metrics, fixed baselines, and fixed stopping rules.

## Primary Question

Can the current layer-27 residual-stream refusal-signal result be strengthened into component-level
or path-level causal evidence?

## Secondary Question

Can any SAE-based feature method outperform mean-difference and logistic-probe baselines under
held-out evaluation without retuning on test results?

## Fixed Hypotheses

H1, residual localization replication:
The train-selected residual-stream layer will show held-out harmful-score suppression under
mean-difference direction ablation. Success requires a harmful-score delta below `-0.05` with a
95% paired-bootstrap CI below zero and absolute benign-score delta at most `0.03`.

H2, component specificity:
A component is component-level evidence only if its held-out harmful-score suppression is at least
50% of the matched residual-stream ablation at the same layer, its CI is below zero, its absolute
benign-score delta is at most `0.03`, and capability-NLL increase is at most `0.05`.

H3, SAE causal utility:
An SAE method beats simple baselines only if it beats both mean-difference and logistic-probe
baselines on the pre-existing specificity/capability utility, has a more negative harmful-score
delta in direct paired comparison, and clears matched random controls with one-sided empirical
`p <= 0.05`.

H4, circuit claim:
The word "circuit" is reserved for a candidate mechanism with necessity, sufficiency, specificity,
and held-out replication evidence. A residual-stream direction alone is not a confirmed circuit.

## Fixed Baselines

Every SAE result is compared against:

- train mean-difference direction;
- train logistic-probe direction;
- isotropic random directions;
- architecture-matched random untrained-SAE latent controls using train-matched centering and
  zeroed encoder bias;
- layerwise residual ablation controls.

## Fixed Run Order

1. Run `configs/qwen2.5-1.5b-expanded-medium.yaml` on the 520-pair controlled corpus.
2. Run the layerwise component scan on that completed artifact.
3. Run the 1,000-pair audit only if the 520-pair run completes without OOM or GPU conflict and
   finishes within a practical single-session budget.
4. Run `configs/qwen2.5-3b-refusal-replication.yaml` only if cached weights load offline and a
   non-conflicting GPU is available.
5. Run SAE stability configs only as fixed seed/scale repetitions, not adaptively selected from
   previous outcomes.

## Fixed Stopping Rules

- Stop a run after two OOM failures at the same scale.
- Stop a run if it conflicts with another user's GPU job.
- Do not change layers, thresholds, scales, feature counts, prompts, bootstrap settings, or success
  criteria after seeing held-out results.
- If the 520-pair run is too slow or unstable, document it as incomplete and do not downsample after
  observing test metrics.
- If the 1,000-pair or 3B run is not feasible, document it as unrun.

## Fixed Metrics

- harmful refusal rate;
- benign over-refusal rate;
- continuous refusal-score gap;
- paired-bootstrap harmful/benign/capability deltas;
- pre-existing specificity/capability utility;
- empirical one-sided random-control p-value;
- SAE reconstruction MSE, explained variance, L0, density, dead-feature fraction, samples per
  feature, top activating examples;
- component-output harmful-score delta as a fraction of residual-stream delta.

## Fixed SAE Stability Grid

Prepared, not adaptively selected:

- seeds: `17`, `23`, `42`;
- expansion factors: `2`, `4`;
- top-k: `32`;
- tokens per prompt: `48`;
- medium corpus max activation samples: `32,000`;
- full corpus max activation samples: `64,000`.

## Reporting Rules

The final report must separate:

- completed evidence;
- failed hypotheses;
- inconclusive results;
- future work;
- unsupported claims.

Negative results are valid results. If SAE methods do not beat mean-difference/probe baselines, say
so. If component evidence remains weak, report a residual-stream refusal signal without claiming a
confirmed circuit.
