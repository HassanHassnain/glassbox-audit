# Glassbox: Causal Activation-Level Audit of Refusal Behavior

## Abstract

Glassbox audits refusal behavior in small open instruction models using held-out causal interventions, sparse-feature tools, and strong baselines. The strongest supported result is a robust late residual-stream refusal-relevant direction in Qwen2.5-1.5B, with partial Qwen2.5-3B replication and OR-Bench external causal transfer. The central negative result is that SAE features beat matched random-SAE controls but do not beat mean-difference/probe baselines under preregistered held-out criteria. Component/path analyses do not confirm a full circuit.

## Method

The controlled corpus pairs harmful prompts with semantically related benign refusal-sensitive prompts. Discovery and SAE training use train records only. Behavior thresholds and steering scales use validation only. Headline metrics use held-out test records only.

Glassbox compares SAE feature interventions against mean-difference directions, logistic-probe directions, isotropic random directions, matched random untrained-SAE latents, layerwise controls, and component-output controls.

## Results

| Evidence block | Main result |
|---|---|
| Qwen2.5-1.5B 1,000-pair audit | layer 27; mean harmful-score delta `-0.136`; SAE `-0.036` |
| SAE stability | 6/6 fixed cells completed; 0/6 preregistered H3 passes |
| OR-Bench causal transfer | SAE toxic refusal-rate delta `-0.12`; mean `-0.68` |
| Qwen2.5-3B replication | late-layer effect replicated; specificity failed due benign/capability shifts |
| Component/path | residual strong; attention/MLP outputs small or non-specific |
| Clean-room rerun | reproduced the Qwen2.5-1.5B audit within fixed tolerances |

## Interpretation

Glassbox supports a residual-stream refusal-signal claim, not a circuit-discovery claim. The SAE result is negative under the preregistered bar: SAE effects are nonzero and better than matched random-SAE controls, but not better than mean/probe baselines.

## Limitations

No head-level or path-level mechanism is confirmed. Non-Qwen replication is prepared but unrun. Real-model artifacts use contrastive log-probability scoring, not generated-answer grading. Larger SAE-scale configs remain unrun.
