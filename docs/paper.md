# Glassbox: A Held-Out Causal Activation Audit of Refusal Behavior

## Abstract

Glassbox evaluates whether sparse autoencoder features provide a better mechanistic account of refusal behavior than simpler residual-stream directions. On a controlled 1,000-pair Qwen2.5-1.5B audit, the strongest supported result is a robust late residual-stream refusal-relevant direction at layer 27. The result partially transfers to Qwen2.5-3B and to OR-Bench external prompts, but the component/path evidence does not establish a circuit. SAE feature interventions are real enough to beat matched random-SAE controls, yet they do not beat mean-difference or probe baselines under the held-out preregistered criterion.

## Research Question

Can an SAE isolate a compact, causally meaningful refusal representation that outperforms simpler activation directions under held-out intervention tests?

## Experimental Design

The audit uses harmful prompts paired with semantically related benign refusal-sensitive prompts. Discovery and SAE training use train records only. Threshold and intervention-scale selection use validation records only. All headline effects are measured on held-out test records.

The intervention set includes SAE feature steering, SAE feature ablation, SAE-direction projection patching, mean-difference direction ablation, linear-probe direction ablation, isotropic random directions, matched random-SAE latent controls, layerwise controls, and component-output localization.

## Primary Results

| Evidence block | Result | Interpretation |
|---|---:|---|
| Qwen2.5-1.5B residual direction | mean harmful-score delta `-0.136` | robust held-out causal effect |
| Qwen2.5-1.5B SAE feature ablation | harmful-score delta `-0.036` | nonzero, but weaker than mean direction |
| Linear probe ablation | harmful-score delta `-0.002` | not the dominant held-out causal readout |
| SAE stability grid | 0/6 H3 passes | no SAE superiority under fixed cells |
| OR-Bench external transfer | SAE toxic refusal-rate delta `-0.12`; mean `-0.68` | transfer present; SAE not mean-superior |
| Qwen2.5-3B | late layer 35 effect, specificity failed | partial family replication |
| Component/path analysis | residual strong, attention/MLP small or non-specific | not a circuit claim |

## Interpretation

The project supports a residual-stream localization claim. It does not support a circuit-discovery claim, and it does not support the claim that SAE features beat simple directions. That negative result is the main scientific value of the repository: the same harness that can surface sparse features can also reject the stronger SAE-superiority story.

## Threats To Validity

- Refusal is measured with a contrastive prefix log-probability score, not a full generated-answer judge.
- Controlled paired prompts reduce topic confounding but are not a substitute for broad naturalistic safety data.
- Component/path analysis is approximate and does not include complete head/path mediation.
- Non-Qwen replication remains unrun.
- SAE training scale, dictionary width, and prompt distribution may affect the negative SAE-vs-baseline conclusion.

## Reproducibility

CPU validation is covered by `make validate`. Real-model reproduction requires GPU access and regenerated ignored artifacts under `artifacts/`. Compact public summaries live under `results/` with stable filenames, and generated artifact hashes live in `results/final/artifact_manifest.json`.
