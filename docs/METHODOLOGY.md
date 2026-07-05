# Methodology

## Research Question

Does a sparse autoencoder identify a compact, causally meaningful refusal representation in a
small instruction-tuned model, and does that representation outperform simpler directions?

The primary claim is deliberately narrow: a set of residual-stream SAE features at one layer
mediates part of the model's refusal behavior on held-out matched prompts.

## Behavior Operationalization

For the Hugging Face backend, refusal is a continuous contrast:

```text
mean log P("I’m sorry, but I can’t" | prompt)
  - mean log P("Sure, here’s" | prompt)
```

The sigmoid of this contrast is the refusal score. This is cheap, deterministic, and supports
intervention sweeps, but it is not a complete semantic judge. A publication run should add blinded
generation grading and report agreement with the prefix score.

When `evaluation.threshold` is null, Glassbox selects the binary refusal threshold by validation
balanced accuracy, then freezes it for held-out test metrics. Continuous-score intervention effects
do not depend on this threshold.

The dataset is paired by intent category. Each harmful prompt has a semantically related benign
counterpart, reducing topic confounding. Splits must be assigned at the pair or prompt-family level.
The controlled real-audit corpus contains 120 pairs across 40 scenario families; all paraphrases of
a family remain in one split.

## Discovery Protocol

1. Capture the final prompt-token residual stream at configured layers on train.
2. Rank layers by separation strength: absolute Cohen's d of the harmful-vs-benign
   mean-difference projection multiplied by the mean-difference norm.
3. Train a top-k SAE on multiple token activations from train prompts at the selected layer after
   mean centering, or load a predeclared SAELens public SAE at its configured hook layer. Final
   prompt-token activations remain the feature-ranking and intervention site.
4. Rank SAE features or public SAE latents by harmful-vs-benign activation effect size on train.
5. Form an SAE direction from the positively associated top feature decoder vectors.
6. Fit a linear-probe direction and compute the raw mean-difference direction on the same data.

Validation selects intervention scale after the discovery method is frozen. The held-out test split
is not used for layer, feature, or scale selection.

## Causal Tests

- **Steering:** add a scaled unit direction at the selected layer.
- **SAE feature ablation:** encode the hidden state, zero selected features, decode the change, and
  add only that decoded delta to the original state.
- **Direction ablation:** remove the selected SAE subspace projection.
- **Activation patching:** replace the SAE-direction projection with the discovery-set benign mean.
- **Baseline ablation:** remove the mean-difference and linear-probe directions using the same
  held-out protocol, allowing a simpler method to supply the strongest causal explanation.

All tests run on held-out prompts. The main effect is the paired change in harmful refusal score.
Benign score and capability NLL changes measure specificity and damage.

For the Hugging Face backend, behavioral scoring applies the intervention from the final prompt
token through the teacher-forced continuation. Capability perplexity applies the same intervention
tokenwise across the prompt, providing a general language-model degradation estimate rather than a
behavior-specific score.

## Baselines and Controls

- Mean-difference direction
- Logistic linear-probe direction
- Isotropic random directions at the same layer and scale
- Decoder directions from a randomly initialized, architecture-matched SAE for locally trained
  SAEs, or random latents from the same public dictionary for SAELens SAE-zoo runs
- Mean-difference direction ablation at every scanned layer

Random directions are tested under steering and direction ablation. For locally trained SAEs,
random untrained-SAE decoder directions are tested under steering, while their corresponding
random latents are tested with actual encode-zero-decode feature ablation. For public SAELens
SAEs, matched controls draw seeded random latents from the same public dictionary rather than a
separate untrained SAE.
Selected methods are also compared directly with paired method-vs-method intervals, rather than
only independently against the unmodified baseline.

The A4 feature-budget frontier evaluates fixed SAE latent budgets
`{1, 2, 5, 10, 20, 50, 100, 500, 1000}`. Top-k rows use train-only feature-effect ranking. The
weighted rows fit L1-regularized logistic weights on train SAE activations only, then test the
corresponding decoder-space direction on held-out prompts.

The important comparison is not only maximum effect. Report the Pareto trade-off between harmful
behavior change, benign behavior change, and capability cost. Hyperparameter selection must use
validation data or a pre-registered scale, never the held-out test set.

## Statistical Protocol

Glassbox reports paired non-parametric bootstrap confidence intervals clustered by matched
`pair_id`, so harmful/benign counterparts are resampled together when a metric spans both sides of
the pair. A6 also emits `equivalence_analysis.json` with preregistered TOST equivalence tests,
one-sided SAE-inferiority tests against mean/probe baselines, and a paired-normal power analysis
for the configured minimum effect size. For a serious claim:

- use at least 1,000 held-out matched pairs;
- bootstrap at the prompt-family level when variants share a source;
- freeze equivalence/inferiority margins and power assumptions before held-out reporting;
- run at least three model/SAE seeds;
- correct for multiple feature or layer searches;
- replicate on a second prompt distribution.

## Known Limitations

- Final-token residual activations may miss token-distributed or attention-head circuits.
- Prefix log-probability scoring is an imperfect refusal metric.
- A single SAE can be unstable across seeds and widths.
- Ablation can move hidden states off distribution.
- Matched prompt quality strongly affects apparent feature specificity.
