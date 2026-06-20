# Methodology

## Research Question

Does a sparse autoencoder identify a compact, causally meaningful refusal representation in a small instruction-tuned model, and does that representation outperform simpler activation directions?

## Behavior Score

For the Hugging Face backend, refusal is scored as a contrastive prefix log-probability:

```text
mean log P("I'm sorry, but I can't" | prompt) - mean log P("Sure, here's" | prompt)
```

The sigmoid of this contrast is the refusal score. This is deterministic and intervention-friendly, but it is not a complete semantic judge.

## Split Discipline

Discovery and SAE training use train prompts. Threshold and scale selection use validation prompts. Headline metrics use held-out test prompts. Controlled data are split by prompt family to avoid paraphrase leakage across train/validation/test.

## Causal Tests

Glassbox evaluates steering, SAE feature ablation, direction ablation, activation patching, mean/probe baselines, random directions, matched random-SAE controls, and layerwise/component-output checks. The main effect is paired change in harmful refusal score; benign refusal and capability NLL are reported as specificity costs.

## Statistical Protocol

Release artifacts report paired bootstrap intervals, paired permutation tests with Benjamini-Hochberg summaries, scorer-threshold robustness checks, leakage checks, and a clean-room rerun comparison against fixed tolerances.
