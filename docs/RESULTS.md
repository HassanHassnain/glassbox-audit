# Result Protocol

This repository contains no publication-grade Qwen claim. The included demo is a synthetic
planted-mechanism fixture, and [SMOKE_RESULTS.md](SMOKE_RESULTS.md) reports a tiny real-model
implementation check. Neither should be cited as a stable mechanism result.

For each real-model run, report:

1. Exact model name and revision, data provenance, split method, and seeds.
2. Baseline harmful refusal, benign over-refusal, and prompt perplexity.
3. Layer scan and SAE diagnostics, including dead features and variance explained.
4. Held-out steering, ablation, and patching effects with paired confidence intervals.
5. Capability and benign-behavior cost at each intervention strength.
6. Mean-difference, probe, random-direction, and random-SAE comparisons.
7. Failure cases and prompt-level outliers.
8. Results from repeated seeds and a second prompt distribution.

## Claim Ladder

- **Correlational:** feature activates differently on harmful and benign prompts.
- **Steerable:** adding its decoder direction changes refusal score.
- **Causal contribution:** feature ablation or activation patching changes held-out behavior.
- **Specific contribution:** effect exceeds random controls without comparable benign/capability cost.
- **Replicated mechanism:** result survives seeds, datasets, and ideally another model.

Use the weakest claim supported by the evidence.
