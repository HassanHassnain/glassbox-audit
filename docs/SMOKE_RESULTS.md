# Smoke Results

These measurements validate the implementation. They are not publication-grade findings: the
included dataset has only 16 matched pairs per train/validation/test split, the SAE is trained on
only 16 activation vectors, and only one seed was run.

## Qwen2.5-1.5B-Instruct

Run on June 14, 2026 with `configs/qwen2.5-1.5b-refusal-smoke.yaml` on one RTX 3090.

| Item | Measurement |
|---|---:|
| Located residual-stream layer | 24 |
| Validation-selected refusal threshold | 0.8594 |
| Test harmful refusal rate | 87.5% |
| Test benign over-refusal rate | 62.5% |
| Continuous behavior gap | 0.103 |
| SAE variance explained | 0.980 |
| SAE dead-feature fraction | 0.955 |

### Causal Results

Paired bootstrap intervals use the smoke config's 100 resamples.

| Intervention | Harmful-score delta | Benign-score delta | Capability-NLL delta |
|---|---:|---:|---:|
| SAE feature ablation | -0.019 `[-0.029, -0.009]` | -0.000 | -0.003 |
| SAE activation patch | -0.128 `[-0.171, -0.089]` | -0.021 | +0.243 |
| SAE direction ablation | -0.029 `[-0.059, +0.003]` | +0.059 | +0.104 |
| Mean-difference ablation | -0.045 `[-0.087, -0.004]` | +0.072 | +0.132 |
| Linear-probe ablation | +0.009 `[-0.012, +0.034]` | +0.078 | +0.081 |

The feature ablation is small but selective and larger than the four matched smoke controls, whose
harmful-score deltas range from `-0.0006` to `+0.0002`. Stronger interventions also produce clear
collateral effects. The high dead-feature fraction is expected from training a 3,072-feature SAE on
only 16 examples and is a warning against overinterpreting the result.

### Honest Read

This smoke run establishes that activation collection, SAE training, feature selection, held-out
interventions, controls, confidence intervals, and capability-cost measurement all function on the
real model. It does not establish a stable refusal mechanism. The next defensible experiment needs a
larger paired dataset, multiple SAE seeds, and a semantic refusal grader.

## Synthetic Fixture

The committed `artifacts/demo` run correctly locates the planted layer. It also demonstrates a
negative SAE comparison: the mean-difference direction gives the strongest causal ablation, while
SAE feature ablation is null. This is intentional evidence that the reporting path does not force an
SAE-positive conclusion.
