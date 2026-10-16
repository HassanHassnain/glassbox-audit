# Final Reviewer Report

## Abstract

Glassbox is a preregistered causal activation-level audit of refusal behavior in small Qwen
instruction models. Phase 4 completed a 1,000-pair controlled Qwen2.5-1.5B audit, the full six-cell
SAE seed/width stability grid, OR-Bench external causal-transfer evaluation, and component/path
approximation runs. The main positive result is a robust late residual-stream refusal-relevant
direction. The main negative results are equally important: the project does not confirm a full
circuit, and SAE features do not beat mean-difference/probe baselines under preregistered held-out
criteria.

Machine-readable summary: [`results/final/phase4_outcomes.json`](../results/final/phase4_outcomes.json)

Phase 5 release hardening added statistical tests, scorer robustness, leakage checks, and fixed-grid
dose response. Phase 6 completed the clean-room Qwen2.5-1.5B 1,000-pair rerun with
interruption-safe checkpoints. It reproduced Phase 4 within fixed tolerances and did not change the
supported claim.

## Preregistration

Phase 4 hypotheses, metrics, baselines, and stopping rules were frozen in
[PHASE4_PREREGISTERED_VALIDATION.md](PHASE4_PREREGISTERED_VALIDATION.md) before the new Phase 4
results were analyzed. No layer, feature, threshold, scale, prompt, seed, width, or criterion was
retuned after seeing held-out outcomes.

## Completed Evidence

### Phase 5 Release Hardening

Artifacts:

- `results/final/statistical_tests.json`
- `results/final/scorer_robustness.json`
- `results/final/data_leakage_audit.json`
- `results/final/dose_response.json`
- `results/final/reproducibility_diff.json`
- `results/final/phase5_release_summary.json`
- `results/final/phase6_reproducibility_diff.json`
- `results/final/resume_capability_check.json`
- `results/final/artifact_hashes.json`

Phase 5/6 paired permutation/BH summaries preserve the Phase 4 interpretation: residual
mean-difference effects are robust, SAE effects are nonzero, and SAE does not beat mean/probe under
H3. Fixed contrastive scorer-threshold variants preserve the headline ordering. The controlled
1,000-pair corpus passes formal split/leakage checks. Dose-response curves show a tiny
SAE-favorable utility point, but not direct harmful-score superiority.

The Phase 6 clean-room Qwen rerun completed on `CUDA_VISIBLE_DEVICES=1`. It wrote all required
artifacts and 128/128 random-control checkpoints. It reproduced Phase 4 exactly on the key frozen
metrics reported in the reproducibility diff.

### 1,000-Pair Qwen2.5-1.5B Internal Audit

Artifact: `artifacts/qwen-expanded-1000-phase4`

| Metric | Value |
|---|---:|
| Held-out test records | 500 |
| Selected layer | 27 |
| Baseline harmful refusal | 64.0% |
| Baseline benign over-refusal | 14.8% |
| Mean-difference harmful-score delta | -0.136 `[-0.143, -0.128]` |
| Mean-difference benign-score delta | -0.015 |
| Mean-difference capability-NLL delta | -0.016 |
| SAE feature-ablation harmful-score delta | -0.036 `[-0.039, -0.034]` |
| SAE training vectors | 47,645 |
| SAE dead-feature fraction | 3.7% |

H1 internal residual robustness passed. The selected late residual layer remained 27, and the
mean-difference residual intervention produced a large held-out harmful-refusal effect with low
benign/capability cost.

The legacy utility flag for SAE was favorable in this run, but preregistered H3 did not pass:
SAE feature ablation remained much weaker than mean-difference on direct harmful-score suppression.

### SAE Stability Grid

Artifact: `results/sae-stability/phase4_full_grid_summary.json`

All six fixed cells completed. No failed seed or width was excluded.

| Summary | Value |
|---|---:|
| Completed cells | 6 / 6 |
| H3 passing cells | 0 |
| SAE harmful-score delta range | -0.061 to -0.016 |
| SAE harmful-score delta mean | -0.037 |
| SAE harmful-score delta stdev | 0.014 |
| Dead-feature fraction range | 14.0% to 48.8% |

H2 failed. SAE feature effects were real relative to matched random-SAE controls, but no completed
cell beat mean-difference on direct harmful-score suppression.

### External OR-Bench Causal Transfer

Preferred artifact: `artifacts/external-causal/or_bench_qwen15b_1000_phase4.json`

OR-Bench rows are unpaired, so this is reported separately from the paired controlled-corpus audit.
Frozen controlled-corpus interventions were applied without OR-Bench rediscovery.

| Method | Toxic refusal-rate delta | Hard-benign refusal-rate delta | Harmful-score delta | Utility |
|---|---:|---:|---:|---:|
| SAE feature ablation | -0.12 | -0.09 | -0.046 | +0.0062 |
| Mean-difference ablation | -0.68 | -0.55 | -0.163 | +0.0060 |
| Linear-probe ablation | +0.00 | +0.00 | -0.004 | +0.0003 |

External causal transfer is present: frozen interventions reduce OR-Bench refusal scores and SAE
feature ablation beats random direction and matched random-SAE controls (`p=0.030`). However,
mean-difference remains far stronger on direct toxic refusal-score suppression. This is not an
SAE-superiority result.

### Component And Path Evidence

Artifacts:

- `artifacts/qwen-expanded-medium-preregistered/path_component_phase4.json`
- `artifacts/qwen-expanded-1000-phase4/path_component_phase4.json`

At 1,000 pairs, layer-27 residual ablation remained strong (`-0.136` harmful-score delta). Direct
attention-output and MLP-output ablations at layer 27 were small or non-specific:

| Component | Harmful-score delta | Benign-score delta | Capability-NLL delta |
|---|---:|---:|---:|
| residual | -0.136 | -0.015 | -0.016 |
| attention output | -0.002 | +0.001 | +0.001 |
| MLP output | +0.007 | +0.023 | +0.004 |

Residual projection patching was strong, including harmful-to-benign projection patching
(`-0.169` harmful-score delta) and benign-to-harmful projection patching (`+0.210` benign-score
delta), but sufficiency-like patching had large collateral shifts. This supports residual-stream
localization, not a confirmed circuit.

## Failed Hypotheses

- H2 SAE stability and utility failed: 0/6 stability cells passed preregistered H3.
- H4 component/path evidence failed for a confirmed circuit: no clean component/path mechanism met
  necessity, sufficiency, specificity, controls, and replication criteria.
- SAE did not beat mean/probe under preregistered held-out criteria, despite beating matched
  random-SAE controls.

## Inconclusive Results

- Non-Qwen replication remains unrun because Gemma/Llama weights were not present in the local
  cache and gated-download troubleshooting was intentionally avoided.
- Head-level and full path-patching claims remain unrun because the wrapper does not expose stable
  per-head/path hooks without architecture-specific surgery.
- Optional larger 96-token SAE scale configs are prepared but unrun.

## Strongest Supported Claim

Qwen2.5-1.5B has a robust late residual-stream refusal-relevant direction on the controlled paired
corpus, with external OR-Bench causal transfer and partial Qwen2.5-3B replication. Glassbox does
not confirm a full circuit, and SAE features do not beat mean/probe baselines under preregistered
held-out criteria.

## Unsupported Claims

The current evidence does not support:

- "the refusal circuit";
- a head-level or path-level mechanism;
- SAE superiority over mean/probe baselines;
- non-Qwen replication results.

## Reproduce Core Phase 4 Runs

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  glassbox run \
  --config configs/qwen2.5-1.5b-expanded-audit.yaml \
  --output artifacts/qwen-expanded-1000-phase4

CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  glassbox run \
  --config configs/qwen2.5-1.5b-stability-seed23-exp4.yaml \
  --output artifacts/sae-stability/qwen15b-medium-seed23-exp4

CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  glassbox external-causal \
  --artifacts artifacts/qwen-expanded-1000-phase4 \
  --inputs data/or_bench_toxic.normalized.jsonl data/or_bench_hard1k.normalized.jsonl \
  --output artifacts/external-causal/or_bench_qwen15b_1000_phase4.json \
  --max-records-per-label 100 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  glassbox path-component \
  --artifacts artifacts/qwen-expanded-1000-phase4 \
  --layers 27 \
  --components residual,attention,mlp \
  --device cuda:0 \
  --output-name path_component_phase4.json
```
