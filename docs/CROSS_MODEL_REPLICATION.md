# Cross-Model Replication

**Status:** Qwen2.5-3B replication completed; Phase 4 found no cached non-Qwen weights, so
Gemma/Llama remain prepared but unrun.

Local cache inspection found cached Qwen2.5 weights:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`

Gemma-2-2B-it and Llama-3.2-1B-Instruct were not found in the local Hugging Face cache during this
update. No gated-download troubleshooting was attempted.

## Prepared Configs

| Model | Config | Status |
|---|---|---|
| Qwen2.5-3B-Instruct | `configs/qwen2.5-3b-refusal-replication.yaml` | completed from cached weights |
| Gemma-2-2B-it | `configs/gemma-2-2b-it-refusal-replication.yaml` | prepared, weights not cached |
| Llama-3.2-1B-Instruct | `configs/llama-3.2-1b-instruct-refusal-replication.yaml` | prepared, weights not cached/gated |

Example command for the cached 3B candidate:

```bash
CUDA_VISIBLE_DEVICES=1 GLASSBOX_MODEL_DEVICE=cuda:0 GLASSBOX_SAE_DEVICE=cuda:0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  glassbox run \
  --config configs/qwen2.5-3b-refusal-replication.yaml \
  --output artifacts/qwen2.5-3b-replication
```

A completed replication should report the same method table as the main Qwen1.5B audit:

- SAE feature ablation and steering;
- mean-difference direction;
- logistic probe direction;
- isotropic random directions;
- architecture-matched random untrained-SAE latent controls;
- layerwise ablation controls;
- confidence intervals and paired comparisons.

## Completed Qwen2.5-3B Result

Artifact: `artifacts/qwen2.5-3b-replication-preregistered`

- selected layer: 35;
- held-out test records: 60;
- baseline harmful refusal / benign over-refusal: 100.0% / 3.3%;
- mean-difference ablation harmful-score delta: `-0.090 [-0.103, -0.076]`;
- mean-difference benign-score delta: `+0.390`;
- mean-difference capability-NLL delta: `+0.164`;
- SAE feature-ablation harmful-score delta: `-0.006 [-0.007, -0.005]`;
- matched random-SAE empirical p: `0.024`;
- preregistered H1 specificity: failed;
- preregistered H3 SAE-win criterion: failed.

The 3B model has causal late-layer interventions, but the strongest simple intervention is not
specific and the SAE does not beat the simple baselines on direct harmful-score suppression. See
[NEXT_PHASE_REPORT.md](NEXT_PHASE_REPORT.md).

## Phase 4 Status

Phase 4 strengthened the Qwen2.5-1.5B side with a 1,000-pair audit and external causal transfer.
It did not add a non-Qwen result. The local cache contained Qwen2.5-1.5B, Qwen2.5-3B, and
Qwen2.5-32B only; Qwen2.5-32B is outside the target 1-4B replication scope for this project phase.
