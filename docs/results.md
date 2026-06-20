# Results

## Final Supported Claim

Glassbox found a robust late residual-stream refusal-relevant direction in Qwen2.5-1.5B, with partial Qwen2.5-3B replication and external causal transfer, but did not confirm a full circuit. SAE features beat matched random-SAE controls but did not beat mean/probe baselines under preregistered held-out criteria.

## Compact Matrix

| Result family | Status | Main readout |
|---|---|---|
| Qwen2.5-1.5B expanded audit | completed | layer 27 replicated; mean ablation `-0.136`; SAE ablation `-0.036`; H1 passed, H3 failed |
| SAE stability grid | completed | 6/6 fixed cells; SAE deltas `-0.061` to `-0.016`; 0/6 H3 passes |
| OR-Bench external causal transfer | completed | SAE toxic refusal-rate delta `-0.12`; mean `-0.68`; SAE not mean-superior |
| Qwen2.5-3B replication | partial | late layer 35 effect, but specificity failed |
| Component/path analysis | completed | residual layer 27 strong; attention/MLP outputs small or non-specific |
| Clean-room reproducibility | completed | reproduced target layer, baseline rates, intervention deltas, and H3 failure within fixed tolerances |

## What Failed

- SAE superiority over mean/probe under preregistered H3.
- Component/path criteria for a circuit-level mechanism.
- Clean specificity in the Qwen2.5-3B replication.
- Non-Qwen replication, because prepared Gemma/Llama configs were not run.
- Response-level judging, because final real-model artifacts rely on contrastive scoring.

## Machine-Readable Evidence

The compact committed evidence lives under `results/final`, `results/sae-stability`, `results/external-causal`, `results/component-path`, and `results/cross-model`. Heavy generated artifacts, tensors, checkpoints, prompt outputs, and model caches are excluded.
