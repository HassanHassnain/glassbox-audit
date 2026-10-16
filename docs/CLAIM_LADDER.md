# Claim Ladder

Glassbox uses a conservative claim ladder so causal interventions are not mistaken for a confirmed
mechanistic circuit.

| Level | Claim | Required Evidence | Current Status |
|---:|---|---|---|
| 1 | Behavior-only score pattern | Frozen scoring on held-out or external prompts | Supported for controlled corpus and OR-Bench behavior-only checks |
| 2 | Residual-stream causal direction | Held-out residual intervention with uncertainty and controls | Supported most strongly in Qwen2.5-1.5B |
| 3 | Component-output localization | Attention/MLP/residual component interventions with specificity | Partially tested; no clean attention/MLP component confirmed |
| 4 | Candidate path/component mechanism | Necessity, sufficiency, specificity, controls, and replication | Not yet supported |
| 5 | Confirmed circuit | Component/path mechanism with necessity, sufficiency, specificity, random controls, and held-out/cross-model replication | Not supported |

## Operational Rule

Use "residual-stream refusal-relevant direction" for the main positive finding. Use "candidate
mechanism" only for hypotheses that have some component/path evidence. Do not use "confirmed
circuit" unless Level 5 passes.

## Why This Matters

A steering or ablation direction can be useful without proving a circuit. Sparse features can be
real relative to random controls while still failing to outperform simpler mean/probe baselines.
Glassbox reports both facts rather than smoothing them into a flattering story.
