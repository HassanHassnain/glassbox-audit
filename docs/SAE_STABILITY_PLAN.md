# SAE Stability Plan

This hardening phase uses the fixed grid in `configs/sae_stability_grid.yaml`.

Full grid:

- seeds: `17`, `23`, `42`;
- expansion factors: `2`, `4`;
- top-k: `32`;
- tokens per prompt: `48`;
- dataset: `data/refusal_controlled_v2_520.jsonl`;
- random controls: `24`;
- steering scales: `[-4, -2, -1, 1, 2, 4]`.

## Fixed Feasible Subset

The full six-cell grid would require six full 520-pair Qwen2.5-1.5B audits. Before looking at any
new stability outcomes, this phase fixes the largest practical subset as:

| Cell | Status |
|---|---|
| seed 17, expansion 2 | already completed as `artifacts/qwen-expanded-medium-preregistered` |
| seed 23, expansion 2 | run in hardening phase |
| seed 17, expansion 4 | run in hardening phase |
| seed 23, expansion 4 | unrun unless compute remains |
| seed 42, expansion 2 | unrun unless compute remains |
| seed 42, expansion 4 | unrun unless compute remains |

The subset was chosen to test one seed perturbation and one width perturbation from the completed
expanded audit. It is not chosen based on SAE outcomes.

## Fixed Interpretation

For each completed cell, report:

- SAE feature-ablation harmful-score delta;
- matched random-SAE empirical p-value;
- direct comparison against mean-difference and logistic-probe baselines;
- whether preregistered H3 passes.

H3 passes only if the SAE beats mean-difference and logistic probe on utility, direct harmful-score
suppression, and matched random controls. If SAE improves over random controls but remains weaker
than mean/probe, H3 fails.
