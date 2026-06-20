# Artifact Manifest

## Committed Evidence

| Path | Purpose |
|---|---|
| `results/final/phase4_outcomes.json` | frozen final outcome matrix |
| `results/final/phase6_reproducibility_diff.json` | clean-room fixed-tolerance comparison |
| `results/final/statistical_tests.json` | paired permutation/BH summaries |
| `results/final/scorer_robustness.json` | fixed threshold scorer robustness |
| `results/final/data_leakage_audit.json` | split/data leakage checks |
| `results/final/dose_response.json` | fixed-grid dose response summary |
| `results/sae-stability/phase4_full_grid_summary.json` | six-cell SAE stability grid |
| `results/external-causal/or_bench_qwen15b_1000_phase4_summary.json` | compact OR-Bench causal transfer summary |
| `results/component-path/phase4_component_path_summary.json` | residual/component/path summary |
| `results/cross-model/qwen2.5-3b-replication-summary.json` | compact Qwen2.5-3B replication summary |

## Excluded Generated Artifacts

The public repository intentionally excludes full `artifacts/` runs, tensors, checkpoints, prompt-output records, model caches, large generated corpora, local logs, and phase scratch folders. Regenerate them with `make reproduce-cleanroom`, `make reproduce-external`, and the commands in `data/README.md`.
