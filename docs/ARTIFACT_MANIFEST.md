# Artifact Manifest

This manifest lists the release-critical artifacts and their role. Hashes are generated into
`results/final/artifact_hashes.json` and rendered in `docs/tables/artifact_hashes.md`.

| Path | Role |
|---|---|
| `artifacts/qwen-expanded-1000-phase4/summary.json` | Frozen Phase 4 headline audit |
| `artifacts/qwen-expanded-1000-phase4/interventions.json` | Frozen Phase 4 causal tests and controls |
| `artifacts/qwen-expanded-1000-phase4/path_component_phase4.json` | Frozen component/path approximation |
| `artifacts/external-causal/or_bench_qwen15b_1000_phase4.json` | Frozen OR-Bench external causal transfer |
| `results/final/statistical_tests.json` | Phase 5/6 statistical tests |
| `results/final/scorer_robustness.json` | Contrastive-threshold scorer robustness |
| `results/final/data_leakage_audit.json` | Split/leakage audit |
| `results/final/dose_response.json` | Fixed-grid dose-response summary |
| `results/final/resume_capability_check.json` | Checkpoint/resume capability check |
| `results/final/phase6_reproducibility_diff.json` | Phase 6 clean-room diff or incomplete status |
| `results/final/phase6_validation_summary.json` | Final validation summary |
| `results/final/artifact_hashes.json` | Release checksums |

## Phase 6 Clean-Room Required Artifacts

The Phase 6 clean-room artifact at `artifacts/phase6-cleanroom/qwen15b-1000` is complete only if
all of the following exist:

- `summary.json`
- `interventions.json`
- `failure_analysis.json`
- `method_comparisons.json`
- `layer_scan.json`
- `layerwise_controls.json`
- `sae_metrics.json`
- `statistical_tests.json`
- `manifest.json`
- `protocol.json`

This release has all required files present.
