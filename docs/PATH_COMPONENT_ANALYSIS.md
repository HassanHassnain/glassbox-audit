# Path And Component Analysis

**Status:** completed for 520-pair and 1,000-pair Qwen2.5-1.5B artifacts.

Artifacts:

- `artifacts/qwen-expanded-medium-preregistered/path_component_phase4.json`
- `artifacts/qwen-expanded-1000-phase4/path_component_phase4.json`
- `results/component-path/phase4_component_path_summary.json`
- `docs/tables/component_path_phase4.md`

Glassbox supports the strongest clean analysis exposed by the current model wrapper:

- residual-stream direction ablation;
- attention-output direction ablation;
- MLP-output direction ablation;
- residual projection patching from harmful to benign train projection;
- residual projection patching from benign to harmful train projection.

The wrapper does not expose stable per-head output tensors or full path-patching contracts without
architecture-specific surgery. Those results are not claimed.

## 1,000-Pair Target-Layer Result

| Component | Harmful-score delta | Benign-score delta | Capability-NLL delta |
|---|---:|---:|---:|
| residual | -0.136 | -0.015 | -0.016 |
| attention output | -0.002 | +0.001 | +0.001 |
| MLP output | +0.007 | +0.023 | +0.004 |

Residual projection patching at layer 27:

- harmful-to-benign projection: harmful-score delta `-0.169`, refusal-rate delta `-0.528`;
- benign-to-harmful projection: benign-score delta `+0.210`, refusal-rate delta `+0.824`;
- benign-to-harmful patching had large capability/collateral shifts.

## Interpretation

The 1,000-pair result strengthens residual-stream localization but does not establish a confirmed
circuit. Attention and MLP output ablations do not mediate the main effect cleanly. Residual
projection patching shows necessity-like and sufficiency-like behavior, but it is still
activation-space projection evidence rather than path-level mediation with random path controls.

## Claim Rule

Large component effects are not sufficient for a circuit claim. A path/component mechanism requires
necessity, sufficiency, specificity, random controls, and held-out replication. Phase 4 does not
meet that bar.
