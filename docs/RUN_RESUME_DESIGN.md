# Run Resume Design

Phase 6 adds interruption-safe progress tracking to the main audit runner without changing the
public artifact contract.

## Stage Markers

Each run writes `checkpoints/*.complete.json` markers and `checkpoints/stage_timings.json` for:

- data load and split validation;
- model load;
- layer scan;
- SAE training;
- feature discovery;
- validation threshold/scale selection plus baseline scoring;
- held-out steering sweeps;
- held-out causal tests;
- random direction and matched random-SAE controls;
- layerwise controls;
- failure analysis and final artifact generation.

The active stage is written to `checkpoints/current_stage.json`, and a compact progress log is
written to `checkpoints/progress.log`.

## Random-Control Resume

Random controls are the long tail of the Qwen audit. Each random direction and matched random-SAE
control writes a separate JSON checkpoint under:

```text
checkpoints/random_controls/
```

On rerun, existing control checkpoint files are loaded instead of recomputed. This makes the
previous Phase 5 failure mode recoverable: interruption during random controls no longer loses every
completed control.

## Finalization Rule

Final headline files such as `summary.json`, `interventions.json`, `method_comparisons.json`, and
`failure_analysis.json` are still written only after all expected stages finish. A directory with
only checkpoint files is therefore a partial run, not a completed audit.

## Capability Check

The toy checkpoint audit at `artifacts/phase6-checkpoint-toy` verifies stage markers, random-control
checkpoints, final artifact presence, and protocol metadata. Machine-readable status:

```text
results/final/resume_capability_check.json
```
