# Release Notes

## Phase 6 Release Freeze

Phase 6 focuses on clean-room reproducibility and release hardening. It does not introduce new
mechanistic claims.

Added:

- interruption-safe stage markers and per-random-control checkpoints in the main audit runner;
- `docs/RUN_RESUME_DESIGN.md`;
- `results/final/resume_capability_check.json`;
- Phase 6 clean-room artifact root: `artifacts/phase6-cleanroom/qwen15b-1000`;
- release Make targets:
  - `make reproduce-cleanroom`;
  - `make reproduce-external`;
  - `make final-report`;
  - `make release-check`.

Unchanged claim:

> Glassbox found a robust late residual-stream refusal-relevant direction in Qwen2.5-1.5B, with
> partial Qwen2.5-3B replication and external causal transfer, but did not confirm a full circuit.
> SAE features beat matched random-SAE controls but did not beat mean/probe baselines under
> preregistered held-out criteria.

Clean-room status:

- `artifacts/phase6-cleanroom/qwen15b-1000` completed;
- 128/128 random-control checkpoints were written;
- Phase 6 reproduced Phase 4 within fixed tolerance bands;
- the final supported claim is unchanged.

See `docs/REPRODUCIBILITY_REPORT.md` and `results/final/phase6_reproducibility_diff.json`.
