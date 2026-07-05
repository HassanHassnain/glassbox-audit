# Contributing

Keep changes scoped, tested, and explicit about evidence quality. New behavior metrics should include
a validation argument; new interpretability methods should use the shared held-out intervention and
control protocol. Never commit model weights, private datasets, or results presented without their
manifest.

Before opening a pull request:

```bash
ruff check .
pytest
```

## Leaderboard Submissions

Public causal-handle leaderboard submissions must include a complete artifact directory, not just a
summary table. At minimum include:

- `manifest.json` with `schema_version: "1.0"` and `preregistration_hash`;
- `summary.json`;
- `protocol.json`;
- `interventions.json` with held-out `causal_tests` and `random_controls`;
- any files named in the artifact hashes.

Validate a submission with:

```bash
python scripts/build_leaderboard.py --artifacts-root path/to/submission --strict
```

Do not submit rows whose thresholds, feature choices, scales, or controls were tuned after looking
at the held-out test split.
