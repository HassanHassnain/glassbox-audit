# Causal-Handle Leaderboard

The B3 leaderboard is a static JSON + HTML view over completed Glassbox artifacts. Rows are causal
handles: `(model, behavior, method, SAE checkpoint)` with held-out effect, specificity, capability
cost, matched-control evidence, verdict, and artifact hashes.

Build it with:

```bash
python scripts/build_leaderboard.py \
  --artifacts-root results \
  --output-json results/leaderboard/leaderboard.json \
  --output-html docs/leaderboard/index.html
```

Use `--strict` for submission validation when reviewing a new leaderboard artifact. Strict mode
requires the artifact directory to include:

- `manifest.json` with `schema_version: "1.0"` and `preregistration_hash`;
- `summary.json`;
- `protocol.json`;
- `interventions.json` with `causal_tests` and `random_controls`.

The builder does not run models. It only reads frozen evidence, computes row verdicts, hashes the
source files, and writes a static HTML table suitable for GitHub Pages.
