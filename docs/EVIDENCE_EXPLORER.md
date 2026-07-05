# Evidence Explorer

The C1 evidence explorer is a static GitHub Pages-ready view of the claim ladder. It groups frozen
artifact evidence by ladder level and renders an interactive HTML table from a JSON bundle.

Build it with:

```bash
python scripts/build_evidence_explorer.py \
  --artifacts-root results \
  --output-json results/evidence_explorer/explorer.json \
  --output-html docs/evidence/index.html
```

The JSON bundle is canonical. The HTML embeds the bundle and runs without a backend.

Evidence cells currently include:

- baseline behavior metrics;
- held-out causal intervention rows;
- external causal-transfer rows;
- component-localization rows;
- component/path summaries;
- explicit failed Level 5 confirmed-circuit status.
