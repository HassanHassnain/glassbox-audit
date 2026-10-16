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
- A2 cross-model completed and access-failed cells;
- A3 source-separation and endpoint-transfer cells;
- A1 SAE-zoo access failures;
- A7 sycophancy baseline and intervention cells;
- A8 completed or explicitly unrun robustness cells;
- B2 mechanism-drift verdicts;
- explicit failed Level 5 confirmed-circuit status.

Aggregate-only cells never acquire a stronger verdict than their source supports.
Point-estimate-only A2 intervention rows are informational, absent A8 corpus cells
are `unrun`, and access failures are distinct from failed empirical criteria.
