# Data Leakage Audit

- Controlled dataset: `data/refusal_controlled_v2_1000.jsonl`
- SHA256: `1c912d796753c4cd29104d0ed78c05b1c585394d98f67e513d92e3c9166276b6`
- Exact/normalized cross-split duplicates: `0` / `0`
- Family overlap across splits: `0`
- Pair overlap across splits: `0`
- Harmful/benign identical-prompt pairs: `0`
- Near-duplicate cross-split prompt warnings: `0`
- External exact normalized overlaps: `0`
- Formal leakage pass: `True`

Near-duplicate warnings are reported for reviewer inspection but are not treated as a hard failure unless family, pair, exact, or normalized prompt leakage is found.
