# A2 Model Sweep

The A2 sweep preregisters the named cross-model refusal grid before any cell runs.

- Frozen preregistration: `docs/preregistration_model_sweep.frozen.json`
- Hash: `7c1d0dc4a2792b81d1d3a1a959772af040359f83229e17a3e07f3e7cd59f754c`
- Config grid: `configs/a2_model_sweep/*.yaml`
- Aggregate summary: `results/model-sweep/a2_summary.json`
- Per-model evidence stubs or completed snapshots: `results/model-sweep/<model_slug>.json`
- Table: `docs/tables/model_sweep.md`
- Figure: `docs/figures/model_sweep.png`

## Current Result

Six of eleven frozen cells completed with 250 held-out pairs and 24 control draws:
Qwen2.5 0.5B, 1.5B, 3B, 7B, and 14B plus Mistral-7B. Every completed
cell selects the final scanned layer. The five Llama/Gemma cells are published as
`access_denied` failures.

The SAE-minus-mean harmful-score gap is favorable to SAE only at Qwen-0.5B and is
strongly mean-favorable at larger completed cells. Utility flags at Qwen-7B and
Qwen-14B reflect specificity costs in the mean intervention, not stronger SAE
harmful-score suppression. The cross-scale result is non-monotone. Equivalence and
inferiority fields for these examined outcomes are `post_hoc_sensitivity` analyses.

## Commands

Regenerate the predeclared configs:

```bash
python -m glassbox_audit.cli model-sweep write-configs \
  --output-dir configs/a2_model_sweep \
  --dataset-path data/refusal_controlled_v2_1000.jsonl \
  --preregistration-path docs/preregistration_model_sweep.frozen.json \
  --device cuda:0 \
  --sae-device cuda:0
```

Run the full grid in the preregistered order:

```bash
bash scripts/sweep_models.sh
```

Summarize completed and unrun cells:

```bash
python -m glassbox_audit.cli model-sweep summarize \
  --configs-dir configs/a2_model_sweep \
  --artifacts-root artifacts/a2_model_sweep \
  --output-json results/model-sweep/a2_summary.json \
  --output-table docs/tables/model_sweep.md \
  --output-figure docs/figures/model_sweep.png
```

## Fixed Budget Rule

Each config sets `sae.samples_per_feature: 8.0`, `sae.expansion_factor: 2`, and
`sae.tokens_per_prompt: 64`. At runtime, `glassbox run` computes:

```text
activation_sample_cap = hidden_size * expansion_factor * samples_per_feature
```

If a config also sets `sae.max_activation_samples`, the stricter cap wins. The effective cap is
written into `protocol.json`.

## Reporting Rule

The summary keeps every predeclared model visible. Cells without a completed artifact remain
`unrun` or `failed`. The sweep driver continues after a nonzero cell exit and writes a `failure.json` record;
failures discovered outside the driver can be published with `glassbox model-sweep mark-failed`.
Failed rows remain in the table and aggregate JSON before any scale-trend claim is made.
