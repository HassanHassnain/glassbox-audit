# Reproducibility

## Run Identity

Every output directory contains `manifest.json` with the full experiment config, dataset counts,
runtime versions, seed, and an explicit evidence label. Keep each seed in a separate output
directory and do not overwrite runs used in a report.

The bundled Qwen configs use `empirical_smoke_fixture` because they point at the tiny included
dataset. Change the label only after replacing it with a defensible evaluation dataset.

The bounded controlled audit uses `configs/qwen2.5-1.5b-real-audit.yaml`. Real-audit mode enforces
train-selected layers, validation-selected thresholds, layerwise controls, multi-token SAE
training, minimum random controls, and minimum bootstrap samples.

## Recommended Environment

- Python 3.10 or 3.11
- PyTorch 2.2+
- Transformers 4.45+
- CUDA GPU with enough memory for the selected dtype

Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui,accelerate]"
pytest
```

## Determinism

The runner seeds Python and PyTorch. The toy backend is deterministic on CPU. GPU kernels and model
quantization can still introduce small differences. Record GPU type, driver, CUDA version, exact
model revision, and any quantization setting for real-model claims.

## Data Discipline

The included fixture has 48 records and exists only for smoke testing. Build larger matched pairs
from an approved source:

```json
{"pair_id":"example-001","category":"cyber","harmful_prompt":"...","benign_prompt":"..."}
```

```bash
glassbox build-pairs --source data/source.jsonl --output data/refusal_pairs_large.jsonl --seed 17
glassbox validate-data --data data/refusal_pairs_large.jsonl
```

Review automatically assigned splits for prompt-family leakage before running experiments.

Reproduce the committed bounded audit:

```bash
glassbox build-real-audit-data --output data/refusal_controlled_v1.jsonl
CUDA_VISIBLE_DEVICES=0 glassbox run \
  --config configs/qwen2.5-1.5b-real-audit.yaml \
  --output artifacts/qwen-real-audit
CUDA_VISIBLE_DEVICES=0 glassbox refresh-controls \
  --artifacts artifacts/qwen-real-audit \
  --control random-sae \
  --device cuda:0
glassbox paper --artifacts artifacts/qwen-real-audit --docs docs
```

## Suggested Real-Model Run Matrix

| Axis | Minimum |
|---|---|
| Model seeds / SAE seeds | 3 |
| Held-out matched pairs | 1,000 |
| SAE widths | 4x, 8x |
| SAE sparsity | at least 2 top-k values |
| Control directions | 100 per setting |
| Prompt distributions | 2 |

Do not tune intervention scale on the test set. Use a validation-set Pareto criterion and report
the frozen scale on test.

The committed bounded audit's random untrained-SAE latent controls have been refreshed with the
trained SAE activation center, zeroed encoder bias, and literal encode -> zero selected latent ->
decode ablation. The refresh does not retune layers, features, thresholds, or scales.
