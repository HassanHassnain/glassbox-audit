# Reproducibility

## CPU Validation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make validate
make report
make release-check
```

`make validate` runs unit tests, lint, compileall, and the deterministic toy audit. It does not require GPUs or model weights.

## Real-Model Clean-Room Rerun

```bash
pip install -e ".[dev,accelerate,data]"
CUDA_VISIBLE_DEVICES=0 make reproduce-cleanroom
```

This regenerates `data/refusal_controlled_v2_1000.jsonl` and writes the Qwen2.5-1.5B audit to `artifacts/qwen2.5-1.5b-expanded-audit`.

Expected shape: 1,000 matched pairs, 2,000 records, selected layer 27, residual mean-difference ablation stronger than SAE feature ablation, and H3 remaining failed.

## External Causal Transfer

After regenerating OR-Bench normalized inputs and the frozen Qwen artifact:

```bash
CUDA_VISIBLE_DEVICES=0 make reproduce-external
```

OR-Bench rows are unpaired; report them separately from controlled paired-corpus effects.

## Frozen Release Diff

The clean-room Qwen2.5-1.5B rerun reproduced the frozen audit within fixed tolerances. See `results/final/phase6_reproducibility_diff.json` for exact fields and tolerances.
