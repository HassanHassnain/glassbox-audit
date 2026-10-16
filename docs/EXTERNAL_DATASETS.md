# External Datasets

Glassbox has an external refusal/safety dataset loader in `glassbox_audit.external_data`.
It supports offline JSONL, JSON, and CSV files, plus a Hugging Face normalization command when the
`datasets` package and network/cache access are available.

## Normalized Schema

Every external row is normalized to:

- `prompt`
- `behavior_label`
- `harmful`
- `split`
- `source`
- `category`
- `family_id`
- `pair_id`
- `metadata`

The existing audit pipeline requires paired harmful/benign rows. If an external dataset provides
`pair_id` plus a `harmful` flag, it can be exported into the Glassbox `PromptRecord` schema.
Unpaired external datasets can still be evaluated with `glassbox external-evaluate`, but that mode
is behavior-only and does not support causal SAE/intervention claims.

## Fixture Validation

The repository includes `data/external_refusal_fixture.jsonl` only as a loader fixture. It has 12
records across train/validation/test and is not a benchmark.

```bash
glassbox normalize-external-data \
  --source data/external_refusal_fixture.jsonl \
  --output data/external_refusal_fixture.normalized.jsonl \
  --glassbox-output data/external_refusal_fixture.glassbox.jsonl
```

This command was run successfully and produced 12 normalized records plus 12 paired Glassbox
records.

## Plugging In A Real Dataset

For a local JSONL or CSV dataset:

```bash
glassbox normalize-external-data \
  --source /path/to/refusal_dataset.jsonl \
  --output data/my_external.normalized.jsonl \
  --glassbox-output data/my_external.glassbox.jsonl \
  --prompt-field prompt \
  --label-field category \
  --harmful-field harmful \
  --split-field split \
  --source-field source \
  --category-field category \
  --family-field family_id \
  --pair-field pair_id
```

Then point an audit config at `data/my_external.glassbox.jsonl`. For a serious external validation,
use family-disjoint train/validation/test splits and do not tune thresholds or intervention scales
on the held-out test split.

For Hugging Face subsets where harmfulness is defined by the subset rather than a per-row field:

```bash
glassbox download-hf-external-data \
  --dataset bench-llm/or-bench \
  --subset or-bench-hard-1k \
  --split train \
  --output data/or_bench_hard1k.normalized.jsonl \
  --default-harmful false \
  --source-name bench-llm/or-bench:or-bench-hard-1k
```

## Current External Status

OR-Bench hard and toxic subsets were downloaded and normalized during final hardening. They are not
matched pairs, so they were evaluated only with the behavior-only external evaluator.

See [EXTERNAL_VALIDATION.md](EXTERNAL_VALIDATION.md) for the bounded Qwen2.5-1.5B OR-Bench result:
95% refusal on sampled toxic prompts and 81% refusal on sampled hard-benign prompts. This is an
over-refusal warning, not a causal or SAE result.
