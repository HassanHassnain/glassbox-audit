# Data

The repository commits only tiny fixtures used by tests and the CPU toy pipeline.

Committed fixtures:

| Path | Purpose |
|---|---|
| `data/fixtures/refusal_pairs_tiny.jsonl` | 24 harmful/benign matched prompt pairs for offline tests |
| `data/fixtures/external_refusal_fixture.jsonl` | tiny normalized-external-loader fixture |

Large controlled and external datasets are generated locally and ignored by git.

## Controlled Corpus

```bash
glassbox build-real-audit-data \
  --output data/refusal_controlled_v2_1000.jsonl \
  --pairs-per-family 25 \
  --source-name glassbox_controlled_refusal_v2_1000
```

Expected generated corpus shape: 1,000 matched pairs, 2,000 prompt records, 40 prompt families, with family-disjoint train/validation/test splits.

## OR-Bench Normalization

```bash
glassbox download-hf-external-data \
  --dataset bench-llm/or-bench \
  --subset or-bench-toxic \
  --split train \
  --output data/or_bench_toxic.normalized.jsonl \
  --default-harmful true \
  --source-name bench-llm/or-bench:or-bench-toxic

glassbox download-hf-external-data \
  --dataset bench-llm/or-bench \
  --subset or-bench-hard-1k \
  --split train \
  --output data/or_bench_hard1k.normalized.jsonl \
  --default-harmful false \
  --source-name bench-llm/or-bench:or-bench-hard-1k
```

`results/final/artifact_hashes.json` records hashes for the frozen generated artifacts used in the release claim. The full tensors, checkpoints, and model outputs are intentionally excluded from the public repository.
