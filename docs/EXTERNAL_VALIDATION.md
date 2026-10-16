# External Validation

**Status:** OR-Bench normalized and evaluated as behavior-only external validation.

Glassbox can normalize offline JSONL/JSON/CSV files and Hugging Face dataset subsets. In this
hardening phase, local cache inspection found no paired naturalistic refusal benchmark already
present. OR-Bench was downloaded from Hugging Face after a sandbox DNS failure was retried with
approved network access.

Downloaded normalized files:

- `data/or_bench_hard1k.normalized.jsonl`: 1,319 benign hard over-refusal prompts;
- `data/or_bench_toxic.normalized.jsonl`: 655 toxic prompts.

These files are not matched harmful/benign pairs. Therefore, they were not exported into the
Glassbox paired causal-audit schema and were not used for SAE/intervention claims.

## Bounded Behavior-Only Check

Artifact: `artifacts/external/or_bench_qwen15b_behavior_eval.json`  
Snapshot: `results/external/or_bench_qwen15b_behavior_eval_summary.json`  
Model/threshold source: `artifacts/qwen-expanded-medium-preregistered`

The run sampled 100 toxic and 100 hard-benign prompts with seed 17 and applied the frozen
Qwen2.5-1.5B expanded-audit refusal threshold.

| Subset | Records | Refusal rate | Mean refusal score |
|---|---:|---:|---:|
| OR-Bench toxic | 100 | 95.0% | 0.946 |
| OR-Bench hard benign | 100 | 81.0% | 0.908 |

Interpretation: the frozen refusal score detects refusal on toxic prompts, but also flags a very
high rate of hard benign prompts. This supports a cautious over-refusal warning and does not change
the causal claims from the controlled paired corpus.

## Reproduce

```bash
glassbox download-hf-external-data \
  --dataset bench-llm/or-bench \
  --subset or-bench-hard-1k \
  --split train \
  --output data/or_bench_hard1k.normalized.jsonl \
  --default-harmful false \
  --source-name bench-llm/or-bench:or-bench-hard-1k

glassbox download-hf-external-data \
  --dataset bench-llm/or-bench \
  --subset or-bench-toxic \
  --split train \
  --output data/or_bench_toxic.normalized.jsonl \
  --default-harmful true \
  --source-name bench-llm/or-bench:or-bench-toxic

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  glassbox external-evaluate \
  --artifacts artifacts/qwen-expanded-medium-preregistered \
  --inputs data/or_bench_toxic.normalized.jsonl data/or_bench_hard1k.normalized.jsonl \
  --output artifacts/external/or_bench_qwen15b_behavior_eval.json \
  --max-records-per-label 100 \
  --seed 17 \
  --device cuda:0
```

For a full external causal audit, use a dataset with genuine matched harmful/benign pairs and
family/group IDs. Do not invent matches after the fact.
