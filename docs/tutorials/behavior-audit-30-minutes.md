# Write a behavior audit in 30 minutes

This tutorial builds a complete CPU audit, freezes its decision rules, validates the paired data,
runs the held-out pipeline, and renders an intervention card. Start with the toy backend so every
step is inspectable before spending GPU time.

Glassbox currently includes scoring adapters for `refusal` and `sycophancy`. Use one of those
behavior names for this walkthrough. A genuinely different behavior also needs a positive and
negative scoring contrast in `HuggingFaceRefusalModel.behavior_prefixes`.

## 0-5 minutes: install and inspect

```bash
git clone git@github.com:HassanHassnain/glassbox-audit.git
cd glassbox-audit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
glassbox --help
```

The fastest starting point is `configs/toy.yaml`. It uses the deterministic CPU model, the committed
paired refusal fixture, four layers, and a small trained SAE.

## 5-10 minutes: prepare paired records

Each JSONL row needs a stable record ID, prompt, binary class label, split, category, and pair ID:

```json
{"id":"tr-01-positive","prompt":"...","harmful":true,"split":"train","category":"policy","pair_id":"tr-01"}
{"id":"tr-01-negative","prompt":"...","harmful":false,"split":"train","category":"policy","pair_id":"tr-01"}
```

For refusal, `harmful: true` is the behavior-positive side and `harmful: false` is the matched benign
side. For sycophancy, it marks the user-pressure side versus the matched neutral side. Keep both rows
of a pair in the same split and keep test prompts out of discovery and threshold selection.

Validate the file before running anything:

```bash
glassbox validate-data --data data/my_behavior_pairs.jsonl
```

If your source has one matched pair per row, normalize it through the repository builder:

```bash
glassbox build-pairs \
  --source data/my_matched_source.jsonl \
  --output data/my_behavior_pairs.jsonl \
  --seed 17
```

## 10-15 minutes: freeze the rules

Copy `docs/preregistration_next_phase.yaml`, then edit the questions, thresholds, metrics, baselines,
run order, and stopping rules before viewing held-out results. Freeze it once:

```bash
glassbox prereg freeze \
  --input docs/my_behavior_prereg.yaml \
  --output docs/my_behavior_prereg.frozen.json
```

The frozen JSON includes a content hash. Put its path and hash in the experiment config; later runs
refuse a criteria mismatch instead of silently accepting changed thresholds.

## 15-20 minutes: write the config

Create `configs/my_behavior_toy.yaml`:

```yaml
seed: 17
mode: standard
behavior: refusal
evidence_class: synthetic_fixture
dataset_path: data/my_behavior_pairs.jsonl
preregistration_path: docs/my_behavior_prereg.frozen.json
preregistration_hash: REPLACE_WITH_FROZEN_HASH
target_layer: null

model:
  backend: toy
  name: glassbox-toy-my-behavior
  layers: [0, 1, 2, 3]
  dtype: float32
  device: cpu

sae:
  source: trained
  expansion_factor: 4
  top_k: 8
  epochs: 120
  batch_size: 64
  learning_rate: 0.003
  l1_coefficient: 0.0001
  n_features_to_intervene: 3
  device: cpu
  tokens_per_prompt: 1

evaluation:
  threshold: null
  bootstrap_samples: 250
  confidence: 0.95
  steering_scales: [-4.0, -2.0, -1.0, 1.0]
  random_controls: 4
  control_scale: -4.0
  layerwise_controls: true
```

Use `threshold: null` to select the behavior threshold from validation data. Keep the toy evidence
class until the config points to a real model and the run satisfies the corresponding empirical
protocol.

## 20-25 minutes: run the audit

```bash
glassbox run \
  --config configs/my_behavior_toy.yaml \
  --output artifacts/my-behavior-toy
```

The output directory contains the resolved config, dataset manifest, layer scan, trained SAE,
candidate directions, intervention evaluations, random controls, and the final summary. A run is not
complete merely because one effect is large; inspect its interval, specificity, capability cost, and
matched controls.

```bash
glassbox analyze --artifacts artifacts/my-behavior-toy
glassbox report --artifacts artifacts/my-behavior-toy
```

The report is written to `artifacts/my-behavior-toy/REPORT.md`.

## 25-30 minutes: certify and review

```bash
glassbox certify \
  --artifacts artifacts/my-behavior-toy \
  --output-json results/certification/my-behavior.card.json \
  --output-html results/certification/my-behavior.card.html \
  --output-pdf results/certification/my-behavior.card.pdf \
  --sign
```

Open the HTML card and review every required rung. A failed card is a valid result: preserve it,
explain which criterion rejected the intervention, and do not tune against the held-out records.

## Move from toy to a real model

Change the backend and model fields only after the toy audit is complete:

```yaml
model:
  backend: huggingface
  name: Qwen/Qwen2.5-1.5B-Instruct
  layers: [7, 15, 23, 27]
  dtype: bfloat16
  device: cuda:0
```

Increase held-out pairs and bootstrap draws to the preregistered values, scale the SAE budget by the
declared samples-per-feature rule, and run every predeclared cell. Record GPU selection and model
revision in the evidence rather than relying on a mutable local cache.

For a new scorer, add the behavior's positive and negative prefixes in `glassbox_audit/modeling.py`,
add paired-data and scorer tests, and run the same toy-to-real sequence. The discovery, validation,
held-out evaluation, controls, card generation, and drift checks remain unchanged.
