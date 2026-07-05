# Mechanism Drift

`glassbox drift` checks whether a previously accepted intervention still passes after a model
version changes.

## Completed Artifact Comparison

Use this when both model versions already have Glassbox artifacts:

```bash
python -m glassbox_audit.cli drift \
  --source-artifacts artifacts/qwen-real-audit \
  --new-artifacts artifacts/qwen2.5-3b-replication-preregistered \
  --output-json results/drift/qwen15b_to_qwen3b.json \
  --output-doc results/drift/qwen15b_to_qwen3b.md
```

This mode compares the old and new audit-card rungs. It is the right path for cross-model evidence
that was already run and frozen.

## Frozen-Handle Model Swap

Use this when the source artifact's hidden size, target layer, frozen directions, and SAE are
compatible with the new checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m glassbox_audit.cli drift \
  --source-artifacts artifacts/qwen-real-audit \
  --new-config configs/qwen2.5-1.5b-real-audit.yaml \
  --device cuda:0 \
  --output-json results/drift/qwen15b_checkpoint_update.json \
  --output-doc results/drift/qwen15b_checkpoint_update.md
```

This mode reloads the source artifact's records, threshold, selected features, SAE, train-derived
directions, and validation-selected steering scales, then evaluates them on the new model without
rediscovery or retuning.

## Report Fields

The JSON report includes:

- `drift_status`: `stable`, `changed`, `degraded`, or `improved`;
- `mechanism_survives`: whether the new artifact/model passes the required rungs;
- source and new model identities;
- per-rung old/new pass states;
- full new baseline and intervention outputs for model-swap mode.

If the new model has a different hidden size than the frozen directions or SAE, model-swap mode
fails closed with an explicit compatibility error. Use completed-artifact comparison for
architecture-changing upgrades.
