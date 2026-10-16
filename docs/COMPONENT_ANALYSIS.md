# Component Analysis

Glassbox now includes a bounded component-output localization mode:

```bash
CUDA_VISIBLE_DEVICES=1 GLASSBOX_MODEL_DEVICE=cuda:0 \
  glassbox component-localize \
  --artifacts artifacts/qwen-real-audit \
  --components residual,attention,mlp \
  --direction mean_difference \
  --device cuda:0
```

The completed Qwen run used the frozen layer 27, frozen train-derived mean-difference direction,
frozen validation-selected behavior threshold, and the 60-record held-out test split. It did not
retune discovery features, layers, thresholds, or scales.

## Result

See [component_localization.json](../results/qwen-real-audit/component_localization.json) and
[the table](tables/component_localization.md).

The residual-stream intervention reproduces the strong held-out mean-difference ablation:
`-0.161 [-0.179, -0.139]` harmful-score delta. Attention-output and MLP-output ablations at the
same layer are much smaller: roughly `-0.001` each.

## Interpretation

This strengthens the original layer-localized result by testing submodule outputs, but it does not
establish a complete circuit. The current evidence supports:

- a layer-27 residual-stream refusal-relevant direction;
- weak direct evidence that the layer-27 attention or MLP output alone mediates the full effect;
- a candidate mechanism that should be followed up with path patching, head-level hooks, and
  multi-layer mediation tests before using the word "circuit" as a confirmed claim.

## Preregistered Layerwise Component Scans

The next phase also ran frozen layerwise component scans:

- Qwen2.5-1.5B expanded: residual layer 27 remained strongest (`-0.141` harmful-score delta).
  The largest attention/MLP output effects were much smaller and failed the preregistered H2
  specificity rule.
- Qwen2.5-3B: layer-35 MLP intervention was large (`-0.640` harmful-score delta), but also changed
  benign scores by `-0.299` and capability NLL by `+0.313`, so it was not specificity-preserving.

Conclusion: the project has residual-stream and component-output causal evidence, but not
necessity/sufficiency/specificity/path replication evidence strong enough to call this a confirmed
refusal circuit.
