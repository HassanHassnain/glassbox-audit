# A5 Head Path Analysis

`glassbox head-path` adds a head-level verification layer on top of a frozen audit artifact.

```bash
python -m glassbox_audit.cli head-path \
  --artifacts artifacts/qwen-real-audit \
  --layers 27 \
  --top-k 20 \
  --device cuda:0 \
  --output-name head_path_analysis.json \
  --output-table docs/tables/head_path.md
```

## What It Does

1. Loads the frozen artifact direction, threshold, records, and baseline outputs.
2. Collects train attention-output head slices at the configured layer.
3. Shortlists heads by train projection effect along the mean-difference refusal direction.
4. Verifies the shortlist on held-out test records using:
   - head-slice direction ablation;
   - head-slice projection patching to the train benign mean.
5. Applies Benjamini-Hochberg correction over verified head-ablation tests.
6. Writes `head_path_analysis.json`, `results/head-path/*_summary.json`, and a Markdown table.

## Scope

This is generic attention-output head-slice evidence. It is useful for asking whether any attention
output slice carries the refusal direction under the same held-out protocol. It is not a
model-specific pre-output attention-head path patcher, and it does not by itself establish a
confirmed circuit claim.
