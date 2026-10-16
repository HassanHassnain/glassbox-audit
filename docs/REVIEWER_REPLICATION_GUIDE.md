# Reviewer Replication Guide

## Fast Checks

```bash
make validate
```

This runs tests, linting, compilation, a toy audit, and workbench CLI validation.

## Clean-Room Qwen Audit

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  make reproduce-cleanroom
```

Notes:

- Use a nonzero GPU unless GPU 0 is confirmed free.
- PyTorch CUDA availability is the source of truth; NVML/`nvidia-smi` may fail.
- The run writes checkpoints under `artifacts/phase6-cleanroom/qwen15b-1000/checkpoints`.
- A run is complete only when `summary.json`, `interventions.json`, `failure_analysis.json`,
  `method_comparisons.json`, `layer_scan.json`, `layerwise_controls.json`, `sae_metrics.json`,
  `manifest.json`, and `protocol.json` all exist.

## Reports

```bash
make final-report
```

If the Phase 6 clean-room artifact is incomplete, the report command falls back to frozen Phase 4
artifacts and does not claim clean-room reproduction. In this release, the Phase 6 clean-room
artifact is complete and the reproducibility diff is available at
`results/final/phase6_reproducibility_diff.json`.
