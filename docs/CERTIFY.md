# B1 Certification Cards

`glassbox certify` emits an Intervention Audit Card from either a completed artifact or a config-backed
audit run.

```bash
python -m glassbox_audit.cli certify \
  --artifacts artifacts/qwen-real-audit \
  --output-json results/certification/qwen-real-audit.card.json \
  --output-html results/certification/qwen-real-audit.card.html \
  --output-pdf results/certification/qwen-real-audit.card.pdf \
  --sign \
  --git-ref v0.1.0
```

For a fresh run, pass the normal audit config and an artifact destination:

```bash
python -m glassbox_audit.cli certify \
  --config configs/qwen2.5-1.5b-real-audit.yaml \
  --output-artifacts artifacts/certify/qwen2.5-1.5b-real-audit \
  --candidate-intervention candidate.json \
  --output-json results/certification/qwen2.5-1.5b-real-audit.card.json \
  --output-html results/certification/qwen2.5-1.5b-real-audit.card.html \
  --output-pdf results/certification/qwen2.5-1.5b-real-audit.card.pdf \
  --sign
```

The card includes:

- schema version and card type;
- model identity hash, behavior, evidence class, dataset summary, corpus hash, and preregistration hash;
- train/validation/test protocol counts;
- rung verdicts for separation, held-out causal effect, specificity, capability cost, matched controls, baseline superiority, equivalence/inferiority tests, and optional transfer;
- hashes for source evidence files and optional external transfer artifacts;
- candidate intervention metadata and referenced-file hashes;
- rendered HTML and PDF summaries;
- optional SHA-256 card digest.

## Candidate Intervention Spec

`--candidate-intervention` accepts a JSON object with one of these `type` values:

- `artifact_method`
- `direction_pt`
- `steering_vector_pt`
- `sae_latents`
- `callable`

Examples:

```json
{"type": "artifact_method", "method": "mean_difference_direction_ablation"}
```

```json
{
  "type": "direction_pt",
  "path": "directions/refusal_mean_direction.pt",
  "layer": 27,
  "mode": "ablate",
  "evidence_method": "mean_difference_direction_ablation"
}
```

```json
{
  "type": "sae_latents",
  "sae_path": "sae.pt",
  "feature_indices": [1509, 1129, 524],
  "evidence_method": "sae_feature_ablation"
}
```

`certify` does not import callable specs or load tensor files while rendering a card. It records the
declared intervention interface, hashes referenced files, and certifies the matching evidence row in
the frozen artifact. Heavy model execution stays in the normal audit pipeline.

External causal-transfer artifacts can be included with `--external-artifacts path/to/transfer.json`.
When supplied, transfer becomes part of the required overall verdict.
