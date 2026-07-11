# A3 Developmental Checkpoint Audit

A3 asks when harmful-versus-benign representation separation appears during pretraining and whether
a direction discovered at checkpoint `t` transfers to the final architecture-matched endpoint.

## Frozen design

The preregistered grid contains twelve cells:

| Family | Source checkpoints | Endpoint | Claim scope |
|---|---:|---|---|
| Pythia-410M | 6, from 2.1B to 299.9B tokens | Pythia-410M `main` | Final-base representation transfer |
| OLMo-7B-0724 | 6, from 4B to 2,729B tokens | OLMo-7B-0724-Instruct-hf | Final-instruct causal transfer |

Pythia has public pretraining checkpoints but no official architecture-matched instruction endpoint.
Its cells therefore do not support a refusal-behavior emergence claim. OLMo supplies both public
pretraining checkpoints and an official instruct endpoint, so those cells test the stronger claim.

The grid is in `configs/developmental_audit.yaml`. Criteria were frozen before any real cell in
`docs/preregistration_developmental.frozen.json`.

## Current Result

All twelve cells completed and every source checkpoint passes representation
separation. No cell passes the complete endpoint-transfer bundle. Pythia's strongest
target effect is `-0.0364` at step 128000 with direction cosine `0.961`, but nominal
control `p=.04` becomes `q=.096` over all twelve cells. OLMo endpoint cosines remain
between `-0.019` and `0.110`, with harmful-score effects between approximately
`-0.0010` and `+0.0011`.

The committed aggregate is `results/developmental/a3_summary.json`; the table and
figure are `docs/tables/developmental_audit.md` and
`docs/figures/developmental_audit.png`.

## Protocol

For each checkpoint, Glassbox:

1. collects the predeclared layers on training pairs;
2. selects the layer and mean-difference direction using training records only;
3. measures frozen source separation on validation pairs;
4. saves the direction tensor, source revision, selected layer, train record IDs, and tensor hash;
5. unloads the source model before loading the final endpoint;
6. selects the endpoint behavior threshold on validation records;
7. ablates the frozen checkpoint direction on held-out test pairs;
8. compares it with the endpoint-native train direction and 24 matched orthogonal random directions;
9. writes pair-bootstrap confidence intervals and an empirical control p-value.

The source models are processed before the endpoint is loaded, so a 7B source and target are never
required in memory together.

## Run

Run one revision first:

```bash
python -m glassbox_audit.cli developmental-audit run \
  --manifest configs/developmental_audit.yaml \
  --family pythia-410m \
  --revision step1000 \
  --output-root artifacts/developmental \
  --source-device cuda:0 \
  --target-device cuda:0
```

Run the complete families with `make developmental-pythia` and `make developmental-olmo`. Recheck
free GPU memory before each family; do not launch if another process has occupied the selected
device.

Publish every completed and unrun cell:

```bash
make developmental-summary
```

This writes `results/developmental/a3_summary.json`, one stable JSON per checkpoint,
`docs/tables/developmental_audit.md`, and a three-panel figure once completed evidence exists.

## Evidence contract

Each completed cell stores:

- `direction.pt`: frozen CPU direction, source identity/revision, layer, record IDs, and digest;
- `source_layer_scan.json`: train-only layer scan;
- `outputs.json`: record-level endpoint outputs for baseline, transferred, native, and controls;
- `evidence.json`: aggregate separation, causal transfer, confidence intervals, controls, hashes,
  claim scope, and pass/fail verdict.

Every failed, null, or infeasible cell remains in the summary. The held-out result never changes the
checkpoint grid, layer candidates, controls, or success thresholds.
