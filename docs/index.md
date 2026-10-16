# Glassbox

Glassbox is a held-out causal audit harness for activation-level interpretability claims. It tests
candidate mechanisms against simple baselines, matched controls, specificity checks, capability
costs, and frozen success criteria.

The central question is deliberately narrow:

> Does this candidate intervention survive a preregistered causal test on records that were not
> used to discover or tune it?

[Run your first audit](tutorials/behavior-audit-30-minutes.md){ .md-button .md-button--primary }
[Read the methodology](METHODOLOGY.md){ .md-button }

## What the harness enforces

1. Discovery uses the training split only.
2. Thresholds and intervention scales are selected on validation data.
3. Headline effects and confidence intervals come from the held-out test split.
4. Sparse methods face mean-direction, probe, random-direction, and matched random-SAE controls.
5. Evidence files carry stable names, schema versions, hashes, and explicit evidence classes.

## What ships with the repository

| Surface | Purpose |
|---|---|
| `glassbox run` | Execute discovery, intervention, controls, and held-out evaluation. |
| `glassbox prereg freeze` | Freeze criteria and embed a tamper-evident hash in later evidence. |
| `glassbox certify` | Render a signed intervention audit card as JSON, HTML, and PDF. |
| `glassbox drift` | Re-test a frozen mechanism after a model or artifact update. |
| Evidence explorer | Trace claim-ladder verdicts back to cells, intervals, and controls. |

The committed toy backend has a planted mechanism and runs the same pipeline on CPU. Its artifacts
are labeled `synthetic_fixture`; they test the harness, not a language-model claim.

## Current scientific result

The main Qwen audit supports a robust late residual-stream refusal direction. Six accessible A2
models, including Mistral, complete at 250 held-out pairs and all select the final scanned layer;
five gated cells remain explicit access failures. All twelve developmental cells complete, but none
passes the frozen endpoint-transfer bundle. The tested sycophancy corpus fails baseline separation,
and no full circuit is confirmed. Equivalence tests on already examined Qwen outcomes are labeled
post-hoc sensitivity analyses. Glassbox keeps every negative, failed, and unrun cell because
rejecting an intervention is part of the product.
