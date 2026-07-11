# Glassbox: A Held-Out, Baseline-Anchored Causal Audit Protocol

## Abstract

Glassbox audits behavior mechanisms in open language models using held-out causal interventions,
sparse-feature tools, and strong baselines. The strongest supported result is a robust late
residual-stream refusal-relevant direction in Qwen2.5-1.5B, with external OR-Bench transfer and
final-layer separation in six accessible cross-model cells, including Mistral. SAE features are
causally active in several cells but show no monotone scaling advantage over mean/probe baselines.
All twelve developmental cells separate source data, yet none passes the frozen endpoint-transfer
bundle. Head-slice analyses remain too small for a circuit claim, and the sycophancy instantiation
fails baseline separation. Equivalence/inferiority calculations on already examined Qwen outcomes
are reported only as post-hoc sensitivity analyses.

## Method

The controlled corpus uses harmful prompts paired with benign refusal-sensitive prompts. Discovery
and SAE training use train records only. Behavior thresholds and steering scales use validation only.
Headline metrics use held-out test records only.

The audit compares:

- SAE feature ablation and steering;
- mean-difference direction ablation and steering;
- logistic-probe direction ablation and steering;
- random directions;
- architecture-matched random untrained-SAE latent controls;
- layerwise and component-output controls.

## Results

| Evidence block | Result |
|---|---|
| Qwen2.5-1.5B 1,000-pair audit | layer 27; mean harmful-score delta `-0.136`; SAE `-0.036` |
| SAE stability | 6/6 fixed cells completed; 0 H3 passes |
| OR-Bench causal transfer | SAE toxic refusal-rate delta `-0.12`; mean `-0.68` |
| Component/path | residual strong; attention/MLP outputs small or non-specific |
| Scorer robustness | headline ordering survives fixed contrastive threshold variants |
| Leakage audit | controlled 1,000-pair corpus passes formal split-integrity checks |
| Dose response | tiny SAE utility-favorable grid point, but H3 remains failed |
| Phase 6 clean-room rerun | completed; reproduced Phase 4 within fixed tolerances |
| A2 cross-model sweep | 6/11 completed; 5 gated cells published as access failures; all completed cells select the final scanned layer |
| A3 developmental audit | 12/12 completed; all source separations pass; 0/12 endpoint-transfer bundles pass |
| A4 feature budgets | fixed 1–1,000 latent frontier completed; no general SAE reversal |
| A5 head slices | five BH-significant layer-27 slices, all practically tiny |
| A7 sycophancy | baseline behavior gap `-0.00483`, balanced accuracy `0.48`; no mechanism interpretation |
| B2 LoRA drift | frozen mechanism loses specificity; `degraded`, `mechanism_survives: false` |

## Interpretation

Glassbox supports a residual-stream refusal-signal claim, not a confirmed circuit claim. The SAE
result remains mixed and baseline-dependent: effects can be real, and Qwen-0.5B even exceeds the
mean direction on harmful-score suppression, but no monotone scale law or field-level SAE advantage
is supported. Utility flags at Qwen-7B/14B reflect mean-direction specificity costs and do not erase
the SAE's weaker harmful-score suppression. Historical Qwen equivalence tests are post-hoc; the
prospective procedure is frozen only for genuinely unseen SAE-zoo v2 cells.

## Limitations

- No confirmed head-level or path-level mechanism.
- Five A2 Llama/Gemma cells and both A1 public-SAE cells require model-access approval; they are
  failures of access, not empirical failures.
- A8 remains unrun because the approved local shifted corpus is absent.
- The completed sycophancy corpus does not induce a held-out baseline signal in Qwen2.5-1.5B.
- Real-model artifacts use contrastive log-prob scoring, not generated-answer grading.
- Response-level keyword scorer robustness remains limited because the completed real-model audit
  used contrastive scoring and did not enable generation logging.

## Reproducibility

Core release commands:

```bash
make validate
make reproduce-core
make reproduce-external
make report
```

When GPUs are unavailable, `make report` can still regenerate Phase 5 offline statistical, scorer,
leakage, and dose-response reports from committed artifacts.
The A2 and A3 aggregate tables and figures regenerate with `make model-sweep` after model execution
and `make developmental-summary`; static cards, leaderboard, and the 165-cell evidence explorer have
dedicated Make targets.
