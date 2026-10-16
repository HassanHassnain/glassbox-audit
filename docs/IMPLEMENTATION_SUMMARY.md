# Implementation Summary

Glassbox is a backend-neutral held-out causal activation-level audit pipeline for refusal behavior.
Phase 5/6 added adversarial reproducibility and release hardening while preserving the Phase 4
interpretation: no layer, threshold, scale, feature, prompt, seed, or metric was retuned after
seeing held-out results.

## Implemented

- versioned 120-pair, 520-pair, and 1,000-pair controlled refusal corpora with family-disjoint splits;
- external JSONL/JSON/CSV normalization, Hugging Face subset normalization, and behavior-only
  external evaluation for unpaired datasets;
- strict train-only discovery/SAE training, validation-only threshold/scale selection, and held-out
  test reporting;
- token-activation top-k SAE training with activation subsampling and diagnostics;
- layer localization, layerwise causal controls, and component-output localization;
- SAE steering, feature ablation, direction ablation, activation patching, and benign-mean patching;
- mean-difference, logistic-probe, random-direction, and architecture-matched random-untrained-SAE
  controls;
- matched random-SAE latent controls with trained activation centering, zeroed encoder bias, and
  literal encode -> zero selected latent -> decode intervention;
- paired-bootstrap confidence intervals, direct method comparisons, failure analysis, and negative
  result reporting;
- fixed SAE stability grid completed across seeds/widths;
- external causal-transfer evaluation for frozen controlled-corpus interventions on OR-Bench;
- residual projection-patching and component-output path approximation;
- paired sign-flip permutation tests and Benjamini-Hochberg correction for headline effects;
- scorer-threshold robustness checks and manual-review disagreement export;
- formal split/data leakage checks for controlled and external datasets;
- fixed-grid dose-response summaries and figures;
- Phase 5 release docs, claim audit, reviewer Q&A, Makefile release targets, and validation summary;
- Phase 6 interruption-safe checkpoints, stage timings, per-random-control resume files, artifact
  hashes, and clean-room reproducibility diff;
- Streamlit workbench plus paper-style docs, static figures, tables, and machine-readable snapshots.

## Actually Run

All GPU runs used PyTorch CUDA visibility rather than NVML because `nvidia-smi`/NVML was unreliable.
Runs were pinned to `CUDA_VISIBLE_DEVICES=1` as an isolated RTX 3090; the user noted another GPU was
occupied by Tamila and it was not touched.
Phase 4 used physical GPUs 1, 2, and 3 while leaving GPU 0 alone after PyTorch showed lower free
memory there.

Completed earlier:

- toy demo and tests;
- Qwen2.5-1.5B bounded 120-pair real audit;
- matched-center random-SAE control refresh for the bounded audit;
- bounded component-output localization;
- 520-pair and 1,000-pair controlled corpora generated and validated.

Completed preregistered/final hardening:

- 520-pair Qwen2.5-1.5B expanded audit: `artifacts/qwen-expanded-medium-preregistered`;
- layerwise component scan for the 520-pair Qwen2.5-1.5B artifact;
- cached Qwen2.5-3B replication: `artifacts/qwen2.5-3b-replication-preregistered`;
- layerwise component scan for Qwen2.5-3B;
- SAE stability seed23/2x: `artifacts/sae-stability/qwen15b-medium-seed23-exp2`;
- SAE stability seed17/4x: `artifacts/sae-stability/qwen15b-medium-seed17-exp4`;
- OR-Bench hard/toxic normalization from Hugging Face;
- bounded OR-Bench behavior-only evaluation: `artifacts/external/or_bench_qwen15b_behavior_eval.json`.

Completed Phase 4:

- 1,000-pair Qwen2.5-1.5B audit: `artifacts/qwen-expanded-1000-phase4`;
- remaining stability cells seed23/4x, seed42/2x, and seed42/4x;
- full stability summary: `results/sae-stability/phase4_full_grid_summary.json`;
- OR-Bench external causal transfer from 520-pair and 1,000-pair artifacts;
- path/component approximation for 520-pair and 1,000-pair artifacts;
- final outcome snapshot: `results/final/phase4_outcomes.json`.

Completed Phase 5 offline hardening:

- controlled 1,000-pair corpus rebuilt at
  `artifacts/phase5-cleanroom/refusal_controlled_v2_1000.rebuilt.jsonl`;
- paired permutation/BH statistical summary:
  `results/final/statistical_tests.json`;
- scorer robustness summary and disagreement CSV:
  `results/final/scorer_robustness.json`,
  `data/manual_review/refusal_scorer_disagreements.csv`;
- data leakage/split-integrity audit:
  `results/final/data_leakage_audit.json`;
- fixed-grid dose-response summary and figures:
  `results/final/dose_response.json`,
  `docs/figures/dose_response_*.png`;
- release summary and validation summary:
  `results/final/phase5_release_summary.json`,
  `results/final/validation_summary.json`.

Attempted but incomplete in Phase 5:

- clean-room Qwen2.5-1.5B 1,000-pair audit:
  `artifacts/phase5-cleanroom/qwen15b-1000`;
- the run was retried outside the sandbox on `CUDA_VISIBLE_DEVICES=1` after sandbox PyTorch reported
  no CUDA devices;
- the run progressed into random-control evaluation but was interrupted before the runner wrote
  final artifacts, so no clean-room numeric result or reproducibility diff is claimed.

Completed Phase 6:

- clean-room Qwen2.5-1.5B 1,000-pair rerun:
  `artifacts/phase6-cleanroom/qwen15b-1000`;
- all required final artifacts present, including `summary.json`, `interventions.json`,
  `failure_analysis.json`, `method_comparisons.json`, `layer_scan.json`, `layerwise_controls.json`,
  `sae_metrics.json`, `statistical_tests.json`, `manifest.json`, and `protocol.json`;
- 128/128 random-control checkpoints written under
  `artifacts/phase6-cleanroom/qwen15b-1000/checkpoints/random_controls`;
- Phase 6 reproduced Phase 4 within fixed tolerances:
  `results/final/phase6_reproducibility_diff.json`;
- artifact hashes written to `results/final/artifact_hashes.json`.

## Headline Results

Bounded Qwen2.5-1.5B:

- selected layer 27;
- mean-difference ablation harmful-score delta `-0.161 [-0.180, -0.141]`;
- SAE feature-ablation harmful-score delta `-0.020 [-0.023, -0.017]`;
- matched random-SAE latent-control empirical `p=0.024`;
- SAE did not beat simple baselines.

520-pair Qwen2.5-1.5B:

- selected layer 27;
- mean-difference ablation harmful-score delta `-0.141 [-0.155, -0.128]`;
- SAE feature-ablation harmful-score delta `-0.035 [-0.039, -0.032]`;
- SAE beat matched random-SAE controls but failed preregistered H3 against mean/probe.

Qwen2.5-3B:

- selected layer 35;
- mean-difference ablation harmful-score delta `-0.090 [-0.103, -0.076]`;
- mean-difference benign-score delta `+0.390`, capability-NLL delta `+0.164`;
- SAE feature-ablation harmful-score delta `-0.006 [-0.007, -0.005]`;
- specificity and H3 failed.

1,000-pair Qwen2.5-1.5B:

- selected layer 27;
- mean-difference harmful-score delta `-0.136 [-0.143, -0.128]`;
- SAE feature-ablation harmful-score delta `-0.036 [-0.039, -0.034]`;
- residual robustness H1 passed;
- SAE H3 failed under the direct harmful-score criterion.

SAE stability full grid:

- seed17/2x SAE harmful delta `-0.035`;
- seed23/2x SAE harmful delta `-0.061`;
- seed42/2x SAE harmful delta `-0.045`;
- seed17/4x SAE harmful delta `-0.016`;
- seed23/4x SAE harmful delta `-0.033`;
- seed42/4x SAE harmful delta `-0.031`;
- 0/6 completed stability cells passed preregistered H3.

External behavior-only:

- OR-Bench toxic refusal rate 95.0% on 100 sampled prompts;
- OR-Bench hard-benign refusal rate 81.0% on 100 sampled prompts;
- unpaired external data, so no causal/intervention or SAE claim.

External causal transfer:

- preferred 1,000-pair-source OR-Bench artifact completed;
- SAE feature ablation toxic refusal-rate delta `-0.12`, hard-benign refusal-rate delta `-0.09`;
- mean-difference toxic refusal-rate delta `-0.68`, hard-benign refusal-rate delta `-0.55`;
- SAE beat random controls but did not beat mean-difference on direct toxic score suppression.

Phase 5 hardening:

- statistical tests: internal, OR-Bench external causal, and component/path harmful-score deltas
  have paired permutation summaries with BH correction; significance does not change the method
  ordering;
- scorer robustness: fixed contrastive threshold variants preserve the headline ordering; keyword
  response scoring is unavailable on current real-model artifacts because generated responses were
  not stored;
- leakage audit: controlled 1,000-pair corpus passes exact, normalized, family, pair, harmful/benign
  prompt, near-duplicate, and external exact-overlap checks;
- dose response: SAE steering has a small fixed-grid specificity-utility-favorable point, but H3
  remains failed because direct harmful-score suppression is still weaker than mean/probe.

Phase 6 clean-room:

- selected layer `27`, matching Phase 4 exactly;
- baseline harmful refusal `64.0%` and benign over-refusal `14.8%`, matching Phase 4 exactly;
- mean-difference harmful-score delta `-0.135550`, matching Phase 4 exactly;
- SAE feature-ablation harmful-score delta `-0.036339`, matching Phase 4 exactly;
- matched random-SAE empirical p-value `0.0303`, matching Phase 4 exactly;
- H3 direct SAE-vs-mean/probe status remained `False`.

## Failed And Inconclusive Hypotheses

Failed:

- H2 component specificity: no clean attention/MLP component met the preregistered rule;
- H3 SAE causal utility: failed on Qwen2.5-1.5B, Qwen2.5-3B, and every completed stability cell;
- H4 confirmed-circuit/path evidence: failed in Phase 4;
- Qwen2.5-3B specificity: late interventions changed refusal but with large collateral shifts.

Inconclusive/unrun:

- Gemma/Llama configs are prepared but weights were not locally available and no gated download was
  attempted;
- optional 96-token SAE scale configs are configured but unrun;
- no head-level/full path-patching artifact is reported.
- Phase 5 clean-room Qwen rerun is configured and attempted but incomplete; it does not establish a
  new reproducibility result. Phase 6 supersedes this with a completed clean-room rerun.
- Phase 6 response-level keyword scoring is unavailable for the completed real-model artifact
  because generated responses were not enabled; generation logging is implemented as an opt-in
  future-run feature.

## Strongest Supported Claim

Qwen2.5-1.5B has a replicated late residual-stream refusal-relevant direction on the controlled
paired corpus. The project did not confirm a full circuit, and SAE features did not outperform
mean-difference/probe baselines under preregistered held-out criteria.

## Validation

Final Phase 6 validation is recorded in `results/final/phase6_validation_summary.json` and
completed through `make release-check`:

- `python -m pytest`: 26 passed;
- `ruff check .`: passed;
- `python -m compileall -q glassbox_audit tests`: passed;
- `python -m glassbox_audit.cli run --config configs/toy.yaml --output artifacts/demo`: completed;
- `python -m glassbox_audit.cli workbench --help`: completed;
- `bash scripts/31_phase6_final_report.sh`: completed;
- workbench artifact load for `artifacts/demo`,
  `artifacts/phase6-cleanroom/qwen15b-1000`, and `artifacts/qwen-expanded-1000-phase4`: passed;
- final JSON sanity check for `results/final/*.json`: passed after the Phase 6 summary was added;
- all required Phase 6 clean-room artifacts were present.
