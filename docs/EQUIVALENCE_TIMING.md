# Equivalence Analysis Timing

Equivalence and inferiority tests are interpreted according to when their bounds were fixed, not
merely whether the code is deterministic.

## Existing Qwen Results

The Qwen2.5-1.5B outcomes were examined before A6 equivalence and inferiority testing was added.
All equivalence outputs derived from those results are therefore post-hoc sensitivity analyses.
They can quantify how conclusions change under stated margins, but they are not confirmatory
preregistered evidence.

The June 19 Phase 4 preregistration remains unchanged:

- `docs/PHASE4_PREREGISTERED_VALIDATION.md`: SHA-256
  `29d7d05dceb6a5a297b61535ddfc960c76fb35bdff6c896190736fe431b96b26`
- `results/final/phase4_preregistration.json`: SHA-256
  `58e83dab9c503fa264c64d09fae1a75693eadc856e6a02c22de4733783148b2b`

Neither file is amended to imply that A6 was part of the June 19 plan.

## Prospective Cells

A result may use `analysis_design: prospective_preregistered` only when its equivalence bounds and
power assumptions were frozen before that cell's first empirical outcome and the artifact carries
the matching `equivalence_preregistration_hash`.

The SAE-zoo v2 freeze satisfies that timing requirement for its still-unrun model cells:

- file: `docs/preregistration_sae_zoo_v2.frozen.json`
- hash: `0053eb4e7c10bbf21bf6d3a975c9057e9f35dfefbb4d142523a2c69d7937851f`
- prior information at freeze time: model-access and model/SAE alignment failures only; no SAE-zoo
  held-out model outcome existed

The original SAE-zoo freeze is retained rather than rewritten. The v2 file explicitly records why
it supersedes v1.

## Enforcement

- Existing artifacts default to `post_hoc_sensitivity`.
- Prospective mode fails without an equivalence preregistration hash.
- The historical migration script can only mark results post-hoc; it cannot assign prospective
  status.
- Reports, audit cards, and public tables surface the timing label.
