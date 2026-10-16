# Post-Roadmap Validation

## Scope

This note records the full-compute and publication pass after A1-A8, B1-B5, and
C1-C4 were implemented. Work remains on the local `roadmap-a6` branch. Nothing was
pushed or published remotely.

## Completed Evidence

- A1: both prospectively frozen 131k SAE-zoo cells are published as access failures.
  The Gemma model is gated; the Llama release is gated and targets the base model.
- A2: 6/11 cells completed with 250 held-out pairs and 24 control draws each.
  Qwen2.5 0.5B, 1.5B, 3B, 7B, and 14B plus Mistral-7B completed. Five
  Llama/Gemma cells remain explicit `access_denied` failures.
- A3: 12/12 developmental cells completed. Every source-separation rung passes;
  0/12 endpoint-transfer bundles pass. The strongest nominal control results become
  `q=.096` over the full family.
- A4: full feature-budget frontiers are present in completed real-model artifacts.
- A5: five layer-27 head slices survive BH correction, with effects from
  `-0.000059` to `-0.000364`; no circuit claim is supported.
- A6: existing Qwen equivalence/inferiority analyses are labeled
  `post_hoc_sensitivity`. Prospective timing is restricted to separately frozen,
  unseen cells.
- A7: the full Qwen sycophancy cell completed. Behavior gap is `-0.00483` and
  balanced accuracy is `0.48`, so causal interpretation is blocked.
- A8: the four frozen aggregate-only rows were regenerated as `unrun`. The approved
  local shifted corpus is absent; no substitute corpus was created.
- B1-B5: audit cards, full LoRA drift, leaderboard, three real backend comparisons,
  packaging, and preregistration-as-code outputs are present. The LoRA drift result
  is `degraded` because specificity fails under frozen handles.
- C1: the evidence explorer now contains 165 source-hashed cells, including A1-A3,
  A7-A8, and B2 aggregates.
- C2-C4: the two-card gallery, 90-second certify demo, and local documentation source
  were refreshed.

## Runtime Findings

- `nvidia-smi` cannot initialize NVML in this environment, but PyTorch CUDA execution
  works when run outside the sandbox.
- GPU availability was checked with `torch.cuda.mem_get_info` before jobs started.
- Qwen2.5-14B completed with Hugging Face `balanced` model sharding across two RTX
  3090 GPUs. This validates model-parallel inference; the pipeline does not implement
  distributed data-parallel SAE training.
- Qwen2.5-14B initially exhausted memory only during whole-dataset SAE metric
  evaluation. Metrics now stream in training-sized batches; the full cell completed
  without reducing samples, budgets, or controls.
- The OLMo developmental family completed on one RTX 3090. Source checkpoints were
  unloaded before the instruct endpoint was loaded.

## Statistical Timing

The June 19 files remain byte-for-byte unchanged:

- `docs/PHASE4_PREREGISTERED_VALIDATION.md`:
  `29d7d05dceb6a5a297b61535ddfc960c76fb35bdff6c896190736fe431b96b26`
- `results/final/phase4_preregistration.json`:
  `58e83dab9c503fa264c64d09fae1a75693eadc856e6a02c22de4733783148b2b`

The prospective SAE-zoo v2 freeze is
`0053eb4e7c10bbf21bf6d3a975c9057e9f35dfefbb4d142523a2c69d7937851f`.

## Publication Outputs

- `results/model-sweep/a2_summary.json`
- `docs/tables/model_sweep.md`
- `docs/figures/model_sweep.png`
- `results/developmental/a3_summary.json`
- `docs/tables/developmental_audit.md`
- `docs/figures/developmental_audit.png`
- `results/drift/qwen15b_lora_model_swap.json`
- `results/evidence_explorer/explorer.json`
- `docs/evidence/index.html`
- `README.md`, `docs/FINAL_PAPER.md`, and `docs/paper.md`

## Validation Policy

The repository-wide suite previously passed 57 tests, Ruff, strict MkDocs, and pip
dependency checks after the equivalence-timing correction. Per the later instruction
not to run tests unless critical for implementation, those routine suites were not
repeated after the final documentation and explorer updates. The required A2/A3
full runs and all publication builders were executed directly.

## Remaining External Actions

- Obtain Hugging Face approval before running the five frozen A2 failures or two A1
  SAE-zoo cells. Do not change their frozen configs or criteria.
- Supply the approved local A8 corpus before running its four cells. Do not synthesize
  a replacement from attack templates.
- Hosted Colab timing, PyPI publication, GitHub Pages deployment, and any remote push
  require an explicit owner decision. None was attempted.
- The branch has unrelated untracked files from the public-main lineage. They were
  intentionally not staged. Do not use `git add -A`.
