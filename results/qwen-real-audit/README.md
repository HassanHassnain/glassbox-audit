# Qwen2.5-1.5B Bounded Real-Audit Snapshot

This directory contains the lightweight, machine-readable result snapshot used to generate the
paper-style documentation in `docs/`.

Included:

- full configuration/runtime manifest and held-out protocol;
- layer localization and layerwise causal controls;
- SAE diagnostics and ranked feature statistics;
- all intervention sweeps, causal tests, random controls, and prompt-level scores;
- direct paired method comparisons and generated failure analysis.

Excluded:

- `sae.pt` and `discovery_tensors.pt`, which are generated locally and omitted from Git because they
  contain large model-derived tensors.

Evidence class: `empirical_bounded_real_audit`. This is a controlled, family-disjoint held-out audit,
not a publication-grade multi-seed or naturalistic-benchmark replication.

The random untrained-SAE latent controls were refreshed with train-matched activation centering,
zeroed encoder bias, and literal encode -> zero selected latent -> decode ablation. No feature,
layer, threshold, or intervention scale was retuned for this refresh.
