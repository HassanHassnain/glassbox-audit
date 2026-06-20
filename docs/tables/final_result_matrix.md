| Evidence block | Status | Key result | Claim |
|---|---|---|---|
| Bounded Qwen2.5-1.5B audit | completed | layer 27; mean Δ -0.161; SAE Δ -0.020 | early held-out real-model signal; SAE weaker than mean |
| 520-pair Qwen2.5-1.5B | completed | mean Δ -0.141; SAE Δ -0.035 | preregistered medium audit replicated late residual signal |
| 1,000-pair Qwen2.5-1.5B | completed | mean Δ -0.136; SAE Δ -0.036; layer 27 | H1 residual robustness passed; SAE still weaker than mean direct |
| SAE stability full grid | completed | 6/6 cells, H3 pass cells 0; SAE Δ range -0.061 to -0.016 | H2 failed |
| External behavior-only | completed | OR-Bench toxic refusal 95-96%; hard-benign 81-82% | over-refusal warning; no causal claim |
| OR-Bench external causal | completed | SAE toxic refusal-rate Δ -0.12; mean Δ -0.68 | Transfer present; SAE not mean-superior |
| Component/path | completed | residual layer-27 strong; attention/MLP small or non-specific | no confirmed circuit |
| Cross-model | partial | Qwen2.5-3B completed earlier; no non-Qwen cached weights | partial Qwen-only replication |
| Scorer robustness | completed | fixed threshold variants preserve method ordering; keyword response scorer unavailable | scorer threshold not driving headline |
| Leakage audit | completed | no exact/normalized/family/pair split leakage; no external exact overlap | controlled corpus passes formal checks |
| Dose response | completed | tiny SAE utility-favorable grid point; H3 still failed | descriptive specificity nuance, not SAE superiority |
| Phase 5 clean-room rerun | attempted incomplete | interrupted before artifact writeout during random-control evaluation | preserved as incomplete; no claim update |
| Phase 6 clean-room rerun | completed | layer 27; mean Δ -0.135550; SAE Δ -0.036339; matched random-SAE p=0.0303 | reproduced Phase 4 within fixed tolerance; claim unchanged |
