# Backend Validation

Qwen2.5-0.5B, 240 controlled records, held-out residual audit.

| Backend | Layer | Baseline gap | Mean ablation delta | SAE ablation delta | |Mean diff vs HF| |
|---|---:|---:|---:|---:|---:|
| `huggingface` | 21 | 0.101519 | -0.101000 | -0.080854 | 0.000000 |
| `nnsight` | 21 | 0.101519 | -0.101000 | -0.080854 | 0.000000 |
| `transformer-lens` | 21 | 0.102111 | -0.101781 | -0.064033 | 0.000781 |

Hugging Face and nnsight are numerically identical. TransformerLens selects the same layer and closely matches baseline and mean-direction effects; its SAE fit differs because bridge activations are not bit-identical.
