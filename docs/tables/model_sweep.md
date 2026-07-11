# A2 Model Sweep

| Status | Model | Params B | Target layer | Depth frac | Refusal d | SAE delta | SAE-mean gap | Test pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| completed | Qwen/Qwen2.5-0.5B-Instruct | 0.5 | 21 | 1.000 | 3.398 | -0.0750 | -0.0295 | 250 |
| failed | meta-llama/Llama-3.2-1B-Instruct | 1.0 |  |  |  |  |  |  |
| completed | Qwen/Qwen2.5-1.5B-Instruct | 1.5 | 27 | 1.000 | 2.657 | -0.0235 | +0.1120 | 250 |
| failed | google/gemma-2-2b-it | 2.0 |  |  |  |  |  |  |
| completed | Qwen/Qwen2.5-3B-Instruct | 3.0 | 35 | 1.000 | 4.122 | -0.0986 | +0.1256 | 250 |
| failed | meta-llama/Llama-3.2-3B-Instruct | 3.0 |  |  |  |  |  |  |
| completed | Qwen/Qwen2.5-7B-Instruct | 7.0 | 27 | 1.000 | 6.215 | -0.0218 | +0.6527 | 250 |
| completed | mistralai/Mistral-7B-Instruct-v0.3 | 7.0 | 31 | 1.000 | 3.767 | -0.0051 | +0.1728 | 250 |
| failed | meta-llama/Llama-3.1-8B-Instruct | 8.0 |  |  |  |  |  |  |
| failed | google/gemma-2-9b-it | 9.0 |  |  |  |  |  |  |
| completed | Qwen/Qwen2.5-14B-Instruct | 14.0 | 47 | 1.000 | 11.276 | -0.0012 | +0.1977 | 250 |
