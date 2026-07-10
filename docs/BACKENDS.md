# Model backends

Glassbox uses the same held-out discovery, intervention, control, and reporting pipeline for all
model backends. The backend only changes how residual activations are captured and edited.

## Install

```bash
pip install "glassbox-audit[backends,accelerate]"
```

Choose one backend in the experiment YAML:

```yaml
model:
  backend: transformer_lens  # huggingface | transformer_lens | nnsight
  name: Qwen/Qwen2.5-1.5B-Instruct
  revision: null
  layers: [8, 16, 24]
  dtype: float16
  device: cuda:0
```

## Capability matrix

| Operation | `huggingface` | `transformer_lens` | `nnsight` |
|---|---:|---:|---:|
| Residual capture | native forward hooks | native `TransformerBridge` cache | native trace saves |
| Residual intervention | native forward hooks | native `run_with_hooks` | native trace assignment |
| Standard held-out audit | yes | yes | yes |
| SAE training activations | yes | yes | yes |
| Attention/MLP localization | yes | no | no |
| Head-slice analysis | yes | no | no |
| Intervention-time text generation | yes | no | no |

The ecosystem backends fail explicitly for component, head, and intervention-time generation
requests. They do not fall back silently to raw hooks. Normal audit scoring does not generate text,
so the standard residual audit remains available on all three engines.

## Backend notes

`transformer_lens` wraps the already loaded Hugging Face model with the current
`TransformerBridge` API. Glassbox uses `blocks.{layer}.hook_out`, the bridge alias for the
post-block residual stream. This backend supports architectures recognized by the installed
TransformerLens release.

`nnsight` wraps the already loaded model with a local `LanguageModel` tracer. Glassbox saves or
edits the output of each configured decoder block and saves logits from `lm_head`. Remote nnsight
execution is not enabled by the audit runner.

`huggingface` remains the required backend for A5 component and attention-head analysis. It is also
the least restrictive choice for architectures not yet represented in TransformerLens or nnsight.

## Real-model validation

All three engines completed the same bounded held-out audit on Qwen2.5-0.5B and 240 controlled
records. Hugging Face and nnsight were numerically identical. TransformerLens selected the same
layer; its absolute difference from Hugging Face was `0.00059` for baseline behavior gap,
`0.00078` for mean-direction harmful-score delta, and `0.01682` for the retrained SAE delta.

- [Comparison table](tables/backend_validation.md)
- Machine-readable result: `results/backend-validation/qwen25-05b.json`
- Figure: `docs/figures/backend_validation.png`

## Package verification

`glassbox installed-smoke --output <dir>` runs a complete toy audit from data embedded in the
installed wheel. Maintainers can build the sdist and wheel, run `twine check`, import from the built
wheel rather than the source tree, and execute that audit with:

```bash
make package-check
```

Building and checking a distribution does not upload it. Publishing to PyPI remains a separate
maintainer action.
