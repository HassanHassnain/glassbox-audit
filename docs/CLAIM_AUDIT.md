# Claim Audit

This page audits what Glassbox is allowed to claim after Phase 5 release hardening.

## Supported

- Qwen2.5-1.5B has a robust late residual-stream refusal-relevant direction on the controlled
  paired refusal corpus.
- The direction is causally useful under held-out intervention tests.
- Qwen2.5-3B provides partial Qwen-family replication, but with worse specificity.
- Frozen controlled-corpus interventions transfer to OR-Bench in the behavior-score sense.
- SAE feature interventions beat matched random-SAE latent controls in several artifacts.
- SAE feature interventions do not beat mean-difference/probe baselines under preregistered H3.
- Component/path evidence supports residual-stream localization, not a confirmed circuit.

## Failed

- SAE superiority over mean/probe under held-out preregistered criteria.
- Confirmed component/path mechanism.
- Clean specificity in the Qwen2.5-3B replication.
- Response-keyword scorer robustness on current real-model artifacts, because generated responses
  are not stored.

## Inconclusive

- Non-Qwen replication.
- Head-level/path-patching evidence.
- Whether larger SAE training scales would improve the negative SAE-vs-mean conclusion.
- Generation-based refusal judging.

## Unsupported Phrases

Do not write:

- "the refusal circuit";
- "SAE discovered the refusal mechanism";
- "SAE beats baselines";
- "external validation confirms the circuit";
- "Gemma/Llama replication completed".

Preferred wording:

> Glassbox found a robust late residual-stream refusal-relevant direction with partial Qwen-family
> replication and external causal transfer, but did not confirm a full circuit. SAE features beat
> matched random-SAE controls but did not beat mean/probe baselines under preregistered held-out
> criteria.
