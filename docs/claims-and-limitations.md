# Claims And Limitations

## Supported

- Qwen2.5-1.5B has a robust late residual-stream refusal-relevant direction on the controlled paired refusal corpus.
- The direction is causally useful under held-out interventions.
- Qwen2.5-3B provides partial Qwen-family replication, with weaker specificity.
- Frozen controlled-corpus interventions transfer to OR-Bench in behavior-score terms.
- SAE feature interventions beat matched random-SAE latent controls in several artifacts.
- SAE feature interventions do not beat mean-difference/probe baselines under preregistered H3.
- Component/path evidence supports residual-stream localization, not circuit discovery.

## Unsupported

- A completed circuit-level mechanism.
- SAE superiority over simple directions.
- Completed Gemma or Llama replication.
- External validation as proof of a circuit.
- Generated-answer refusal judging for the final real-model artifacts.

## Claim Ladder

| Level | Meaning | Release status |
|---|---|---|
| Correlational | feature/direction separates harmful from benign prompts | supported |
| Steerable | adding a direction changes refusal score | supported |
| Causal contribution | ablation/patching changes held-out behavior | supported for residual directions and weaker SAE effects |
| Specific contribution | effect beats random controls without comparable benign/capability cost | mixed |
| SAE beats simple baselines | SAE beats mean/probe under preregistered held-out criteria | failed |
| Confirmed circuit | path/component mechanism with necessity, sufficiency, specificity, and replication | failed |
