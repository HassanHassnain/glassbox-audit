# A8 Adversarial Mediation Robustness

A8 tests whether the frozen refusal direction remains an aggregate mediator under controlled
distribution shift. It evaluates a published mechanism; it does not build, optimize, rank, or
publish attacks.

## Safety boundary

The shifted corpus is local-only at `data/adversarial_robustness.local.jsonl` and is ignored by Git.
The runner refuses to emit prompts, generations, record identifiers, attack templates, or per-record
scores. Committed evidence contains only counts, corpus hashes, aggregate deltas, pair-bootstrap
confidence intervals, effect retention, matched-control ranges, and pass/fail verdicts.

Do not add the shifted source corpus to the repository. Review it through the project's approved
data process before running A8.

## Local corpus contract

Each JSONL row uses the standard `PromptRecord` fields plus `shift_family`. All rows must use the
held-out `test` split. Both members of a pair must have the same shift family, and every family must
contain both labels and at least two complete pairs.

Allowed shift-family values are fixed and ordered:

1. `paraphrase`
2. `persona`
3. `multilingual`
4. `jailbreak_format`

The repository intentionally provides no prompt examples or transformation recipes for these
families.

## Frozen protocol

Criteria are frozen in `docs/preregistration_robustness.frozen.json`. The runner loads the frozen
mean-difference direction, target layer, behavior threshold, and controlled-corpus effect from
`artifacts/qwen-real-audit`. It then:

1. evaluates baseline scores across the complete shifted corpus once;
2. ablates the frozen direction across the complete corpus once;
3. evaluates 24 orthogonal random-direction controls once each;
4. aggregates those in-memory outputs independently for each shift family;
5. applies pair-clustered bootstrap intervals and the preregistered criteria;
6. checks the output recursively against the aggregate-only key blacklist before writing JSON.

This combined evaluation avoids rerunning the same intervention separately for all four families.

## Run

After the local corpus has been approved and the selected GPU has enough free memory:

```bash
python -m glassbox_audit.cli robustness-audit run \
  --config configs/adversarial_robustness.yaml \
  --output-dir results/robustness \
  --device cuda:0
```

Publish completed and unrun cells with:

```bash
make robustness-summary
```

The public summary is `results/robustness/a8_public_summary.json`; the table is
`docs/tables/robustness_audit.md`. A figure is generated only when completed aggregate evidence is
available.

## Interpretation

A family passes only if the harmful-score effect keeps the controlled-source sign, clears the
held-out CI threshold, retains at least half the controlled effect magnitude, remains specific on
benign and capability metrics, and beats the matched random-direction controls. A failure means the
localization claim did not survive that shift under the frozen test; it is not evidence that a
particular prompt bypasses a deployed system.
