# Extending Glassbox

## Add a Model Backend

Implement `AuditModel.collect` and `AuditModel.evaluate` in `glassbox_audit/modeling.py`. The
collector must return one `[hidden_size]` vector per prompt and layer. Evaluation must apply the
provided intervention at the same activation site used for discovery.

## Add a Behavior

Create a matched dataset and replace the behavior contrast in the model adapter. Keep scoring
continuous where possible, validate it against human or model grading, and define a distinct
capability metric.

## Add an Interpretability Method

Return one or more activation-space directions plus any method-specific ablation callable. Register
them in `pipeline.py` so they use the same held-out evaluation and control protocol.

Natural extensions include transcoders, cross-layer transcoders, attention-head patching,
distributed alignment search, and feature-labeling pipelines.

## Scale the Workbench

The current Streamlit app reads JSON artifacts. For millions of feature examples, store activation
examples in Parquet or DuckDB and add lazy queries rather than expanding `features.json`.
