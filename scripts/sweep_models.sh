#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-configs/a2_model_sweep}"
OUTPUT_ROOT="${2:-artifacts/a2_model_sweep}"

mkdir -p "$OUTPUT_ROOT"

for config in "$CONFIG_DIR"/*.yaml; do
  [ -e "$config" ] || {
    echo "No configs found in $CONFIG_DIR" >&2
    exit 1
  }
  slug="$(basename "$config" .yaml)"
  python -m glassbox_audit.cli run \
    --config "$config" \
    --output "$OUTPUT_ROOT/$slug"
done

python -m glassbox_audit.cli model-sweep summarize \
  --configs-dir "$CONFIG_DIR" \
  --artifacts-root "$OUTPUT_ROOT" \
  --output-json results/model-sweep/a2_summary.json \
  --output-table docs/tables/model_sweep.md \
  --output-figure docs/figures/model_sweep.png
