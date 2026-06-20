#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PYRELEASE'
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
errors: list[str] = []
scanned_roots = ["README.md", "docs", "scripts", "configs", "src", "tests", "data", "results"]
skip_parts = {"__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
private_terms = ["/" + "Users/", "/" + "home/", "/" + "private" + "/var", r"[A-Za-z]:\\Users\\"]
private_path = re.compile("|".join(re.escape(term) if not term.startswith("[") else term for term in private_terms))
banned = [
    "the " + "refusal circuit",
    "sae discovered " + "the refusal mechanism",
    "sae beats " + "baselines",
    "external validation confirms " + "the circuit",
    "gemma/llama replication " + "completed",
    "gemma replication " + "completed",
    "llama replication " + "completed",
    "syco" + "phancy",
    "sand" + "bagging",
]
phase_name = re.compile(r"phase[-_ ]?\d")
for item in scanned_roots:
    path = root / item
    if not path.exists():
        errors.append(f"missing required path: {item}")
        continue
    candidates = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    for file_path in candidates:
        rel = file_path.relative_to(root)
        if any(part in skip_parts for part in rel.parts):
            continue
        rel_text = str(rel).lower()
        if rel.parts and rel.parts[0] in {"results", "docs"} and phase_name.search(rel_text):
            errors.append(f"phase-numbered public filename: {rel}")
        if file_path.stat().st_size > 5_000_000 and "results" not in rel.parts:
            errors.append(f"unexpected large public file: {rel}")
        if file_path.suffix.lower() in {".md", ".py", ".sh", ".yaml", ".yml", ".toml"}:
            text = file_path.read_text(errors="ignore")
            if private_path.search(text):
                errors.append(f"private absolute path found: {rel}")
            lowered = text.lower()
            for phrase in banned:
                if phrase in lowered:
                    errors.append(f"banned claim phrase {phrase!r} in {rel}")
checkpoint_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".npy", ".npz"}
try:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
except Exception:
    tracked = []
for name in tracked:
    p = Path(name)
    if any(part in skip_parts for part in p.parts):
        errors.append(f"tracked cache/build path: {name}")
    if p.parts[:1] == ("artifacts",) and name != "artifacts/.gitkeep":
        errors.append(f"tracked generated artifact: {name}")
    if p.suffix.lower() in checkpoint_suffixes:
        errors.append(f"tracked tensor/checkpoint file: {name}")
for rel in [
    "results/final/claim_summary.json",
    "results/final/qwen2_5_1_5b_audit.json",
    "results/final/reproducibility.json",
    "results/final/statistical_tests.json",
    "results/sae-stability/stability_grid.json",
    "results/external-causal/or_bench_qwen15b_1000_summary.json",
    "results/component-path/component_path_summary.json",
    "results/cross-model/qwen2_5_3b_replication.json",
]:
    if not (root / rel).exists():
        errors.append(f"missing public result summary: {rel}")
if errors:
    for error in errors:
        print(f"release-check: {error}", file=sys.stderr)
    sys.exit(1)
print("release-check: passed")
PYRELEASE
