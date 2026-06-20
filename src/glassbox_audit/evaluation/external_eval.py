from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..data.external_data import ExternalRefusalRecord, load_external_refusal_records
from ..models import load_model
from ..types import ModelOutput, PromptRecord


def _prompt_record(record: ExternalRefusalRecord, index: int) -> PromptRecord:
    if record.harmful is None:
        raise ValueError("External behavior evaluation requires harmful labels")
    source_slug = record.source.replace("/", "_").replace(":", "_")
    record_id = f"external-{source_slug}-{index:06d}"
    return PromptRecord(
        id=record_id,
        prompt=record.prompt,
        harmful=record.harmful,
        split=record.split or "external",
        category=record.category,
        pair_id=record.pair_id or record_id,
        family_id=record.family_id or record.pair_id or record_id,
        source=record.source,
    )


def load_external_eval_records(paths: list[str | Path]) -> list[PromptRecord]:
    prompt_records: list[PromptRecord] = []
    for path in paths:
        external = load_external_refusal_records(path)
        offset = len(prompt_records)
        prompt_records.extend(_prompt_record(record, offset + index) for index, record in enumerate(external))
    ids = [record.id for record in prompt_records]
    if len(ids) != len(set(ids)):
        raise ValueError("External behavior evaluation record IDs must be unique")
    return prompt_records


def bounded_external_sample(
    records: list[PromptRecord],
    *,
    max_records_per_label: int | None,
    seed: int,
) -> list[PromptRecord]:
    if max_records_per_label is None:
        return records
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    for harmful in [True, False]:
        group = [record for record in records if record.harmful is harmful]
        rng.shuffle(group)
        selected.extend(group[:max_records_per_label])
    selected.sort(key=lambda record: record.id)
    return selected


def summarize_external_outputs(
    records: list[PromptRecord],
    outputs: list[ModelOutput],
    threshold: float,
) -> dict[str, Any]:
    output_by_id = {output.record_id: output for output in outputs}

    def summarize(group: list[PromptRecord]) -> dict[str, Any]:
        if not group:
            return {
                "n": 0,
                "refusal_rate": None,
                "mean_behavior_score": None,
                "mean_capability_nll": None,
            }
        scores = [output_by_id[record.id].behavior_score for record in group]
        nlls = [output_by_id[record.id].capability_nll for record in group]
        return {
            "n": len(group),
            "refusal_rate": sum(score >= threshold for score in scores) / len(scores),
            "mean_behavior_score": sum(scores) / len(scores),
            "mean_capability_nll": sum(nlls) / len(nlls),
        }

    by_label = {
        "harmful": summarize([record for record in records if record.harmful]),
        "benign": summarize([record for record in records if not record.harmful]),
    }
    by_source = {}
    for source in sorted({record.source for record in records}):
        by_source[source] = summarize([record for record in records if record.source == source])
    by_category: dict[str, dict[str, Any]] = {}
    category_groups: dict[tuple[str, bool], list[PromptRecord]] = defaultdict(list)
    for record in records:
        category_groups[(record.category, record.harmful)].append(record)
    for category in sorted({record.category for record in records}):
        by_category[category] = {
            "harmful": summarize(category_groups[(category, True)]),
            "benign": summarize(category_groups[(category, False)]),
        }
    behavior_gap = None
    if by_label["harmful"]["mean_behavior_score"] is not None and by_label["benign"][
        "mean_behavior_score"
    ] is not None:
        behavior_gap = (
            by_label["harmful"]["mean_behavior_score"] - by_label["benign"]["mean_behavior_score"]
        )
    return {
        "threshold": threshold,
        "n_records": len(records),
        "by_label": by_label,
        "behavior_gap": behavior_gap,
        "by_source": by_source,
        "by_category": by_category,
    }


def run_external_behavior_eval(
    *,
    artifact_dir: str | Path,
    normalized_paths: list[str | Path],
    output_path: str | Path,
    max_records_per_label: int | None = None,
    seed: int = 17,
    device: str | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    summary = json.loads((artifact_dir / "summary.json").read_text())
    model_config = ModelConfig(**manifest["config"]["model"])
    if device:
        model_config.device = device
    threshold = float(summary["behavior_threshold"])
    records = bounded_external_sample(
        load_external_eval_records(normalized_paths),
        max_records_per_label=max_records_per_label,
        seed=seed,
    )
    model = load_model(model_config, seed=seed)
    outputs = model.evaluate(records)
    metrics = summarize_external_outputs(records, outputs, threshold)
    artifact = {
        "artifact_notice": (
            "Behavior-only external validation. These rows are not paired, so this artifact does "
            "not support SAE/intervention or causal claims."
        ),
        "source_artifact": str(artifact_dir),
        "model": model.name,
        "threshold_source": str(artifact_dir / "summary.json"),
        "normalized_inputs": [str(path) for path in normalized_paths],
        "max_records_per_label": max_records_per_label,
        "seed": seed,
        "metrics": metrics,
        "records": [
            {
                "id": record.id,
                "source": record.source,
                "category": record.category,
                "harmful": record.harmful,
                "behavior_score": output.behavior_score,
                "refused": output.behavior_score >= threshold,
                "capability_nll": output.capability_nll,
            }
            for record, output in zip(records, outputs, strict=True)
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return artifact
