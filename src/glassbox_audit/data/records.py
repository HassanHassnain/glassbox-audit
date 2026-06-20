from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ..types import PromptRecord


def load_records(path: str | Path) -> list[PromptRecord]:
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(PromptRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
    validate_records(records)
    return records


def validate_records(records: list[PromptRecord]) -> None:
    if not records:
        raise ValueError("Dataset is empty")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset record IDs must be unique")
    valid_splits = {"train", "validation", "test"}
    unknown = {record.split for record in records} - valid_splits
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")

    pair_labels: dict[tuple[str, str], set[bool]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        pair_labels[(record.split, record.pair_id)].add(record.harmful)
        family_splits[record.family_id].add(record.split)
    incomplete = [key for key, labels in pair_labels.items() if labels != {False, True}]
    if incomplete:
        raise ValueError(f"Each pair must contain harmful and benign prompts: {incomplete[:5]}")
    leaking = [family_id for family_id, splits in family_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"Prompt families must not cross splits: {leaking[:5]}")


def split_records(records: list[PromptRecord], split: str) -> list[PromptRecord]:
    return [record for record in records if record.split == split]


def dataset_summary(records: list[PromptRecord]) -> dict[str, object]:
    return {
        "n_records": len(records),
        "splits": dict(Counter(record.split for record in records)),
        "labels": {
            "harmful": sum(record.harmful for record in records),
            "benign": sum(not record.harmful for record in records),
        },
        "categories": dict(Counter(record.category for record in records)),
        "n_pairs": len({(record.split, record.pair_id) for record in records}),
        "n_families": len({record.family_id for record in records}),
        "sources": dict(Counter(record.source for record in records)),
        "family_split_leakage": False,
    }
