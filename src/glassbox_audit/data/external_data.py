from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..types import PromptRecord
from .records import validate_records


@dataclass(frozen=True)
class ExternalRefusalRecord:
    prompt: str
    behavior_label: str
    harmful: bool | None = None
    split: str | None = None
    source: str = "external"
    category: str = "external"
    family_id: str | None = None
    pair_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "harmful", "unsafe", "refusal-relevant"}:
        return True
    if text in {"0", "false", "no", "n", "benign", "safe"}:
        return False
    raise ValueError(f"Cannot parse boolean harmful flag from {value!r}")


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            raw = json.load(handle)
        return raw if isinstance(raw, list) else list(raw["data"])
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported external dataset file type: {path.suffix}")


def load_external_refusal_records(
    path: str | Path,
    *,
    prompt_field: str = "prompt",
    label_field: str = "behavior_label",
    harmful_field: str = "harmful",
    split_field: str = "split",
    source_field: str = "source",
    category_field: str = "category",
    family_field: str = "family_id",
    pair_field: str = "pair_id",
    source_name: str = "external",
    default_harmful: bool | None = None,
) -> list[ExternalRefusalRecord]:
    rows = _read_rows(path)
    records = []
    reserved = {
        prompt_field,
        label_field,
        harmful_field,
        split_field,
        source_field,
        category_field,
        family_field,
        pair_field,
    }
    for index, row in enumerate(rows):
        if prompt_field not in row:
            raise ValueError(f"Missing prompt field {prompt_field!r} in row {index}")
        label = str(row.get(label_field, row.get(category_field, "unknown")))
        records.append(
            ExternalRefusalRecord(
                prompt=str(row[prompt_field]),
                behavior_label=label,
                harmful=(
                    _parse_bool(row.get(harmful_field))
                    if row.get(harmful_field) not in {None, ""}
                    else default_harmful
                ),
                split=str(row[split_field]) if row.get(split_field) not in {None, ""} else None,
                source=str(row.get(source_field) or source_name),
                category=str(row.get(category_field) or label),
                family_id=str(row[family_field]) if row.get(family_field) not in {None, ""} else None,
                pair_id=str(row[pair_field]) if row.get(pair_field) not in {None, ""} else None,
                metadata={key: value for key, value in row.items() if key not in reserved},
            )
        )
    return records


def write_external_normalized(records: list[ExternalRefusalRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def external_to_glassbox_pairs(
    records: list[ExternalRefusalRecord],
    destination: str | Path,
    *,
    default_split: str = "test",
) -> int:
    output = []
    for index, record in enumerate(records):
        if record.harmful is None:
            raise ValueError("Glassbox paired export requires a harmful/benign flag for every record")
        pair_id = record.pair_id or record.family_id or f"external-pair-{index:06d}"
        family_id = record.family_id or pair_id
        split = record.split or default_split
        output.append(
            {
                "id": f"{pair_id}-{'h' if record.harmful else 'b'}",
                "prompt": record.prompt,
                "harmful": record.harmful,
                "split": split,
                "category": record.category,
                "pair_id": pair_id,
                "family_id": family_id,
                "source": record.source,
            }
        )
    prompt_records = [PromptRecord.from_dict(row) for row in output]
    validate_records(prompt_records)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(output)


def load_huggingface_refusal_records(
    dataset_name: str,
    *,
    split: str,
    source_name: str | None = None,
    subset: str | None = None,
    default_harmful: bool | None = None,
    **field_kwargs: Any,
) -> list[ExternalRefusalRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the optional `datasets` package to load Hugging Face datasets") from exc
    rows = list(load_dataset(dataset_name, subset, split=split))
    tmp = Path("/tmp/glassbox_external_hf_rows.jsonl")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return load_external_refusal_records(
        tmp,
        source_name=source_name or dataset_name,
        split_field=field_kwargs.pop("split_field", "__missing_split__"),
        default_harmful=default_harmful,
        **field_kwargs,
    )
