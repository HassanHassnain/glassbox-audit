from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    harmful: bool
    split: str
    category: str
    pair_id: str
    family_id: str = ""
    source: str = "unknown"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PromptRecord":
        pair_id = str(row["pair_id"])
        return cls(
            id=str(row["id"]),
            prompt=str(row["prompt"]),
            harmful=bool(row["harmful"]),
            split=str(row["split"]),
            category=str(row["category"]),
            pair_id=pair_id,
            family_id=str(row.get("family_id", pair_id)),
            source=str(row.get("source", "unknown")),
        )


@dataclass(frozen=True)
class ModelOutput:
    record_id: str
    behavior_score: float
    capability_nll: float
    response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "behavior_score": self.behavior_score,
            "capability_nll": self.capability_nll,
            "response": self.response,
        }
