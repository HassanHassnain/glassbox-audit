from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = {
    "schema_version": str,
    "name": str,
    "hypotheses": (dict, list),
    "run_order": list,
    "success_criteria": dict,
    "stopping_rules": list,
}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validate_jsonable(value: Any, path: str) -> list[str]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return []
    if isinstance(value, list):
        errors = []
        for index, item in enumerate(value):
            errors.extend(_validate_jsonable(item, f"{path}[{index}]"))
        return errors
    if isinstance(value, dict):
        errors = []
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(f"{path} contains non-string key {key!r}")
            errors.extend(_validate_jsonable(item, f"{path}.{key}"))
        return errors
    return [f"{path} has unsupported type {type(value).__name__}"]


def validate_preregistration(payload: dict[str, Any]) -> list[str]:
    errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in payload:
            errors.append(f"missing required field: {field}")
            continue
        if not isinstance(payload[field], expected_type):
            errors.append(f"{field} must be {expected_type}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    for field in ["hypotheses", "run_order", "success_criteria", "stopping_rules"]:
        if field in payload and not payload[field]:
            errors.append(f"{field} must not be empty")
    if "thresholds" in payload and not isinstance(payload["thresholds"], dict):
        errors.append("thresholds must be a mapping when present")
    errors.extend(_validate_jsonable(payload, "$"))
    return errors


def load_preregistration_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Preregistration YAML must contain a top-level mapping")
    errors = validate_preregistration(payload)
    if errors:
        raise ValueError("Invalid preregistration: " + "; ".join(errors))
    return payload


def preregistration_hash(payload: dict[str, Any]) -> str:
    errors = validate_preregistration(payload)
    if errors:
        raise ValueError("Invalid preregistration: " + "; ".join(errors))
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def freeze_preregistration(
    source_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(source_path)
    payload = load_preregistration_yaml(source_path)
    digest = preregistration_hash(payload)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_hash": digest,
        "source": str(source_path),
        "preregistration": payload,
    }
    if output_path is not None:
        _write_json(output_path, frozen)
    return frozen


def load_frozen_preregistration(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        frozen = json.load(handle)
    if not isinstance(frozen, dict):
        raise ValueError("Frozen preregistration must be a JSON object")
    if "preregistration" not in frozen or "preregistration_hash" not in frozen:
        raise ValueError("Frozen preregistration must contain preregistration and preregistration_hash")
    payload = frozen["preregistration"]
    if not isinstance(payload, dict):
        raise ValueError("Frozen preregistration payload must be an object")
    digest = preregistration_hash(payload)
    if digest != frozen["preregistration_hash"]:
        raise ValueError(
            "Frozen preregistration hash mismatch: "
            f"stored {frozen['preregistration_hash']}, computed {digest}"
        )
    return frozen


def preregistration_metadata(
    path: str | Path | None,
    expected_hash: str | None,
) -> dict[str, Any] | None:
    if path is None:
        if expected_hash is not None:
            raise ValueError("preregistration_hash requires preregistration_path")
        return None
    frozen = load_frozen_preregistration(path)
    digest = str(frozen["preregistration_hash"])
    if expected_hash is not None and expected_hash != digest:
        raise ValueError(
            "Configured preregistration_hash does not match frozen criteria: "
            f"configured {expected_hash}, frozen {digest}"
        )
    payload = frozen["preregistration"]
    return {
        "schema_version": str(frozen.get("schema_version", SCHEMA_VERSION)),
        "hash": digest,
        "path": str(path),
        "name": str(payload.get("name", "")),
    }
