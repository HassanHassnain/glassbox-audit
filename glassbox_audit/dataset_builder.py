from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _split_for(pair_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()[:8], 16) % 10
    return "train" if value < 6 else "validation" if value < 8 else "test"


def build_paired_dataset(source: str | Path, destination: str | Path, seed: int = 17) -> int:
    """Convert neutral source rows into the paired Glassbox JSONL schema.

    Expected source fields are harmful_prompt, benign_prompt, and category. pair_id is optional.
    This function never generates a benign counterpart automatically because doing so can silently
    create label leakage or semantically unmatched pairs.
    """

    output = []
    with Path(source).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = str(row.get("pair_id", f"pair-{index:06d}"))
            split = str(row.get("split", _split_for(pair_id, seed)))
            category = str(row["category"])
            for suffix, harmful, prompt_field in [
                ("h", True, "harmful_prompt"),
                ("b", False, "benign_prompt"),
            ]:
                output.append(
                    {
                        "id": f"{pair_id}-{suffix}",
                        "prompt": str(row[prompt_field]),
                        "harmful": harmful,
                        "split": split,
                        "category": category,
                        "pair_id": pair_id,
                    }
                )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(output)
