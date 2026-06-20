from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    if not values:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(statistic(sample))
    alpha = (1 - confidence) / 2
    return {
        "estimate": statistic(values),
        "low": percentile(draws, alpha),
        "high": percentile(draws, 1 - alpha),
    }


def paired_delta_ci(
    before: Sequence[float],
    after: Sequence[float],
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    if len(before) != len(after):
        raise ValueError("Paired samples must have equal length")
    return bootstrap_ci(
        [right - left for left, right in zip(before, after, strict=True)],
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def paired_permutation_test(
    before: Sequence[float],
    after: Sequence[float],
    *,
    alternative: str = "two-sided",
    samples: int = 10000,
    seed: int = 0,
) -> dict[str, float | str | int]:
    """Sign-flip paired permutation test for a mean paired delta.

    The observed statistic is mean(after - before). For a harmful-score suppression test,
    use ``alternative="less"`` because more negative deltas mean stronger suppression.
    """

    if len(before) != len(after):
        raise ValueError("Paired samples must have equal length")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be one of: two-sided, less, greater")
    if not before:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "alternative": alternative,
            "samples": samples,
        }
    deltas = [right - left for left, right in zip(before, after, strict=True)]
    observed = mean(deltas)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        signed = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        statistic = mean(signed)
        if alternative == "less":
            extreme += statistic <= observed
        elif alternative == "greater":
            extreme += statistic >= observed
        else:
            extreme += abs(statistic) >= abs(observed)
    return {
        "statistic": observed,
        "p_value": (extreme + 1) / (samples + 1),
        "alternative": alternative,
        "samples": samples,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in original order."""

    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = n - rank_from_end + 1
        running = min(running, p_value * n / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def cohen_d(positive: Sequence[float], negative: Sequence[float]) -> float:
    if len(positive) < 2 or len(negative) < 2:
        return 0.0
    pos_mean, neg_mean = mean(positive), mean(negative)
    pos_var = sum((x - pos_mean) ** 2 for x in positive) / (len(positive) - 1)
    neg_var = sum((x - neg_mean) ** 2 for x in negative) / (len(negative) - 1)
    pooled = math.sqrt((pos_var + neg_var) / 2)
    return (pos_mean - neg_mean) / pooled if pooled > 1e-12 else 0.0
