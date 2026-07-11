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


def grouped_bootstrap_ci(
    values: Sequence[float],
    group_ids: Sequence[str],
    statistic: Callable[[Sequence[float]], float] = mean,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap complete groups, preserving linked observations such as prompt pairs."""

    if len(values) != len(group_ids):
        raise ValueError("Values and group IDs must have equal length")
    if not values:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    grouped: dict[str, list[float]] = {}
    for value, group_id in zip(values, group_ids, strict=True):
        grouped.setdefault(str(group_id), []).append(value)
    keys = list(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [
            value
            for _ in keys
            for value in grouped[keys[rng.randrange(len(keys))]]
        ]
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
    *,
    group_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    if len(before) != len(after):
        raise ValueError("Paired samples must have equal length")
    deltas = [right - left for left, right in zip(before, after, strict=True)]
    if group_ids is not None:
        return grouped_bootstrap_ci(
            deltas,
            group_ids,
            samples=samples,
            confidence=confidence,
            seed=seed,
        )
    return bootstrap_ci(
        deltas,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )


def sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def one_sample_mean_test(
    values: Sequence[float],
    null_value: float,
    *,
    alternative: str = "two-sided",
) -> dict[str, float | str]:
    """Normal-approximation test for a mean, used for prespecified inferiority margins."""

    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be one of: two-sided, less, greater")
    estimate = mean(values)
    standard_error = (
        sample_standard_deviation(values) / math.sqrt(len(values)) if values else float("nan")
    )
    if not values:
        z = p_value = float("nan")
    elif standard_error <= 1e-12:
        z = 0.0 if estimate == null_value else math.copysign(1e12, estimate - null_value)
        if alternative == "greater":
            p_value = 0.0 if estimate > null_value else 1.0
        elif alternative == "less":
            p_value = 0.0 if estimate < null_value else 1.0
        else:
            p_value = 0.0 if estimate != null_value else 1.0
    else:
        z = (estimate - null_value) / standard_error
        if alternative == "greater":
            p_value = 1 - normal_cdf(z)
        elif alternative == "less":
            p_value = normal_cdf(z)
        else:
            p_value = 2 * min(normal_cdf(z), 1 - normal_cdf(z))
    return {
        "estimate": estimate,
        "null_value": null_value,
        "standard_error": standard_error,
        "z": z,
        "p_value": p_value,
        "alternative": alternative,
    }


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


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm familywise-error adjusted p-values in original order."""

    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    running = 0.0
    for rank, (index, p_value) in enumerate(indexed):
        running = max(running, (n - rank) * p_value)
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
