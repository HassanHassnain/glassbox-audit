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
            for _key in keys
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
    return bootstrap_ci(deltas, samples=samples, confidence=confidence, seed=seed)


def sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_ppf(probability: float) -> float:
    """Approximate the inverse standard-normal CDF without a SciPy dependency."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    low, high = 0.02425, 1 - 0.02425
    if probability < low:
        q = math.sqrt(-2 * math.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if probability > high:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def one_sample_mean_test(
    values: Sequence[float],
    null_value: float,
    *,
    alternative: str = "two-sided",
) -> dict[str, float | str]:
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be one of: two-sided, less, greater")
    estimate = mean(values)
    sd = sample_standard_deviation(values)
    se = sd / math.sqrt(len(values)) if values else float("nan")
    if not values:
        z = float("nan")
        p_value = float("nan")
    elif se <= 1e-12:
        if alternative == "greater":
            p_value = 0.0 if estimate > null_value else 1.0
        elif alternative == "less":
            p_value = 0.0 if estimate < null_value else 1.0
        else:
            p_value = 0.0 if estimate != null_value else 1.0
        z = 0.0 if estimate == null_value else math.copysign(1e12, estimate - null_value)
    else:
        z = (estimate - null_value) / se
        if alternative == "greater":
            p_value = 1 - normal_cdf(z)
        elif alternative == "less":
            p_value = normal_cdf(z)
        else:
            p_value = 2 * min(normal_cdf(z), 1 - normal_cdf(z))
    return {
        "estimate": estimate,
        "null_value": null_value,
        "standard_error": se,
        "z": z,
        "p_value": p_value,
        "alternative": alternative,
    }


def tost_equivalence_test(
    values: Sequence[float],
    lower_bound: float,
    upper_bound: float,
    *,
    alpha: float = 0.05,
    samples: int = 1000,
    seed: int = 0,
    group_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    lower = one_sample_mean_test(values, lower_bound, alternative="greater")
    upper = one_sample_mean_test(values, upper_bound, alternative="less")
    ci_confidence = 1 - 2 * alpha
    ci = (
        grouped_bootstrap_ci(values, group_ids, samples=samples, confidence=ci_confidence, seed=seed)
        if group_ids is not None
        else bootstrap_ci(values, samples=samples, confidence=ci_confidence, seed=seed)
    )
    equivalent = (
        float(lower["p_value"]) <= alpha
        and float(upper["p_value"]) <= alpha
        and float(ci["low"]) > lower_bound
        and float(ci["high"]) < upper_bound
    )
    return {
        "estimate": mean(values),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "alpha": alpha,
        "ci_confidence": ci_confidence,
        "ci": ci,
        "lower_one_sided_p": lower["p_value"],
        "upper_one_sided_p": upper["p_value"],
        "p_value": max(float(lower["p_value"]), float(upper["p_value"])),
        "equivalent": equivalent,
    }


def normal_approx_power(
    *,
    n: int,
    effect: float,
    standard_deviation: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be one of: two-sided, less, greater")
    effect = abs(effect)
    if standard_deviation <= 1e-12:
        return 1.0 if effect > 0 else alpha
    noncentrality = effect / (standard_deviation / math.sqrt(n))
    if alternative == "two-sided":
        critical = normal_ppf(1 - alpha / 2)
        return (1 - normal_cdf(critical - noncentrality)) + normal_cdf(
            -critical - noncentrality
        )
    critical = normal_ppf(1 - alpha)
    return 1 - normal_cdf(critical - noncentrality)


def required_n_for_power(
    *,
    effect: float,
    standard_deviation: float,
    target_power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> int | None:
    if effect <= 0 or standard_deviation <= 0:
        return None
    if not 0 < target_power < 1:
        raise ValueError("target_power must be between 0 and 1")
    if alternative == "two-sided":
        critical = normal_ppf(1 - alpha / 2)
    elif alternative in {"less", "greater"}:
        critical = normal_ppf(1 - alpha)
    else:
        raise ValueError("alternative must be one of: two-sided, less, greater")
    n = math.ceil(((critical + normal_ppf(target_power)) * standard_deviation / effect) ** 2)
    while normal_approx_power(
        n=n,
        effect=effect,
        standard_deviation=standard_deviation,
        alpha=alpha,
        alternative=alternative,
    ) < target_power:
        n += 1
    return n


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
