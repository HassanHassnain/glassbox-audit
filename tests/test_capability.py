import math

import pytest

from glassbox_audit.evaluation.capability import frozen_token_blocks, summarize_lm_conditions


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [len(word) for word in text.split()]}


def test_frozen_token_blocks_are_deterministic_and_hash_pinned():
    texts = [" ".join(f"word{index}" for index in range(200))]
    first, first_meta = frozen_token_blocks(
        texts, WordTokenizer(), block_size=8, max_blocks=5, seed=17
    )
    second, second_meta = frozen_token_blocks(
        texts, WordTokenizer(), block_size=8, max_blocks=5, seed=17
    )
    assert first == second
    assert first_meta == second_meta
    assert len(first) == 5
    assert all(len(block) == 9 for block in first)
    assert len(first_meta["subset_token_sha256"]) == 64


def test_capability_summary_uses_paired_token_weighted_bootstrap():
    baseline = [
        {"nll_sum": 20.0, "token_count": 10},
        {"nll_sum": 60.0, "token_count": 20},
    ]
    mean = [
        {"nll_sum": 21.0, "token_count": 10},
        {"nll_sum": 62.0, "token_count": 20},
    ]
    sae = [
        {"nll_sum": 20.5, "token_count": 10},
        {"nll_sum": 61.0, "token_count": 20},
    ]
    result = summarize_lm_conditions(
        {
            "baseline": baseline,
            "mean_difference_direction_ablation": mean,
            "sae_feature_ablation": sae,
        },
        bootstrap_samples=200,
        confidence=0.95,
        seed=3,
    )
    assert result["baseline"]["nll_ci"]["estimate"] == pytest.approx(80 / 30)
    mean_delta = result["mean_difference_direction_ablation"]["delta_nll_ci"]
    assert mean_delta["estimate"] == pytest.approx(3 / 30)
    assert result["mean_difference_direction_ablation"]["perplexity_ratio_ci"][
        "estimate"
    ] == pytest.approx(math.exp(3 / 30))
    assert mean_delta["low"] <= mean_delta["estimate"] <= mean_delta["high"]


def test_capability_summary_rejects_unpaired_conditions():
    with pytest.raises(ValueError, match="same non-zero number"):
        summarize_lm_conditions(
            {
                "baseline": [{"nll_sum": 1.0, "token_count": 1}],
                "sae_feature_ablation": [],
            },
            bootstrap_samples=10,
            confidence=0.95,
            seed=1,
        )
