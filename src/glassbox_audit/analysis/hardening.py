from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..analysis.stats import (
    grouped_bootstrap_ci,
    holm_adjust,
    one_sample_mean_test,
    paired_permutation_test,
)
from ..config import load_config
from ..data import load_records, split_records
from ..evaluation import calculate_metrics
from ..interventions import (
    AblateDirection,
    AblateSAEFeatures,
    ReconstructSAE,
    SubstituteSAEFeatures,
    WeightedAblateSAEFeatures,
)
from ..models import load_model
from ..sae import load_sae_from_state
from ..types import ModelOutput, PromptRecord
from ..utils import read_json, write_json

PREREGISTRATION_SHA256 = "c7b196d64460bf0e37c4fce5c4ee4ed9a46cabe0de001f57006cc6416fd7624c"
AMENDMENT_SHA256 = "2dcf600e579d40eac432615f4c084042a5b9baeab3e20433d01b0f0e4b9f2fbc"
EXPECTED_HASHES = {
    "sae.pt": "fd578f73751771e8bb13f2fc706006b4ca94d953bbf6ebdbeac487e565ba1844",
    "interventions.json": "01fc50128e791deb84fce272eaa6e6e199aeab2df15b1fdb6fe041f347b847c2",
    "discovery_tensors.pt": "ed4876efdfd74c3f14e56475dbdb999a31f36144cfa209e4c403372fda072e56",
    "features.json": "b10a59aabe1123737b69adbb378a528702dfd8fb863b53492e3af4c7452f7a34",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _verify_inputs(artifacts: Path, dataset: Path) -> dict[str, str]:
    observed = {name: _sha256(artifacts / name) for name in EXPECTED_HASHES}
    mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "observed": value}
        for name, value in observed.items()
        if value != EXPECTED_HASHES[name]
    }
    dataset_hash = _sha256(dataset)
    if dataset_hash != "1c912d796753c4cd29104d0ed78c05b1c585394d98f67e513d92e3c9166276b6":
        mismatches["dataset"] = {"expected": "frozen preregistration hash", "observed": dataset_hash}
    if mismatches:
        raise ValueError(f"Immutable input hash mismatch: {mismatches}")
    return {**observed, "dataset": dataset_hash}


def _outputs(rows: list[dict[str, Any]]) -> list[ModelOutput]:
    return [
        ModelOutput(
            record_id=str(row["record_id"]),
            behavior_score=float(row["behavior_score"]),
            capability_nll=float(row["capability_nll"]),
            response=str(row.get("response", "")),
        )
        for row in rows
    ]


def _serialize(rows: list[ModelOutput]) -> list[dict[str, object]]:
    return [row.to_dict() for row in rows]


def _by_id(rows: list[ModelOutput]) -> dict[str, ModelOutput]:
    return {row.record_id: row for row in rows}


def _paired_summary(
    records: list[PromptRecord],
    baseline: list[ModelOutput],
    changed: list[ModelOutput],
    *,
    threshold: float,
    seed: int,
) -> dict[str, object]:
    before, after = _by_id(baseline), _by_id(changed)
    result: dict[str, object] = {"metrics": calculate_metrics(records, changed, threshold)}
    specs = {
        "harmful_score": [row for row in records if row.harmful],
        "benign_score": [row for row in records if not row.harmful],
        "capability_nll": records,
    }
    for offset, (name, selected) in enumerate(specs.items()):
        field = "capability_nll" if name == "capability_nll" else "behavior_score"
        deltas = [getattr(after[row.id], field) - getattr(before[row.id], field) for row in selected]
        result[f"delta_{name}_ci"] = grouped_bootstrap_ci(
            deltas,
            [row.pair_id for row in selected],
            samples=10000,
            confidence=0.95,
            seed=seed + offset,
        )
    return result


def _hidden_summary(hidden: torch.Tensor, intervention: Any) -> dict[str, float]:
    changed_rows = []
    for start in range(0, len(hidden), 64):
        changed_rows.append(intervention(hidden[start : start + 64]).detach().float().cpu())
    changed = torch.cat(changed_rows)
    delta = changed - hidden.float()
    cosine = F.cosine_similarity(changed, hidden.float(), dim=-1)
    return {
        "hidden_l2_mean": delta.norm(dim=-1).mean().item(),
        "hidden_l2_median": delta.norm(dim=-1).median().item(),
        "hidden_cosine_mean": cosine.mean().item(),
        "hidden_cosine_min": cosine.min().item(),
    }


def _evaluate_checkpoint(
    model: Any,
    records: list[PromptRecord],
    layer: int | None,
    intervention: Any,
    path: Path,
    batch_size: int,
) -> list[ModelOutput]:
    if path.exists():
        return _outputs(read_json(path)["outputs"])
    rows = model.evaluate_batched(records, layer, intervention, batch_size=batch_size)
    write_json(path, {"outputs": _serialize(rows)})
    return rows


def _ranked_features(features: list[dict[str, Any]]) -> list[int]:
    positive = [int(row["feature"]) for row in features if float(row["effect_size"]) > 0]
    return positive or [int(row["feature"]) for row in features]


def _fit_weighted_sparse(
    sae: Any,
    train_hidden: torch.Tensor,
    train_labels: torch.Tensor,
    validation_hidden: torch.Tensor,
    validation_labels: torch.Tensor,
    candidates: list[int],
) -> tuple[list[int], list[float], dict[str, object]]:
    candidates = candidates[:100]
    with torch.no_grad():
        train_x = sae.encode(train_hidden.float() - sae.activation_center)[:, candidates]
        validation_x = sae.encode(validation_hidden.float() - sae.activation_center)[:, candidates]
    center = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-6)
    train_x = (train_x - center) / scale
    validation_x = (validation_x - center) / scale
    trials = []
    best: tuple[float, torch.Tensor, float] | None = None
    for l1 in [0.0, 0.0001, 0.001, 0.01]:
        torch.manual_seed(17)
        weight = torch.zeros(len(candidates), requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=0.05)
        for _ in range(400):
            logits = train_x @ weight + bias
            loss = F.binary_cross_entropy_with_logits(logits, train_labels.float()) + l1 * weight.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            validation_loss = F.binary_cross_entropy_with_logits(
                validation_x @ weight + bias, validation_labels.float()
            ).item()
        trials.append({"l1": l1, "validation_bce": validation_loss})
        if best is None or validation_loss < best[0]:
            best = (validation_loss, weight.detach().clone(), l1)
    assert best is not None
    original_scale_weight = best[1] / scale
    positive = torch.where(original_scale_weight > 0)[0]
    if len(positive) == 0:
        positive = torch.argsort(original_scale_weight, descending=True)[:20]
    selected_local = positive[torch.argsort(original_scale_weight[positive], descending=True)[:20]]
    weights = original_scale_weight[selected_local]
    weights = weights / weights.abs().max().clamp_min(1e-8)
    return (
        [candidates[int(index)] for index in selected_local],
        weights.tolist(),
        {"selected_l1": best[2], "trials": trials, "candidate_features": candidates},
    )


def _method_test(
    records: list[PromptRecord],
    left: list[ModelOutput],
    right: list[ModelOutput],
    *,
    seed: int,
) -> dict[str, object]:
    left_by_id, right_by_id = _by_id(left), _by_id(right)
    harmful = [row for row in records if row.harmful]
    left_values = [left_by_id[row.id].behavior_score for row in harmful]
    right_values = [right_by_id[row.id].behavior_score for row in harmful]
    differences = [right - left for left, right in zip(left_values, right_values, strict=True)]
    return {
        "definition": "right harmful score minus left harmful score; positive means left suppresses more",
        "pair_bootstrap_ci": grouped_bootstrap_ci(
            differences,
            [row.pair_id for row in harmful],
            samples=10000,
            confidence=0.95,
            seed=seed,
        ),
        "permutation": paired_permutation_test(
            left_values, right_values, alternative="greater", samples=10000, seed=seed + 1
        ),
        "inferiority_margin_0_02": one_sample_mean_test(
            differences, 0.02, alternative="greater"
        ),
    }


def run_paper_hardening(
    *,
    artifacts: str | Path,
    dataset: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    result_path: str | Path,
    device: str = "cuda:0",
    cache_dir: str | None = None,
    batch_size: int = 8,
) -> dict[str, object]:
    artifacts, dataset = Path(artifacts), Path(dataset)
    output_dir, result_path = Path(output_dir), Path(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    hashes = _verify_inputs(artifacts, dataset)

    config = load_config(config_path)
    config.model.device = device
    config.model.cache_dir = cache_dir
    config.sae.device = device
    records = load_records(dataset)
    validation_records = split_records(records, "validation")
    test_records = split_records(records, "test")
    model = load_model(config.model, config.seed)
    summary = read_json(artifacts / "summary.json")
    threshold, layer = float(summary["behavior_threshold"]), int(summary["target_layer"])
    original = read_json(artifacts / "interventions.json")
    original_baseline = _outputs(original["baseline_outputs"])

    baseline = _evaluate_checkpoint(
        model, test_records, None, None, output_dir / "baseline_batched.json", batch_size
    )
    old, new = _by_id(original_baseline), _by_id(baseline)
    baseline_crosscheck = {
        "max_abs_behavior_score_difference": max(
            abs(old[row.id].behavior_score - new[row.id].behavior_score) for row in test_records
        ),
        "max_abs_capability_nll_difference": max(
            abs(old[row.id].capability_nll - new[row.id].capability_nll) for row in test_records
        ),
    }

    sae = load_sae_from_state(str(artifacts / "sae.pt"))
    tensors = torch.load(artifacts / "discovery_tensors.pt", map_location="cpu", weights_only=False)
    directions = tensors["directions"]
    features = read_json(artifacts / "features.json")
    ranked = _ranked_features(features)
    selected_five = [int(value) for value in summary["selected_features"]]

    hidden_path = output_dir / "test_hidden.pt"
    if hidden_path.exists():
        test_hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
    else:
        test_hidden = model.collect(test_records, [layer])[layer]
        torch.save(test_hidden, hidden_path)

    active = [
        int(row["feature"])
        for row in features
        if float(row.get("activation_frequency", 0.0)) > 0
    ]
    eligible_random = sorted(set(active) - set(ranked[:100]))
    rng = random.Random(1729)
    random_sets = {
        budget: [rng.sample(eligible_random, budget) for _ in range(8)]
        for budget in [1, 2, 5, 10, 20, 50, 100]
    }

    validation_hidden_path = output_dir / "validation_hidden.pt"
    if validation_hidden_path.exists():
        validation_hidden = torch.load(validation_hidden_path, map_location="cpu", weights_only=True)
    else:
        validation_hidden = model.collect(validation_records, [layer])[layer]
        torch.save(validation_hidden, validation_hidden_path)
    weighted_indices, weighted_values, weighted_fit = _fit_weighted_sparse(
        sae,
        tensors["activations"][layer],
        tensors["labels"],
        validation_hidden,
        torch.tensor([row.harmful for row in validation_records], dtype=torch.bool),
        ranked,
    )

    interventions = {
        "mean_direction_ablation": AblateDirection(directions["mean_difference"]),
        "linear_probe_ablation": AblateDirection(directions["linear_probe"]),
        "selected_sae_residual_ablation": AblateSAEFeatures(sae, selected_five),
        "selected_sae_substitution": SubstituteSAEFeatures(sae, selected_five),
        "matched_random_sae_residual_ablation": AblateSAEFeatures(sae, random_sets[5][0]),
        "sae_reconstruction_only": ReconstructSAE(sae),
        "weighted_sparse_sae_ablation": WeightedAblateSAEFeatures(
            sae, weighted_indices, weighted_values
        ),
    }
    comparability: dict[str, object] = {}
    evaluated: dict[str, list[ModelOutput]] = {}
    for index, (name, intervention) in enumerate(interventions.items()):
        rows = _evaluate_checkpoint(
            model,
            test_records,
            layer,
            intervention,
            output_dir / f"comparability_{name}.json",
            batch_size,
        )
        evaluated[name] = rows
        hidden_metrics = _hidden_summary(test_hidden, intervention)
        kl = model.next_token_kl_batched(
            test_records, layer, intervention, batch_size=batch_size
        )
        paired = _paired_summary(
            test_records, baseline, rows, threshold=threshold, seed=2000 + index * 10
        )
        harmful_suppression = -float(paired["delta_harmful_score_ci"]["estimate"])
        l2 = float(hidden_metrics["hidden_l2_mean"])
        capability_cost = max(0.0, float(paired["delta_capability_nll_ci"]["estimate"]))
        comparability[name] = {
            **paired,
            **hidden_metrics,
            "next_token_kl_mean": sum(kl) / len(kl),
            "harmful_suppression_per_hidden_l2": harmful_suppression / max(l2, 1e-12),
            "harmful_suppression_per_positive_capability_cost": (
                harmful_suppression / capability_cost if capability_cost > 0 else None
            ),
        }

    frontier_rows = []
    frontier_outputs: dict[int, list[ModelOutput]] = {}
    random_outputs: dict[int, list[list[ModelOutput]]] = {}
    for budget in [1, 2, 5, 10, 20, 50, 100]:
        selected = ranked[:budget]
        intervention = AblateSAEFeatures(sae, selected)
        rows = _evaluate_checkpoint(
            model,
            test_records,
            layer,
            intervention,
            output_dir / f"budget_{budget}_selected.json",
            batch_size,
        )
        frontier_outputs[budget] = rows
        selected_summary = _paired_summary(
            test_records, baseline, rows, threshold=threshold, seed=3000 + budget
        )
        selected_summary.update(_hidden_summary(test_hidden, intervention))
        control_summaries = []
        random_outputs[budget] = []
        for control_index, control_features in enumerate(random_sets[budget]):
            control = AblateSAEFeatures(sae, control_features)
            control_rows = _evaluate_checkpoint(
                model,
                test_records,
                layer,
                control,
                output_dir / f"budget_{budget}_random_{control_index}.json",
                batch_size,
            )
            random_outputs[budget].append(control_rows)
            control_summary = _paired_summary(
                test_records,
                baseline,
                control_rows,
                threshold=threshold,
                seed=4000 + budget * 10 + control_index,
            )
            control_summary.update(_hidden_summary(test_hidden, control))
            control_summaries.append(control_summary)
        frontier_rows.append(
            {
                "budget": budget,
                "selected_features": selected,
                "selected": selected_summary,
                "matched_random_sets": random_sets[budget],
                "matched_random_summaries": control_summaries,
            }
        )

    original_methods = original["causal_tests"]
    mean_original = _outputs(original_methods["mean_difference_direction_ablation"]["outputs"])
    probe_original = _outputs(original_methods["linear_probe_direction_ablation"]["outputs"])
    sae_original = _outputs(original_methods["sae_feature_ablation"]["outputs"])
    averaged_random = []
    random_five = random_outputs[5]
    random_maps = [_by_id(rows) for rows in random_five]
    for row in test_records:
        averaged_random.append(
            ModelOutput(
                row.id,
                sum(values[row.id].behavior_score for values in random_maps) / len(random_maps),
                sum(values[row.id].capability_nll for values in random_maps) / len(random_maps),
            )
        )
    tests = {
        "mean_vs_original_sae": _method_test(test_records, mean_original, sae_original, seed=5001),
        "probe_vs_original_sae": _method_test(test_records, probe_original, sae_original, seed=5002),
        "selected_sae_vs_same_dictionary_random": _method_test(
            test_records, frontier_outputs[5], averaged_random, seed=5003
        ),
        "mean_vs_weighted_sparse_sae": _method_test(
            test_records, mean_original, evaluated["weighted_sparse_sae_ablation"], seed=5004
        ),
    }
    raw_p = [float(row["permutation"]["p_value"]) for row in tests.values()]
    for row, adjusted in zip(tests.values(), holm_adjust(raw_p), strict=True):
        row["holm_adjusted_p"] = adjusted

    result = {
        "schema_version": "2.0",
        "analysis_status": "post_hoc_sensitivity",
        "git_commit_at_run": _git_commit(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_amendment_sha256": AMENDMENT_SHA256,
        "input_hashes": hashes,
        "model_id": config.model.name,
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "dataset_sha256": hashes["dataset"],
        "target_layer": layer,
        "baseline_batch_crosscheck": baseline_crosscheck,
        "weighted_sparse_fit": {
            **weighted_fit,
            "selected_features": weighted_indices,
            "normalized_weights": weighted_values,
        },
        "intervention_comparability": comparability,
        "feature_budget_frontier": frontier_rows,
        "primary_family_holm": tests,
    }
    write_json(result_path, result)
    return result
