from pathlib import Path

import torch

from glassbox_audit.data import load_records, split_records
from glassbox_audit.interventions import AddDirection
from glassbox_audit.models import HuggingFaceRefusalModel, ToyRefusalModel

ROOT = Path(__file__).parents[1]


def test_toy_planted_direction_is_causal():
    records = split_records(load_records(ROOT / "data/fixtures/refusal_pairs_tiny.jsonl"), "test")
    model = ToyRefusalModel(seed=17)
    baseline = model.evaluate(records)
    reduced = model.evaluate(records, 2, AddDirection(model.behavior_direction, -3))
    harmful = [index for index, record in enumerate(records) if record.harmful]
    assert sum(reduced[i].behavior_score for i in harmful) < sum(
        baseline[i].behavior_score for i in harmful
    )


def test_hf_hook_changes_requested_position_range_without_loading_model():
    adapter = object.__new__(HuggingFaceRefusalModel)
    layer = torch.nn.Identity()
    adapter._layers = lambda: [layer]
    hidden = torch.zeros(1, 5, 2)
    with adapter._intervention_hook(0, lambda value: value + 1, start_position=2):
        changed = layer(hidden)
    assert torch.all(changed[:, :2] == 0)
    assert torch.all(changed[:, 2:] == 1)


def test_hf_component_hook_changes_submodule_without_loading_model():
    adapter = object.__new__(HuggingFaceRefusalModel)
    block = torch.nn.Module()
    block.mlp = torch.nn.Identity()
    adapter._layers = lambda: [block]
    hidden = torch.zeros(1, 4, 2)
    with adapter._component_intervention_hook(0, "mlp", lambda value: value + 2, start_position=1):
        changed = block.mlp(hidden)
    assert torch.all(changed[:, :1] == 0)
    assert torch.all(changed[:, 1:] == 2)
