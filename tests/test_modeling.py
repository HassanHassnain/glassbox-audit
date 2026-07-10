from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from glassbox_audit.data import load_records, split_records
from glassbox_audit.interventions import AddDirection, HeadSliceIntervention
from glassbox_audit.modeling import (
    HuggingFaceRefusalModel,
    NNsightRefusalModel,
    ToyRefusalModel,
    TransformerLensRefusalModel,
)

ROOT = Path(__file__).parents[1]


def test_toy_planted_direction_is_causal():
    records = split_records(load_records(ROOT / "data/refusal_pairs.jsonl"), "test")
    model = ToyRefusalModel(seed=17)
    baseline = model.evaluate(records)
    reduced = model.evaluate(records, 2, AddDirection(model.behavior_direction, -3))
    harmful = [index for index, record in enumerate(records) if record.harmful]
    assert sum(reduced[i].behavior_score for i in harmful) < sum(
        baseline[i].behavior_score for i in harmful
    )


def test_toy_sycophancy_signal_is_held_out_and_causal():
    records = split_records(load_records(ROOT / "data/sycophancy_controlled_v1.jsonl"), "test")
    model = ToyRefusalModel(seed=17, behavior="sycophancy")
    baseline = model.evaluate(records)
    reduced = model.evaluate(records, 2, AddDirection(model.behavior_direction, -3))
    pressured = [index for index, record in enumerate(records) if record.harmful]
    neutral = [index for index, record in enumerate(records) if not record.harmful]
    pressured_mean = sum(baseline[index].behavior_score for index in pressured) / len(pressured)
    neutral_mean = sum(baseline[index].behavior_score for index in neutral) / len(neutral)
    assert pressured_mean - neutral_mean > 0.3
    assert sum(reduced[index].behavior_score for index in pressured) < sum(
        baseline[index].behavior_score for index in pressured
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


def test_hf_prompt_ids_accepts_chat_template_batch_encoding():
    transformers = __import__("transformers", fromlist=["BatchEncoding"])
    adapter = object.__new__(HuggingFaceRefusalModel)
    adapter.model = torch.nn.Linear(1, 1)
    expected = torch.tensor([[1, 2, 3]])
    adapter.tokenizer = SimpleNamespace(
        chat_template="template",
        apply_chat_template=lambda *args, **kwargs: transformers.BatchEncoding(
            {"input_ids": expected}
        ),
    )
    assert torch.equal(adapter._prompt_ids("test"), expected)


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


def test_hf_attention_head_slice_hook_changes_selected_head_only():
    adapter = object.__new__(HuggingFaceRefusalModel)
    adapter.name = "fake"
    adapter.hidden_size = 4
    adapter.model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=2))
    block = torch.nn.Module()
    block.self_attn = torch.nn.Identity()
    adapter._layers = lambda: [block]
    hidden = torch.zeros(1, 3, 4)
    intervention = HeadSliceIntervention(1, 2, lambda value: value + 5)
    with adapter._component_intervention_hook(0, "attention", intervention, start_position=1):
        changed = block.self_attn(hidden)
    assert torch.all(changed[:, :1] == 0)
    assert torch.all(changed[:, 1:, :2] == 0)
    assert torch.all(changed[:, 1:, 2:] == 5)


def _tiny_llama():
    transformers = __import__("transformers", fromlist=["LlamaConfig", "LlamaForCausalLM"])
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return transformers.LlamaForCausalLM(config).eval()


def test_transformer_lens_backend_runs_native_residual_intervention():
    pytest.importorskip("transformer_lens")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = __import__("transformers", fromlist=["PreTrainedTokenizerFast"])
    from transformer_lens.model_bridge import TransformerBridge

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "test": 4},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    raw = _tiny_llama()
    adapter = object.__new__(TransformerLensRefusalModel)
    adapter.transformer_lens = TransformerBridge.boot_transformers(
        "glassbox-tiny-llama",
        hf_model=raw,
        tokenizer=tokenizer,
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 4, 4, 2]])
    baseline = adapter._forward_logits(input_ids, None, None, 0)
    changed = adapter._forward_logits(
        input_ids,
        0,
        lambda hidden: torch.zeros_like(hidden),
        0,
    )
    assert baseline.shape == changed.shape == (1, 4, 32)
    assert not torch.allclose(baseline, changed)
    adapter._prompt_ids = lambda _prompt: input_ids
    captures = adapter._collect_tokens([SimpleNamespace(prompt="test")], [0, 1], 2)
    assert captures[0].shape == captures[1].shape == (2, 16)


def test_nnsight_backend_runs_native_residual_intervention():
    nnsight = pytest.importorskip("nnsight")
    raw = _tiny_llama()
    adapter = object.__new__(NNsightRefusalModel)
    adapter.model = raw
    adapter.nnsight = nnsight.LanguageModel(raw, tokenizer=None, dispatch=True)
    input_ids = torch.tensor([[1, 4, 4, 2]])
    baseline = adapter._forward_logits(input_ids, None, None, 0)
    changed = adapter._forward_logits(
        input_ids,
        0,
        lambda hidden: torch.zeros_like(hidden),
        0,
    )
    assert baseline.shape == changed.shape == (1, 4, 32)
    assert not torch.allclose(baseline, changed)
    adapter._prompt_ids = lambda _prompt: input_ids
    captures = adapter._collect_tokens([SimpleNamespace(prompt="test")], [0, 1], 2)
    assert captures[0].shape == captures[1].shape == (2, 16)
