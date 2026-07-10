#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_rows(path: Path, seed: int, max_records: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] == "train"]
    random.Random(seed).shuffle(rows)
    return rows[:max_records]


def token_ids(encoded) -> torch.Tensor:
    return encoded.input_ids if hasattr(encoded, "input_ids") else encoded


def training_example(tokenizer, row: dict, max_length: int, device: str):
    response = (
        "I cannot assist with that request."
        if row["harmful"]
        else "I can help with that request."
    )
    user = {"role": "user", "content": row["prompt"]}
    prompt = token_ids(
        tokenizer.apply_chat_template([user], add_generation_prompt=True, return_tensors="pt")
    )
    full = token_ids(
        tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": response}],
            add_generation_prompt=False,
            return_tensors="pt",
        )
    )[:, :max_length]
    labels = full.clone()
    labels[:, : min(prompt.shape[-1], labels.shape[-1])] = -100
    if not bool((labels != -100).any()):
        return None
    return full.to(device), labels.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the B2 LoRA drift demonstration")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--data", default="data/refusal_controlled_v2_1000.jsonl")
    parser.add_argument("--output", default="artifacts/lora/qwen25-15b-refusal-shifted")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_path = Path(args.data)
    rows = load_train_rows(data_path, args.seed, args.max_records)
    if not rows:
        raise ValueError("No train rows found")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        dtype=torch.float16,
        device_map={"": args.device},
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    losses = []
    completed_steps = 0
    example_index = 0
    while completed_steps < args.max_steps:
        row = rows[example_index % len(rows)]
        example_index += 1
        example = training_example(tokenizer, row, args.max_length, args.device)
        if example is None:
            continue
        input_ids, labels = example
        loss = model(input_ids=input_ids, labels=labels).loss
        (loss / args.gradient_accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        if example_index % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1

    output = Path(args.output)
    adapter_dir = output / "adapter"
    merged_dir = output / "merged"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    trainable, total = model.get_nb_trainable_parameters()
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(merged_dir)
    manifest = {
        "schema_version": "1.0",
        "base_model": args.base_model,
        "dataset": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "seed": args.seed,
        "records_available": len(rows),
        "optimizer_steps": completed_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0},
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
