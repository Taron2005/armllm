# scripts/debug_batch.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from arm_llm.collator import CausalLMCollator
from arm_llm.config import load_yaml_config, require_key
from arm_llm.data import (
    label_mask_report,
    load_jsonl_dataset,
    tokenize_dataset,
    validate_text_dataset,
)
from arm_llm.model import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug dynamic padding batch.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="How many batches to inspect.",
    )
    return parser.parse_args()


def decode_active_label_text(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> str:
    """
    Decode only tokens participating in the loss.

    Important:
    We decode input_ids at positions where labels != -100.
    This shows what the model is being trained to predict.
    """
    active_positions = labels != -100
    active_input_ids = input_ids[active_positions].tolist()

    if not active_input_ids:
        return ""

    return tokenizer.decode(active_input_ids, skip_special_tokens=False)


def sanity_check_batch(batch: dict[str, torch.Tensor]) -> None:
    """
    Check basic batch correctness.
    """
    required_keys = {"input_ids", "attention_mask", "labels"}

    missing = required_keys - set(batch.keys())
    if missing:
        raise KeyError(f"Batch is missing keys: {missing}")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids and attention_mask shapes differ: "
            f"{input_ids.shape} vs {attention_mask.shape}"
        )

    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels shapes differ: "
            f"{input_ids.shape} vs {labels.shape}"
        )

    if input_ids.dtype != torch.long:
        raise TypeError(f"input_ids dtype must be torch.long, got {input_ids.dtype}")

    if attention_mask.dtype != torch.long:
        raise TypeError(
            f"attention_mask dtype must be torch.long, got {attention_mask.dtype}"
        )

    if labels.dtype != torch.long:
        raise TypeError(f"labels dtype must be torch.long, got {labels.dtype}")

    # All padded positions must have label -100.
    padded_positions = attention_mask == 0

    if padded_positions.any():
        padded_labels = labels[padded_positions]
        if not torch.all(padded_labels == -100):
            raise ValueError("Some padded positions have labels != -100.")

    # Each row should have at least one active label token.
    active_per_row = (labels != -100).sum(dim=1)

    zero_active_rows = (active_per_row == 0).sum().item()

    if zero_active_rows > 0:
        raise ValueError(
            f"{zero_active_rows} rows in batch have zero active label tokens."
        )


def print_batch_report(
    batch: dict[str, torch.Tensor],
    tokenizer,
    batch_index: int,
) -> None:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size, seq_len = input_ids.shape

    total_tokens = input_ids.numel()
    real_tokens = attention_mask.sum().item()
    pad_tokens = total_tokens - real_tokens
    active_label_tokens = (labels != -100).sum().item()

    print("=" * 100)
    print(f"Batch {batch_index} report")
    print("=" * 100)
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"attention_mask shape: {tuple(attention_mask.shape)}")
    print(f"labels shape: {tuple(labels.shape)}")
    print(f"batch_size: {batch_size}")
    print(f"seq_len after dynamic padding: {seq_len}")
    print(f"total tokens in batch: {total_tokens}")
    print(f"real tokens: {real_tokens}")
    print(f"pad tokens: {pad_tokens}")
    print(f"active label tokens: {active_label_tokens}")
    print(f"active label percent: {100 * active_label_tokens / total_tokens:.2f}%")

    print("\nFirst row active-label text preview:")
    text = decode_active_label_text(
        tokenizer=tokenizer,
        input_ids=input_ids[0],
        labels=labels[0],
    )

    print(text[:1500])


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_config = require_key(config, "model")
    data_config = require_key(config, "data")
    sequence_config = require_key(config, "sequence")
    loss_config = require_key(config, "loss")
    batching_config = require_key(config, "batching")

    train_file = data_config["train_file"]
    text_field = data_config["dataset_text_field"]
    max_seq_length = int(sequence_config["max_seq_length"])

    batch_size = int(batching_config.get("train_batch_size", 1))
    pad_to_multiple_of = batching_config.get("pad_to_multiple_of", None)

    if pad_to_multiple_of is not None:
        pad_to_multiple_of = int(pad_to_multiple_of)

    tokenizer = load_tokenizer(model_config)

    # For causal LM training, right padding is the usual default.
    tokenizer.padding_side = "right"

    dataset = load_jsonl_dataset(train_file)
    validate_text_dataset(dataset, text_field, "train")

    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        text_field=text_field,
        max_seq_length=max_seq_length,
        loss_config=loss_config,
        dataset_name="train",
    )

    label_mask_report(tokenized_dataset, "train")

    collator = CausalLMCollator(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print("=" * 100)
    print("Inspecting dynamic batches")
    print("=" * 100)

    for batch_index, batch in enumerate(dataloader):
        sanity_check_batch(batch)
        print_batch_report(batch, tokenizer, batch_index)

        if batch_index + 1 >= args.num_batches:
            break

    print("=" * 100)
    print("Batch debug finished successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()