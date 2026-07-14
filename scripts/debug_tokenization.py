# scripts/debug_tokenization.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from arm_llm.config import load_yaml_config, require_key
from arm_llm.data import (
    label_mask_report,
    load_jsonl_dataset,
    tokenize_dataset,
    validate_text_dataset,
)
from arm_llm.model import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug tokenization and label masking.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def decode_labeled_tokens(tokenizer, input_ids: list[int], labels: list[int]) -> str:
    """
    Decode only the tokens that are active in the loss.
    This helps us visually check assistant_only masking.
    """
    labeled_ids = [
        input_id
        for input_id, label in zip(input_ids, labels)
        if label != -100
    ]

    if not labeled_ids:
        return ""

    return tokenizer.decode(labeled_ids, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model_config = require_key(config, "model")
    data_config = require_key(config, "data")
    sequence_config = require_key(config, "sequence")
    loss_config = require_key(config, "loss")

    train_file = data_config["train_file"]
    eval_file = data_config["eval_file"]
    text_field = data_config["dataset_text_field"]
    max_seq_length = int(sequence_config["max_seq_length"])

    tokenizer = load_tokenizer(model_config)

    train_dataset = load_jsonl_dataset(train_file)
    eval_dataset = load_jsonl_dataset(eval_file)

    validate_text_dataset(train_dataset, text_field, "train")
    validate_text_dataset(eval_dataset, text_field, "eval")

    tokenized_train = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        text_field=text_field,
        max_seq_length=max_seq_length,
        loss_config=loss_config,
        dataset_name="train",
    )

    tokenized_eval = tokenize_dataset(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        text_field=text_field,
        max_seq_length=max_seq_length,
        loss_config=loss_config,
        dataset_name="eval",
    )

    label_mask_report(tokenized_train, "train")
    label_mask_report(tokenized_eval, "eval")

    print("=" * 100)
    print("Visual check: first 3 labeled assistant-only regions")
    print("=" * 100)

    for i in range(min(3, len(tokenized_train))):
        row = tokenized_train[i]

        labeled_text = decode_labeled_tokens(
            tokenizer=tokenizer,
            input_ids=row["input_ids"],
            labels=row["labels"],
        )

        print(f"\n--- Example {i}: active loss text ---")
        print(labeled_text[:1500])

    print("=" * 100)
    print("Tokenization debug finished successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()