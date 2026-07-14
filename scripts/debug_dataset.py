# armenian_llm_training/scripts/debug_dataset.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

# Allows running the script without installing the package yet.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from arm_llm.config import load_yaml_config, require_key
from arm_llm.data import (
    load_jsonl_dataset,
    print_preview,
    token_length_report,
    validate_text_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate staged Armenian SFT dataset.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    tokenizer_name = require_key(config, "model.tokenizer_name")
    train_file = require_key(config, "data.train_file")
    eval_file = require_key(config, "data.eval_file")
    text_field = require_key(config, "data.dataset_text_field")
    max_seq_length = int(require_key(config, "sequence.max_seq_length"))
    num_preview_examples = int(config.get("debug", {}).get("num_preview_examples", 3))

    print("=" * 100)
    print("Loading tokenizer")
    print("=" * 100)
    print(f"Tokenizer: {tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )

    print(f"tokenizer.eos_token: {tokenizer.eos_token!r}")
    print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    print(f"tokenizer.pad_token: {tokenizer.pad_token!r}")
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

    qwen_eos_token = "<|im_end|>"
    qwen_eos_id = tokenizer.convert_tokens_to_ids(qwen_eos_token)

    if qwen_eos_id is None or qwen_eos_id == tokenizer.unk_token_id:
        raise ValueError(f"Could not find Qwen EOS token in tokenizer: {qwen_eos_token}")

    print(f"Qwen <|im_end|> token id: {qwen_eos_id}")

    print("=" * 100)
    print("Loading datasets")
    print("=" * 100)

    train_dataset = load_jsonl_dataset(train_file)
    eval_dataset = load_jsonl_dataset(eval_file)

    print(f"Train rows: {len(train_dataset)}")
    print(f"Train columns: {train_dataset.column_names}")
    print(f"Eval rows: {len(eval_dataset)}")
    print(f"Eval columns: {eval_dataset.column_names}")

    validate_text_dataset(train_dataset, text_field, "train")
    validate_text_dataset(eval_dataset, text_field, "eval")

    print_preview(
        dataset=train_dataset,
        text_field=text_field,
        num_examples=num_preview_examples,
        dataset_name="train",
    )

    token_length_report(
        dataset=train_dataset,
        tokenizer=tokenizer,
        text_field=text_field,
        max_seq_length=max_seq_length,
        dataset_name="train",
    )

    token_length_report(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        text_field=text_field,
        max_seq_length=max_seq_length,
        dataset_name="eval",
    )

    print("=" * 100)
    print("Dataset validation finished successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()