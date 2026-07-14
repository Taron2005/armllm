# scripts/debug_model.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from arm_llm.config import load_yaml_config, require_key
from arm_llm.model import (
    add_lora_adapters,
    load_base_model,
    load_tokenizer,
    print_gpu_memory,
    print_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug native HF QLoRA model loading.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def validate_cuda() -> None:
    print("=" * 100)
    print("CUDA check")
    print("=" * 100)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU: {gpu_name}")
    print(f"Compute capability: {capability}")
    print(f"VRAM: {total_vram:.2f} GB")
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")

    if capability[0] < 6:
        raise RuntimeError("This GPU is too old for BitsAndBytes NF4/FP4 quantization.")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    validate_cuda()

    model_config = require_key(config, "model")
    lora_config = require_key(config, "lora")

    print("=" * 100)
    print("Loading tokenizer")
    print("=" * 100)

    tokenizer = load_tokenizer(model_config)

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"EOS token: {tokenizer.eos_token!r}, id={tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token!r}, id={tokenizer.pad_token_id}")

    print_gpu_memory("Before model loading")

    print("=" * 100)
    print("Loading base model with native HF Transformers")
    print("=" * 100)

    model = load_base_model(model_config)

    # Make model config consistent with tokenizer.
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print_gpu_memory("After base model loading")

    print("=" * 100)
    print("Adding LoRA adapters with PEFT")
    print("=" * 100)

    model = add_lora_adapters(model, lora_config)

    print_trainable_parameters(model)
    print_gpu_memory("After LoRA adapter loading")

    print("=" * 100)
    print("Model debug finished successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()