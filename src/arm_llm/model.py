# src/arm_llm/model.py

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def attach_or_load_lora_adapters(model: Any, lora_config: dict[str, Any]) -> Any:
    """
    Attach a new LoRA adapter or load an existing trainable adapter.

    Two cases:

    1. Stage 1:
        adapter_checkpoint is null/missing
        -> create new LoRA adapters

    2. Stage 2 / Stage 3:
        adapter_checkpoint points to previous adapter directory
        -> load that adapter and keep it trainable

    This is different from resume_from_checkpoint.
    - adapter_checkpoint = continue from previous stage with new optimizer/LR/scheduler
    - resume_from_checkpoint = resume interrupted same run
    """
    if not bool(lora_config.get("enabled", True)):
        return model

    model = prepare_model_for_kbit_training(model)

    adapter_checkpoint = lora_config.get("adapter_checkpoint", None)

    if adapter_checkpoint:
        print(f"Loading trainable LoRA adapter from: {adapter_checkpoint}")
        model = PeftModel.from_pretrained(
            model,
            adapter_checkpoint,
            is_trainable=True,
        )
        return model

    print("Creating new LoRA adapters.")
    peft_config = build_lora_config(lora_config)
    model = get_peft_model(model, peft_config)

    return model

def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """
    Convert config string to torch dtype.

    For GTX 1080 Ti, use float16.
    Do not use bfloat16 on this GPU.
    """
    normalized = dtype_name.lower()

    if normalized in {"float16", "fp16"}:
        return torch.float16

    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16

    if normalized in {"float32", "fp32"}:
        return torch.float32

    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def build_quantization_config(model_config: dict[str, Any]) -> BitsAndBytesConfig | None:
    """
    Build BitsAndBytesConfig for QLoRA.

    This is native Hugging Face Transformers quantization.
    No Unsloth is used here.
    """
    quant_config = model_config.get("quantization", {})
    load_in_4bit = bool(quant_config.get("load_in_4bit", False))

    if not load_in_4bit:
        return None

    compute_dtype = resolve_torch_dtype(
        quant_config.get("bnb_4bit_compute_dtype", "float16")
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(
            quant_config.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_config: dict[str, Any]) -> Any:
    """
    Load tokenizer from Hugging Face.

    We also force Qwen's <|im_end|> as PAD/EOS later.
    """
    tokenizer_name = model_config.get("tokenizer_name") or model_config["name"]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )

    setup_qwen_tokenizer_tokens(tokenizer)

    return tokenizer


def setup_qwen_tokenizer_tokens(tokenizer: Any) -> None:
    """
    Qwen chat-formatted data uses <|im_end|>.

    We make sure PAD token exists.
    For causal LM training, using EOS as PAD is common.
    """
    qwen_eos_token = "<|im_end|>"
    qwen_eos_id = tokenizer.convert_tokens_to_ids(qwen_eos_token)

    if qwen_eos_id is None or qwen_eos_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(
            f"Qwen EOS token {qwen_eos_token!r} was not found in tokenizer vocab."
        )

    tokenizer.eos_token = qwen_eos_token
    tokenizer.eos_token_id = qwen_eos_id
    tokenizer.pad_token = qwen_eos_token
    tokenizer.pad_token_id = qwen_eos_id


def load_base_model(model_config: dict[str, Any]) -> Any:
    """
    Load base causal LM with optional 4-bit quantization.

    Important:
    - Do not call model.cuda() manually.
    - device_map='auto' lets Transformers/Accelerate place the model.
    """
    model_name = model_config["name"]
    torch_dtype = resolve_torch_dtype(model_config.get("torch_dtype", "float16"))
    quantization_config = build_quantization_config(model_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        dtype=torch_dtype,
        quantization_config=quantization_config,
        # device_map="auto",
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    return model


def build_lora_config(lora_config: dict[str, Any]) -> LoraConfig:
    """
    Build PEFT LoRA config.
    """
    return LoraConfig(
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("alpha", 16)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
        target_modules=list(lora_config["target_modules"]),
    )


def add_lora_adapters(model: Any, lora_config: dict[str, Any]) -> Any:
    """
    Prepare quantized model for k-bit training and attach LoRA adapters.
    """
    if not bool(lora_config.get("enabled", True)):
        return model

    # Required for stable k-bit adapter training.
    model = prepare_model_for_kbit_training(model)

    peft_config = build_lora_config(lora_config)
    model = get_peft_model(model, peft_config)

    return model


def print_trainable_parameters(model: Any) -> None:
    """
    Print trainable vs total parameters.

    PEFT models usually already have print_trainable_parameters(),
    but this fallback is useful for clarity.
    """
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        return

    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    percent = 100 * trainable / total if total > 0 else 0

    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable percent: {percent:.4f}%")


def print_gpu_memory(prefix: str) -> None:
    """
    Print current and max GPU memory usage.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print("=" * 100)
    print(prefix)
    print("=" * 100)
    print(f"Allocated: {allocated:.3f} GB")
    print(f"Reserved: {reserved:.3f} GB")
    print(f"Max allocated: {max_allocated:.3f} GB")
    print(f"Max reserved: {max_reserved:.3f} GB")