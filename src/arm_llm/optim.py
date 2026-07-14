# src/arm_llm/optim.py

from __future__ import annotations

from typing import Any

import torch


def get_trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """
    Return only parameters that require gradients.

    In QLoRA/LoRA training, this should mostly be LoRA adapter weights,
    not the frozen base model.
    """
    return [param for param in model.parameters() if param.requires_grad]


def build_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Build optimizer from config.

    First baseline:
        AdamW

    Why AdamW:
        It is the standard stable optimizer for Transformer fine-tuning.
    """
    name = optimizer_config.get("name", "adamw").lower()

    learning_rate = float(optimizer_config.get("learning_rate", 2e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    beta1 = float(optimizer_config.get("beta1", 0.9))
    beta2 = float(optimizer_config.get("beta2", 0.999))
    eps = float(optimizer_config.get("eps", 1e-8))

    params = get_trainable_parameters(model)

    if not params:
        raise ValueError(
            "No trainable parameters found. "
            "This usually means LoRA adapters were not attached correctly."
        )

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {name}. "
        "Currently supported: adamw"
    )