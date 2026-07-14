# src/arm_llm/schedulers.py

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler


def compute_warmup_steps(
    scheduler_config: dict[str, Any],
    num_training_steps: int,
) -> int:
    """
    Compute warmup steps.

    Priority:
        1. If warmup_steps is set, use it.
        2. Otherwise use warmup_ratio * num_training_steps.
    """
    warmup_steps = scheduler_config.get("warmup_steps", None)

    if warmup_steps is not None:
        return int(warmup_steps)

    warmup_ratio = float(scheduler_config.get("warmup_ratio", 0.0))

    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be between 0 and 1, got {warmup_ratio}")

    return int(num_training_steps * warmup_ratio)


def linear_warmup_factor(
    current_step: int,
    warmup_steps: int,
) -> float:
    """
    Linear warmup from 0 to 1.
    """
    if warmup_steps <= 0:
        return 1.0

    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))

    return 1.0


def build_warmup_only_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> LambdaLR:
    """
    Warmup-only schedule.

    LR behavior:
        0 -> 1 during warmup
        stays at 1 after warmup

    If warmup_steps == total training steps,
    the whole stage is one long warmup.
    """

    def lr_lambda(current_step: int) -> float:
        return linear_warmup_factor(current_step, warmup_steps)

    return LambdaLR(optimizer, lr_lambda)

def cosine_decay_factor(
    progress: float,
    min_lr_ratio: float,
) -> float:
    """
    Cosine decay from 1.0 to min_lr_ratio.

    progress:
        0.0 -> start of decay
        1.0 -> end of decay
    """
    progress = min(max(progress, 0.0), 1.0)

    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_value


def linear_decay_factor(
    progress: float,
    min_lr_ratio: float,
) -> float:
    """
    Linear decay from 1.0 to min_lr_ratio.
    """
    progress = min(max(progress, 0.0), 1.0)

    return 1.0 - progress * (1.0 - min_lr_ratio)


def sine_decay_factor(
    progress: float,
    min_lr_ratio: float,
) -> float:
    """
    Sine-style decay from 1.0 to min_lr_ratio.

    This is a custom schedule, useful for experimentation.
    It decays slowly at first, faster later.
    """
    progress = min(max(progress, 0.0), 1.0)

    sine_value = math.sin((1.0 - progress) * math.pi / 2.0)

    return min_lr_ratio + (1.0 - min_lr_ratio) * sine_value


def build_cosine_min_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """
    Warmup + cosine decay to non-zero minimum LR.

    Schedule:
        warmup: 0 -> 1
        decay:  1 -> min_lr_ratio
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return linear_warmup_factor(current_step, warmup_steps)

        decay_steps = max(1, num_training_steps - warmup_steps)
        progress = (current_step - warmup_steps) / decay_steps

        return cosine_decay_factor(progress, min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


def build_sine_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """
    Warmup + custom sine decay.

    This exists because you said you want to easily test
    schedules like linear vs sine.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return linear_warmup_factor(current_step, warmup_steps)

        decay_steps = max(1, num_training_steps - warmup_steps)
        progress = (current_step - warmup_steps) / decay_steps

        return sine_decay_factor(progress, min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


def build_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    stable_ratio: float,
    decay_ratio: float,
    min_lr_ratio: float,
    decay_type: str,
) -> LambdaLR:
    """
    WSD = Warmup Stable Decay.

    Schedule:
        1. Warmup: 0 -> 1
        2. Stable: stay at 1
        3. Decay: 1 -> min_lr_ratio

    This is useful when you want a 3-stage schedule.
    """
    if stable_ratio < 0 or decay_ratio < 0:
        raise ValueError("stable_ratio and decay_ratio must be non-negative.")

    if stable_ratio + decay_ratio > 1.0:
        raise ValueError("stable_ratio + decay_ratio must be <= 1.0.")

    remaining_steps = max(1, num_training_steps - warmup_steps)

    stable_steps = int(remaining_steps * stable_ratio)
    decay_steps = int(remaining_steps * decay_ratio)

    # If ratios leave unused steps, give them to stable phase.
    used_steps = stable_steps + decay_steps
    if used_steps < remaining_steps:
        stable_steps += remaining_steps - used_steps

    decay_start = warmup_steps + stable_steps
    decay_end = decay_start + max(1, decay_steps)

    decay_type = decay_type.lower()

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return linear_warmup_factor(current_step, warmup_steps)

        if current_step < decay_start:
            return 1.0

        progress = (current_step - decay_start) / max(1, decay_end - decay_start)

        if decay_type == "linear":
            return linear_decay_factor(progress, min_lr_ratio)

        if decay_type == "cosine":
            return cosine_decay_factor(progress, min_lr_ratio)

        if decay_type == "sine":
            return sine_decay_factor(progress, min_lr_ratio)

        raise ValueError(
            f"Unsupported WSD decay_type: {decay_type}. "
            "Use linear, cosine, or sine."
        )

    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build LR scheduler from config.

    Supported:
        - linear
        - cosine
        - cosine_min_lr
        - wsd
        - sine_decay

    linear/cosine use HF get_scheduler.
    custom ones use PyTorch LambdaLR.
    """
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be positive, got {num_training_steps}")

    name = scheduler_config.get("name", "linear").lower()
    warmup_steps = compute_warmup_steps(scheduler_config, num_training_steps)

    min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.0))

    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be between 0 and 1, got {min_lr_ratio}")

    if name in {"linear", "cosine", "constant", "constant_with_warmup"}:
        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    if name == "warmup_only":
        return build_warmup_only_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
        )
        
    if name == "cosine_min_lr":
        return build_cosine_min_lr_scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

    if name == "sine_decay":
        return build_sine_decay_scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

    if name == "wsd":
        return build_wsd_scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            stable_ratio=float(scheduler_config.get("stable_ratio", 0.8)),
            decay_ratio=float(scheduler_config.get("decay_ratio", 0.15)),
            min_lr_ratio=min_lr_ratio,
            decay_type=scheduler_config.get("decay_type", "cosine"),
        )

    raise ValueError(
    f"Unsupported scheduler: {name}. "
    "Supported: linear, cosine, constant, constant_with_warmup, "
    "warmup_only, cosine_min_lr, wsd, sine_decay"
    )