# src/arm_llm/training_math.py

from __future__ import annotations

import math


def compute_num_update_steps_per_epoch(
    num_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> int:
    """
    Compute optimizer update steps per epoch.

    Important:
    forward/backward mini-steps are not the same as optimizer steps.

    Example:
        num_examples = 2003
        batch_size = 1
        grad_accum = 8

        mini-batches per epoch = 2003
        optimizer steps per epoch = ceil(2003 / 8) = 251
    """
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")

    if per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive.")

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")

    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    effective_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * world_size
    )

    return math.ceil(num_examples / effective_batch_size)


def compute_num_training_steps(
    num_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    world_size: int = 1,
) -> int:
    """
    Compute total optimizer update steps.
    """
    steps_per_epoch = compute_num_update_steps_per_epoch(
        num_examples=num_examples,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
    )

    return math.ceil(steps_per_epoch * num_train_epochs)


def compute_effective_batch_size(
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> int:
    """
    Effective batch size in examples.
    """
    return per_device_train_batch_size * gradient_accumulation_steps * world_size