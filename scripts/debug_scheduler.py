# scripts/debug_scheduler.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from arm_llm.config import load_yaml_config, require_key
from arm_llm.optim import build_optimizer
from arm_llm.schedulers import build_scheduler, compute_warmup_steps
from arm_llm.training_math import (
    compute_effective_batch_size,
    compute_num_training_steps,
    compute_num_update_steps_per_epoch,
)


class TinyModel(torch.nn.Module):
    """
    Tiny model only for testing optimizer/scheduler code.

    We do not need to load Qwen just to check LR behavior.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug optimizer and LR scheduler.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2003,
        help="Use stage1 train row count by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    optimizer_config = require_key(config, "optimizer")
    scheduler_config = require_key(config, "scheduler")
    batching_config = require_key(config, "batching")
    training_config = require_key(config, "training")

    batch_size = int(batching_config["train_batch_size"])
    grad_accum = int(training_config["gradient_accumulation_steps"])
    epochs = float(training_config["num_train_epochs"])

    effective_batch_size = compute_effective_batch_size(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        world_size=1,
    )

    steps_per_epoch = compute_num_update_steps_per_epoch(
        num_examples=args.num_examples,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        world_size=1,
    )

    num_training_steps = compute_num_training_steps(
        num_examples=args.num_examples,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        world_size=1,
    )

    warmup_steps = compute_warmup_steps(
        scheduler_config=scheduler_config,
        num_training_steps=num_training_steps,
    )

    print("=" * 100)
    print("Training step math")
    print("=" * 100)
    print(f"num_examples: {args.num_examples}")
    print(f"per_device_train_batch_size: {batch_size}")
    print(f"gradient_accumulation_steps: {grad_accum}")
    print(f"effective_batch_size: {effective_batch_size}")
    print(f"num_train_epochs: {epochs}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"num_training_steps: {num_training_steps}")
    print(f"warmup_steps: {warmup_steps}")

    model = TinyModel()
    optimizer = build_optimizer(model, optimizer_config)

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_config=scheduler_config,
        num_training_steps=num_training_steps,
    )

    base_lr = optimizer_config["learning_rate"]

    print("=" * 100)
    print("LR schedule preview")
    print("=" * 100)
    print(f"scheduler: {scheduler_config['name']}")
    print(f"base LR: {base_lr}")

    selected_steps = sorted(
        set(
            [
                0,
                1,
                warmup_steps,
                int(num_training_steps * 0.25),
                int(num_training_steps * 0.50),
                int(num_training_steps * 0.75),
                num_training_steps - 1,
            ]
        )
    )

    lr_by_step: dict[int, float] = {}

    for step in range(num_training_steps):
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if step in selected_steps:
            lr_by_step[step] = current_lr

    for step, lr in lr_by_step.items():
        print(f"step {step:>6}: lr = {lr:.10f}")

    print("=" * 100)
    print("Scheduler debug finished successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()