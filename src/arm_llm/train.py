# src/arm_llm/train.py

from __future__ import annotations #this is just for type hints

import argparse #this is used to read the config from the command line
import os #os is used for making env variables for Wandb enabled or disbaled for example
import sys
from pathlib import Path #this is used just for making the output dirs
from typing import Any

import torch #cuda checks and and the dtype config is done with torch here 
from transformers import Trainer, TrainingArguments, set_seed

from arm_llm.collator import CausalLMCollator #the collator is used to do masking on the inputs so that only the assistant answer is useed for loss compuatation
from arm_llm.config import load_yaml_config, require_key #just loads the yaml config reuire key is about these project, model, lora, data, sequence, loss, batching, optimizer, scheduler, training
from arm_llm.data import (
    label_mask_report,
    load_jsonl_dataset, #reads the dataset 
    tokenize_dataset,  
    validate_text_dataset, #checks if the dataset has teh column named text_field and that rows are not empty
    
)
from arm_llm.data import filter_zero_label_rows #this just handles the samples that have not active labels
from arm_llm.model import (
    attach_or_load_lora_adapters,
    load_base_model, 
    load_tokenizer,
    print_gpu_memory,
    print_trainable_parameters,
)
from arm_llm.optim import build_optimizer
from arm_llm.schedulers import build_scheduler, compute_warmup_steps
from arm_llm.training_math import (
    compute_effective_batch_size,
    compute_num_training_steps,
    compute_num_update_steps_per_epoch,
)

from arm_llm.callbacks import CustomEvaluationCallback
from arm_llm.custom_eval import CustomBenchmarkEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native HF QLoRA SFT training.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def setup_wandb(wandb_config: dict[str, Any]) -> list[str]:
    """
    Configure W&B environment.

    Returns value for TrainingArguments.report_to.
    """
    enabled = bool(wandb_config.get("enabled", False))

    if not enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return []

    project = wandb_config.get("project")
    run_name = wandb_config.get("run_name")
    entity = wandb_config.get("entity")

    if project:
        os.environ["WANDB_PROJECT"] = str(project)

    if run_name:
        os.environ["WANDB_NAME"] = str(run_name)

    if entity:
        os.environ["WANDB_ENTITY"] = str(entity)

    return ["wandb"]


def validate_cuda_for_training(training_config: dict[str, Any]) -> None:
    """
    Basic CUDA/GPU check.

    This script is meant for GPU training.
    """
    print("=" * 100)
    print("CUDA training check")
    print("=" * 100)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training requires GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU: {gpu_name}")
    print(f"Compute capability: {capability}")
    print(f"VRAM: {total_vram:.2f} GB")
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")

    if bool(training_config.get("bf16", False)):
        print("WARNING: bf16=True. GTX 1080 Ti should use bf16=False.")


def print_training_plan(
    *,
    num_examples: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    steps_per_epoch: int,
    total_steps: int,
    warmup_steps: int,
    effective_batch_size: int,
    optimizer_config: dict[str, Any],
    scheduler_config: dict[str, Any],
) -> None:
    print("=" * 100)
    print("Training plan")
    print("=" * 100)
    print(f"Train examples: {num_examples}")
    print(f"Per-device batch size: {batch_size}")
    print(f"Gradient accumulation steps: {grad_accum}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Optimizer steps per epoch: {steps_per_epoch}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Optimizer: {optimizer_config.get('name')}")
    print(f"Peak LR: {optimizer_config.get('learning_rate')}")
    print(f"Scheduler: {scheduler_config.get('name')}")
    print(f"Min LR ratio: {scheduler_config.get('min_lr_ratio', None)}")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    project_config = require_key(config, "project")
    model_config = require_key(config, "model")
    lora_config = require_key(config, "lora")
    data_config = require_key(config, "data")
    sequence_config = require_key(config, "sequence")
    loss_config = require_key(config, "loss")
    batching_config = require_key(config, "batching")
    optimizer_config = require_key(config, "optimizer")
    scheduler_config = require_key(config, "scheduler")
    training_config = require_key(config, "training")
    wandb_config = config.get("wandb", {"enabled": False})
    custom_eval_config = config.get("custom_evaluation", {"enabled": False})

    seed = int(project_config.get("seed", 3407))
    set_seed(seed)

    validate_cuda_for_training(training_config)

    report_to = setup_wandb(wandb_config)

    output_dir = Path(training_config["output_dir"])
    final_adapter_dir = Path(training_config["final_adapter_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    text_field = data_config["dataset_text_field"]
    max_seq_length = int(sequence_config["max_seq_length"])

    print("=" * 100)
    print("Loading tokenizer")
    print("=" * 100)
    tokenizer = load_tokenizer(model_config)
    tokenizer.padding_side = "right"

    print(f"EOS token: {tokenizer.eos_token!r}, id={tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token!r}, id={tokenizer.pad_token_id}")

    print("=" * 100)
    print("Loading raw datasets")
    print("=" * 100)

    train_dataset = load_jsonl_dataset(data_config["train_file"])
    eval_dataset = load_jsonl_dataset(data_config["eval_file"])

    validate_text_dataset(train_dataset, text_field, "train")
    validate_text_dataset(eval_dataset, text_field, "eval")

    print(f"Train rows: {len(train_dataset)}")
    print(f"Eval rows: {len(eval_dataset)}")

    print("=" * 100)
    print("Tokenizing datasets and building labels")
    print("=" * 100)

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
    tokenized_train = filter_zero_label_rows(
        tokenized_dataset=tokenized_train,
        dataset_name="train",
    )

    tokenized_eval = filter_zero_label_rows(
        tokenized_dataset=tokenized_eval,
        dataset_name="eval",
    )
    label_mask_report(tokenized_train, "train")
    label_mask_report(tokenized_eval, "eval")

    batch_size = int(batching_config["train_batch_size"])
    eval_batch_size = int(batching_config["eval_batch_size"])
    grad_accum = int(training_config["gradient_accumulation_steps"])
    epochs = float(training_config["num_train_epochs"])

    effective_batch_size = compute_effective_batch_size(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        world_size=1,
    )

    steps_per_epoch = compute_num_update_steps_per_epoch(
        num_examples=len(tokenized_train),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        world_size=1,
    )

    total_training_steps = compute_num_training_steps(
        num_examples=len(tokenized_train),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        world_size=1,
    )

    warmup_steps = compute_warmup_steps(
        scheduler_config=scheduler_config,
        num_training_steps=total_training_steps,
    )

    print_training_plan(
        num_examples=len(tokenized_train),
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_training_steps,
        warmup_steps=warmup_steps,
        effective_batch_size=effective_batch_size,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )

    print_gpu_memory("Before model load")

    print("=" * 100)
    print("Loading base model")
    print("=" * 100)

    model = load_base_model(model_config)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if bool(training_config.get("gradient_checkpointing", True)):
        model.config.use_cache = False

    print_gpu_memory("After base model load")

    print("=" * 100)
    print("Attaching/loading LoRA adapter")
    print("=" * 100)

    model = attach_or_load_lora_adapters(model, lora_config)

    print_trainable_parameters(model)
    print_gpu_memory("After LoRA adapter setup")

    pad_to_multiple_of = batching_config.get("pad_to_multiple_of", None)
    if pad_to_multiple_of is not None:
        pad_to_multiple_of = int(pad_to_multiple_of)

    data_collator = CausalLMCollator(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    print("=" * 100)
    print("Building optimizer and scheduler")
    print("=" * 100)

    optimizer = build_optimizer(model, optimizer_config)

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_config=scheduler_config,
        num_training_steps=total_training_steps,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,

        fp16=bool(training_config.get("fp16", True)),
        bf16=bool(training_config.get("bf16", False)),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),

        logging_steps=int(training_config.get("logging_steps", 5)),
        eval_strategy="steps",
        eval_steps=int(training_config.get("eval_steps", 50)),

        save_strategy="steps",
        save_steps=int(training_config.get("save_steps", 50)),
        save_total_limit=int(training_config.get("save_total_limit", 3)),

        report_to=report_to,
        run_name=wandb_config.get("run_name") if report_to else None,

        remove_unused_columns=False,
        dataloader_pin_memory=False,

        learning_rate=float(optimizer_config.get("learning_rate", 2e-4)),

        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )

    if bool(custom_eval_config.get("enabled", False)):
        custom_evaluator = CustomBenchmarkEvaluator(
            config=custom_eval_config,
            tokenizer=tokenizer,
        )

        trainer.add_callback(
            CustomEvaluationCallback(
                evaluator=custom_evaluator,
                log_to_wandb=bool(report_to),
            )
        )

    resume_from_checkpoint = training_config.get("resume_from_checkpoint", None)

    print("=" * 100)
    print("Starting training")
    print("=" * 100)

    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )

    print("=" * 100)
    print("Training finished")
    print("=" * 100)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    print("=" * 100)
    print("Running final evaluation")
    print("=" * 100)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("=" * 100)
    print(f"Saving final adapter to: {final_adapter_dir}")
    print("=" * 100)

    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    print_gpu_memory("After training")

    print("=" * 100)
    print("Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()