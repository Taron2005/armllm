from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import wandb
from peft import PeftModel

from arm_llm.config import load_yaml_config
from arm_llm.model import load_base_model, load_tokenizer


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


@dataclass
class WikiExample:
    example_id: str
    language: str
    prompt_ids: list[int]
    prompt: str
    reference: str


@dataclass
class CheckpointInfo:
    name: str
    path: Path
    step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base model and LoRA checkpoints on Wikipedia continuations."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-language", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--reference-tokens", type=int, default=64)

    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def text_similarity(reference: str, prediction: str) -> float:
    """
    Simple lexical similarity from 0 to 1.

    This is not semantic similarity.
    The W&B table is the main evaluation result.
    """
    return SequenceMatcher(
        None,
        normalize_text(reference),
        normalize_text(prediction),
    ).ratio()


def load_wiki_examples(
    *,
    wikipedia_path: str,
    tokenizer: Any,
    samples_per_language: int,
    prompt_tokens: int,
    reference_tokens: int,
    seed: int,
    text_field: str,
    lang_field: str,
    languages: dict[str, str],
) -> list[WikiExample]:
    df = pd.read_parquet(wikipedia_path)

    examples: list[WikiExample] = []

    required_tokens = prompt_tokens + reference_tokens

    for language_name, language_value in languages.items():
        language_df = df[
            df[lang_field] == language_value
        ].dropna(subset=[text_field])

        # Fixed order so every evaluation uses the same texts.
        language_df = language_df.sample(
            frac=1.0,
            random_state=seed,
        )

        added = 0

        for row_index, row in language_df.iterrows():
            text = str(row[text_field]).strip()

            token_ids = tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]

            if len(token_ids) < required_tokens:
                continue

            prompt_ids = token_ids[:prompt_tokens]

            reference_ids = token_ids[
                prompt_tokens : prompt_tokens + reference_tokens
            ]

            prompt = tokenizer.decode(
                prompt_ids,
                skip_special_tokens=True,
            ).strip()

            reference = tokenizer.decode(
                reference_ids,
                skip_special_tokens=True,
            ).strip()

            examples.append(
                WikiExample(
                    example_id=f"{language_value}-{row_index}",
                    language=language_name,
                    prompt_ids=prompt_ids,
                    prompt=prompt,
                    reference=reference,
                )
            )

            added += 1

            if added >= samples_per_language:
                break

        if added < samples_per_language:
            raise ValueError(
                f"Could only create {added} examples for {language_name}. "
                f"Requested {samples_per_language}."
            )

    return examples


def get_final_step(output_dir: Path) -> int:
    trainer_state_path = output_dir / "trainer_state.json"

    if not trainer_state_path.exists():
        return 0

    with trainer_state_path.open("r", encoding="utf-8") as file:
        trainer_state = json.load(file)

    return int(trainer_state.get("global_step", 0))


def discover_checkpoints(
    output_dir: Path,
    final_adapter_dir: Path,
) -> list[CheckpointInfo]:
    checkpoints: list[CheckpointInfo] = []

    for checkpoint_path in output_dir.glob("checkpoint-*"):
        if not checkpoint_path.is_dir():
            continue

        match = CHECKPOINT_PATTERN.fullmatch(
            checkpoint_path.name
        )

        if match is None:
            continue

        # Make sure this is a PEFT adapter checkpoint.
        if not (
            checkpoint_path / "adapter_config.json"
        ).exists():
            continue

        checkpoints.append(
            CheckpointInfo(
                name=checkpoint_path.name,
                path=checkpoint_path,
                step=int(match.group(1)),
            )
        )

    checkpoints.sort(key=lambda item: item.step)

    # Also evaluate the explicitly saved final adapter.
    if (
        final_adapter_dir.exists()
        and (final_adapter_dir / "adapter_config.json").exists()
    ):
        checkpoints.append(
            CheckpointInfo(
                name="final_adapter",
                path=final_adapter_dir,
                step=get_final_step(output_dir),
            )
        )

    if not checkpoints:
        raise ValueError(
            f"No LoRA checkpoints found inside {output_dir}"
        )

    return checkpoints


def generate_answer(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> str:
    device = next(model.parameters()).device

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    prompt_length = input_ids.shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, prompt_length:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


def adapter_name(checkpoint: CheckpointInfo) -> str:
    return f"adapter_{checkpoint.step}_{checkpoint.name}"


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)

    project_config = config["project"]
    model_config = config["model"]
    training_config = config["training"]
    wandb_config = config["wandb"]

    custom_eval_config = config["custom_evaluation"]
    wikipedia_config = custom_eval_config["wikipedia"]

    seed = int(project_config.get("seed", 3407))

    output_dir = Path(training_config["output_dir"])
    final_adapter_dir = Path(
        training_config["final_adapter_dir"]
    )

    print("=" * 100)
    print("Loading tokenizer")
    print("=" * 100)

    tokenizer = load_tokenizer(model_config)

    examples = load_wiki_examples(
        wikipedia_path=wikipedia_config["path"],
        tokenizer=tokenizer,
        samples_per_language=args.samples_per_language,
        prompt_tokens=args.prompt_tokens,
        reference_tokens=args.reference_tokens,
        seed=seed,
        text_field=wikipedia_config.get(
            "text_field",
            "text",
        ),
        lang_field=wikipedia_config.get(
            "lang_field",
            "lang",
        ),
        languages=wikipedia_config.get(
            "languages",
            {
                "armenian": "hy",
                "english": "en",
            },
        ),
    )

    checkpoints = discover_checkpoints(
        output_dir=output_dir,
        final_adapter_dir=final_adapter_dir,
    )

    print("=" * 100)
    print("Evaluation plan")
    print("=" * 100)
    print(f"Wikipedia examples: {len(examples)}")
    print(f"Checkpoints: {len(checkpoints)}")

    for checkpoint in checkpoints:
        print(
            f"{checkpoint.name}: "
            f"step={checkpoint.step}, "
            f"path={checkpoint.path}"
        )

    print("=" * 100)
    print("Loading 4-bit base model")
    print("=" * 100)

    base_model = load_base_model(model_config)
    base_model.eval()
    base_model.config.use_cache = True

    # ---------------------------------------------------------
    # Generate base-model answers only once.
    # ---------------------------------------------------------

    print("=" * 100)
    print("Generating base-model answers")
    print("=" * 100)

    base_answers: dict[str, str] = {}
    base_similarities: dict[str, float] = {}

    for example in examples:
        base_answer = generate_answer(
            model=base_model,
            tokenizer=tokenizer,
            prompt_ids=example.prompt_ids,
            max_new_tokens=args.reference_tokens,
        )

        base_answers[example.example_id] = base_answer

        base_similarities[example.example_id] = (
            text_similarity(
                example.reference,
                base_answer,
            )
        )

        print(
            f"{example.example_id}: "
            f"base similarity="
            f"{base_similarities[example.example_id]:.4f}"
        )

    # ---------------------------------------------------------
    # Initialize W&B.
    # ---------------------------------------------------------

    run = wandb.init(
        project=wandb_config["project"],
        name=f"{wandb_config['run_name']}-wiki-checkpoints",
        job_type="checkpoint-evaluation",
        config={
            "base_model": model_config["name"],
            "wikipedia_path": wikipedia_config["path"],
            "samples_per_language": args.samples_per_language,
            "prompt_tokens": args.prompt_tokens,
            "reference_tokens": args.reference_tokens,
            "checkpoint_count": len(checkpoints),
        },
    )

    run.define_metric("checkpoint_step")

    run.define_metric(
        "wiki_checkpoint/*",
        step_metric="checkpoint_step",
    )

    columns = [
        "checkpoint_step",
        "checkpoint_name",
        "example_id",
        "language",
        "prompt",
        "reference_continuation",
        "base_generation",
        "checkpoint_generation",
        "base_similarity",
        "checkpoint_similarity",
        "improvement_over_base",
    ]

    table_rows: list[list[Any]] = []

    # ---------------------------------------------------------
    # Load the first LoRA checkpoint.
    # ---------------------------------------------------------

    first_checkpoint = checkpoints[0]
    first_adapter_name = adapter_name(first_checkpoint)

    model = PeftModel.from_pretrained(
        base_model,
        str(first_checkpoint.path),
        adapter_name=first_adapter_name,
        is_trainable=False,
    )

    model.eval()
    model.config.use_cache = True

    # ---------------------------------------------------------
    # Evaluate all checkpoints.
    # ---------------------------------------------------------

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        current_adapter_name = adapter_name(checkpoint)

        if checkpoint_index > 0:
            model.load_adapter(
                str(checkpoint.path),
                adapter_name=current_adapter_name,
                is_trainable=False,
            )

        # load_adapter does not automatically activate the adapter.
        model.set_adapter(current_adapter_name)
        model.eval()

        checkpoint_similarities: list[float] = []

        print("=" * 100)
        print(
            f"Evaluating {checkpoint.name} "
            f"at step {checkpoint.step}"
        )
        print("=" * 100)

        for example in examples:
            checkpoint_answer = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=example.prompt_ids,
                max_new_tokens=args.reference_tokens,
            )

            base_answer = base_answers[
                example.example_id
            ]

            base_similarity = base_similarities[
                example.example_id
            ]

            checkpoint_similarity = text_similarity(
                example.reference,
                checkpoint_answer,
            )

            improvement = (
                checkpoint_similarity
                - base_similarity
            )

            checkpoint_similarities.append(
                checkpoint_similarity
            )

            table_rows.append(
                [
                    checkpoint.step,
                    checkpoint.name,
                    example.example_id,
                    example.language,
                    example.prompt,
                    example.reference,
                    base_answer,
                    checkpoint_answer,
                    base_similarity,
                    checkpoint_similarity,
                    improvement,
                ]
            )

            print(
                f"{example.example_id}: "
                f"checkpoint={checkpoint_similarity:.4f}, "
                f"improvement={improvement:+.4f}"
            )

        mean_checkpoint_similarity = (
            sum(checkpoint_similarities)
            / len(checkpoint_similarities)
        )

        mean_base_similarity = (
            sum(base_similarities.values())
            / len(base_similarities)
        )

        run.log(
            {
                "checkpoint_step": checkpoint.step,
                "wiki_checkpoint/base_similarity": (
                    mean_base_similarity
                ),
                "wiki_checkpoint/checkpoint_similarity": (
                    mean_checkpoint_similarity
                ),
                "wiki_checkpoint/improvement_over_base": (
                    mean_checkpoint_similarity
                    - mean_base_similarity
                ),
            }
        )

    comparison_table = wandb.Table(
        columns=columns,
        data=table_rows,
    )

    run.log(
        {
            "wiki_checkpoint/comparisons": (
                comparison_table
            )
        }
    )

    run.finish()

    print("=" * 100)
    print("Wikipedia checkpoint evaluation finished")
    print("=" * 100)


if __name__ == "__main__":
    main()