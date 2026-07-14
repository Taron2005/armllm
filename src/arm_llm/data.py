# src/arm_llm/data.py

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


IGNORE_INDEX = -100


def load_jsonl_dataset(file_path: str | Path) -> Dataset:
    """
    Load a JSONL dataset using Hugging Face datasets.

    Expected:
        {"text": "...already Qwen chat formatted..."}
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return load_dataset(
        "json",
        data_files=str(file_path),
        split="train",
    )


def validate_text_dataset(
    dataset: Dataset,
    text_field: str,
    dataset_name: str,
) -> None:
    """
    Validate that the dataset has the already-formatted text field.

    Your staged dataset already contains:
        {"text": "...Qwen chat template..."}
    """
    if text_field not in dataset.column_names:
        raise ValueError(
            f"{dataset_name} is missing text field {text_field!r}. "
            f"Found columns: {dataset.column_names}"
        )

    if len(dataset) == 0:
        raise ValueError(f"{dataset_name} is empty.")

    bad_empty = 0
    bad_not_string = 0
    missing_qwen_markers = 0

    for row in dataset:
        value = row[text_field]

        if not isinstance(value, str):
            bad_not_string += 1
            continue

        if not value.strip():
            bad_empty += 1
            continue

        if "<|im_start|>" not in value or "<|im_end|>" not in value:
            missing_qwen_markers += 1

    if bad_not_string:
        raise ValueError(f"{dataset_name}: {bad_not_string} rows have non-string {text_field!r}.")

    if bad_empty:
        raise ValueError(f"{dataset_name}: {bad_empty} rows have empty {text_field!r}.")

    if missing_qwen_markers:
        raise ValueError(
            f"{dataset_name}: {missing_qwen_markers} rows do not look Qwen-chat-formatted. "
            "Expected <|im_start|> and <|im_end|> markers."
        )


def print_preview(
    dataset: Dataset,
    text_field: str,
    num_examples: int,
    dataset_name: str,
) -> None:
    print("=" * 100)
    print(f"Preview: {dataset_name}")
    print("=" * 100)

    for i in range(min(num_examples, len(dataset))):
        text = dataset[i][text_field]
        print(f"\n--- Example {i} ---")
        print(text[:1500])


def token_length_report(
    dataset: Dataset,
    tokenizer: Any,
    text_field: str,
    max_seq_length: int,
    dataset_name: str,
) -> None:
    """
    Compute basic token length statistics.
    """
    lengths: list[int] = []

    for row in dataset:
        token_ids = tokenizer(
            row[text_field],
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(token_ids))

    if not lengths:
        raise ValueError(f"{dataset_name}: no token lengths computed.")

    lengths_sorted = sorted(lengths)

    def percentile(p: float) -> int:
        index = int((len(lengths_sorted) - 1) * p)
        return lengths_sorted[index]

    over_limit = sum(length > max_seq_length for length in lengths)

    print("=" * 100)
    print(f"Token length report: {dataset_name}")
    print("=" * 100)
    print(f"Rows: {len(lengths)}")
    print(f"Min: {min(lengths)}")
    print(f"P50: {percentile(0.50)}")
    print(f"P90: {percentile(0.90)}")
    print(f"P95: {percentile(0.95)}")
    print(f"P99: {percentile(0.99)}")
    print(f"Max: {max(lengths)}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Rows over max_seq_length: {over_limit} / {len(lengths)}")
    print(f"Percent over max_seq_length: {100 * over_limit / len(lengths):.2f}%")


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    """
    Find the first occurrence of pattern in sequence starting from start.
    Return -1 if not found.
    """
    if not pattern:
        return -1

    max_start = len(sequence) - len(pattern)

    for i in range(start, max_start + 1):
        if sequence[i : i + len(pattern)] == pattern:
            return i

    return -1


def build_full_text_labels(input_ids: list[int]) -> list[int]:
    """
    Full-text causal LM labels.

    The model is trained on every token.
    """
    return input_ids.copy()


def build_assistant_only_labels(
    input_ids: list[int],
    response_marker_ids: list[int],
    end_marker_ids: list[int],
) -> list[int]:
    """
    Build labels where only assistant response tokens are trained.

    Everything else becomes -100.

    Example text:

        <|im_start|>user
        ...
        <|im_end|>
        <|im_start|>assistant
        ANSWER TOKENS
        <|im_end|>

    We label:
        ANSWER TOKENS + <|im_end|>

    We mask:
        user tokens, role tokens, prompt markers.
    """
    labels = [IGNORE_INDEX] * len(input_ids)

    search_start = 0

    while True:
        response_marker_start = _find_subsequence(
            input_ids,
            response_marker_ids,
            start=search_start,
        )

        if response_marker_start == -1:
            break

        answer_start = response_marker_start + len(response_marker_ids)

        end_marker_start = _find_subsequence(
            input_ids,
            end_marker_ids,
            start=answer_start,
        )

        if end_marker_start == -1:
            # If the answer was truncated and <|im_end|> is missing,
            # train until the end of the sequence.
            answer_end = len(input_ids)
            search_start = len(input_ids)
        else:
            # Include <|im_end|> in labels so the model learns when to stop.
            answer_end = end_marker_start + len(end_marker_ids)
            search_start = answer_end

        for i in range(answer_start, answer_end):
            labels[i] = input_ids[i]

    return labels


def tokenize_one_example(
    example: dict[str, Any],
    tokenizer: Any,
    text_field: str,
    max_seq_length: int,
    loss_mode: str,
    response_marker_ids: list[int],
    end_marker_ids: list[int],
) -> dict[str, list[int]]:
    """
    Tokenize one row and create labels.

    This function does not pad.
    Padding will be done dynamically by the collator.
    """
    text = example[text_field]

    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    if loss_mode == "full_text":
        labels = build_full_text_labels(input_ids)

    elif loss_mode == "assistant_only":
        labels = build_assistant_only_labels(
            input_ids=input_ids,
            response_marker_ids=response_marker_ids,
            end_marker_ids=end_marker_ids,
        )

    else:
        raise ValueError(
            f"Unsupported loss mode: {loss_mode}. "
            "Use 'full_text' or 'assistant_only'."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Any,
    text_field: str,
    max_seq_length: int,
    loss_config: dict[str, Any],
    dataset_name: str,
) -> Dataset:
    """
    Tokenize a dataset and create labels.
    """
    loss_mode = loss_config.get("mode", "assistant_only")

    response_marker = loss_config.get("response_marker", "<|im_start|>assistant\n")
    end_marker = loss_config.get("end_marker", "<|im_end|>")

    response_marker_ids = tokenizer(
        response_marker,
        add_special_tokens=False,
    )["input_ids"]

    end_marker_ids = tokenizer(
        end_marker,
        add_special_tokens=False,
    )["input_ids"]

    print("=" * 100)
    print(f"Tokenizing dataset: {dataset_name}")
    print("=" * 100)
    print(f"Loss mode: {loss_mode}")
    print(f"Response marker: {response_marker!r}")
    print(f"Response marker ids: {response_marker_ids}")
    print(f"End marker: {end_marker!r}")
    print(f"End marker ids: {end_marker_ids}")

    tokenized = dataset.map(
        lambda example: tokenize_one_example(
            example=example,
            tokenizer=tokenizer,
            text_field=text_field,
            max_seq_length=max_seq_length,
            loss_mode=loss_mode,
            response_marker_ids=response_marker_ids,
            end_marker_ids=end_marker_ids,
        ),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {dataset_name}",
    )

    return tokenized


def label_mask_report(
    tokenized_dataset: Dataset,
    dataset_name: str,
) -> None:
    """
    Print how many tokens are actually used for loss.

    If assistant_only mode works, not all labels should be active.
    """
    total_tokens = 0
    active_label_tokens = 0
    zero_active_rows = 0

    for row in tokenized_dataset:
        labels = row["labels"]
        total_tokens += len(labels)

        row_active = sum(label != IGNORE_INDEX for label in labels)
        active_label_tokens += row_active

        if row_active == 0:
            zero_active_rows += 1

    active_percent = 100 * active_label_tokens / total_tokens if total_tokens else 0

    print("=" * 100)
    print(f"Label mask report: {dataset_name}")
    print("=" * 100)
    print(f"Rows: {len(tokenized_dataset)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Active label tokens: {active_label_tokens}")
    print(f"Active label percent: {active_percent:.2f}%")
    print(f"Rows with zero active labels: {zero_active_rows}")

    if zero_active_rows > 0:
        print(
            "WARNING: Some rows have zero active labels. "
            "This usually means the assistant marker was not found after tokenization/truncation."
        )

def filter_zero_label_rows(
    tokenized_dataset: Dataset,
    dataset_name: str,
) -> Dataset:
    """
    Remove rows where all labels are -100.

    Such rows produce NaN loss because CrossEntropyLoss has no valid target tokens.
    """
    before = len(tokenized_dataset)

    filtered = tokenized_dataset.filter(
        lambda row: any(label != IGNORE_INDEX for label in row["labels"]),
        desc=f"Filtering zero-label rows from {dataset_name}",
    )

    after = len(filtered)
    removed = before - after

    print("=" * 100)
    print(f"Zero-label filter: {dataset_name}")
    print("=" * 100)
    print(f"Before: {before}")
    print(f"After: {after}")
    print(f"Removed: {removed}")

    if after == 0:
        raise ValueError(
            f"{dataset_name}: all rows were removed after zero-label filtering."
        )

    return filtered