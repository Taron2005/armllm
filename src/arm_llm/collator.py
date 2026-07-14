# src/arm_llm/collator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CausalLMCollator:
    """
    Dynamic padding collator for causal LM SFT.

    Input features are already tokenized and contain:
        input_ids: list[int]
        attention_mask: list[int]
        labels: list[int]

    This collator pads a batch to the longest sequence in that batch.

    Padding rules:
        input_ids      -> tokenizer.pad_token_id
        attention_mask -> 0
        labels         -> -100

    Why labels use -100:
        PyTorch CrossEntropyLoss ignores -100 by default.
        So padded tokens do not affect the loss.
    """

    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None. "
                "Set tokenizer.pad_token/tokenizer.pad_token_id before using the collator."
            )

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive or None.")

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty batch passed to collator.")

        required_keys = {"input_ids", "attention_mask", "labels"}

        for i, feature in enumerate(features):
            missing = required_keys - set(feature.keys())
            if missing:
                raise KeyError(f"Feature {i} is missing keys: {missing}")

            input_len = len(feature["input_ids"])
            mask_len = len(feature["attention_mask"])
            label_len = len(feature["labels"])

            if not (input_len == mask_len == label_len):
                raise ValueError(
                    f"Feature {i} has inconsistent lengths: "
                    f"input_ids={input_len}, "
                    f"attention_mask={mask_len}, "
                    f"labels={label_len}"
                )

        max_length = max(len(feature["input_ids"]) for feature in features)

        if self.pad_to_multiple_of is not None:
            max_length = self._round_up_to_multiple(
                value=max_length,
                multiple=self.pad_to_multiple_of,
            )

        input_ids_batch: list[list[int]] = []
        attention_mask_batch: list[list[int]] = []
        labels_batch: list[list[int]] = []

        for feature in features:
            input_ids_batch.append(
                self._pad_list(
                    values=feature["input_ids"],
                    target_length=max_length,
                    pad_value=self.tokenizer.pad_token_id,
                )
            )

            attention_mask_batch.append(
                self._pad_list(
                    values=feature["attention_mask"],
                    target_length=max_length,
                    pad_value=0,
                )
            )

            labels_batch.append(
                self._pad_list(
                    values=feature["labels"],
                    target_length=max_length,
                    pad_value=self.label_pad_token_id,
                )
            )

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }

    def _pad_list(
        self,
        values: list[int],
        target_length: int,
        pad_value: int,
    ) -> list[int]:
        """
        Pad one list according to tokenizer.padding_side.
        """
        pad_length = target_length - len(values)

        if pad_length < 0:
            raise ValueError(
                f"target_length={target_length} is smaller than values length={len(values)}"
            )

        if pad_length == 0:
            return values

        padding = [pad_value] * pad_length
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        if padding_side == "right":
            return values + padding

        if padding_side == "left":
            return padding + values

        raise ValueError(f"Unsupported tokenizer.padding_side: {padding_side}")

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        """
        Round sequence length up to a multiple.

        Example:
            value=513, multiple=8 -> 520
        """
        remainder = value % multiple

        if remainder == 0:
            return value

        return value + multiple - remainder