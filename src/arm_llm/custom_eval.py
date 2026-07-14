# src/arm_llm/custom_eval.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F


def safe_exp(value: float, max_value: float = 20.0) -> float:
    """
    Safe exp for perplexity.

    If loss is too large, exp(loss) can overflow.
    """
    return float(math.exp(min(value, max_value)))


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get device where model parameters live.
    """
    return next(model.parameters()).device


def deterministic_sample_df(
    df: pd.DataFrame,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Fixed sample for reproducible benchmark curves.

    We do not change the dataset. We only select a deterministic subset in memory.
    """
    if max_samples <= 0:
        return df.iloc[0:0]

    if len(df) <= max_samples:
        return df.reset_index(drop=True)

    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def pad_token_id_lists(
    sequences: list[list[int]],
    pad_token_id: int,
    label_pad_token_id: int = -100,
) -> dict[str, torch.Tensor]:
    """
    Dynamic padding for benchmark LM blocks.

    input_ids:
        padded with tokenizer.pad_token_id

    attention_mask:
        1 for real token, 0 for pad

    labels:
        same as input_ids for real tokens
        -100 for pad tokens
    """
    if not sequences:
        raise ValueError("No sequences to pad.")

    max_len = max(len(x) for x in sequences)

    input_ids_batch: list[list[int]] = []
    attention_mask_batch: list[list[int]] = []
    labels_batch: list[list[int]] = []

    for ids in sequences:
        pad_len = max_len - len(ids)

        input_ids_batch.append(ids + [pad_token_id] * pad_len)
        attention_mask_batch.append([1] * len(ids) + [0] * pad_len)
        labels_batch.append(ids + [label_pad_token_id] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }


class CustomBenchmarkEvaluator:
    """
    Custom benchmark evaluator for Armenian LLM training.

    It supports:

    1. Wikipedia LM evaluation:
        - plain text causal LM loss
        - perplexity
        - split by language

    2. Tatoeba cross-lingual similarity:
        - Armenian sentence embedding
        - English sentence embedding
        - cosine similarity

    This class only reads benchmark files.
    It does not modify benchmark datasets.
    """

    def __init__(
        self,
        config: dict[str, Any],
        tokenizer: Any,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer

        self.enabled = bool(config.get("enabled", False))
        self.eval_steps = int(config.get("eval_steps", 100))
        self.seed = int(config.get("seed", 3407))
        self.batch_size = int(config.get("batch_size", 1))

        if self.eval_steps <= 0:
            raise ValueError("custom_evaluation.eval_steps must be positive.")

        if self.batch_size <= 0:
            raise ValueError("custom_evaluation.batch_size must be positive.")

        self.wikipedia_config = config.get("wikipedia", {})
        self.tatoeba_config = config.get("tatoeba", {})

        self.wikipedia_df: pd.DataFrame | None = None
        self.tatoeba_pairs: list[tuple[str, str]] = []

        if self.enabled:
            self._load_benchmarks()

    def _load_benchmarks(self) -> None:
        if bool(self.wikipedia_config.get("enabled", False)):
            path = Path(self.wikipedia_config["path"])
            if not path.exists():
                raise FileNotFoundError(f"Wikipedia benchmark not found: {path}")

            self.wikipedia_df = pd.read_parquet(path)

            text_field = self.wikipedia_config.get("text_field", "text")
            lang_field = self.wikipedia_config.get("lang_field", "lang")

            required = {text_field, lang_field}
            missing = required - set(self.wikipedia_df.columns)
            if missing:
                raise ValueError(
                    f"Wikipedia benchmark missing columns: {missing}. "
                    f"Found columns: {list(self.wikipedia_df.columns)}"
                )

        if bool(self.tatoeba_config.get("enabled", False)):
            path = Path(self.tatoeba_config["path"])
            if not path.exists():
                raise FileNotFoundError(f"Tatoeba benchmark not found: {path}")

            df = pd.read_parquet(path)

            translation_field = self.tatoeba_config.get("translation_field", "translation")
            armenian_key = self.tatoeba_config.get("armenian_key", "hy")
            english_key = self.tatoeba_config.get("english_key", "en")
            max_pairs = int(self.tatoeba_config.get("max_pairs", 128))

            if translation_field not in df.columns:
                raise ValueError(
                    f"Tatoeba benchmark missing column {translation_field!r}. "
                    f"Found columns: {list(df.columns)}"
                )

            pairs: list[tuple[str, str]] = []

            sampled_df = deterministic_sample_df(
                df=df,
                max_samples=max_pairs,
                seed=self.seed,
            )

            for row in sampled_df.itertuples(index=False):
                translation = getattr(row, translation_field)

                if not isinstance(translation, dict):
                    continue

                hy = translation.get(armenian_key)
                en = translation.get(english_key)

                if not hy or not en:
                    continue

                pairs.append((str(hy), str(en)))

            self.tatoeba_pairs = pairs

    def evaluate(self, model: torch.nn.Module, global_step: int) -> dict[str, float]:
        """
        Run all enabled custom evaluations.

        Returns a flat metrics dict ready for W&B logging.
        """
        if not self.enabled:
            return {}

        was_training = model.training
        model.eval()

        metrics: dict[str, float] = {}

        try:
            with torch.no_grad():
                if bool(self.wikipedia_config.get("enabled", False)):
                    metrics.update(self.evaluate_wikipedia(model))

                if bool(self.tatoeba_config.get("enabled", False)):
                    metrics.update(self.evaluate_tatoeba(model))

            metrics["custom_evaluation/global_step"] = float(global_step)
            return metrics

        finally:
            if was_training:
                model.train()

    def evaluate_wikipedia(self, model: torch.nn.Module) -> dict[str, float]:
        """
        Evaluate plain-text LM loss and perplexity on Wikipedia.

        Produces:
            - separate Armenian loss/perplexity
            - separate English loss/perplexity
            - token-weighted average loss/perplexity
            - macro average loss/perplexity

        This is evaluation only:
            - no generation
            - no chat template
            - no assistant-only masking
            - no backpropagation
        """
        if self.wikipedia_df is None:
            return {}

        text_field = self.wikipedia_config.get("text_field", "text")
        lang_field = self.wikipedia_config.get("lang_field", "lang")

        languages = self.wikipedia_config.get(
            "languages",
            {
                "armenian": "hy",
                "english": "en",
            },
        )

        max_samples = int(self.wikipedia_config.get("max_samples_per_language", 64))
        max_chunks = int(self.wikipedia_config.get("max_chunks_per_language", 128))
        block_size = int(self.wikipedia_config.get("block_size", 512))

        metrics: dict[str, float] = {}

        weighted_loss_numerator = 0.0
        weighted_target_tokens = 0

        language_losses: list[float] = []
        language_perplexities: list[float] = []

        for language_name, language_value in languages.items():
            lang_df = self.wikipedia_df[self.wikipedia_df[lang_field] == language_value]

            lang_df = deterministic_sample_df(
                df=lang_df,
                max_samples=max_samples,
                seed=self.seed,
            )

            texts = lang_df[text_field].dropna().astype(str).tolist()

            result = self._language_modeling_loss(
                model=model,
                texts=texts,
                block_size=block_size,
                max_chunks=max_chunks,
            )

            if result is None:
                continue

            loss, target_tokens = result
            perplexity = safe_exp(loss)

            prefix = f"custom_evaluation/wikipedia_{language_value}"

            metrics[f"{prefix}_loss"] = float(loss)
            metrics[f"{prefix}_perplexity"] = float(perplexity)
            metrics[f"{prefix}_target_tokens"] = float(target_tokens)

            weighted_loss_numerator += loss * target_tokens
            weighted_target_tokens += target_tokens

            language_losses.append(float(loss))
            language_perplexities.append(float(perplexity))

        # Token-weighted average across Armenian + English.
        # This is the mathematically better global LM loss.
        if weighted_target_tokens > 0:
            avg_loss = weighted_loss_numerator / weighted_target_tokens
            avg_perplexity = safe_exp(avg_loss)

            metrics["custom_evaluation/wikipedia_avg_loss"] = float(avg_loss)
            metrics["custom_evaluation/wikipedia_avg_perplexity"] = float(avg_perplexity)
            metrics["custom_evaluation/wikipedia_total_target_tokens"] = float(weighted_target_tokens)

        # Macro average gives equal weight to each language, regardless of token count.
        # Useful for dashboards because Armenian and English are equally visible.
        if language_losses:
            macro_avg_loss = sum(language_losses) / len(language_losses)
            macro_avg_perplexity = safe_exp(macro_avg_loss)

            metrics["custom_evaluation/wikipedia_macro_avg_loss"] = float(macro_avg_loss)
            metrics["custom_evaluation/wikipedia_macro_avg_perplexity"] = float(macro_avg_perplexity)

            # This is only the arithmetic mean of per-language perplexities.
            # It is useful as a dashboard number, but avg_perplexity above is more correct.
            metrics["custom_evaluation/wikipedia_macro_avg_perplexity_raw"] = float(
                sum(language_perplexities) / len(language_perplexities)
            )

        return metrics
    def _language_modeling_loss(
        self,
        model: torch.nn.Module,
        texts: list[str],
        block_size: int,
        max_chunks: int,
    ) -> tuple[float, int] | None:
        """
        Compute weighted causal LM cross-entropy loss over fixed text chunks.

        Returns:
            (mean_loss, total_target_tokens)

        The model internally shifts labels for causal LM loss.
        For a block of length N, there are N - 1 predicted target tokens.
        """
        device = get_model_device(model)
        pad_id = int(self.tokenizer.pad_token_id)

        blocks: list[list[int]] = []

        for text in texts:
            token_ids = self.tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]

            if len(token_ids) < 2:
                continue

            for start in range(0, len(token_ids), block_size):
                block = token_ids[start : start + block_size]

                if len(block) < 2:
                    continue

                blocks.append(block)

                if len(blocks) >= max_chunks:
                    break

            if len(blocks) >= max_chunks:
                break

        if not blocks:
            return None

        total_loss = 0.0
        total_target_tokens = 0

        for start in range(0, len(blocks), self.batch_size):
            batch_blocks = blocks[start : start + self.batch_size]

            batch = pad_token_id_lists(
                sequences=batch_blocks,
                pad_token_id=pad_id,
                label_pad_token_id=-100,
            )

            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )

            batch_target_tokens = sum(max(0, len(ids) - 1) for ids in batch_blocks)
            batch_loss = float(outputs.loss.detach().float().item())

            total_loss += batch_loss * batch_target_tokens
            total_target_tokens += batch_target_tokens

        if total_target_tokens == 0:
            return None

        mean_loss = total_loss / total_target_tokens

        return float(mean_loss), int(total_target_tokens)
    def evaluate_tatoeba(self, model: torch.nn.Module) -> dict[str, float]:
        """
        Evaluate Armenian-English representation similarity.

        For each Tatoeba translation pair:
            Armenian sentence -> mean-pooled hidden-state embedding
            English sentence  -> mean-pooled hidden-state embedding
            cosine similarity between the two embeddings

        Final metric:
            mean similarity over all evaluated pairs.
        """
        if not self.tatoeba_pairs:
            return {}

        block_size = int(self.tatoeba_config.get("block_size", 256))

        similarities: list[float] = []

        for start in range(0, len(self.tatoeba_pairs), self.batch_size):
            batch_pairs = self.tatoeba_pairs[start : start + self.batch_size]

            hy_texts = [pair[0] for pair in batch_pairs]
            en_texts = [pair[1] for pair in batch_pairs]

            hy_embeddings = self._sentence_embeddings(
                model=model,
                texts=hy_texts,
                block_size=block_size,
            )

            en_embeddings = self._sentence_embeddings(
                model=model,
                texts=en_texts,
                block_size=block_size,
            )

            # Embeddings are already normalized, so dot product = cosine similarity.
            pair_similarities = (hy_embeddings * en_embeddings).sum(dim=-1)

            similarities.extend(
                pair_similarities.detach().float().cpu().tolist()
            )

        if not similarities:
            return {}

        mean_similarity = sum(similarities) / len(similarities)

        if len(similarities) > 1:
            variance = sum((x - mean_similarity) ** 2 for x in similarities) / len(similarities)
            std_similarity = math.sqrt(variance)
        else:
            std_similarity = 0.0

        return {
            "custom_evaluation/tatoeba_similarity_mean": float(mean_similarity),
            "custom_evaluation/tatoeba_similarity_std": float(std_similarity),
            "custom_evaluation/tatoeba_similarity_min": float(min(similarities)),
            "custom_evaluation/tatoeba_similarity_max": float(max(similarities)),
            "custom_evaluation/tatoeba_pairs": float(len(similarities)),
        }
    def _sentence_embeddings(
        self,
        model: torch.nn.Module,
        texts: list[str],
        block_size: int,
    ) -> torch.Tensor:
        """
        Mean-pool last hidden states over non-padding tokens.
        """
        device = get_model_device(model)

        tokenized = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=block_size,
            padding=True,
            return_tensors="pt",
        )

        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        outputs = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1].float()
        mask = tokenized["attention_mask"].unsqueeze(-1).float()

        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return F.normalize(pooled, p=2, dim=-1)