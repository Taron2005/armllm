# src/arm_llm/callbacks.py

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from arm_llm.custom_eval import CustomBenchmarkEvaluator


class CustomEvaluationCallback(TrainerCallback):
    """
    Runs custom benchmark evaluation every N optimizer steps.

    This is separate from normal Trainer eval_loss.

    Normal eval:
        uses the SFT validation split

    Custom eval:
        uses Wikipedia and Tatoeba benchmark datasets
    """

    def __init__(
        self,
        evaluator: CustomBenchmarkEvaluator,
        log_to_wandb: bool,
    ) -> None:
        self.evaluator = evaluator
        self.log_to_wandb = log_to_wandb
        self.last_logged_step = -1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        step = int(state.global_step)

        if not self._should_run(step):
            return control

        model = kwargs.get("model")
        if model is None:
            print("Custom evaluation skipped: model not found in callback kwargs.")
            return control

        self._run_and_log(model=model, step=step)

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        step = int(state.global_step)

        model = kwargs.get("model")
        if model is None:
            return control

        if step != self.last_logged_step:
            self._run_and_log(model=model, step=step)

        return control

    def _should_run(self, step: int) -> bool:
        if not self.evaluator.enabled:
            return False

        if step <= 0:
            return False

        if step == self.last_logged_step:
            return False

        return step % self.evaluator.eval_steps == 0

    def _run_and_log(self, model: Any, step: int) -> None:
        print("=" * 100)
        print(f"Running custom benchmark evaluation at step {step}")
        print("=" * 100)

        metrics = self.evaluator.evaluate(
            model=model,
            global_step=step,
        )

        self.last_logged_step = step

        if not metrics:
            print("Custom benchmark evaluation returned no metrics.")
            return

        for key, value in metrics.items():
            print(f"{key}: {value}")

        if self.log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    payload = {
                        **metrics,
                        "train/global_step": step,
                    }

                    wandb.log(payload, step=step)   # <-- pin to the real optimizer step

                    print(
                        f"Logged {len(metrics)} custom metrics to W&B "
                        f"at optimizer step {step}."
                    )
                else:
                    print("W&B run is not active; custom metrics printed only.")

            except Exception as exc:
                print(f"Could not log custom metrics to W&B: {exc}")