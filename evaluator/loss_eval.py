import math

import wandb
from evaluator.base_evaluator import BaseEvaluator


class LossEvaluator(BaseEvaluator):
    def __init__(self, config, model, tokenizer, test_dataloader, fs_dataloader=None, **kwargs):
        super().__init__(config, model, tokenizer, test_dataloader, fs_dataloader, **kwargs)
        self.main_metric = "loss"
        self.generation_evaluation = False

    def reset_metrics(self):
        self.metrics = {
            "loss": 0,
            "total": 0,
        }

    def get_prediction(self, outputs, inputs):
        return {
            "loss": outputs["loss"].item(),
        }

    def evaluate_item(self, predicted_label, expected_label, full_item):
        self.metrics["loss"] += predicted_label["loss"]
        self.metrics["total"] += 1
        return {
            "loss": predicted_label["loss"],
        }

    def pbar_update(self, pbar):
        loss_avg = (
            self.metrics["loss"] / self.metrics["total"] if self.metrics["total"] > 0 else math.nan
        )
        pbar.set_description(f"Loss: {loss_avg:.4f}")

    def log_metrics(self, prefix="", step=None):
        loss_avg = (
            self.metrics["loss"] / self.metrics["total"] if self.metrics["total"] > 0 else math.nan
        )
        wandb.log(
            {
                f"{prefix}loss": loss_avg,
            },
            step=step,
        )

    def print_metrics(self, prefix=""):
        loss_avg = (
            self.metrics["loss"] / self.metrics["total"] if self.metrics["total"] > 0 else math.nan
        )
        print(f"{prefix}Loss: {loss_avg:.4f}")

    def finalize_metrics(self):
        pass
