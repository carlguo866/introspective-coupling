import wandb

from evaluator.base_evaluator import BaseEvaluator
from evaluator.causal_exact_match_eval import CausalExactMatchEvaluator
from evaluator.loss_eval import LossEvaluator

EVALUATOR_MAPPING = {
    "loss": LossEvaluator,
    "exact_match": CausalExactMatchEvaluator,
}


class MixedTypeEvaluator(BaseEvaluator):
    def __init__(
        self, config, model, tokenizer, test_dataloader, fs_dataloader=None, **kwargs
    ):
        super().__init__(
            config, model, tokenizer, test_dataloader, fs_dataloader, **kwargs
        )
        # self.options = config["options"]
        # self.label_map = {op: i for i, op in enumerate(self.options)}
        self.generate_limit = config.get(
            "generate_limit", 100
        )  # Longer for open-ended responses

        self.question_evaluators = {}
        self.any_continuous_output = False
        # Create specialized evaluators for different question types
        for question_type in self.config["question_types"]:
            question_config = self.config["question_types"][question_type]
            if "evaluation_type" not in question_config:
                continue
            self.question_evaluators[question_type] = EVALUATOR_MAPPING[
                question_config["evaluation_type"]
            ](
                question_config,
                model,
                tokenizer,
                test_dataloader,
                fs_dataloader,
                **kwargs,
                # inexact="inexact" in question_config["evaluation_type"],
            )
            self.any_continuous_output |= (
                getattr(self.question_evaluators[question_type], "output_type", None)
                == "continuous"
            )
        self.output_type = "continuous" if self.any_continuous_output else "text"

    def reset_metrics(self):
        """Reset all metrics for each question type"""
        for evaluator in self.question_evaluators.values():
            evaluator.reset_metrics()

        # Initialize metrics for each question type
        self.metrics = {
            "total_by_type": {qtype: 0 for qtype in self.question_evaluators.keys()},
            "metric_by_type": {qtype: 0 for qtype in self.question_evaluators.keys()},
            # "total_by_type_layer": {},
            # "metric_by_type_layer": {},
            "total": 0,
        }

    def get_prediction(self, outputs, inputs):
        question_type = inputs.get("question_type", None)
        if question_type is None:
            print(f"Warning: question_type missing in inputs. Keys: {list(inputs.keys())}")
            return None, None
        if question_type not in self.question_evaluators:
            print(f"Warning: question_type '{question_type}' not in question_evaluators. Available: {list(self.question_evaluators.keys())}")
            return None, None
        return self.question_evaluators[question_type].get_prediction(
            outputs, inputs
        )


    def evaluate_item(self, predicted_label, expected_label, full_item):
        """Evaluate based on question type"""
        question_type = full_item["question_type"]
        self.metrics["total"] += 1

        if question_type not in self.question_evaluators:
            print(f"Warning: question_type '{question_type}' not in question_evaluators. Available: {list(self.question_evaluators.keys())}")
            return None
        
        # Always increment total_by_type to track that we saw this question type
        # This ensures metrics are logged even if evaluation fails
        if question_type not in self.metrics["total_by_type"]:
            self.metrics["total_by_type"][question_type] = 0
        if question_type not in self.metrics["metric_by_type"]:
            self.metrics["metric_by_type"][question_type] = 0
        
        try:
            increment_metric = self.question_evaluators[question_type].evaluate_item(
                predicted_label, expected_label, full_item
            )
        except Exception as e:
            print(f"Error evaluating {question_type}: {e}")
            increment_metric = None


        if increment_metric is not None:
            self.metrics["total_by_type"][question_type] += 1
            if self.question_evaluators[question_type].main_metric in increment_metric:
                self.metrics["metric_by_type"][question_type] += increment_metric[
                    self.question_evaluators[question_type].main_metric
                ]
        # self.metrics["correct"] = self.question_evaluators[question_type].metrics["correct"]

        return increment_metric

    def finalize_metrics(self):
        """Calculate final metrics"""
        # Calculate main for each question type
        self.metrics["metric"] = sum(self.metrics["metric_by_type"].values())
        for qtype in self.question_evaluators.keys():
            self.question_evaluators[qtype].finalize_metrics()
            if self.metrics["total_by_type"][qtype] > 0:
                self.metrics[f"{qtype}_metric"] = (
                    self.metrics["metric_by_type"][qtype]
                    / self.metrics["total_by_type"][qtype]
                )
            else:
                self.metrics[f"{qtype}_metric"] = 0.0


    def pbar_update(self, pbar):
        """Update progress bar with current metrics"""
        if self.metrics["total"] > 0:
            # Get main metrics by type
            type_metrics = {
                qtype: (
                    self.metrics["metric_by_type"][qtype]
                    / self.metrics["total_by_type"][qtype]
                    if self.metrics["total_by_type"][qtype] > 0
                    else 0.0
                )
                for qtype in self.question_evaluators.keys()
                if self.metrics["total_by_type"][qtype] > 0
            }

            # Get counts by type
            type_counts = {
                qtype: self.metrics["total_by_type"][qtype]
                for qtype in self.question_evaluators.keys()
                if self.metrics["total_by_type"][qtype] > 0
            }

            pbar.set_description(
                " | ".join(
                    [
                        f"{qtype[:3]}: {metric:.2f} ({count})"
                        for qtype, metric, count in zip(
                            type_metrics.keys(),
                            type_metrics.values(),
                            type_counts.values(),
                        )
                    ]
                )
            )

    def log_metrics(self, prefix="", step=None):
        """Log metrics to wandb"""
        # Log overall metrics
        log_metrics = {}

        # Log metrics for each question type
        # print all question types and their metrics
        print(f"Question types and their metrics: {self.metrics}")
        print(f"Total by type: {self.metrics['total_by_type']}")
        print(f"Metric by type: {self.metrics['metric_by_type']}")
        print(f"Question evaluators keys: {self.question_evaluators.keys()}")
        for qtype in self.question_evaluators.keys():
            total_for_qtype = self.metrics["total_by_type"][qtype]
            metric_key = f"{qtype}_metric"
            if total_for_qtype > 0:
                if metric_key in self.metrics:
                    log_metrics[prefix + f"{qtype}/metric"] = self.metrics[metric_key]
                else:
                    # Debug: Metric key doesn't exist - finalize_metrics might not have been called
                    print(f"Warning: {prefix}{qtype} has {total_for_qtype} examples but {metric_key} not in metrics. Available keys: {list(self.metrics.keys())}")
                # log_metrics[prefix + f"count_{qtype}"] = self.metrics["total_by_type"][qtype]
            else:
                # Debug: Log that this question type had no examples
                print(f"Warning: {prefix}{qtype} had 0 examples evaluated (total_by_type={total_for_qtype})")
            self.question_evaluators[qtype].log_metrics(prefix=prefix + f"{qtype}/", step=step)

        # for qtype in self.metrics["total_by_type_layer"].keys():
        #     for layer in self.metrics["total_by_type_layer"][qtype].keys():
        #         if self.metrics["total_by_type_layer"][qtype][layer] > 0:
        #             log_metrics[prefix + f"{qtype}/metric/layer_{layer}"] = (
        #                 self.metrics[f"{qtype}_metric_by_type_layer"][layer]
        #             )
        #         else:
        #             log_metrics[prefix + f"{qtype}/metric/layer_{layer}"] = 0.0
        #         # log_metrics[prefix + f"count_{qtype}_layer_{layer}"] = self.metrics["total_by_type_layer"][qtype][layer]

        wandb.log(log_metrics, step=step)

    def print_metrics(self, prefix=""):
        """Print metrics to console"""
        print(f"{prefix}Results by question type:")
        for qtype in self.question_evaluators.keys():
            if self.metrics["total_by_type"][qtype] > 0:
                print(
                    f"  {qtype}: {self.metrics[f'{qtype}_metric']:.4f} ({self.metrics['metric_by_type'][qtype]}/{self.metrics['total_by_type'][qtype]})"
                )
            self.question_evaluators[qtype].print_metrics(prefix=f"    {qtype} ")

