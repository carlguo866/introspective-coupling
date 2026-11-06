import json
import os
from typing import Dict, Any

import torch
from tqdm import tqdm
from transformers import TrainerCallback


class BaseEvaluator:
    def __init__(
        self,
        config,
        model,
        tokenizer,
        test_dataloader,
        fs_dataloader=None,
        cap_at_100: bool = False,
        **kwargs,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader
        self.fs_dataloader = fs_dataloader
        self.generate_limit = config.get("generate_limit", 20)
        self.main_metric = None  # Set in subclasses
        self.greater_is_better = None  # Set in subclasses
        self.generation_evaluation = True
        self.cap_at_100 = cap_at_100

    def get_prediction(self, outputs, inputs) -> tuple[str, Any]:
        return (
            self.tokenizer.decode(
                outputs["sequences"][len(inputs["input_ids_without_answer"]) :],
                skip_special_tokens=True,
            ).strip(),
            None,
        )

    def evaluate(self, fewshot=False, save_file=None):
        """
        Evaluate the model on the test set.
        """
        predictions = []
        self.reset_metrics()
        self.model.eval()
        # if hasattr(self.model, "generate"):
        #     self.model.loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        if fewshot:
            pbar = tqdm(self.fs_dataloader, desc="Evaluating")
        else:
            pbar = tqdm(self.test_dataloader, desc="Evaluating")

        for batch in pbar:
            if self.cap_at_100 and len(predictions) > 100:
                break
            activations_mask = batch.get("activations_mask", None)
            if hasattr(self.model, "generate"):  # and self.generation_evaluation:
                with torch.no_grad():
                    # get loss - use existing activations_mask from dataloader
                    if activations_mask is not None:
                        activations_mask = activations_mask.to(self.model.device)

                    model_output_dict = self.model(
                        input_ids=batch["input_ids"].to(self.model.device),
                        attention_mask=batch["attention_mask"].to(self.model.device),
                        labels=batch["labels"].to(self.model.device),
                        activations_mask=activations_mask,
                        extra_args=batch.get("extra_args", []),
                    )
                    # Only generate for explanation examples (~activations_mask)
                    if activations_mask is not None and activations_mask.all():
                        # All examples are activation prediction - no generation needed
                        generation_outputs = None
                    elif activations_mask is not None and activations_mask.any():
                        # Mixed batch - only generate for explanation examples
                        explanation_mask = ~activations_mask
                        if explanation_mask.any():
                            # transformers v5 leaves config.architectures None on
                            # wrapper-constructed models; fall back to the class name.
                            _archs = self.model.config.architectures or [
                                type(self.model).__name__
                            ]
                            if any(a.startswith("ContinuousQwen3") for a in _archs):
                                stop_strings = None
                            elif "decode_layer_token" in batch["task"]:
                                stop_strings = ["."]
                                # self.generate_limit = 1
                            else:
                                # For causal tasks, use [END] as stop string instead of eos_token
                                stop_strings = [" [END]", "\n", ">>>."]

                            # Generate only for explanation examples
                            generation_outputs = self.model.generate(
                                input_ids=batch["input_ids_without_answer"].to(
                                    self.model.device
                                )[explanation_mask],
                                attention_mask=batch[
                                    "attention_mask_without_answer"
                                ].to(self.model.device)[explanation_mask],
                                max_new_tokens=self.generate_limit,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=(
                                    None
                                    if "causal" in batch["task"]
                                    else self.tokenizer.eos_token_id
                                ),
                                do_sample=False,
                                temperature=None,
                                top_p=None,
                                return_dict_in_generate=True,
                                output_hidden_states=True,
                                stop_strings=stop_strings,
                                tokenizer=self.tokenizer,
                                extra_args=batch.get("extra_args", []),
                                # **batch.get("extra_args", {}),
                            )
                        else:
                            generation_outputs = None
                    else:
                        # No activations_mask - generate for all (old behavior)
                        # (see note above on v5 architectures=None + Qwen3.5)
                        _archs = self.model.config.architectures or [
                            type(self.model).__name__
                        ]
                        if any(a.startswith("ContinuousQwen3") for a in _archs):
                            stop_strings = None
                        elif "decode_layer_token" in batch["task"]:
                            stop_strings = ["."]
                        # self.generate_limit = 1
                        else:
                            # For causal tasks, use [END] as stop string instead of eos_token
                            stop_strings = [" [END]", "\n", ">>>."]
                        generation_outputs = self.model.generate(
                            input_ids=batch["input_ids_without_answer"].to(
                                self.model.device
                            ),
                            attention_mask=batch["attention_mask_without_answer"].to(
                                self.model.device
                            ),
                            max_new_tokens=self.generate_limit,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=(
                                None
                                if "causal" in batch["task"]
                                else self.tokenizer.eos_token_id
                            ),
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            stop_strings=stop_strings,
                            tokenizer=self.tokenizer,
                            extra_args=batch.get("extra_args", []),
                        )

                    if generation_outputs is not None:
                        generation_outputs["loss"] = model_output_dict["loss"]
            # Track indices for generation outputs
            generation_idx = 0
            metadata = None
            for i in range(len(batch["expected_label"])):
                item_features = {key: batch[key][i] for key in batch.keys()}

                # Check if this is an activation prediction example
                is_activation_example = (
                    activations_mask is not None
                    and i < len(activations_mask)
                    and activations_mask[i].item()
                )

                if hasattr(self.model, "generate") and generation_outputs is not None:
                    # For explanation examples: use generation outputs
                    try:
                        output_features = {}
                        for key in generation_outputs.keys():
                            # past_key_values: no evaluator consumes it, and
                            # slicing it per-item breaks on transformers v5
                            # hybrid caches (e.g. Qwen3.5 DeltaNet layers have
                            # no .keys/.values tensors to iterate).
                            if key in ("hidden_states", "loss", "past_key_values"):
                                continue
                            output_features[key] = generation_outputs[key][
                                generation_idx
                            ]
                        generation_idx += 1  # Increment for next explanation example
                    except Exception as e:
                        # More informative error handling
                        print(
                            f"Error processing generation_outputs for batch item {i}: {e}"
                        )
                        print(
                            f"Available output keys: {list(generation_outputs.keys()) if generation_outputs else 'None'}"
                        )
                        import sys
                        if sys.stdin.isatty():
                            # Interactive — drop to pdb; headless runs must not
                            # die with bdb.BdbQuit, the fallback below handles it.
                            raise
                        output_features = {
                            key: (
                                generation_outputs[key][generation_idx]
                                if generation_outputs and key in generation_outputs
                                else None
                            )
                            for key in ["sequences"]
                        }
                        generation_idx += 1

                    predicted_label, metadata = self.get_prediction(
                        output_features, item_features
                    )
                    if predicted_label is None:
                        continue
                    metrics = self.evaluate_item(
                        predicted_label, batch["expected_label"][i], item_features
                    )
                elif hasattr(self.model, "get_prediction"):
                    # nearest neighbor model
                    predicted_label, metadata = self.model.get_prediction(item_features)
                    if predicted_label is None:
                        continue
                    metrics = self.evaluate_item(
                        predicted_label, batch["expected_label"][i], item_features
                    )
                else:
                    # Handle cases where model doesn't have generate or get_prediction methods
                    print(
                        f"Warning: Model does not have generate() or get_prediction() methods"
                    )
                    predicted_label = None
                    metrics = None
                prediction = {}
                if metadata is not None:
                    prediction["metadata"] = metadata
                prediction["predicted_label"] = predicted_label
                prediction["expected_label"] = batch["expected_label"][i]
                prediction["is_activation_example"] = is_activation_example
                if metadata is not None:
                    prediction["metadata"] = metadata

                def filter_tensors(obj):
                    """Recursively filter out torch.Tensor objects from nested structures."""
                    if isinstance(obj, torch.Tensor):
                        return None
                    elif isinstance(obj, dict):
                        return {
                            k: v
                            for k, v in ((k, filter_tensors(v)) for k, v in obj.items())
                            if v is not None
                        }
                    elif isinstance(obj, (list, tuple)):
                        filtered = [filter_tensors(item) for item in obj]
                        return [item for item in filtered if item is not None]
                    else:
                        return obj

                filtered_features = filter_tensors(item_features)
                prediction.update(filtered_features)
                if metrics is not None:
                    prediction.update(metrics)
                predictions.append(prediction)
                if getattr(self, "output_type", "text") == "text" and save_file:
                    with open(save_file, "a") as f:
                        try:
                            f.write(json.dumps(predictions[-1]) + "\n")
                        except Exception as e:
                            raise
                            print(f"Error writing to file: {e}")
                            print(f"Prediction: {predictions[-1]}")

            self.pbar_update(pbar)
        self.finalize_metrics()
        if getattr(self, "output_type", "text") != "text" and save_file:
            predictions = [p["predicted_label"] for p in predictions]
            with open(save_file.replace(".json", "_preds.pt"), "wb") as f:
                torch.save(predictions, f)
            # Save additional data if available
            try:
                # Save activations if available
                if (
                    predictions
                    and "extra_args" in predictions[0]
                    and "activations" in predictions[0].get("extra_args", {})
                ):
                    activations = [
                        p["extra_args"]["activations"]
                        for p in predictions
                        if "extra_args" in p and "activations" in p["extra_args"]
                    ]
                    if activations:
                        with open(
                            save_file.replace(".json", "_activations.pt"), "wb"
                        ) as f:
                            torch.save(activations, f)
            except Exception as e:
                print(f"Warning: Could not save additional evaluation data: {e}")
        self.model.train()
        # if hasattr(self.model, "generate"):
        #     self.model.loss_function = ForCausalLMLoss
        return predictions

    def parse_predicted_label(self, predicted_label):
        pass

    def pbar_update(self, pbar):
        pass

    def log_metrics(self, prefix="", step=None):
        pass

    def print_metrics(self, prefix=""):
        pass

    def reset_metrics(self):
        """
        Reset metrics to 0
        """

    def finalize_metrics(self):
        """
        Normalize metrics on evaluation end
        """

    def evaluate_item(self, predicted_label, expected_label, full_item):
        """
        Update metrics for a single item.
        """


class EvaluationCallback(TrainerCallback):
    """Callback to evaluate the model at the end of each epoch."""

    def __init__(
        self,
        evaluators: Dict[str, BaseEvaluator],
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        debug_mode=False,
    ):
        self.evaluators = evaluators
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.debug_mode = debug_mode

    def on_step_end(self, args, state, control, **kwargs):
        predictions_dir = os.path.join(os.getcwd(), "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        if self.eval_strategy == "steps" and state.global_step % self.eval_steps == 0:
            
            for evaluator_name, evaluator in self.evaluators.items():
                evaluator.evaluate(
                    fewshot=False,
                    save_file=os.path.join(predictions_dir, f"{evaluator_name}_step_{state.global_step}_predictions.json"),
                )
                if not self.debug_mode:
                    evaluator.log_metrics(
                        prefix=f"per_step_{evaluator_name}_",
                        step=int(state.global_step),
                    )
                evaluator.print_metrics(
                    prefix=f"Step {state.global_step} - {evaluator_name}: "
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        predictions_dir = os.path.join(os.getcwd(), "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        if self.eval_strategy == "epoch":
            for evaluator_name, evaluator in self.evaluators.items():
                evaluator.evaluate(
                    fewshot=False,
                    save_file=os.path.join(predictions_dir, f"{evaluator_name}_epoch_{state.epoch}_predictions.json"),
                )
                if not self.debug_mode:
                    evaluator.log_metrics(
                        prefix=f"per_epoch_{evaluator_name}_",
                        step=int(state.global_step),
                        # epoch=int(state.epoch)
                    )
                evaluator.print_metrics(
                    prefix=f"Epoch {state.epoch} - {evaluator_name}: "
                )

        return control
