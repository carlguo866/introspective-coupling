import argparse
import os
import shutil

import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from dataloaders import TASK_DATALOADER_MAPPING, TASK_DATASET_MAPPING, BaseDataCollator, MixedDataset
from evaluator import EVALUATOR_MAPPING, EvaluationCallback
from model.utils import make_model


def maybe_reset_features(feature_descriptor):
    if feature_descriptor is not None:
        feature_descriptor.reset_features()


class LossLoggingCallback(TrainerCallback):
    """Custom callback to log detailed loss components."""

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        """Called when logging occurs - extract and log detailed losses."""
        if (
            logs is not None
            and hasattr(model, "last_outputs")
            and model.last_outputs is not None
        ):
            outputs = model.last_outputs
            extra_logs = {}

            # Extract individual loss components
            if hasattr(outputs, "text_loss") and outputs.text_loss is not None:
                extra_logs["train/text_loss"] = round(outputs.text_loss.item(), 4)

            if (
                hasattr(outputs, "regularization_loss")
                and outputs.regularization_loss is not None
            ):
                extra_logs["train/regularization_loss"] = round(outputs.regularization_loss.item(), 4)

            if hasattr(outputs, "kl_loss") and outputs.kl_loss is not None:
                extra_logs["train/kl_loss"] = round(outputs.kl_loss.item(), 4)

            if (
                hasattr(outputs, "top1_match_by_qt")
                and outputs.top1_match_by_qt
            ):
                for qt, (correct, total) in outputs.top1_match_by_qt.items():
                    if total > 0:
                        extra_logs[f"train/top1_match/{qt}"] = round(
                            correct / total, 4
                        )

            if hasattr(outputs, "causal_loss") and outputs.causal_loss is not None:
                extra_logs["train/causal_loss"] = round(outputs.causal_loss.item(), 4)

            # Mutate logs in-place; HF's WandbCallback (reordered to fire
            # AFTER us in train.py) will sync these to wandb in its single
            # auto-stepped wandb.log call. Calling wandb.log directly with
            # step=state.global_step is unsafe because HF's WandbCallback
            # logs without step=, which auto-bumps wandb's internal step
            # past us — our explicit step gets rejected as stale.
            logs.update(extra_logs)


def _format_causal_change_string(orig_letter: str, change_letter: str) -> str:
    """Mirror dataloaders/causal.py::_format_causal_change for the rollout
    consistency check. Inputs are single-letter answers (e.g. "A", "B").
    Output matches the supervision string the dataset would produce."""
    orig = "Answer: " + orig_letter.strip()
    change = "Answer: " + change_letter.strip()
    if orig == change:
        return f"The output would remain unchanged from <<<{orig}>>>."
    return f"The most likely output would change to <<<{change}>>>."


class RolloutConsistencyCallback(TrainerCallback):
    """Per-step self-consistency check for `generative_explanation`.

    For every gen_explanation row in the current micro-batch:
      1. Look up the model's argmax letter at the supervised answer position
         for the matching orig_match / change_match rows in the same batch
         (paired by `(extra_args.input, extra_args.hint)`).
      2. For any orig/change letter not present in the batch, run an
         on-the-fly forward over the stashed orig/change prompt strings
         (added to extra_args by dataloaders/causal.py for gen_explanation
         rows) to read the predicted letter.
      3. Run autoregressive `model.generate(...)` on the gen_explanation
         input_ids_without_answer.
      4. Construct the expected string via `_format_causal_change_string` and
         compare lowercased prefixes.
      5. Log `train/rollout_match/generative_explanation` and the orphan
         fallback count to wandb (rank 0 only).

    Gated by `explanation_behavior_match: true` in YAML — propagated through
    `model.log_explanation_behavior_match` which causes
    `model/continuous_base.py::forward` to populate `model.last_inputs`.
    """

    def __init__(self, predictor_tokenizer, generate_limit: int = 20, flip_explanation_direction: bool = False):
        self.tokenizer = predictor_tokenizer
        self.generate_limit = generate_limit
        # Gen rows built with flip_explanation_direction=true have
        # original_response/ablated_response swapped relative to the prompts.
        self.flip_explanation_direction = flip_explanation_direction
        # Stash metrics computed in on_step_end; consumed (and cleared) in on_log.
        self._pending_metrics = None

    @staticmethod
    def _example_key(extra_args_row):
        if not isinstance(extra_args_row, dict):
            return None
        inp = extra_args_row.get("input")
        hint = extra_args_row.get("hint")
        if inp is None or hint is None:
            return None
        return (inp, hint)

    def _decode_letter(self, token_id):
        if token_id is None:
            return None
        return self.tokenizer.decode([int(token_id)]).strip()

    @staticmethod
    def _normalize_letter(resp):
        # Expected labels arrive as " A" on orig/change rows and "Answer: A"
        # on gen rows — reduce both to the bare letter.
        if not isinstance(resp, str):
            return None
        letter = resp.strip()
        if letter.lower().startswith("answer:"):
            letter = letter[len("answer:"):].strip()
        return letter or None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if model is None or not hasattr(model, "last_inputs") or model.last_inputs is None:
            return
        # Only fire at the same cadence as the trainer's on_log so the metric
        # lands on the same wandb step as basic_logs / extra_logs.
        logging_steps = getattr(args, "logging_steps", 1) or 1
        if state.global_step % logging_steps != 0:
            return

        last = model.last_inputs
        question_types = last.get("question_type", [])
        extra_args_list = last.get("extra_args", [])
        pred_token_ids = last.get("pred_token_ids", [])
        input_ids_woa = last.get("input_ids_without_answer")
        attn_mask_woa = last.get("attention_mask_without_answer")

        if input_ids_woa is None or attn_mask_woa is None or not question_types:
            return

        # Build {key: {qt: letter}} from in-batch orig/change rows, plus the
        # frozen M0 expected letters from the same rows' extra_args (identical
        # on the orig and change row of a pair, so either row suffices).
        in_batch = {}
        expected_in_batch = {}
        for i, qt in enumerate(question_types):
            if qt not in ("orig_match", "change_match"):
                continue
            if i >= len(extra_args_list) or i >= len(pred_token_ids):
                continue
            key = self._example_key(extra_args_list[i])
            letter = self._decode_letter(pred_token_ids[i])
            if key is None or letter is None:
                continue
            in_batch.setdefault(key, {})[qt] = letter
            if key not in expected_in_batch:
                row = extra_args_list[i]
                exp_orig = self._normalize_letter(row.get("original_response"))
                exp_change = self._normalize_letter(row.get("ablated_response"))
                if exp_orig and exp_change:
                    expected_in_batch[key] = {"orig": exp_orig, "change": exp_change}

        gen_indices = [i for i, qt in enumerate(question_types) if qt == "generative_explanation"]
        if not gen_indices:
            return

        device = next(model.parameters()).device
        was_training = model.training
        # Snapshot last_outputs / last_inputs because every model(...) and
        # model.generate(...) call below overwrites them; LossLoggingCallback
        # reads last_outputs in on_log AFTER on_step_end, so we must restore.
        saved_last_outputs = getattr(model, "last_outputs", None)
        saved_last_inputs = getattr(model, "last_inputs", None)
        model.eval()
        try:
            # ---- 1. Fill orphan letters via fallback forwards ----
            orphan_count = 0
            fallback_specs = []  # (gen_i, "orig"|"change", prompt_text)
            letters_by_gen = {}  # gen_i -> {"orig": letter, "change": letter}
            for gen_i in gen_indices:
                row = extra_args_list[gen_i] if gen_i < len(extra_args_list) else None
                key = self._example_key(row)
                in_b = in_batch.get(key, {}) if key is not None else {}
                letters_by_gen[gen_i] = {
                    "orig": in_b.get("orig_match"),
                    "change": in_b.get("change_match"),
                }
                if isinstance(row, dict):
                    if letters_by_gen[gen_i]["orig"] is None and row.get("orig_match_prompt"):
                        fallback_specs.append((gen_i, "orig", row["orig_match_prompt"]))
                        orphan_count += 1
                    if letters_by_gen[gen_i]["change"] is None and row.get("change_match_prompt"):
                        fallback_specs.append((gen_i, "change", row["change_match_prompt"]))
                        orphan_count += 1

            if fallback_specs:
                fallback_prompts = [s[2] for s in fallback_specs]
                tokenized = self.tokenizer(
                    fallback_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(device)
                with torch.inference_mode():
                    fb_out = model(
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                        return_dict=True,
                    )
                # Last non-pad position per row → argmax → decoded letter.
                seq_lens = tokenized["attention_mask"].sum(dim=-1) - 1
                for j, (gen_i, which, _prompt) in enumerate(fallback_specs):
                    pos = int(seq_lens[j].item())
                    pred_id = int(fb_out.logits[j, pos, :].argmax().item())
                    letter = self.tokenizer.decode([pred_id]).strip()
                    letters_by_gen[gen_i][which] = letter

            # ---- 2. Identify scored gen rows (have both letters) ----
            # Denominator is the count of gen rows we successfully scored
            # (fallback should fill orphans in practice, so this normally
            # equals the total number of gen rows in the batch).
            total_gen = len(gen_indices)
            scored = [
                gi
                for gi in gen_indices
                if letters_by_gen[gi]["orig"] and letters_by_gen[gi]["change"]
            ]

            # ---- 2b. Total behavioral match: the model's CURRENT answers on
            # both the cue-included and cue-ablated prompts equal the frozen
            # M0 labels (per-example joint — drifts if EITHER flips).
            # Observation-only; reuses letters already computed above.
            behavior_correct = 0
            behavior_total = 0
            for gi in scored:
                row = extra_args_list[gi] if gi < len(extra_args_list) else None
                key = self._example_key(row)
                expected = expected_in_batch.get(key) if key is not None else None
                if expected is None and isinstance(row, dict):
                    # Fall back to the gen row's own labels ("Answer: X" form;
                    # swapped when the dataset flips explanation direction).
                    exp_orig = self._normalize_letter(row.get("original_response"))
                    exp_change = self._normalize_letter(row.get("ablated_response"))
                    if self.flip_explanation_direction:
                        exp_orig, exp_change = exp_change, exp_orig
                    if exp_orig and exp_change:
                        expected = {"orig": exp_orig, "change": exp_change}
                if not expected:
                    continue
                behavior_total += 1
                if (
                    letters_by_gen[gi]["orig"] == expected["orig"]
                    and letters_by_gen[gi]["change"] == expected["change"]
                ):
                    behavior_correct += 1
            behavior_metrics = {"train/total_behavior_match/count": behavior_total}
            if behavior_total > 0:
                behavior_metrics["train/total_behavior_match"] = round(
                    behavior_correct / behavior_total, 4
                )

            if not scored:
                self._pending_metrics = {
                    "train/rollout_match/orphan_fallback_count": orphan_count,
                    "train/rollout_match/scored_count": 0,
                    "train/rollout_match/total_gen_count": total_gen,
                    **behavior_metrics,
                }
                return

            # ---- 3. Generate rollouts for gen rows ----
            scored_tensor = torch.tensor(scored, device=input_ids_woa.device)
            gen_input_ids = input_ids_woa.index_select(0, scored_tensor).to(device)
            gen_attn = attn_mask_woa.index_select(0, scored_tensor).to(device)
            with torch.inference_mode():
                gen_out = model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attn,
                    max_new_tokens=self.generate_limit,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            # Slice off the prompt — keep only newly generated tokens.
            new_tokens = gen_out[:, gen_input_ids.size(1):]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            # ---- 4. Compare against constructed expected ----
            num_correct = 0
            for k, gen_i in enumerate(scored):
                orig_letter = letters_by_gen[gen_i]["orig"]
                change_letter = letters_by_gen[gen_i]["change"]
                expected = _format_causal_change_string(orig_letter, change_letter).lower().strip()
                pred = decoded[k].lower().strip()
                if pred.startswith(expected):
                    num_correct += 1

            metric = num_correct / len(scored)
            self._pending_metrics = {
                "train/rollout_match/generative_explanation": round(metric, 4),
                "train/rollout_match/orphan_fallback_count": orphan_count,
                "train/rollout_match/scored_count": len(scored),
                "train/rollout_match/total_gen_count": total_gen,
                **behavior_metrics,
            }
        except Exception as e:
            print(f"[RolloutConsistencyCallback] step {state.global_step} skipped: {e}", flush=True)
        finally:
            # Restore so LossLoggingCallback sees the training step's outputs.
            model.last_outputs = saved_last_outputs
            model.last_inputs = saved_last_inputs
            if was_training:
                model.train()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # Mutate logs in-place so HF's WandbCallback (reordered to fire AFTER
        # us) syncs the rollout metrics in its single auto-stepped wandb.log.
        if logs is None or self._pending_metrics is None:
            return
        if state.is_world_process_zero:
            logs.update(self._pending_metrics)
        self._pending_metrics = None


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, metric_for_best_model="eval_loss", greater_is_better=False):
        self.patience = patience
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.patience_counter = 0
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return
            
        # Look for the metric in logs - try different possible formats
        current_metric = None
        metric_keys = [
            self.metric_for_best_model,
            f"eval_{self.metric_for_best_model}",
            f"test_{self.metric_for_best_model}",
        ]
        
        for key in metric_keys:
            if key in logs:
                current_metric = logs[key]
                break
                
        # If we can't find any eval metric, try to use any eval metric available
        if current_metric is None:
            eval_metrics = {k: v for k, v in logs.items() if k.startswith("eval_") or k.startswith("test_") or k.startswith("per_epoch_") or k.startswith("per_step_")}
            if eval_metrics:
                # Use the first available eval metric
                metric_key = list(eval_metrics.keys())[0]
                current_metric = eval_metrics[metric_key]
                if self.best_metric is None:  # First time, update the metric name
                    print(f"Early stopping using metric: {metric_key}")
                    self.metric_for_best_model = metric_key
        
        if current_metric is None:
            return
            
        # Check if this is the best metric we've seen
        is_better = False
        if self.best_metric is None:
            is_better = True
        elif self.greater_is_better:
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric
            
        if is_better:
            self.best_metric = current_metric
            self.patience_counter = 0
            print(f"New best {self.metric_for_best_model}: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            print(f"No improvement in {self.metric_for_best_model} for {self.patience_counter}/{self.patience} evaluations")
            
        # Stop training if patience exceeded
        if self.patience_counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} evaluations without improvement")
            control.should_training_stop = True


class NoOptimizerCheckpointCallback(TrainerCallback):
    """Callback to remove optimizer state from checkpoints after they're saved."""
    
    def on_save(self, args, state, control, **kwargs):
        """Remove optimizer state files after checkpoint is saved."""
        if state.is_world_process_zero:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            
            # Remove optimizer state if it exists
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)
                print(f"Removed optimizer state from checkpoint: {checkpoint_dir}")
            
            # Remove scheduler state if it exists
            if os.path.exists(scheduler_path):
                os.remove(scheduler_path)
        
        return control


class ActivationExplainCallback(TrainerCallback):
    def __init__(
        self,
        feature_descriptor,
        k=10,
        recompute_per_epoch=False,
        recompute_per_step=None,
    ):
        self.feature_descriptor = feature_descriptor
        self.k = k
        self.recompute_per_epoch = recompute_per_epoch
        self.recompute_per_step = recompute_per_step

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.recompute_per_epoch:
            # If interval is not set, cache at every epoch
            # maybe_cache_activation_explain(self.train_dataset, k=self.k)
            # maybe_recompute_features(self.train_dataset)
            maybe_reset_features(self.feature_descriptor)
            # for ds in self.test_datasets.values():
            #     # maybe_cache_activation_explain(ds, k=self.k)
            #     if ds.feature_descriptor is not None:
            #         ds.feature_descriptor.recompute_features()

    def on_step_end(self, args, state, control, **kwargs):
        if self.recompute_per_step and state.global_step % self.recompute_per_step == 0:
            # maybe_cache_activation_explain(self.train_dataset, k=self.k)
            maybe_reset_features(self.feature_descriptor)
            # for ds in self.test_datasets.values():
            #     # maybe_cache_activation_explain(ds, k=self.k)
            #     maybe_recompute_features(ds)


def train(config, args):
    # Handle argparse overrides for hyperparameters
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if hasattr(args, "learning_rate") and args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if hasattr(args, "batch_size") and args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if hasattr(args, "weight_decay") and args.weight_decay is not None:
        config["train"]["weight_decay"] = args.weight_decay
    if hasattr(args, "lr_scheduler_type") and args.lr_scheduler_type is not None:
        config["train"]["lr_scheduler_type"] = args.lr_scheduler_type
    if hasattr(args, "warmup_steps") and args.warmup_steps is not None:
        config["train"]["warmup_steps"] = args.warmup_steps
    if hasattr(args, "warmup_ratio") and args.warmup_ratio is not None:
        config["train"]["warmup_ratio"] = args.warmup_ratio
    if hasattr(args, "num_epochs") and args.num_epochs is not None:
        config["train"]["num_epochs"] = args.num_epochs
    if hasattr(args, "regularization_weight") and args.regularization_weight is not None:
        for task_name in config["train"]["tasks"]:
            if "question_types" in config["train"]["tasks"][task_name]:
                if "orig_match" in config["train"]["tasks"][task_name]["question_types"]:
                    config["train"]["tasks"][task_name]["question_types"]["orig_match"]["weight"] = args.regularization_weight
                if "change_match" in config["train"]["tasks"][task_name]["question_types"]:
                    config["train"]["tasks"][task_name]["question_types"]["change_match"]["weight"] = args.regularization_weight
        print(f"Applied regularization_weight={args.regularization_weight} to orig_match and change_match question types")
    if hasattr(args, "explanation_weight") and args.explanation_weight is not None:
        for task_name in config["train"]["tasks"]:
            qts = config["train"]["tasks"][task_name].get("question_types", {})
            if "generative_explanation" in qts:
                qts["generative_explanation"]["weight"] = args.explanation_weight
        print(f"Applied explanation_weight={args.explanation_weight} to generative_explanation question type")

    # Propagate KL regularization settings to config (CLI args override YAML)
    if getattr(args, "use_kl_regularization", False):
        config["use_kl_regularization"] = True
    else:
        config.setdefault("use_kl_regularization", False)
    if getattr(args, "kl_temperature", 1.0) != 1.0:
        config["kl_temperature"] = args.kl_temperature
    else:
        config.setdefault("kl_temperature", 1.0)
    if getattr(args, "kl_topk", 128) != 128:
        config["kl_topk"] = args.kl_topk
    else:
        config.setdefault("kl_topk", 128)
    if getattr(args, "log_top1_match", False):
        config["log_top1_match"] = True
    else:
        config.setdefault("log_top1_match", False)

    # `explanation_behavior_match` is YAML-only (no CLI flag). When true, both
    # top1_match and the rollout consistency check fire every step.
    if config.get("explanation_behavior_match", False):
        config["log_top1_match"] = True
        config["log_explanation_behavior_match"] = True
    else:
        config.setdefault("log_explanation_behavior_match", False)

    # CE loss normalization: default to "supervised" when KL regularization is on
    # (recent KL configs were tuned to that scale), else "seq_len" (legacy
    # behavior — divides per-sample CE by full seq_len, matches pre-Apr-27 runs).
    default_norm = "supervised" if config.get("use_kl_regularization", False) else "seq_len"
    config.setdefault("loss_normalize_by", default_norm)

    if hasattr(args, "test_upsample_factor") and args.test_upsample_factor is not None:
        for section in ["train", "test"]:
            if section in config and "tasks" in config[section]:
                for task_name in config[section]["tasks"]:
                    task_config = config[section]["tasks"][task_name]
                    if "question_types" in task_config:
                        for qtype in task_config["question_types"]:
                            if task_config["question_types"][qtype] is None:
                                task_config["question_types"][qtype] = {}
                            if "split_upsample_factors" not in task_config["question_types"][qtype]:
                                task_config["question_types"][qtype]["split_upsample_factors"] = {}
                            task_config["question_types"][qtype]["split_upsample_factors"]["test"] = args.test_upsample_factor
        print(f"Applied test_upsample_factor={args.test_upsample_factor} to all question types")

    """
    Step 1: Initialize models + tokenizer
    """
    if args.debug:
        config["train"]["num_samples"] = 100
        config["train"]["batch_size"] = 5
        config["test"]["num_samples"] = 100
        config["test"]["batch_size"] = 5
        config["train"]["eval_steps"] = 20
        for task in config["test"]["tasks"]:
            config["test"]["tasks"][task]["num_samples"] = 100
        config["num_epochs"] = 1
    else:
        shutil.rmtree(config["output_dir"], ignore_errors=True)

    # Save the final config (with all CLI overrides applied) to output_dir
    os.makedirs(config["output_dir"], exist_ok=True)
    saved_config_path = os.path.join(config["output_dir"], "config.yaml")
    with open(saved_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    predictor_tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_path", config["model_path"]),
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        config.get("target_tokenizer_path", config["target_model_path"]),
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
    target_tokenizer.pad_token = target_tokenizer.eos_token

    # Initialize models using the refactored make_model function
    wrapped_model, target_model = make_model(
        config=config,
        output_dir=config["output_dir"],
        train_self=args.train_self,
        use_embed_proj=config.get("use_embed_proj", False),
    )
    if args.train_self:
        assert wrapped_model == target_model

    # Set KL regularization config on model — set on the outer wrapper so shared_forward
    # sees it, and also on .base_model for PEFT setups where the inner HF model is what runs.
    use_kl = config.get("use_kl_regularization", False)
    kl_temp = config.get("kl_temperature", 1.0)
    log_top1 = config.get("log_top1_match", False)
    log_explanation_behavior_match = config.get("log_explanation_behavior_match", False)
    loss_norm = config.get("loss_normalize_by", "seq_len")
    # Walk the wrapper chain (PEFT adds PeftModel -> LoraModel -> base HF model).
    # shared_forward reads `self.use_kl_regularization` on the innermost HF model,
    # so set the flag on every level to guarantee it reaches that instance.
    _m = wrapped_model
    _seen = set()
    while _m is not None and id(_m) not in _seen:
        _seen.add(id(_m))
        _m.use_kl_regularization = use_kl
        _m.kl_temperature = kl_temp
        _m.log_top1_match = log_top1
        _m.log_explanation_behavior_match = log_explanation_behavior_match
        _m.loss_normalize_by = loss_norm
        _m = getattr(_m, "base_model", None) or getattr(_m, "model", None)

    # wrapped_model.print_trainable_parameters()

    """
    Step 2: Initialize datasets + data collator
    """
    if len(config["train"]["tasks"]) > 1:
        task_type = "mixed"
        question_config = config["train"]
    else:
        task_type = list(config["train"]["tasks"].keys())[0]
        question_config = config["train"]["tasks"][task_type]
    # Propagate KL regularization settings into train task config for dataset access
    question_config["use_kl_regularization"] = config.get("use_kl_regularization", False)
    question_config["kl_topk"] = config.get("kl_topk", 128)
    question_config["use_freeform_target"] = config.get("use_freeform_target", False)
    if config.get("teacher_logits_path"):
        question_config["teacher_logits_path"] = config["teacher_logits_path"]
    if "flip_explanation_direction" in config and "flip_explanation_direction" not in question_config:
        question_config["flip_explanation_direction"] = config["flip_explanation_direction"]
    # Create data collator
    data_collator = TASK_DATALOADER_MAPPING.get(task_type, BaseDataCollator)(
        predictor_tokenizer=predictor_tokenizer,
        target_tokenizer=target_tokenizer,
        question_config=question_config,
    )
    # Online relabeling (paper §4.2 / Fig 5): rebuild explanation targets from the
    # model's current behavior every batch, faithful with probability nu.
    if getattr(args, "online_relabel_nu", None) is not None:
        from dataloaders.online_relabel import OnlineRelabelCollator
        data_collator = OnlineRelabelCollator(
            data_collator,
            model=wrapped_model,
            tokenizer=predictor_tokenizer,
            nu=args.online_relabel_nu,
            seed=args.online_relabel_seed,
            flip_explanation_direction=question_config.get("flip_explanation_direction", False),
        )
        config["online_relabel_nu"] = args.online_relabel_nu
        print(f"Online relabeling enabled: nu={args.online_relabel_nu}")
    train_dataset = TASK_DATASET_MAPPING[task_type](
        "train",
        wrapped_model,
        target_model,
        predictor_tokenizer,
        target_tokenizer,
        config=question_config,
        debug=args.debug,
        self_train=args.train_self,
        model_name=config["model_path"],
        model_cache_dir=config.get("cache_dir", None),
    )
    train_feature_descriptor = None
    train_features = None
    db = None
    subject = None
    percentiles = None
    examples = None
    if isinstance(train_dataset, MixedDataset):
        if not getattr(train_dataset, "all_iterable", False):
            print(train_dataset[0])
        for ds in train_dataset.datasets.values():
            train_feature_descriptor = getattr(
                ds, "feature_descriptor", train_feature_descriptor
            )
            train_features = getattr(ds, "all_features", train_features)
            db = getattr(ds, "db", db)
            subject = getattr(ds, "subject", subject)
            percentiles = getattr(ds, "percentiles", percentiles)
            examples = getattr(ds, "examples", examples)
    else:
        train_feature_descriptor = getattr(
            train_dataset, "feature_descriptor", train_feature_descriptor
        )
        train_features = getattr(train_dataset, "all_features", train_features)
        db = getattr(train_dataset, "db", db)
        subject = getattr(train_dataset, "subject", subject)
        percentiles = getattr(train_dataset, "percentiles", percentiles)
        examples = getattr(train_dataset, "examples", examples)
    test_datasets = {
        task: TASK_DATASET_MAPPING[task](
            "test",
            wrapped_model,
            target_model,
            predictor_tokenizer,
            target_tokenizer,
            config=config["test"]["tasks"][task],
            debug=args.debug,
            self_train=args.train_self,
            feature_descriptor=train_feature_descriptor,
            all_features=train_features,
            db=db,
            subject=subject,
            percentiles=percentiles,
            examples=examples,
            model_name=config["model_path"],
            model_cache_dir=config.get("cache_dir", None),
        )
        for task in config["test"]["tasks"]
    }
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=data_collator,
    )
    test_collators = {
        task: TASK_DATALOADER_MAPPING.get(task, BaseDataCollator)(
            predictor_tokenizer=predictor_tokenizer,
            target_tokenizer=target_tokenizer,
            question_config=config["test"]["tasks"][task],
        )
        for task in config["test"]["tasks"]
    }
    test_dataloaders = {
        task: DataLoader(
            test_datasets[task],
            batch_size=config["test"]["batch_size"],
            collate_fn=test_collators[task],
        )
        for task in config["test"]["tasks"]
    }

    test_evaluators = {
        task: EVALUATOR_MAPPING[config["test"]["tasks"][task]["evaluation_type"]](
            config["test"]["tasks"][task],
            wrapped_model,
            predictor_tokenizer,
            test_dataloaders[task],
            # fs_dataloader=fs_test_dataloaders[task],
            parse_response_func=(
                None
                # parse_activation_explain_response
                # if task == "activation_explain"
                # else None
            ),
            cap_at_100=True,
            feature_descriptor=train_feature_descriptor,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
        )
        for task in config["test"]["tasks"]
    }

    """
    Step 3: Initialize evaluators
    """
    simulator = None
    exemplar_wrapper = None
    train_evaluators = {
        task: EVALUATOR_MAPPING[config["train"]["tasks"][task]["evaluation_type"]](
            config["train"]["tasks"][task],
            wrapped_model,
            predictor_tokenizer,
            train_dataloader,
            parse_response_func=(
                None
                # parse_activation_explain_response
                # if task == "activation_explain"
                # else None
            ),
            cap_at_100=True,
            feature_descriptor=train_feature_descriptor,
            simulator=simulator,
            exemplar_wrapper=exemplar_wrapper,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
        )
        for task in config["train"]["tasks"]
    }
    split_evaluators = {
        "train": train_evaluators,
        "test": test_evaluators,
    }

    eval_callback = EvaluationCallback(
        {
            f"{split}_{task}": split_evaluators[split][task]
            for split in ["train", "test"]
            for task in config[split]["tasks"]
            if getattr(
                split_evaluators[split][task], "generation_evaluation", True
            )
        },
        eval_strategy=config["train"].get("eval_strategy", "epoch"),
        eval_steps=config["train"].get("eval_steps", 500),
        debug_mode=args.debug,
    )

    output_dir = config["output_dir"]
    if not args.debug:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "introspective-coupling"),
            name=output_dir,
            config=config,
        )
        wandb.save(saved_config_path)

    # save test data
    os.makedirs(config["output_dir"], exist_ok=True)
    if args.evaluate_icl:  # or args.debug:
        for task in test_evaluators:
            print(task)
            test_evaluators[task].evaluate(fewshot=True if args.evaluate_icl else False)
            if not args.debug:
                test_evaluators[task].log_metrics(
                    prefix=(
                        f"{task}_icl_before_training/"
                        if args.evaluate_icl
                        else f"{task}_before_training/"
                    )
                )
            test_evaluators[task].print_metrics(
                prefix=(
                    f"Before training ICL ({task}) - "
                    if args.evaluate_icl
                    else f"Before training ({task}) - "
                )
            )
        for task in train_evaluators:
            if not getattr(
                train_evaluators[task], "generation_evaluation", True
            ):
                continue
            print(task)
            train_evaluators[task].evaluate()
            if not args.debug:
                train_evaluators[task].log_metrics(prefix=f"{task}_before_training/")
            train_evaluators[task].print_metrics(prefix=f"Before training ({task}) - ")
    # # save features
    # train_feature_descriptor.save_features()

    """
    Step 4: Initialize trainer
    """
    # Define training arguments with variable learning rate support
    print("Using bf16:", config["train"].get("bf16", True))
    
    # Get learning rate scheduler configuration
    lr_scheduler_type = config["train"].get("lr_scheduler_type", "constant")
    warmup_steps = config["train"].get("warmup_steps", 0)
    warmup_ratio = config["train"].get("warmup_ratio", 0)
    
    # Log learning rate scheduler configuration
    print(f"Learning rate scheduler type: {lr_scheduler_type}")
    if warmup_steps is not None:
        print(f"Warmup steps: {warmup_steps}")
    elif warmup_ratio is not None:
        print(f"Warmup ratio: {warmup_ratio}")
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["train"].get("num_epochs", 100),
        per_device_train_batch_size=config["train"]["batch_size"],
        per_device_eval_batch_size=config["test"]["batch_size"],
        learning_rate=config["train"].get("learning_rate", 5e-5),
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=config["train"].get("weight_decay", 0.01),
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=config["train"].get("logging_steps", 10),
        save_strategy=config["train"].get("save_strategy", "epoch"),
        save_steps=config["train"].get("save_steps", None),
        save_total_limit=config["train"].get("save_total_limit", None),
        save_only_model=config["train"].get("save_only_model", False),
        remove_unused_columns=config["train"].get("remove_unused_columns", False),
        bf16=config["train"].get("bf16", True),  # Enable bfloat16 precision
        report_to="wandb" if not args.debug else "none",
        ddp_find_unused_parameters=config["train"].get(
            "ddp_find_unused_parameters", False
        ),
        max_steps=config["train"].get("max_steps", -1),
        optim=config["train"].get("optim", "adamw_torch"),
        accelerator_config={"dispatch_batches": False},
    )

    # Create callbacks list
    trainer_callbacks = [eval_callback]
    
    # Add callback to remove optimizer state from checkpoints (unless --save_optimizer is set)
    if not args.save_optimizer:
        no_optimizer_callback = NoOptimizerCheckpointCallback()
        trainer_callbacks.append(no_optimizer_callback)
    
    # Add early stopping callback if patience is specified in config
    patience = config["train"].get("early_stopping_patience", None)
    if patience is not None and patience > 0:
        # Try to determine the main metric from evaluators
        main_metric = None
        greater_is_better = False
        
        # Look for the main metric in test evaluators
        for evaluator in test_evaluators.values():
            if hasattr(evaluator, 'main_metric') and evaluator.main_metric:
                main_metric = evaluator.main_metric
                greater_is_better = getattr(evaluator, 'greater_is_better', False)
                break
        
        early_stopping_callback = EarlyStoppingCallback(
            patience=patience,
            metric_for_best_model=main_metric or "eval_loss",
            greater_is_better=greater_is_better
        )
        trainer_callbacks.append(early_stopping_callback)
        print(f"Early stopping enabled with patience={patience}, metric={main_metric or 'eval_loss'}, greater_is_better={greater_is_better}")
    
    if train_feature_descriptor is not None and not args.debug and args.train_self:
        # Determine cache interval and k from config
        recompute_per_epoch = config["train"]["tasks"]["activation_explain"].get(
            "recompute_per_epoch", False
        )
        recompute_per_step = config["train"]["tasks"]["activation_explain"].get(
            "recompute_per_step", None
        )
        num_features = config["train"]["tasks"]["activation_explain"].get(
            "num_features", 10
        )
        cache_callback = ActivationExplainCallback(
            train_feature_descriptor,
            k=num_features,
            recompute_per_epoch=recompute_per_epoch,
            recompute_per_step=recompute_per_step,
        )
        trainer_callbacks.append(cache_callback)
    
    # Add loss logging callback
    loss_logging_callback = LossLoggingCallback()
    trainer_callbacks.append(loss_logging_callback)

    # Per-step rollout consistency check, gated by YAML `explanation_behavior_match`.
    if config.get("explanation_behavior_match", False):
        rollout_callback = RolloutConsistencyCallback(
            predictor_tokenizer=predictor_tokenizer,
            generate_limit=config.get("rollout_generate_limit", 20),
            flip_explanation_direction=config.get("train", {})
            .get("tasks", {})
            .get("causal", {})
            .get("flip_explanation_direction", False),
        )
        trainer_callbacks.append(rollout_callback)
        print("Rollout consistency check enabled (explanation_behavior_match=true)")

    # Create trainer
    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=trainer_callbacks,
        # overwrite_output_dir=False,
    )

    # Reorder callbacks so LossLoggingCallback / RolloutConsistencyCallback
    # fire BEFORE HF's WandbCallback in on_log. Otherwise WandbCallback's
    # wandb.log({...}) (no step=) auto-bumps wandb's internal step past
    # state.global_step, and any subsequent wandb.log(..., step=state.global_step)
    # call we make is silently dropped as "step in the past". The clean fix
    # is to mutate the shared `logs` dict before WandbCallback reads it, so
    # all our extras ride out in HF's single, properly-stepped wandb.log call.
    try:
        from transformers.integrations.integration_utils import WandbCallback as _HFWandbCB
        _cbs = trainer.callback_handler.callbacks
        _wb_cb = next((c for c in _cbs if isinstance(c, _HFWandbCB)), None)
        if _wb_cb is not None:
            _wb_idx = _cbs.index(_wb_cb)
            for _cb in [c for c in (loss_logging_callback, locals().get("rollout_callback")) if c is not None]:
                if _cb in _cbs:
                    _cbs.remove(_cb)
                    _wb_idx = _cbs.index(_wb_cb)
                    _cbs.insert(_wb_idx, _cb)
            print(f"Reordered callbacks: LossLogging/Rollout fire before WandbCallback (idx {_cbs.index(_wb_cb)})")
    except Exception as _e:
        print(f"Warning: could not reorder callbacks ({_e}); wandb extras may not sync")
    # tokenized_prompt = predictor_tokenizer(
    #     train_dataset[0]["prompt"], return_tensors="pt"
    # )
    # tokenized_prompt_without_answer = predictor_tokenizer(
    #     train_dataset[0]["prompt_without_answer"], return_tensors="pt"
    # )
    # output = target_model(**tokenized_prompt)
    # output_generation = wrapped_model.generate(
    #     **tokenized_prompt_without_answer,
    #     do_sample=False,
    #     pad_token_id=predictor_tokenizer.eos_token_id,
    # )
    # print(
    #     predictor_tokenizer.decode(
    #         output_generation[
    #             0, tokenized_prompt_without_answer["input_ids"].shape[1] :
    #         ]
    #     )
    # )
    # # input_tokens_len = tokenized_prompt["expected_label"].shape[1]
    # print(predictor_tokenizer.decode(output.logits[0].argmax(dim=-1)))
    # print(train_dataset[0]["expected_label"])

    torch.cuda.empty_cache()
    # Train the model
    resume_checkpoint = args.resume_from_checkpoint if hasattr(args, 'resume_from_checkpoint') else None
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save the model
    all_tasks = list(test_evaluators.keys()) + list(train_evaluators.keys())
    model_save_path = os.path.join(config["output_dir"], f"{'_'.join(all_tasks)}_model")

    wrapped_model.save_pretrained(model_save_path)
    predictor_tokenizer.save_pretrained(model_save_path)

    for task in test_evaluators:
        test_evaluators[task].evaluate()
        test_evaluators[task].print_metrics(prefix=f"Final evaluation (test) - {task}")

    if args.evaluate_icl:
        for task in test_evaluators:
            test_evaluators[task].evaluate(fewshot=True)
            test_evaluators[task].print_metrics(
                prefix=f"After training ICL ({task}) - "
            )

    wandb.finish()

    return wrapped_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_self", action="store_true", help="Train model to predict self or other"
    )
    parser.add_argument("--evaluate_icl", action="store_true", help="Evaluate ICL")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/qwen_qwen_hint_kl_regularized.yaml",
        help="Path to the question configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (overrides config)")
    parser.add_argument("--lr_scheduler_type", type=str, default=None, help="LR scheduler type (overrides config)")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps (overrides config)")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio (overrides config)")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--regularization_weight", type=float, default=None, help="Weight for orig_match and change_match question types (overrides config)")
    parser.add_argument("--explanation_weight", type=float, default=None, help="Weight for generative_explanation question type (overrides config)")
    parser.add_argument("--use_kl_regularization", action="store_true", help="Use KL divergence instead of CE for orig_match/change_match regularization losses")
    parser.add_argument("--kl_temperature", type=float, default=1.0, help="Temperature for KL divergence regularization (default: 1.0)")
    parser.add_argument("--log_top1_match", action="store_true", help="Log per-question-type student/supervision top-1 agreement to wandb (train/top1_match/{qt})")
    parser.add_argument("--kl_topk", type=int, default=128, help="Number of top-k teacher logits to precompute for KL regularization (default: 128)")
    parser.add_argument("--test_upsample_factor", type=int, default=None, help="Upsample factor for test split (overrides config split_upsample_factors)")
    parser.add_argument("--save_optimizer", action="store_true", help="Keep optimizer state in checkpoints (default: remove to save disk space)")
    parser.add_argument("--online_relabel_nu", type=float, default=None, help="Per-batch online relabeling of explanation targets: fraction faithful to the model's current behavior (paper Fig 5)")
    parser.add_argument("--online_relabel_seed", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from")
    args = parser.parse_args()

    # Load the config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate test_hint_path matches test config's hint_path
    for task_name, train_task_config in config.get("train", {}).get("tasks", {}).items():
        if "test_hint_path" in train_task_config:
            test_task_config = config.get("test", {}).get("tasks", {}).get(task_name, {})
            test_hint_path = test_task_config.get("hint_path")
            train_test_hint_path = train_task_config["test_hint_path"]
            if test_hint_path is None:
                raise ValueError(
                    f"train.tasks.{task_name}.test_hint_path is specified but test.tasks.{task_name}.hint_path is missing"
                )
            if train_test_hint_path != test_hint_path:
                raise ValueError(
                    f"train.tasks.{task_name}.test_hint_path ({train_test_hint_path}) must match "
                    f"test.tasks.{task_name}.hint_path ({test_hint_path})"
                )

    # Train the token attention model
    train(
        config=config,
        args=args,
    )

    # Force-exit to avoid interpreter teardown hangs from lingering
    # distributed / DataLoader / CUDA cleanup. wandb.finish() and the final
    # model save already ran inside train().
    os._exit(0)
