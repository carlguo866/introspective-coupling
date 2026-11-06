import copy
import os
import pickle
import random
import warnings
from typing import Any
import glob
import pandas as pd
import json
import numpy as np
from dataloaders.base_dataset import format_messages
import torch
from dataloaders.base_dataset import BaseDataset, ExampleLabels, BaseDataCollator
from tqdm import tqdm
from torch.utils.data import IterableDataset
from nnsight import LanguageModel
import re

warnings.filterwarnings("ignore")


def _normalize_aita_verdict(text: str) -> str:
    """Reduce a generated AITA verdict (" NTA", " **YTA**", "YTA.") to YTA/NTA; else the stripped text."""
    m = re.search(r"(YTA|NTA)", text.replace("*", ""))
    return m.group(1) if m else text.strip()


class CausalDataset(IterableDataset):
    def __init__(
        self,
        split,
        predictor_model,
        target_model,
        predictor_tokenizer,
        target_tokenizer,
        config,
        num_icl_examples=0,
        all_features=None,
        append_test_data=False,
        **kwargs,
    ):
        # Initialize without loading datasets first
        self.n_changed = config.get("n_changed", None)
        self.n_unchanged = config.get("n_unchanged", None)
        # super().__init__(
        #     split,
        #     predictor_model,
        #     target_model,
        #     predictor_tokenizer,
        #     target_tokenizer,
        #     config,
        #     num_icl_examples,
        #     do_load_datasets=True,
        #     **kwargs,
        # )

        self.data_split = split
        self.debug = kwargs.get("debug", False)
        self.predictor_model = predictor_model
        self.target_model = target_model
        self.predictor_tokenizer = predictor_tokenizer
        self.target_tokenizer = target_tokenizer
        self.config = config
        self.num_icl_examples = num_icl_examples
        self.all_features = all_features
        self.append_test_data = append_test_data or config.get("append_test_data", False)
        self.question_config = config
        self.example_to_dataset_name = {}
        self.self_consistency = self.config.get("self_consistency", False)
        self.use_kl_regularization = self.config.get("use_kl_regularization", False)
        self.kl_topk = self.config.get("kl_topk", 128)
        self.flip_explanation_direction = self.config.get(
            "flip_explanation_direction", False
        )
        # When True, judge_verdict orig_match/change_match training targets become
        # the full freeform model rollout (freeform_response_with_sys/no_sys)
        # instead of the short Yes/No verdict, and KL is regularized per-token.
        self.use_freeform_target = self.config.get("use_freeform_target", False)
        self.precomputed_teacher_logits = None
        if self.use_kl_regularization:
            # Explicit path wins; otherwise fall back to the derived
            # "<data_path>.<suffix>.pt" convention.
            teacher_logits_path = self.config.get("teacher_logits_path", None)
            if not teacher_logits_path:
                data_path = (
                    self.config.get("hint_path", "")
                    or self.config.get("judge_verdict_path", "")
                    or self.config.get("aita_path", "")
                )
                cache_suffix = (
                    f"teacher_logits_freeform_top{self.kl_topk}"
                    if self.use_freeform_target
                    else f"teacher_logits_top{self.kl_topk}"
                )
                teacher_logits_path = f"{data_path}.{cache_suffix}.pt"
            if os.path.exists(teacher_logits_path):
                print(f"Loading precomputed teacher logits from {teacher_logits_path}")
                self.precomputed_teacher_logits = torch.load(
                    teacher_logits_path, map_location="cpu", weights_only=False
                )
                print(f"Loaded {len(self.precomputed_teacher_logits)} entries")
            else:
                print(
                    f"No precomputed teacher logits found at {teacher_logits_path}, "
                    f"will compute on the fly"
                )
        # Max sequence length to prevent OOM - check both prompt and full sequence
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_sequence_length = config.get("max_sequence_length", 1024)
        if self.self_consistency:
            self.nnsight_model = LanguageModel(
                self.target_model,
                tokenizer=self.target_tokenizer,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        # self.num_repeats_per_example = config.get("num_repeats_per_example", 1)
        if kwargs.get("num_samples", None) is None:
            try:
                self.num_samples = config["num_samples"]
            except:
                raise
        else:
            self.num_samples = kwargs.get("num_samples", None)

        self.qtype_to_splits = {
            qtype: self.question_config["question_types"][qtype].get(
                "split", [self.data_split]
            )
            for qtype in self.question_config["question_types"]
        }
        self.split_to_qtype = {}
        for qtype in self.question_config["question_types"]:
            for split in self.qtype_to_splits[qtype]:
                if split not in self.split_to_qtype:
                    self.split_to_qtype[split] = set()
                self.split_to_qtype[split].add(qtype)

        dataset_names = config["dataset"]
        self.icl_examples, self.non_icl_examples = [], []
        self.icl_labels, self.non_icl_labels = [], []
        # self.num_repeats_per_example = len(
        #     self.question_config["question_types"]
        # ) * len(self.split_to_qtype)
        self.num_samples = 0
        self.example_to_dataset_name = {}
        for dataset_name in dataset_names:
            # for qtype in self.question_config["question_types"]:
            for split in self.split_to_qtype:
                # num_dataset_samples = (
                #     self.num_samples // len(dataset_names) // self.num_repeats_per_example
                # )
                # num_dataset_samples = (
                #     self.num_samples
                #     // len(dataset_names)
                #     // len(self.split_to_task_samples[split])
                # )
                if dataset_name.startswith("causal"):
                    (
                        non_icl_examples,
                        non_icl_labels,
                        icl_examples,
                        icl_labels,
                        _,
                    ) = self.load_intervention_data(self.question_config, split)
                elif dataset_name == "hint":
                    (
                        non_icl_examples,
                        non_icl_labels,
                        icl_examples,
                        icl_labels,
                    ) = self.load_hint_data(self.question_config, split)
                elif dataset_name == "aita":
                    (
                        non_icl_examples,
                        non_icl_labels,
                        icl_examples,
                        icl_labels,
                    ) = self.load_aita_data(self.question_config, split)
                elif dataset_name == "judge_verdict":
                    (
                        non_icl_examples,
                        non_icl_labels,
                        icl_examples,
                        icl_labels,
                    ) = self.load_judge_verdict_data(self.question_config, split)
                else:
                    raise ValueError(f"Invalid dataset name: {dataset_name}")

                for i, example in enumerate(non_icl_examples):
                    self.example_to_dataset_name[example] = dataset_name

                    # Get upsample factor for this split
                    upsample_factor = 1
                    for qtype in self.split_to_qtype[split]:
                        qtype_config = self.question_config["question_types"].get(qtype, {})
                        split_upsample_factors = qtype_config.get("split_upsample_factors", {})
                        if split in split_upsample_factors:
                            upsample_factor = max(upsample_factor, split_upsample_factors[split])

                    # Insert the example upsample_factor times
                    for _ in range(upsample_factor):
                        if split != self.data_split:
                            if self.append_test_data:
                                # Append test data after train data for alternating training
                                self.non_icl_examples.append(
                                    (example, self.split_to_qtype[split], split)
                                )
                                self.non_icl_labels.append(non_icl_labels[i])
                            else:
                                # insert into random locations
                                random_idx = np.random.randint(0, len(self.non_icl_examples) + 1)
                                self.non_icl_examples.insert(
                                    random_idx, (example, self.split_to_qtype[split], split)
                                )
                                self.non_icl_labels.insert(random_idx, non_icl_labels[i])
                        else:
                            self.non_icl_examples.append(
                                (example, self.split_to_qtype[split], split)
                            )
                            self.non_icl_labels.append(non_icl_labels[i])
                        self.num_samples += len(self.split_to_qtype[split])

                self.icl_labels.extend(icl_labels)
                self.icl_examples.extend(icl_examples)
        self.task_prompt = ""

        assert len(self.icl_examples) == len(self.icl_labels)
        assert len(self.non_icl_examples) == len(self.non_icl_labels)

        print(f"# Total samples for {self.data_split}: {self.num_samples}")

    def _apply_balanced_sampling(self, df):
        """Apply balanced sampling across exemplars for the entire dataset"""
        if df.empty:
            return df

        # Convert input_tokens to hashable format for grouping
        df["input_tokens_hashable"] = df["input_tokens"].apply(
            lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
        )

        # Group by hashable input_tokens and separate changed/unchanged examples
        changed_examples = []
        unchanged_examples = []
        for group_name, group_df in df.groupby("input_tokens_hashable"):
            # Separate changed and unchanged examples
            changed_examples.extend(group_df[group_df["is_different"]])
            unchanged_examples.extend(group_df[~group_df["is_different"]])

        # Convert lists back to DataFrames for sampling
        changed_df = (
            pd.DataFrame(changed_examples) if changed_examples else pd.DataFrame()
        )
        unchanged_df = (
            pd.DataFrame(unchanged_examples) if unchanged_examples else pd.DataFrame()
        )

        # Strategy: Sample from exemplars first, then from examples within those exemplars
        max_per_exemplar = (
            8  # Maximum examples per exemplar to avoid over-concentration
        )

        sampled_dfs = []

        # Sample changed examples with exemplar distribution constraint
        if not changed_df.empty and self.n_changed is not None and self.n_changed > 0:
            changed_per_exemplar = changed_df.groupby("input_tokens_hashable")

            # Calculate how many examples to take per exemplar
            num_exemplars_with_changed = len(changed_per_exemplar)
            if num_exemplars_with_changed > 0:
                examples_per_exemplar = min(
                    max_per_exemplar,
                    max(1, self.n_changed // num_exemplars_with_changed),
                )

                sampled_changed_parts = []
                for _, group in changed_per_exemplar:
                    n_from_group = min(
                        examples_per_exemplar,
                        len(group),
                        self.n_changed
                        - len(sampled_changed_parts) * examples_per_exemplar,
                    )
                    if n_from_group > 0:
                        sampled_part = group.sample(n=n_from_group, random_state=42)
                        sampled_changed_parts.append(sampled_part)

                sampled_changed = pd.concat(sampled_changed_parts, ignore_index=True)

                # If we still need more examples, sample randomly from remaining
                if len(sampled_changed) < self.n_changed:
                    remaining_needed = self.n_changed - len(sampled_changed)
                    remaining_changed = changed_df[
                        ~changed_df.index.isin(sampled_changed.index)
                    ]
                    if not remaining_changed.empty:
                        additional = remaining_changed.sample(
                            n=min(remaining_needed, len(remaining_changed)),
                            random_state=42,
                        )
                        sampled_changed = pd.concat(
                            [sampled_changed, additional], ignore_index=True
                        )
            else:
                sampled_changed = pd.DataFrame()
        else:
            sampled_changed = pd.DataFrame()

        # Sample unchanged examples with exemplar distribution constraint
        if (
            not unchanged_df.empty
            and self.n_unchanged is not None
            and self.n_unchanged > 0
        ):
            unchanged_per_exemplar = unchanged_df.groupby("input_tokens_hashable")

            # Calculate how many examples to take per exemplar
            num_exemplars_with_unchanged = len(unchanged_per_exemplar)
            if num_exemplars_with_unchanged > 0:
                examples_per_exemplar = min(
                    max_per_exemplar,
                    max(1, self.n_unchanged // num_exemplars_with_unchanged),
                )

                sampled_unchanged_parts = []
                for _, group in unchanged_per_exemplar:
                    n_from_group = min(
                        examples_per_exemplar,
                        len(group),
                        self.n_unchanged
                        - len(sampled_unchanged_parts) * examples_per_exemplar,
                    )
                    if n_from_group > 0:
                        sampled_part = group.sample(n=n_from_group, random_state=42)
                        sampled_unchanged_parts.append(sampled_part)

                sampled_unchanged = pd.concat(
                    sampled_unchanged_parts, ignore_index=True
                )

                # If we still need more examples, sample randomly from remaining
                if len(sampled_unchanged) < self.n_unchanged:
                    remaining_needed = self.n_unchanged - len(sampled_unchanged)
                    remaining_unchanged = unchanged_df[
                        ~unchanged_df.index.isin(sampled_unchanged.index)
                    ]
                    if not remaining_unchanged.empty:
                        additional = remaining_unchanged.sample(
                            n=min(remaining_needed, len(remaining_unchanged)),
                            random_state=42,
                        )
                        sampled_unchanged = pd.concat(
                            [sampled_unchanged, additional], ignore_index=True
                        )
            else:
                sampled_unchanged = pd.DataFrame()
        else:
            sampled_unchanged = pd.DataFrame()

        # Combine the sampled examples
        if not sampled_changed.empty and not sampled_unchanged.empty:
            result_df = pd.concat(
                [sampled_changed, sampled_unchanged], ignore_index=True
            )
        elif not sampled_changed.empty:
            result_df = sampled_changed
        elif not sampled_unchanged.empty:
            result_df = sampled_unchanged
        else:
            result_df = pd.DataFrame()

        # print changed/unchanged
        print(f"# Changed Sampled ({self.data_split}): {len(sampled_changed)}")
        print(f"# Unchanged Sampled ({self.data_split}): {len(sampled_unchanged)}")

        # Drop the temporary hashable column
        if not result_df.empty and "input_tokens_hashable" in result_df.columns:
            result_df = result_df.drop(columns=["input_tokens_hashable"])

        return result_df

    def _sample_examples_per_input(self, df):
        """Sample n_changed and n_unchanged examples per unique input_tokens"""
        return self._apply_balanced_sampling(df)

    def _compute_n_changed_and_n_unchanged(self, all_changed, all_unchanged):
        total_ratio_parts = self.n_changed + self.n_unchanged
        changed_fraction = self.n_changed / total_ratio_parts
        unchanged_fraction = self.n_unchanged / total_ratio_parts

        # Determine how many total samples we can get while maintaining ratio
        max_changed_by_availability = len(all_changed)
        max_unchanged_by_availability = len(all_unchanged)

        # Calculate max total samples we can get while maintaining exact ratio
        if changed_fraction > 0 and unchanged_fraction > 0:
            max_total_by_changed = max_changed_by_availability / changed_fraction
            max_total_by_unchanged = max_unchanged_by_availability / unchanged_fraction
            max_total_samples = int(min(max_total_by_changed, max_total_by_unchanged))
        elif changed_fraction > 0:
            max_total_samples = max_changed_by_availability
        else:
            max_total_samples = max_unchanged_by_availability

        # Calculate actual samples to take
        n_changed_to_sample = int(max_total_samples * changed_fraction)
        n_unchanged_to_sample = int(max_total_samples * unchanged_fraction)
        return n_changed_to_sample, n_unchanged_to_sample

    def load_hint_data(self, question_config, data_split):
        """Load hint data from jsonl files"""
        # Select hint_path based on data_split - use test_hint_path for test split if available
        default_hint_path = "hint_data/mmlu_llama3.1_8b_ablation_prompts.jsonl"
        if data_split == "test" and "test_hint_path" in question_config:
            hint_path = question_config["test_hint_path"]
        else:
            hint_path = question_config.get("hint_path", default_hint_path)

        print(f"Loading hint data from {hint_path} for data_split {data_split}")

        changed_data_list = []
        unchanged_data_list = []
        with open(hint_path, "r") as f:
            for line in f:
                hint_data = json.loads(line)
                if hint_data["changed_pred_bc_hint"]:
                    changed_data_list.append(hint_data)
                else:
                    unchanged_data_list.append(hint_data)
        if self.n_changed is not None and self.n_unchanged is not None:
            n_changed_to_sample, n_unchanged_to_sample = (
                self._compute_n_changed_and_n_unchanged(
                    changed_data_list, unchanged_data_list
                )
            )
            changed_data_list = changed_data_list[:n_changed_to_sample]
            unchanged_data_list = unchanged_data_list[:n_unchanged_to_sample]
        all_data_list = changed_data_list + unchanged_data_list
        random.Random(42).shuffle(all_data_list)

        non_icl_examples, non_icl_labels, icl_examples, icl_labels = [], [], [], []
        for hint_data in all_data_list:
            hint_data["dataset"] = "hint"
            non_icl_examples.append(hint_data["ablation_prompt"])
            non_icl_labels.append(hint_data)
            icl_examples.append(hint_data["ablation_prompt"])
            icl_labels.append(hint_data)
        return non_icl_examples, non_icl_labels, icl_examples, icl_labels

    def load_aita_data(self, question_config, data_split):
        """Load AITA data from jsonl files.

        The intervention is the system prompt: the model's prediction may change
        when the system prompt is present vs absent.

        Fields used from each JSONL entry:
          - system_prompt:          the system prompt for the "with system prompt" condition
          - user_prompt:            the user message (AITA post + answer suffix)
          - with_system_prediction: model prediction WITH system prompt
          - no_system_prediction:   model prediction WITHOUT system prompt
          - changed_pred:           whether removing the system prompt changed the prediction
          - ablation_prompt:        used as example key/identifier
        """
        if data_split == "test" and "test_aita_path" in question_config:
            aita_path = question_config["test_aita_path"]
        else:
            aita_path = question_config.get("aita_path")
        assert aita_path is not None, "aita_path must be specified in config for dataset='aita'"

        print(f"Loading AITA data from {aita_path} for data_split {data_split}")

        changed_data_list = []
        unchanged_data_list = []
        with open(aita_path, "r") as f:
            for line in f:
                aita_data = json.loads(line)
                if aita_data["changed_pred"]:
                    changed_data_list.append(aita_data)
                else:
                    unchanged_data_list.append(aita_data)
        all_data_list = changed_data_list + unchanged_data_list
        random.Random(42).shuffle(all_data_list)

        non_icl_examples, non_icl_labels, icl_examples, icl_labels = [], [], [], []
        for aita_data in all_data_list:
            aita_data["dataset"] = "aita"
            non_icl_examples.append(aita_data["ablation_prompt"])
            non_icl_labels.append(aita_data)
            icl_examples.append(aita_data["ablation_prompt"])
            icl_labels.append(aita_data)
        return non_icl_examples, non_icl_labels, icl_examples, icl_labels

    def load_judge_verdict_data(self, question_config, data_split):
        """Load judge-verdict data from jsonl files.

        Same JSONL schema as `load_aita_data`, but used for the verdict-driven
        4-way classification task. The `changed_pred` field in the JSONL is
        derived from judge verdicts (random_hint_prediction != zeroshot_prediction)
        and is the source of truth for changed/unchanged.

        Path is read from `judge_verdict_path` (or `test_judge_verdict_path`
        for test split), falling back to `aita_path` for backward compatibility
        since the JSONL schemas are identical.
        """
        if data_split == "test" and "test_judge_verdict_path" in question_config:
            data_path = question_config["test_judge_verdict_path"]
        elif "judge_verdict_path" in question_config:
            data_path = question_config["judge_verdict_path"]
        elif data_split == "test" and "test_aita_path" in question_config:
            data_path = question_config["test_aita_path"]
        else:
            data_path = question_config.get("aita_path")
        assert data_path is not None, (
            "judge_verdict_path (or aita_path) must be specified in config "
            "for dataset='judge_verdict'"
        )

        print(f"Loading judge_verdict data from {data_path} for data_split {data_split}")

        changed_data_list = []
        unchanged_data_list = []
        with open(data_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry["changed_pred"]:
                    changed_data_list.append(entry)
                else:
                    unchanged_data_list.append(entry)
        all_data_list = changed_data_list + unchanged_data_list
        random.Random(42).shuffle(all_data_list)

        non_icl_examples, non_icl_labels, icl_examples, icl_labels = [], [], [], []
        for entry in all_data_list:
            entry["dataset"] = "judge_verdict"
            non_icl_examples.append(entry["ablation_prompt"])
            non_icl_labels.append(entry)
            icl_examples.append(entry["ablation_prompt"])
            icl_labels.append(entry)
        return non_icl_examples, non_icl_labels, icl_examples, icl_labels

    def load_intervention_data(self, question_config, data_split):
        """Load intervention data from parquet files"""
        intervention_path = question_config.get(
            "intervention_path", "intervention_data"
        )

        # Find the correct dataset subdirectory
        model_name_key = None
        for model_key in [
            "llama_3.1_8b",
            "llama_3_8b",
            "gemma_2_9b",
        ]:  # Add more as needed
            for dataset_type in ["fineweb", "ultrachat"]:
                for ablation_type in ["zero_ablation", "mean_ablation"]:
                    test_path = os.path.join(
                        intervention_path, f"{model_key}_{dataset_type}_{ablation_type}"
                    )
                    if os.path.exists(test_path):
                        intervention_path = test_path
                        model_name_key = model_key
                        break
                if model_name_key:
                    break
            if model_name_key:
                break

        # Check if exemplar-grouped format exists
        exemplar_path = os.path.join(intervention_path, data_split, "exemplars")
        use_exemplar_format = os.path.exists(exemplar_path)
        cache_path = self.question_config.get("cache_path", None)

        if use_exemplar_format:
            print(f"Using exemplar-grouped format from {exemplar_path}")
            return self._load_exemplar_grouped_data(
                exemplar_path,
                data_split,
                cache_path,
            )
        else:
            print(f"Using legacy format from {intervention_path}")
            return self._load_legacy_intervention_data(intervention_path, data_split)

    def _load_exemplar_grouped_data(self, exemplar_path, data_split, cache_path):
        """Load intervention data from exemplar-grouped parquet files"""
        # TODO create splits by feature

        if cache_path is None:
            cache_path = os.path.join(
                exemplar_path,
                f"combined_df_{data_split}_{self.n_changed}:{self.n_unchanged}.parquet",
            )
            raise ValueError(f"Cache path is not specified")

        if os.path.exists(cache_path):
            return self._process_intervention_dataframe(pd.read_parquet(cache_path))

        all_changed = []
        all_unchanged = []
        max_num_batches = 1 if self.debug else 1000
        total_files_loaded = 0

        # Find all batch files
        batch_files = glob.glob(os.path.join(exemplar_path, "batch_*.parquet"))
        batch_files.sort()

        # Create progress bar
        pbar = tqdm(
            total=min(len(batch_files), max_num_batches),
            desc="Loading exemplar-grouped data",
            unit="files",
        )

        # First pass: collect all interventions across all files
        for batch_file in batch_files:
            try:
                df = pd.read_parquet(batch_file)
            except:
                continue
            total_files_loaded += 1
            if total_files_loaded > max_num_batches:
                break

            # Expand each exemplar's interventions into separate rows
            for _, row in df.iterrows():
                exemplar_text = row["exemplar_text"]
                input_tokens = row["input_tokens"]
                if len(self.predictor_tokenizer.tokenize(exemplar_text)) > 512:
                    continue
                original_continuation = row["original_continuation"]

                # Filter and collect changed interventions
                filtered_changed = [
                    intervention
                    for intervention in row["changed_interventions"].tolist()
                ]
                for intervention in filtered_changed:
                    expanded_row = {
                        "layer": intervention["layer"],
                        "input_tokens": input_tokens,
                        "original_continuation": original_continuation,
                        "ablated_continuation": intervention["ablated_continuation"],
                        "is_different": True,
                    }
                    if "ablated_token_positions" in intervention:
                        expanded_row.update(
                            {
                                "ablated_token_positions": intervention[
                                    "ablated_token_positions"
                                ],
                                "ablated_tokens": intervention["ablated_tokens"],
                            }
                        )
                    if "patch_position" in intervention:
                        expanded_row.update(
                            {
                                "patch_position": intervention["patch_position"],
                                "counterfactual_text": intervention[
                                    "counterfactual_text"
                                ],
                                "gt_original_target": intervention[
                                    "gt_original_target"
                                ],
                                "gt_counterfactual_target": intervention[
                                    "gt_counterfactual_target"
                                ],
                            }
                        )
                    if "feature_idx" in intervention:
                        expanded_row.update(
                            {
                                "feature_idx": intervention["feature_idx"],
                                "description": intervention["description"],
                                "activation_values": intervention["activation_values"],
                                "activating_positions": intervention[
                                    "activating_positions"
                                ],
                            }
                        )
                    all_changed.append(expanded_row)

                # Filter and collect unchanged interventions
                filtered_unchanged = [
                    intervention
                    for intervention in row["unchanged_interventions"].tolist()
                ]
                for intervention in filtered_unchanged:
                    expanded_row = {
                        "layer": intervention["layer"],
                        "input_tokens": input_tokens,
                        "original_continuation": original_continuation,
                        "ablated_continuation": intervention["ablated_continuation"],
                        "is_different": False,
                    }
                    if "ablated_token_positions" in intervention:
                        expanded_row.update(
                            {
                                "ablated_token_positions": intervention[
                                    "ablated_token_positions"
                                ],
                                "ablated_tokens": intervention["ablated_tokens"],
                            }
                        )
                    if "patch_position" in intervention:
                        expanded_row.update(
                            {
                                "patch_position": intervention["patch_position"],
                                "counterfactual_text": intervention[
                                    "counterfactual_text"
                                ],
                                "gt_original_target": intervention[
                                    "gt_original_target"
                                ],
                                "gt_counterfactual_target": intervention[
                                    "gt_counterfactual_target"
                                ],
                            }
                        )
                    if "feature_idx" in intervention:
                        expanded_row.update(
                            {
                                "feature_idx": intervention["feature_idx"],
                                "description": intervention["description"],
                                "activation_values": intervention["activation_values"],
                                "activating_positions": intervention[
                                    "activating_positions"
                                ],
                            }
                        )
                    all_unchanged.append(expanded_row)

            pbar.update(1)
            pbar.set_postfix({"files": total_files_loaded})

        pbar.close()

        if self.n_changed is not None and self.n_unchanged is not None:
            # Now sample globally to achieve target ratio
            # Calculate ratio and scale to use available data effectively

            sampled_changed = []
            n_changed_to_sample, n_unchanged_to_sample = (
                self._compute_n_changed_and_n_unchanged(all_changed, all_unchanged)
            )
            if all_changed and n_changed_to_sample > 0:
                # Save sample to cache
                sampled_changed_cache_path = os.path.join(
                    exemplar_path,
                    f"sampled_changed_{data_split}_{self.n_changed}:{self.n_unchanged}.pkl",
                )
                if os.path.exists(sampled_changed_cache_path):
                    sampled_changed = pickle.load(
                        open(sampled_changed_cache_path, "rb")
                    )
                    print(f"Loaded sampled changed from {sampled_changed_cache_path}")
                else:
                    sampled_changed = random.sample(all_changed, n_changed_to_sample)
                    pickle.dump(sampled_changed, open(sampled_changed_cache_path, "wb"))
                    print(f"Saved sampled changed to {sampled_changed_cache_path}")

            sampled_unchanged = []
            if all_unchanged and n_unchanged_to_sample > 0:
                sampled_unchanged_cache_path = os.path.join(
                    exemplar_path,
                    f"sampled_unchanged_{data_split}_{self.n_changed}:{self.n_unchanged}.pkl",
                )
                if os.path.exists(sampled_unchanged_cache_path):
                    sampled_unchanged = pickle.load(
                        open(sampled_unchanged_cache_path, "rb")
                    )
                    print(
                        f"Loaded sampled unchanged from {sampled_unchanged_cache_path}"
                    )
                else:
                    sampled_unchanged = random.sample(
                        all_unchanged, n_unchanged_to_sample
                    )
                    pickle.dump(
                        sampled_unchanged, open(sampled_unchanged_cache_path, "wb")
                    )
                    print(f"Saved sampled unchanged to {sampled_unchanged_cache_path}")

            print(f"# Total changed ({data_split}) available: {len(all_changed)}")
            print(f"# Total unchanged ({data_split}) available: {len(all_unchanged)}")
            print(f"# Changed sampled ({data_split}): {len(sampled_changed)}")
            print(f"# Unchanged sampled ({data_split}): {len(sampled_unchanged)}")
            print(f"# Target ratio ({data_split}): {self.n_changed}:{self.n_unchanged}")
            if len(sampled_changed) > 0 and len(sampled_unchanged) > 0:
                actual_ratio = len(sampled_changed) / len(sampled_unchanged)
                target_ratio = self.n_changed / self.n_unchanged
                print(
                    f"# Actual ratio achieved ({data_split}): {actual_ratio:.2f} (target: {target_ratio:.2f})"
                )

            # Combine sampled data
            all_sampled = sampled_changed + sampled_unchanged
        else:
            print(f"# Total changed ({data_split}): {len(all_changed)}")
            print(f"# Total unchanged ({data_split}): {len(all_unchanged)}")
            all_sampled = all_changed + all_unchanged

        if all_sampled:
            if os.path.exists(cache_path):
                combined_df = pd.read_parquet(cache_path)
                print(f"Loaded combined df from {cache_path}")
            else:
                combined_df = pd.DataFrame(all_sampled)
                # Shuffle to mix changed and unchanged
                combined_df = combined_df.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )
                combined_df.to_parquet(cache_path)
                print(f"Saved combined df to {cache_path}")
        else:
            combined_df = pd.DataFrame()

        return self._process_intervention_dataframe(combined_df)

    def _load_legacy_intervention_data(self, intervention_path, data_split):
        """Load intervention data from legacy layer-based parquet files"""
        # Load and combine all parquet files
        all_data = []
        batch_num = 0
        max_num_batches = 1 if self.debug else 100
        total_samples_loaded = 0
        total_files_loaded = 0

        # Calculate total expected files for progress bar
        total_expected_files = (
            self.target_model.config.num_hidden_layers * 2 * max_num_batches
        )  # layers * types * batches

        # Create progress bar
        pbar = tqdm(
            total=total_expected_files,
            desc="Loading legacy intervention data",
            unit="files",
            unit_scale=True,
        )

        while batch_num < max_num_batches:
            for layer in range(self.target_model.config.num_hidden_layers):
                # for type in ["changed", "unchanged"]:
                # Find parquet files for this split
                parquet_file = os.path.join(
                    intervention_path,
                    data_split,
                    str(layer),
                    f"batch_{batch_num}.parquet",
                )

                if not os.path.exists(parquet_file):
                    pbar.update(1)  # Update progress even for missing files
                    continue

                # Load and process the parquet file
                df = pd.read_parquet(parquet_file)
                total_files_loaded += 1

                # Sample examples per unique input before adding to all_data
                sampled_df = self._sample_examples_per_input(df)
                # sampled_df = df
                if not sampled_df.empty:
                    all_data.append(sampled_df)
                    total_samples_loaded += len(sampled_df)

                # Update progress bar with current stats
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "samples": total_samples_loaded,
                        "files": total_files_loaded,
                        "batch": batch_num,
                    }
                )
            batch_num += 1

        pbar.close()

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        return self._process_intervention_dataframe(combined_df)

    def _process_intervention_dataframe(self, combined_df):
        """Process the combined intervention dataframe into the format expected by BaseDataset"""
        if combined_df.empty:
            return [], [], [], [], {}

        # Create hashable column for unique counting
        combined_df["input_tokens_hashable"] = combined_df["input_tokens"].apply(
            lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
        )
        num_unique_samples = len(combined_df["input_tokens_hashable"].unique())
        print(
            f"Loaded {len(combined_df)} intervention examples with {num_unique_samples} unique exemplars"
        )

        # # shuffle entries
        # combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        # Convert to the format expected by BaseDataset
        icl_examples = []  # No ICL examples for intervention task
        icl_labels = []
        non_icl_examples = []
        non_icl_labels = []
        example_to_dataset_name = {}

        # Process intervention data
        for i, row in tqdm(
            combined_df.iterrows(),
            total=len(combined_df),
            desc="Processing intervention data",
        ):
            # Create a unique example identifier
            example_id = {
                "layer": (
                    row["layer"].tolist()
                    if isinstance(row["layer"], np.ndarray)
                    else row["layer"]
                ),
                "text": "".join(row["input_tokens"]),
            }
            if "ablated_token_positions" in row:
                example_id["ablated_token_positions"] = row[
                    "ablated_token_positions"
                ].tolist()
            if "feature_idx" in row:
                example_id["feature_idx"] = row["feature_idx"]
            example_id = json.dumps(example_id)

            if (
                self.predictor_tokenizer.eos_token in row["original_continuation"]
                or self.predictor_tokenizer.eos_token in row["ablated_continuation"]
            ):
                continue

            # Store the intervention data as the label
            label = {
                "input_tokens": row["input_tokens"],
                "original_continuation": row["original_continuation"],
                "ablated_continuation": row["ablated_continuation"],
                "chosen_label": self._format_causal_change(
                    row["original_continuation"],
                    row["ablated_continuation"],
                )[2],
                "all_labels": [
                    self._format_causal_change(
                        row["original_continuation"],
                        row["ablated_continuation"],
                    )[2]
                ],
                "is_different": row["is_different"],
                "all_label_scores": {},
                "dataset": f"causal_{self.config['intervention_type']}",
            }
            if self.config["intervention_type"] in [
                "activation_patch",
                "decode_layer_token",
            ]:
                label["patch_position"] = row["patch_position"]
                label["counterfactual_text"] = row["counterfactual_text"]
                label["gt_original_target"] = row["gt_original_target"]
                label["gt_counterfactual_target"] = row["gt_counterfactual_target"]
            if "ablated_token_positions" in row:
                label["ablated_token_positions"] = row["ablated_token_positions"]
            if "ablated_tokens" in row:
                label["ablated_tokens"] = row["ablated_tokens"]
            if "description" in row:
                label["description"] = row["description"]
            if "activating_positions" in row:
                label["activating_positions"] = row["activating_positions"]
            if "activation_values" in row:
                label["activation_values"] = row["activation_values"]
            if "token_position_weights" in row:
                label["token_position_weights"] = row["token_position_weights"]

            non_icl_examples.append(example_id)
            non_icl_labels.append(label)
            # example_to_dataset_name[example_id] =
            example_to_dataset_name[example_id] = (
                f"causal_{self.config['intervention_type']}"
            )

            if self.debug and len(non_icl_examples) >= 1000:
                break

        print(f"Prepared {len(non_icl_examples)} examples for training")
        return (
            non_icl_examples,
            non_icl_labels,
            icl_examples,
            icl_labels,
            example_to_dataset_name,
        )

    def _format_causal_change(self, original_continuation, ablated_continuation):
        """Format the causal change description"""
        if not isinstance(original_continuation, str):
            original_response = self.predictor_tokenizer.convert_tokens_to_string(
                original_continuation
            )
        else:
            original_response = original_continuation
        if not isinstance(ablated_continuation, str):
            new_response = self.predictor_tokenizer.convert_tokens_to_string(
                ablated_continuation
            )
        else:
            new_response = ablated_continuation

        if original_response == new_response:
            return (
                original_response,
                new_response,
                f"The output would remain unchanged from <<<{original_response}>>>.",
                # f"remain unchanged from <<<{original_response}>>>.",
            )
        else:
            return (
                original_response,
                new_response,
                # f"The most likely output would change from <<<{original_response}>>> to <<<{new_response}>>>.",
                f"The most likely output would change to <<<{new_response}>>>.",
                # f"change to <<<{new_response}>>>.",
            )

    def _format_judge_verdict_change(self, with_sys_verdict, no_sys_verdict):
        """4-way classification target: (changed/unchanged) x (refusal/compliance),
        driven by judge verdicts. Format mirrors `_format_causal_change` so the
        existing `<<<...>>>` evaluator regex parses it for free.

        Returns: (payload, formatted_string, is_changed)
        """
        is_changed = with_sys_verdict != no_sys_verdict
        # Payload = noun form of the post-ablation verdict (no system prompt).
        # For unchanged cases this equals the with-sys verdict by definition.
        payload = "refusal" if no_sys_verdict == "Yes" else "compliance"
        if is_changed:
            formatted = f"The most likely output would change to <<<{payload}>>>."
        else:
            formatted = f"The output would remain unchanged from <<<{payload}>>>."
        return payload, formatted, is_changed

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        """Iterate through data and convert to training examples."""
        samples_yielded = 0
        samples_skipped = 0
        pbar = tqdm(
            enumerate(self.non_icl_examples),
            total=len(self.non_icl_examples),
            desc="Generating samples",
        )
        # question_type_sequence = self.split_to_task[self.data_split]
        # Generate samples based on weights
        # For activation prediction: generate one sample per activation
        for s, (example, question_types, split) in pbar:
            pbar.set_postfix(
                {
                    "samples_yielded": samples_yielded,
                    "samples_skipped": samples_skipped,
                }
            )
            for question_type in question_types:
                # Check if we've reached the sample limit
                if self.num_samples is not None and samples_yielded >= self.num_samples:
                    break

                # example = (sample, question_types)
                example_label = self.non_icl_labels[s]

                if example_label.get("judge_skip", False):
                    samples_skipped += 1
                    continue

                # `skip_question_types` is a train-only filter (e.g. jabberwocky
                # rows skip generative_explanation during training because the
                # random_hint/zeroshot Yes/No targets are construction-time
                # placeholders, not real explanations). At test time we still
                # want predictions on those qtypes so eval can measure whether
                # the model recovers the synthetic GT.
                if (
                    self.data_split == "train"
                    and question_type in example_label.get("skip_question_types", ())
                ):
                    samples_skipped += 1
                    continue

                if example_label["dataset"] == "hint":
                    training_example = self.get_labels_hint(
                        example=example,
                        messages_so_far=[],
                        label=example_label,
                        question_type=question_type,
                    )
                elif example_label["dataset"] == "aita":
                    training_example = self.get_labels_aita(
                        example=example,
                        messages_so_far=[],
                        label=example_label,
                        question_type=question_type,
                    )
                elif example_label["dataset"] == "judge_verdict":
                    training_example = self.get_labels_judge_verdict(
                        example=example,
                        messages_so_far=[],
                        label=example_label,
                        question_type=question_type,
                    )
                else:
                    training_example = self.get_labels_intervention(
                        example=example,
                        messages_so_far=[],
                        label=example_label,
                        question_type=question_type,
                    )
                training_example = {
                    "example": example,
                    "prompt": training_example.prompt,
                    "prompt_without_answer": training_example.prompt_without_answer,
                    "expected_label": training_example.expected_label,
                    "evaluation_type": training_example.evaluation_type,
                    "task": training_example.task,
                    "dataset_name": training_example.dataset_name,
                    "question_type": training_example.question_type,
                    "extra_args": training_example.extra_args,
                }
                if isinstance(training_example["prompt"], list):
                    prompt = self.predictor_tokenizer.apply_chat_template(
                        training_example["prompt"], tokenize=False
                    )
                else:
                    prompt = training_example["prompt"]
                
                # Check prompt length to prevent OOM
                prompt_tokens = self.predictor_tokenizer.tokenize(prompt)
                prompt_length = len(prompt_tokens)
                
                # Also check full sequence length (prompt + label) if label exists
                if training_example.get("expected_label"):

                    label_tokens = self.predictor_tokenizer.tokenize(training_example["expected_label"])
                    total_length = prompt_length + len(label_tokens)
                else:
                    total_length = prompt_length
                
                # Skip if prompt or total sequence is too long
                if prompt_length > self.max_prompt_length or total_length > self.max_sequence_length:
                    samples_skipped += 1
                    continue

                yield training_example
                samples_yielded += 1

                # Break outer loop if we've reached the limit
                if self.num_samples is not None and samples_yielded >= self.num_samples:
                    break

    def __getitem__(self, idx):
        return None

    def fill_in_prompt_args(
        self,
        tokenizer,
        prompt_template: list[dict[str, str]] | str,
        prompt_args: dict[str, str],
        expected_label: str | None = None,
        full_assistant_response: bool = False,
    ) -> tuple[list[dict[str, str]] | None, list[dict[str, str]], str | None]:
        # fill in template with args

        if isinstance(prompt_template, str):
            # flat template
            prompt_without_answer = prompt_template.format(**prompt_args)
            if expected_label is None:
                return None, prompt_without_answer, None
            expected_label = expected_label  # + tokenizer.eos_token
            prompt_with_answer = prompt_without_answer + expected_label
            return prompt_with_answer, prompt_without_answer, expected_label

        prompt_with_answer = copy.deepcopy(prompt_template)
        last_user_turn_idx = -1

        for i, message in enumerate(prompt_with_answer):
            if message["role"] == "user":
                last_user_turn_idx = i
            # Format the message with only the args that appear in the message
            format_args = {}
            for arg in prompt_args:
                if f"{{{arg}}}" in message.get("content", ""):
                    format_args[arg] = prompt_args[arg]

            if format_args:  # Only format if we found matching args
                prompt_with_answer[i]["content"] = prompt_with_answer[i][
                    "content"
                ].format(**format_args)

        prompt_without_answer = copy.deepcopy(prompt_with_answer)

        if expected_label is None:
            return None, prompt_without_answer, None

        # Add assistant response
        if last_user_turn_idx == len(prompt_template) - 1:
            # last turn is a user turn
            prompt_with_answer.append({"role": "assistant", "content": expected_label})
            if not full_assistant_response:
                prompt_without_answer.append({"role": "assistant", "content": ""})
            else:
                expected_label = tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": expected_label}], tokenize=False
                )
                # remove system prompt portion and only get final assistant response
                if "<|start_header_id|>" in expected_label:
                    # llama template
                    expected_label_start_idx = expected_label.index(
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                elif "<start_of_turn>" in expected_label:
                    expected_label_start_idx = expected_label.index(
                        "<start_of_turn>model\n"
                    )
                elif "<|im_start|>" in expected_label:
                    expected_label_start_idx = expected_label.index(
                        "<|im_start|>assistant"
                    )
                elif "<|start|>assistant" in expected_label:
                    # gpt-oss / harmony
                    expected_label_start_idx = expected_label.index(
                        "<|start|>assistant"
                    )

                expected_label = expected_label[expected_label_start_idx:]
        else:
            # last turn is an assistant turn
            if "content" in prompt_with_answer[-1]:
                prompt_with_answer[-1]["content"] += expected_label
            else:
                prompt_with_answer[-1]["content"] = expected_label

        try:
            assert prompt_with_answer[:-1] == prompt_without_answer[:-1]
            # assert prompt_with_answer[-1]["content"] != prompt_without_answer[-1].get(
            #     "content", ""
            # )
        except:
            print(prompt_with_answer)
            print(prompt_without_answer)
            print(expected_label)
            raise

        return prompt_with_answer, prompt_without_answer, expected_label

    def recompute_labels_intervention(
        self,
        input_text,
        # messages_so_far,
        label: dict[str, Any] | None = None,
        # question_type=None,
        layer=None,
        # idx=None,
        feature_continuous_token=None,
    ):
        tokens = self.target_tokenizer(input_text, return_tensors="pt")
        special_tokens = [
            self.target_tokenizer.bos_token_id,
            self.target_tokenizer.eos_token_id,
            self.target_tokenizer.pad_token_id,
        ]
        start_token_pos = 0
        while (
            start_token_pos < len(tokens["input_ids"][0])
            and tokens["input_ids"][0][start_token_pos] in special_tokens
        ):
            start_token_pos += 1
        self.target_model.eval()
        with torch.no_grad():
            tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}

            # 1. get original continuation
            original_output = self.target_model(**tokens)
            original_logits_full = original_output.logits[0, -1, :]
            original_next_token = torch.argmax(original_logits_full, dim=-1)
            original_continuation = [
                self.target_tokenizer.convert_tokens_to_string(
                    self.target_tokenizer.convert_ids_to_tokens([original_next_token])
                )
            ]

            # Save top-k teacher logits for KL regularization (original/unablated)
            if self.use_kl_regularization:
                topk_vals, topk_idx = original_logits_full.float().topk(self.kl_topk)
                label["teacher_logits_original"] = (topk_idx.cpu(), topk_vals.cpu())

            # 2. get ablated continuation
            with self.nnsight_model.trace(tokens) as tracer:
                if self.question_config["intervention_type"] == "activation_patch":
                    feature_continuous_token = torch.tensor(
                        feature_continuous_token,
                        device=self.nnsight_model.device,
                        dtype=torch.bfloat16,
                    )
                    assert (
                        self.target_tokenizer.convert_ids_to_tokens(
                            tokens["input_ids"][:, label["patch_position"]["orig_pos"]]
                        )[0]
                        == label["patch_position"]["orig_text_token"]
                    )
                    for l_idx in layer:
                        self.nnsight_model.model.layers[l_idx].output[
                            0, label["patch_position"]["orig_pos"]
                        ] = feature_continuous_token  # .to(self.nnsight_model.device)
                else:
                    for pos, ablated_token_position in enumerate(
                        label["ablated_token_positions"]
                    ):
                        # ablated_token_position = label["ablated_token_positions"][pos]
                        ablated_token_position_padded = (
                            ablated_token_position + start_token_pos
                        )
                        if (
                            self.question_config["intervention_type"]
                            == "layer_ablation"
                        ):
                            self.nnsight_model.model.layers[layer].output[
                                0, ablated_token_position_padded
                            ] = 0
                        else:
                            raise ValueError(
                                f"Invalid intervention type: {self.question_config['intervention_type']}"
                            )
                ablated_logits = self.nnsight_model.lm_head.output[0, -1, :].save()

            ablated_next_token = torch.argmax(ablated_logits, dim=-1)
            ablated_continuation = [
                self.target_tokenizer.convert_tokens_to_string(
                    self.target_tokenizer.convert_ids_to_tokens([ablated_next_token])
                )
            ]

            # Save top-k teacher logits for KL regularization (ablated)
            if self.use_kl_regularization:
                topk_vals, topk_idx = ablated_logits.float().topk(self.kl_topk)
                label["teacher_logits_ablated"] = (topk_idx.cpu(), topk_vals.cpu())

            label["original_continuation"] = original_continuation
            label["ablated_continuation"] = ablated_continuation

        return label

    def recompute_labels_hint(
        self,
        with_hint_text,
        no_hint_text,
        label: dict[str, Any] | None = None,
    ):
        # Some hint datasets (e.g. synthetic jabberwocky) ship with empty
        # *_prompt_full fields and rebuild them lazily via the chat template.
        # Reconstruct here if needed so self_consistency re-eval works.
        if (not with_hint_text) and label is not None:
            with_hint_text = self.target_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": label["system_prompt"]},
                    {"role": "user", "content": label["hint_user_prompt"]},
                    {"role": "assistant", "content": "Answer:"},
                ],
                tokenize=False,
                continue_final_message=True,
            )
        if (not no_hint_text) and label is not None:
            no_hint_text = self.target_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": label["system_prompt"]},
                    {"role": "user", "content": label["original_user_prompt"]},
                    {"role": "assistant", "content": "Answer:"},
                ],
                tokenize=False,
                continue_final_message=True,
            )
        self.target_model.eval()
        with torch.no_grad():
            # 1. get original continuation (with hint)
            tokens = self.target_tokenizer(with_hint_text, return_tensors="pt")
            tokens = {k: v.to(self.target_model.device).long() for k, v in tokens.items()}
            original_output = self.target_model(**tokens)
            original_logits = original_output.logits[0, -1, :]
            original_next_token = torch.argmax(original_logits, dim=-1)
            original_continuation = self.target_tokenizer.convert_tokens_to_string(
                self.target_tokenizer.convert_ids_to_tokens([original_next_token])
            ).strip()

            # Save top-k teacher logits for KL regularization
            if self.use_kl_regularization:
                topk_vals, topk_idx = original_logits.float().topk(self.kl_topk)
                label["teacher_logits_with_hint"] = (topk_idx.cpu(), topk_vals.cpu())

            # 2. get no hint continuation
            tokens = self.target_tokenizer(no_hint_text, return_tensors="pt")
            tokens = {k: v.to(self.target_model.device).long() for k, v in tokens.items()}
            zeroshot_output = self.target_model(**tokens)
            zeroshot_logits = zeroshot_output.logits[0, -1, :]
            zeroshot_next_token = torch.argmax(zeroshot_logits, dim=-1)
            zeroshot_prediction = self.target_tokenizer.convert_tokens_to_string(
                self.target_tokenizer.convert_ids_to_tokens([zeroshot_next_token])
            ).strip()

            # Save top-k teacher logits for KL regularization
            if self.use_kl_regularization:
                topk_vals, topk_idx = zeroshot_logits.float().topk(self.kl_topk)
                label["teacher_logits_no_hint"] = (topk_idx.cpu(), topk_vals.cpu())
        label["random_hint_prediction"] = original_continuation
        label["zeroshot_prediction"] = zeroshot_prediction
        label["changed_pred_bc_hint"] = original_continuation != zeroshot_prediction

        return label

    def _greedy_next_n_tokens(self, text, n):
        """Generate up to *n* tokens greedily, stopping early on EOS.

        Returns a list of decoded token strings (length <= n).
        """
        tokens = self.target_tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.target_model.device)
        attention_mask = tokens["attention_mask"].to(self.target_model.device)
        eos_id = self.target_tokenizer.eos_token_id

        result_strs = []
        for _ in range(n):
            logits = self.target_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[0, -1, :]
            tok_id = torch.argmax(logits, dim=-1)

            if tok_id.item() == eos_id:
                break

            result_strs.append(
                self.target_tokenizer.convert_tokens_to_string(
                    self.target_tokenizer.convert_ids_to_tokens([tok_id])
                ).strip()
            )

            input_ids = torch.cat(
                [input_ids, tok_id.unsqueeze(0).unsqueeze(0)], dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=attention_mask.device, dtype=attention_mask.dtype)], dim=-1
            )

        return result_strs

    def recompute_labels_aita(
        self,
        with_sys_text,
        no_sys_text,
        label: dict[str, Any],
        max_new_tokens: int = 2,
    ):
        """Recompute AITA labels by running the target model on with/without system prompt.

        Takes pre-formatted prompt texts and gets the model's next-token prediction
        for each, analogous to recompute_labels_hint.

        Generates up to ``max_new_tokens`` tokens greedily (controlled by
        ``generate_limit`` in the YAML config), stopping early on EOS.
        """
        self.target_model.eval()
        with torch.no_grad():
            # Qwen3-8B often writes the verdict in markdown bold (" **YTA**"), which
            # 2 tokens truncate to "**Y". Read a few more tokens and reduce to the
            # bare verdict, matching how make_data_scripts/aita_vllm.py stores labels.
            n_tok = max(max_new_tokens, 5)
            with_toks = self._greedy_next_n_tokens(with_sys_text, n_tok)
            with_sys_pred = _normalize_aita_verdict("".join(with_toks))

            no_toks = self._greedy_next_n_tokens(no_sys_text, n_tok)
            no_sys_pred = _normalize_aita_verdict("".join(no_toks))

            # Save top-k teacher logits for KL regularization (first token position)
            if self.use_kl_regularization:
                tokens = self.target_tokenizer(with_sys_text, return_tensors="pt")
                tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
                with_logits = self.target_model(**tokens).logits[0, -1, :]
                topk_vals, topk_idx = with_logits.float().topk(self.kl_topk)
                label["teacher_logits_with_sys"] = (topk_idx.cpu(), topk_vals.cpu())

                tokens = self.target_tokenizer(no_sys_text, return_tensors="pt")
                tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
                no_logits = self.target_model(**tokens).logits[0, -1, :]
                topk_vals, topk_idx = no_logits.float().topk(self.kl_topk)
                label["teacher_logits_no_sys"] = (topk_idx.cpu(), topk_vals.cpu())

        label["with_system_prediction"] = with_sys_pred
        label["no_system_prediction"] = no_sys_pred
        label["changed_pred"] = with_sys_pred != no_sys_pred

        return label

    def extract_prompt_role(
        self,
        input_text,
        role="user",
    ):
        if f"<|start_header_id|>{role}<|end_header_id|>\n\n" in input_text:
            return input_text.split(f"<|start_header_id|>{role}<|end_header_id|>\n\n")[
                1
            ].split("<|eot_id|>")[0]
        elif f"<start_of_turn>{role}\n" in input_text:
            return input_text.split(f"<start_of_turn>{role}\n")[1].split(
                "<end_of_turn>\n"
            )[0]
        elif f"<|im_start|>{role}" in input_text:
            return input_text.split(f"<|im_start|>{role}")[1].split("<|im_end|>\n")[0]
        elif f"<|start|>{role}" in input_text:
            # gpt-oss / harmony: role header is <|start|>{role}[<|channel|>...]
            # <|message|>content<|end|> (assistant final turns end <|return|>).
            segment = input_text.split(f"<|start|>{role}")[1]
            segment = segment.split("<|message|>", 1)[-1]
            return segment.split("<|end|>")[0].split("<|return|>")[0]
        else:
            return input_text

    def remove_hint(self, input_text):
        # Remove hint text matching the pattern: '\nHint: I think the answer is ... but I'm curious to hear what you think.'
        pattern = r"\nHint: I think the answer is \([A-D]\) but I'm curious to hear what you think\."
        return re.sub(pattern, "", input_text)

    def get_labels_hint(
        self,
        example: str,
        messages_so_far: list[dict[str, str]],
        label: dict[str, Any] | None = None,
        question_type=None,
    ):
        """
        example: string (prompt)
        label: dict[str, Any] | None
        """
        question_config = self.question_config["question_types"][question_type]

        chosen_prompt_template = copy.deepcopy(
            random.choice(question_config["prompts"])
        )
        input_text = example
        if isinstance(chosen_prompt_template, str):
            prompt_template = "\n".join([*messages_so_far, chosen_prompt_template])
        else:
            prompt_template = [
                *messages_so_far,
                *chosen_prompt_template["messages"],
            ]
        # if question_type == "change_match":
        #     prompt = self.remove_hint(label["prompt_text"])
        # else:
        if question_type == "change_match":
            prompt = label["original_prompt_full"]
        elif (
            question_type == "generative_explanation"
            and self.flip_explanation_direction
        ):
            prompt = label["original_prompt_full"]
        else:
            prompt = label["hint_prompt_full"]
        # if question_type == "generative_explanation":
        if isinstance(chosen_prompt_template, str):
            if question_type == "generative_explanation":
                # Support both Llama and Qwen prompt endings
                VALID_ENDINGS = ["\n\nAssistant: Answer:", "</think>\n\nAnswer:", "<|channel|>final<|message|>Answer:"]
                matched_ending = None
                for ending in VALID_ENDINGS:
                    if prompt.endswith(ending):
                        matched_ending = ending
                        break
                assert matched_ending is not None, f"Prompt must end with one of {VALID_ENDINGS}, got: ...{prompt[-50:]}"
                prompt = prompt[: -len(matched_ending)]
            prompt_args = {
                "prompt": prompt,
                # "with_hint_answer": f" {label['random_hint_prediction']}",
            }
        elif question_type == "orig_match" or (
            question_type == "generative_explanation"
            and not self.flip_explanation_direction
        ):
            prompt_args = {
                "user_prompt": label["hint_user_prompt"],
                "system_prompt": label["system_prompt"],
                # Carried for templates that reference {hint}; harmless when absent
                # (fill_in_prompt_args only interpolates placeholders in the content).
                "hint": label.get("hint", ""),
            }
        elif question_type == "change_match" or (
            question_type == "generative_explanation"
            and self.flip_explanation_direction
        ):
            prompt_args = {
                "user_prompt": label["original_user_prompt"],
                "system_prompt": label["system_prompt"],
                # Addition-direction generative_explanation builds its prompt from
                # the no-hint question, so the hint letter is otherwise absent and
                # the meta target (random_hint_prediction) can't be predicted.
                # Expose the hint as its own section; fill_in_prompt_args only
                # interpolates it when the template contains {hint} (change_match
                # does not, so it is unaffected).
                "hint": label.get("hint", ""),
            }
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        if self.self_consistency:
            # Force re-template via target_tokenizer.apply_chat_template:
            # `hint_prompt_full` / `original_prompt_full` carry the source
            # jsonl's chat template (always Qwen-templated for the alpha mix),
            # so feeding them straight into a Llama target ran Llama on Qwen
            # markers and produced drifted GT labels. Passing "" triggers the
            # re-template branch in recompute_labels_hint using the raw
            # system / user prompts, which is template-agnostic.
            label = self.recompute_labels_hint(
                with_hint_text="",
                no_hint_text="",
                label=label,
            )
        elif (
            self.use_kl_regularization
            and question_type in ["orig_match", "change_match"]
            and "teacher_logits_with_hint" not in label
        ):
            # Try precomputed teacher logits first, fall back to on-demand computation
            sample_key = label.get("index")
            if (
                self.precomputed_teacher_logits is not None
                and sample_key is not None
                and sample_key in self.precomputed_teacher_logits
            ):
                precomputed = self.precomputed_teacher_logits[sample_key]
                label["teacher_logits_with_hint"] = precomputed["with_hint"]
                label["teacher_logits_no_hint"] = precomputed["no_hint"]
            else:
                # On-demand teacher logit computation
                self.target_model.eval()
                with torch.no_grad():
                    tokens = self.target_tokenizer(label["hint_prompt_full"], return_tensors="pt")
                    tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
                    hint_logits = self.target_model(**tokens).logits[0, -1, :]
                    topk_vals, topk_idx = hint_logits.float().topk(self.kl_topk)
                    label["teacher_logits_with_hint"] = (topk_idx.cpu(), topk_vals.cpu())

                    tokens = self.target_tokenizer(label["original_prompt_full"], return_tensors="pt")
                    tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
                    no_hint_logits = self.target_model(**tokens).logits[0, -1, :]
                    topk_vals, topk_idx = no_hint_logits.float().topk(self.kl_topk)
                    label["teacher_logits_no_hint"] = (topk_idx.cpu(), topk_vals.cpu())

        if question_type == "generative_explanation":
            # `gen_*` overrides exist on alpha-mixed datasets where
            # generative_explanation is supervised against random labels while
            # orig_match / change_match stay anchored to the qwen baseline
            # (see scripts/mix_qwen_random_ht8.py). Falls back to the canonical
            # fields for legacy datasets.
            gen_rh = label.get("gen_random_hint_prediction", label["random_hint_prediction"])
            gen_zs = label.get("gen_zeroshot_prediction", label["zeroshot_prediction"])
            if self.flip_explanation_direction:
                original_response, new_response, response = self._format_causal_change(
                    "Answer: " + gen_zs,
                    "Answer: " + gen_rh,
                )
            else:
                original_response, new_response, response = self._format_causal_change(
                    "Answer: " + gen_rh,
                    "Answer: " + gen_zs,
                )
            if self.predictor_tokenizer.chat_template is None:
                response = " " + response
        elif question_type == "orig_match":
            original_response = " " + label["random_hint_prediction"].strip()
            new_response = " " + label["zeroshot_prediction"].strip()
            response = original_response
        elif question_type == "change_match":
            original_response = " " + label["random_hint_prediction"].strip()
            new_response = " " + label["zeroshot_prediction"].strip()
            response = new_response
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        try:
            prompt_with_answer, prompt_without_answer, expected_label = (
                self.fill_in_prompt_args(
                    self.predictor_tokenizer,
                    prompt_template,
                    prompt_args,
                    response,
                    full_assistant_response=False,
                )
            )
        except:
            raise
        # if question_type == "change_match":
        #     prompt_without_answer = self.remove_hint(prompt_without_answer)
        #     prompt_with_answer = self.remove_hint(prompt_with_answer)

        # Always recompute from actual predictions to avoid stale labels
        label["changed_pred_bc_hint"] = original_response != new_response
        other_features = {
            "input": input_text,
            "original_response": original_response,
            "ablated_response": new_response,
            "hint": label["hint"],
            "changed_pred_bc_hint": label["changed_pred_bc_hint"],
            "is_different": original_response != new_response,
            "question_type_weight": question_config.get("weight", 1.0),
        }

        # Stash orig/change match prompts on gen_explanation rows so the rollout
        # consistency callback can do orphan-fallback forwards without rebuilding
        # prompts from scratch when the matching qt rows aren't co-batched.
        if question_type == "generative_explanation":
            other_features["orig_match_prompt"] = label.get("hint_prompt_full")
            other_features["change_match_prompt"] = label.get("original_prompt_full")

        # Add teacher logits for KL regularization
        if self.use_kl_regularization:
            if question_type == "orig_match":
                other_features["teacher_logits"] = label.get("teacher_logits_with_hint")
            elif question_type == "change_match":
                other_features["teacher_logits"] = label.get("teacher_logits_no_hint")

        if question_type == "orig_match":
            full_prompt = format_messages(
                self.predictor_tokenizer,
                prompt_without_answer,
                remove_final_assistant_eot=True,
                add_final_assistant_header=False,
            )
            # Skip assertion if using data from different tokenizer (e.g., Llama data with Qwen tokenizer)
            # The stored prompt was formatted with a different chat template
            stored_has_qwen = "<|im_start|>" in label["hint_prompt_full"]
            stored_has_llama = "<|start_header_id|>" in label["hint_prompt_full"] or "<start_of_turn>" in label["hint_prompt_full"]
            stored_has_harmony = "<|channel|>" in label["hint_prompt_full"]
            current_template = self.predictor_tokenizer.chat_template or ""
            current_has_qwen = "<|im_start|>" in current_template
            current_has_llama = "<|start_header_id|>" in current_template or "<start_of_turn>" in current_template
            current_has_harmony = "<|channel|>final<|message|>" in current_template

            # Only assert if tokenizers are compatible (both Qwen or both Llama)
            if (stored_has_qwen and current_has_qwen) or (stored_has_llama and current_has_llama) or (stored_has_harmony and current_has_harmony):
                assert label["hint_prompt_full"] == full_prompt
        elif question_type == "change_match":
            full_prompt = format_messages(
                self.predictor_tokenizer,
                prompt_without_answer,
                remove_final_assistant_eot=True,
                add_final_assistant_header=False,
            )
            # Skip assertion if using data from different tokenizer (e.g., Llama data with Qwen tokenizer)
            # The stored prompt was formatted with a different chat template
            stored_has_qwen = "<|im_start|>" in label["original_prompt_full"]
            stored_has_llama = "<|start_header_id|>" in label["original_prompt_full"] or "<start_of_turn>" in label["original_prompt_full"]
            stored_has_harmony = "<|channel|>" in label["original_prompt_full"]
            current_template = self.predictor_tokenizer.chat_template or ""
            current_has_qwen = "<|im_start|>" in current_template
            current_has_llama = "<|start_header_id|>" in current_template or "<start_of_turn>" in current_template
            current_has_harmony = "<|channel|>final<|message|>" in current_template

            # Only assert if tokenizers are compatible (both Qwen or both Llama)
            if (stored_has_qwen and current_has_qwen) or (stored_has_llama and current_has_llama) or (stored_has_harmony and current_has_harmony):
                assert label["original_prompt_full"] == full_prompt
        return ExampleLabels(
            prompt=prompt_with_answer,
            prompt_without_answer=prompt_without_answer,
            expected_label=expected_label,
            evaluation_type=self.question_config["evaluation_type"],
            task=self.question_config["intervention_type"],
            question_type=question_type,
            dataset_name=self.example_to_dataset_name[example],
            extra_args=other_features,
        )

    def get_labels_aita(
        self,
        example: str,
        messages_so_far: list[dict[str, str]],
        label: dict[str, Any] | None = None,
        question_type=None,
    ):
        """Get labels for AITA dataset where the intervention is the system prompt.

        - orig_match:              predict the answer WITH system prompt
        - change_match:            predict the answer WITHOUT system prompt
        - generative_explanation:  predict how the output changes when system prompt is removed

        The prompt templates from the config use {system_prompt} and {user_prompt}.
        For orig_match: system prompt is included.
        For change_match: system prompt is empty (removed).
        For generative_explanation: system prompt is included, and we ask how removing it changes the answer.
        """
        question_config = self.question_config["question_types"][question_type]

        chosen_prompt_template = copy.deepcopy(
            random.choice(question_config["prompts"])
        )
        input_text = example
        if isinstance(chosen_prompt_template, str):
            prompt_template = "\n".join([*messages_so_far, chosen_prompt_template])
        else:
            prompt_template = [
                *messages_so_far,
                *chosen_prompt_template["messages"],
            ]

        # Build prompt_args based on question type
        # Use original_user_prompt for orig/change_match if available (freeform CE),
        # otherwise fall back to user_prompt (meta-question)
        if question_type == "orig_match":
            # WITH system prompt
            prompt_args = {
                "user_prompt": label.get("original_user_prompt", label["user_prompt"]),
                "system_prompt": label["system_prompt"],
            }
        elif question_type == "change_match":
            # WITHOUT system prompt (system prompt removed)
            prompt_args = {
                "user_prompt": label.get("original_user_prompt", label["user_prompt"]),
                "system_prompt": "",
            }
        elif question_type == "generative_explanation":
            # WITH system prompt, asking how removing it changes the answer
            prompt_args = {
                "user_prompt": label["user_prompt"],
                "system_prompt": label["system_prompt"],
            }
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        # Recompute labels using the target model if self_consistency is enabled
        if self.self_consistency:
            generate_limit = question_config.get("generate_limit", 2)
            label = self.recompute_labels_aita(
                with_sys_text=label["with_system_prompt_full"],
                no_sys_text=label["no_system_prompt_full"],
                label=label,
                max_new_tokens=generate_limit,
            )
        elif (
            self.use_kl_regularization
            and question_type in ["orig_match", "change_match"]
            and "teacher_logits_with_sys" not in label
        ):
            # Try precomputed teacher logits first, fall back to on-demand computation
            sample_key = label.get("index")
            if (
                self.precomputed_teacher_logits is not None
                and sample_key is not None
                and sample_key in self.precomputed_teacher_logits
            ):
                precomputed = self.precomputed_teacher_logits[sample_key]
                label["teacher_logits_with_sys"] = precomputed["with_hint"]
                label["teacher_logits_no_sys"] = precomputed["no_hint"]
            else:
                # On-demand teacher logit computation for non-self_consistency AITA path
                self.target_model.eval()
                with torch.no_grad():
                    tokens = self.target_tokenizer(label["with_system_prompt_full"], return_tensors="pt")
                    tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
                    with_logits = self.target_model(**tokens).logits[0, -1, :]
                    topk_vals, topk_idx = with_logits.float().topk(self.kl_topk)
                    label["teacher_logits_with_sys"] = (topk_idx.cpu(), topk_vals.cpu())

                    tokens = self.target_tokenizer(label["no_system_prompt_full"], return_tensors="pt")
                    tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
                    no_logits = self.target_model(**tokens).logits[0, -1, :]
                    topk_vals, topk_idx = no_logits.float().topk(self.kl_topk)
                    label["teacher_logits_no_sys"] = (topk_idx.cpu(), topk_vals.cpu())

        # Build the expected response based on question type
        with_sys_pred = label["with_system_prediction"]
        no_sys_pred = label["no_system_prediction"]

        # aita_freeform_target: the behavior rows' target is the base model's own short reply after "Answer:"
        # (e.g. " **YTA**") with per-token teacher logits, so the KL term covers the whole verdict rather than only
        # its first token (" **"). Explanation targets keep the bare verdict.
        if (
            self.config.get("aita_freeform_target")
            and question_type in ("orig_match", "change_match")
            and label.get("aita_reply_with_sys") is not None
            and label.get("aita_reply_no_sys") is not None
        ):
            original_response = label["aita_reply_with_sys"]
            new_response = label["aita_reply_no_sys"]
            response = original_response if question_type == "orig_match" else new_response
        elif question_type == "generative_explanation":
            original_response, new_response, response = self._format_causal_change(
                "Answer: " + with_sys_pred,
                "Answer: " + no_sys_pred,
            )
            if self.predictor_tokenizer.chat_template is None:
                response = " " + response
        elif question_type == "orig_match":
            original_response = " " + with_sys_pred.strip()
            new_response = " " + no_sys_pred.strip()
            response = original_response
        elif question_type == "change_match":
            original_response = " " + with_sys_pred.strip()
            new_response = " " + no_sys_pred.strip()
            response = new_response
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        try:
            prompt_with_answer, prompt_without_answer, expected_label = (
                self.fill_in_prompt_args(
                    self.predictor_tokenizer,
                    prompt_template,
                    prompt_args,
                    response,
                    full_assistant_response=False,
                )
            )
        except:
            raise

        # Remove system messages with empty content (e.g. change_match removes system prompt)
        if isinstance(prompt_with_answer, list):
            prompt_with_answer = [
                m for m in prompt_with_answer
                if not (m["role"] == "system" and not m.get("content", "").strip())
            ]
        if isinstance(prompt_without_answer, list):
            prompt_without_answer = [
                m for m in prompt_without_answer
                if not (m["role"] == "system" and not m.get("content", "").strip())
            ]

        is_different = original_response != new_response
        other_features = {
            "input": input_text,
            "original_response": original_response,
            "ablated_response": new_response,
            "system_prompt": label["system_prompt"],
            "changed_pred": is_different,
            "is_different": is_different,
            "question_type_weight": question_config.get("weight", 1.0),
            "aita_example_id": input_text,
        }

        # Add teacher logits for KL regularization
        if self.use_kl_regularization:
            if question_type == "orig_match":
                other_features["teacher_logits"] = label.get("teacher_logits_with_sys")
            elif question_type == "change_match":
                other_features["teacher_logits"] = label.get("teacher_logits_no_sys")

        return ExampleLabels(
            prompt=prompt_with_answer,
            prompt_without_answer=prompt_without_answer,
            expected_label=expected_label,
            evaluation_type=self.question_config["evaluation_type"],
            task=self.question_config["intervention_type"],
            question_type=question_type,
            dataset_name=self.example_to_dataset_name[example],
            extra_args=other_features,
        )

    def get_labels_judge_verdict(
        self,
        example: str,
        messages_so_far: list[dict[str, str]],
        label: dict[str, Any] | None = None,
        question_type=None,
    ):
        """Get labels for the judge-verdict (system-prompt-ablation) task.

        Same three question_types as the AITA path, but generative_explanation
        uses judge verdicts (random_hint_prediction / zeroshot_prediction) to
        produce a 4-way classification target. orig_match / change_match keep
        the existing freeform-CE behavior (target = with_system_prediction /
        no_system_prediction respectively) for regularization.
        """
        question_config = self.question_config["question_types"][question_type]

        chosen_prompt_template = copy.deepcopy(
            random.choice(question_config["prompts"])
        )
        input_text = example
        if isinstance(chosen_prompt_template, str):
            prompt_template = "\n".join([*messages_so_far, chosen_prompt_template])
        else:
            prompt_template = [
                *messages_so_far,
                *chosen_prompt_template["messages"],
            ]

        if question_type == "orig_match":
            prompt_args = {
                "user_prompt": label.get("original_user_prompt", label["user_prompt"]),
                "system_prompt": label["system_prompt"],
            }
        elif question_type == "change_match":
            prompt_args = {
                "user_prompt": label.get("original_user_prompt", label["user_prompt"]),
                "system_prompt": "",
            }
        elif question_type == "generative_explanation":
            prompt_args = {
                "user_prompt": label["user_prompt"],
                "system_prompt": label["system_prompt"],
            }
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        with_sys_pred = label["with_system_prediction"]
        no_sys_pred = label["no_system_prediction"]

        # When use_freeform_target is enabled, swap the short Yes/No verdict
        # for the full freeform model rollout for orig_match / change_match.
        # Falls back to the verdict if the freeform field is missing.
        if (
            self.use_freeform_target
            and question_type in ("orig_match", "change_match")
        ):
            with_freeform = label.get("freeform_response_with_sys")
            no_freeform = label.get("freeform_response_no_sys")
            if with_freeform is not None and no_freeform is not None:
                with_sys_pred = with_freeform
                no_sys_pred = no_freeform

        if question_type == "generative_explanation":
            with_verdict = label["random_hint_prediction"]
            no_verdict = label["zeroshot_prediction"]
            payload, formatted, is_changed_verdict = self._format_judge_verdict_change(
                with_verdict, no_verdict
            )
            response = formatted
            if self.predictor_tokenizer.chat_template is None:
                response = " " + response
            original_response = with_sys_pred
            new_response = no_sys_pred
            is_different = is_changed_verdict
        elif question_type == "orig_match":
            if self.use_freeform_target:
                original_response = with_sys_pred
                new_response = no_sys_pred
            else:
                original_response = " " + with_sys_pred.strip()
                new_response = " " + no_sys_pred.strip()
            response = original_response
            is_different = original_response != new_response
        elif question_type == "change_match":
            if self.use_freeform_target:
                original_response = with_sys_pred
                new_response = no_sys_pred
            else:
                original_response = " " + with_sys_pred.strip()
                new_response = " " + no_sys_pred.strip()
            response = new_response
            is_different = original_response != new_response
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        prompt_with_answer, prompt_without_answer, expected_label = (
            self.fill_in_prompt_args(
                self.predictor_tokenizer,
                prompt_template,
                prompt_args,
                response,
                full_assistant_response=False,
            )
        )

        # Strip empty system messages (change_match removes system prompt)
        if isinstance(prompt_with_answer, list):
            prompt_with_answer = [
                m for m in prompt_with_answer
                if not (m["role"] == "system" and not m.get("content", "").strip())
            ]
        if isinstance(prompt_without_answer, list):
            prompt_without_answer = [
                m for m in prompt_without_answer
                if not (m["role"] == "system" and not m.get("content", "").strip())
            ]

        other_features = {
            "input": input_text,
            "original_response": original_response,
            "ablated_response": new_response,
            "system_prompt": label["system_prompt"],
            "changed_pred": is_different,
            "is_different": is_different,
            "question_type_weight": question_config.get("weight", 1.0),
            "aita_example_id": input_text,
        }

        # Attach precomputed teacher logits for KL regularization.
        # In freeform mode these are per-token tensors of shape [T, k];
        # otherwise legacy single-position [k] tensors. The KL loss handles both.
        if (
            self.use_kl_regularization
            and question_type in ("orig_match", "change_match")
            and self.precomputed_teacher_logits is not None
        ):
            sample_key = label.get("index")
            precomputed = self.precomputed_teacher_logits.get(sample_key)
            if precomputed is not None:
                if question_type == "orig_match":
                    other_features["teacher_logits"] = precomputed.get("with_hint")
                else:
                    other_features["teacher_logits"] = precomputed.get("no_hint")

        return ExampleLabels(
            prompt=prompt_with_answer,
            prompt_without_answer=prompt_without_answer,
            expected_label=expected_label,
            evaluation_type=self.question_config["evaluation_type"],
            task=self.question_config["intervention_type"],
            question_type=question_type,
            dataset_name=self.example_to_dataset_name[example],
            extra_args=other_features,
        )

    def get_labels_intervention(
        self,
        example,
        messages_so_far,
        label: dict[str, Any] | None = None,
        question_type=None,
        idx=None,
    ):
        """
        example: (layer, feature_idx)
        messages_so_far: list[dict[str, str]]
        label: dict[str, Any] | None
        question_type: str

        Returns:
            ExampleLabels
        """
        example_dict = json.loads(example)
        layer = example_dict["layer"]
        question_config = self.question_config["question_types"][question_type]

        chosen_prompt_template = copy.deepcopy(
            random.choice(question_config["prompts"])
        )
        input_text = self.predictor_tokenizer.convert_tokens_to_string(
            label["input_tokens"]
        )
        if isinstance(chosen_prompt_template, str):
            prompt_template = "\n".join([*messages_so_far, chosen_prompt_template])
        else:
            prompt_template = [
                *messages_so_far,
                *chosen_prompt_template["messages"],
            ]
        prompt_args = {}
        if self.question_config["intervention_type"] == "activation_patch":
            layer_str = ", ".join(str(layer) for layer in layer)
            text_token = self.target_tokenizer.convert_tokens_to_string(
                [label["patch_position"]["orig_text_token"]]
            )
            ablated_tokens = f"<{text_token}>"
            prompt_args.update({"tokens": ablated_tokens})
        else:
            layer_str = str(layer)
        prompt_args.update(
            {
                "layer": layer_str,
                "input": input_text,
            }
        )

        if self.question_config["intervention_type"] == "activation_patch":
            feature_continuous_token = label["patch_position"]["intervention_vector"]
        else:
            feature_continuous_token = None

        if self.self_consistency:
            label = self.recompute_labels_intervention(
                input_text=input_text,
                # messages_so_far,
                label=label,
                layer=layer,
                # idx,
                feature_continuous_token=feature_continuous_token,
                # layer,
            )
        if question_type == "generative_explanation":
            # Get the input text for the intervention
            # input_text = " ".join(label["input_tokens"])

            original_response, new_response, response = self._format_causal_change(
                label["original_continuation"],
                label["ablated_continuation"],
            )
        elif question_type == "orig_match":
            original_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["original_continuation"]
            )
            new_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["ablated_continuation"]
            )
            response = original_response
        elif question_type == "change_match":
            original_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["original_continuation"]
            )
            new_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["ablated_continuation"]
            )
            response = new_response
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        prompt_with_answer, prompt_without_answer, expected_label = (
            self.fill_in_prompt_args(
                self.predictor_tokenizer,
                prompt_template,
                prompt_args,
                # label["chosen_label"],
                response,
                full_assistant_response=True,
            )
        )
        other_features = {
            "layer": layer,
            "input": input_text,
            "original_response": original_response,
            "ablated_response": new_response,
            "input_tokens": label["input_tokens"].tolist(),
            "is_different": label["is_different"],
            "question_type_weight": question_config.get("weight", 1.0),
        }

        # Add teacher logits for KL regularization
        if self.use_kl_regularization:
            if question_type == "orig_match":
                other_features["teacher_logits"] = label.get("teacher_logits_original")
            elif question_type == "change_match":
                other_features["teacher_logits"] = label.get("teacher_logits_ablated")

        if self.question_config["intervention_type"] == "activation_patch":
            other_features["patch_position"] = {
                "orig_pos": label["patch_position"]["orig_pos"],
                "counterfact_pos": label["patch_position"]["counterfact_pos"],
                "orig_text_token": label["patch_position"]["orig_text_token"],
                "counterfact_text_token": label["patch_position"][
                    "counterfact_text_token"
                ],
            }
            other_features["counterfactual_text"] = label["counterfactual_text"]
            other_features["gt_original_target"] = label["gt_original_target"]
            other_features["gt_counterfactual_target"] = label[
                "gt_counterfactual_target"
            ]

        return ExampleLabels(
            prompt=prompt_with_answer,
            prompt_without_answer=prompt_without_answer,
            expected_label=expected_label,
            evaluation_type=self.question_config["evaluation_type"],
            task=self.question_config["intervention_type"],
            question_type=question_type,
            dataset_name=self.example_to_dataset_name[example],
            extra_args=other_features,
        )


class CausalDataCollator(BaseDataCollator):
    """
    Data collator for causal intervention tasks that handles change_match questions
    by preparing intervention arguments for the model.
    """

    def __init__(self, predictor_tokenizer, target_tokenizer, question_config=None):
        super().__init__(predictor_tokenizer, target_tokenizer)
        self.question_config = question_config or {}

    def __call__(self, features):
        # First call the parent collator to handle standard processing
        batch = super().__call__(features)

        # Check for change_match questions and prepare intervention arguments
        intervention_mask = torch.zeros(len(features), dtype=torch.bool)
        intervention_layers = []
        intervention_positions = []
        intervention_vectors = []

        for i, feature in enumerate(features):
            if (
                feature.get("question_type") == "change_match"
                and feature.get("extra_args") is not None
                and not feature.get("task", "").startswith("hint")
                and "layer" in (feature.get("extra_args") or {})
            ):

                intervention_mask[i] = True
                extra_args = feature["extra_args"]

                # Extract intervention information from extra_args
                layer = extra_args["layer"]
                intervention_layers.append(layer)

                # account for padding
                start_token_pos = 0
                special_token_ids = [
                    self.target_tokenizer.bos_token_id,
                    self.target_tokenizer.eos_token_id,
                    self.target_tokenizer.pad_token_id,
                ]
                while (
                    start_token_pos < len(batch["input_ids"][i])
                    and batch["input_ids"][i][start_token_pos] in special_token_ids
                ):
                    start_token_pos += 1
                if start_token_pos == len(batch["input_ids"][i]):
                    intervention_positions.append([])
                    continue

                # Get ablated token positions
                if "ablated_token_positions" in extra_args:
                    positions = [
                        pos + start_token_pos
                        for pos in extra_args["ablated_token_positions"]
                    ]
                    intervention_positions.append(positions)
                else:
                    intervention_positions.append([])

                # No remaining intervention type carries an intervention vector
                intervention_vectors.append(None)
            else:
                intervention_layers.append(None)
                intervention_positions.append([])
                intervention_vectors.append(None)

        # Add intervention information to batch
        if intervention_mask.any():
            batch["intervention_mask"] = intervention_mask
            batch["intervention_layers"] = intervention_layers
            batch["intervention_positions"] = intervention_positions
            batch["intervention_vectors"] = intervention_vectors

        # Collect question type weights from all features and add to batch
        question_type_weights = []
        for feature in features:
            if feature.get("extra_args") is not None:
                weight = feature["extra_args"].get("question_type_weight", 1.0)
                question_type_weights.append(weight)
            else:
                question_type_weights.append(1.0)  # Default weight

        batch["question_type_weights"] = torch.tensor(
            question_type_weights, dtype=torch.float32
        )

        # Collect teacher logits for KL regularization and remove from extra_args
        # to avoid accumulating tensors in memory across iterations
        teacher_logits_list = []
        has_teacher_logits = False
        for feature in features:
            extra = feature.get("extra_args") or {}
            tl = extra.pop("teacher_logits", None)
            if tl is not None:
                has_teacher_logits = True
                teacher_logits_list.append(tl)
            else:
                teacher_logits_list.append(None)

        if has_teacher_logits:
            batch["teacher_logits_list"] = teacher_logits_list

        return batch
