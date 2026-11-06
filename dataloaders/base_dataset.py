import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import nltk
import torch
from datasets import Dataset, load_dataset
from scripts.task_infer import load_dataset_auto
from transformers import LlamaTokenizer

# Download NLTK data if not already present
try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown")
    nltk.download("punkt")

import json
import os

from nltk.corpus import brown
from tqdm import tqdm


def format_messages(
    tokenizer,
    messages,
    remove_final_assistant_eot=True,
    add_final_assistant_header=True,
    remove_final_user_eot=False,
):
    """Format messages for the tokenizer"""
    if tokenizer.chat_template is None or isinstance(messages, str):
        return messages
    chat = tokenizer.apply_chat_template(messages, tokenize=False)
    if "<|start_header_id|>assistant<|end_header_id|>\\n\\n" in tokenizer.chat_template:
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        eot_id = tokenizer.eos_token
    elif "<start_of_turn>model\n" in tokenizer.chat_template:
        assistant_header = "<start_of_turn>model\n"
        eot_id = "<end_of_turn>\n"
    elif "<|im_start|>assistant" in tokenizer.chat_template:
        assistant_header = "<|im_start|>assistant"
        eot_id = "<|im_end|>\n"
    elif "<|channel|>final<|message|>" in tokenizer.chat_template:
        # gpt-oss / harmony. Header goes straight to the final channel so the
        # model cannot open an analysis (thinking) channel; the template ends
        # the last assistant turn with <|return|>.
        assistant_header = "<|start|>assistant<|channel|>final<|message|>"
        eot_id = "<|return|>"
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer.chat_template}")
    if messages[-1]["role"] == "user" and add_final_assistant_header:
        # add assistant header if user
        chat += assistant_header
    if (
        messages[-1]["role"] == "assistant"
        and chat.endswith(eot_id)
        and remove_final_assistant_eot
    ):
        # remove last eot_id if assistant
        chat = chat[: -len(eot_id)]
    if (
        messages[-1]["role"] == "user"
        and chat.endswith(eot_id)
        and remove_final_user_eot
    ):
        # remove last eot_id if use
        chat = chat[: -len(eot_id)]
    return chat


@dataclass
class ExampleLabels:
    prompt: str | torch.Tensor
    prompt_without_answer: str | torch.Tensor
    expected_label: str | torch.Tensor
    evaluation_type: str
    task: str
    dataset_name: str
    extra_args: Dict = field(default_factory=dict)
    question_type: Optional[str] = None


class BaseDataset(Dataset):
    def __init__(
        self,
        split,
        predictor_model,
        target_model,
        predictor_tokenizer,
        target_tokenizer,
        config,
        num_icl_examples=0,
        num_samples=None,
        debug=False,
        self_train=False,
        do_load_datasets=True,
        **kwargs,
    ):
        self.data_split = split
        self.predictor_model = predictor_model
        self.target_model = target_model
        self.predictor_tokenizer = predictor_tokenizer
        self.target_tokenizer = target_tokenizer
        self.num_icl_examples = num_icl_examples
        self.question_config = config
        self.load_question_config()
        if num_samples is None:
            try:
                self.num_samples = config["num_samples"]
            except:
                raise
        else:
            self.num_samples = num_samples
        self.num_repeats_per_example = (
            len(self.question_config["question_types"])
            if "question_types" in self.question_config
            else 1
        )
        self.artifacts_dir = self.question_config.get(
            "artifacts_dir", "."
        )
        self.debug = debug
        self.self_train = self_train
        for k, v in kwargs.items():
            setattr(self, k, v)

        if "dataset" in config and do_load_datasets:
            self.load_datasets(config["dataset"])

    def load_datasets(self, dataset_names):
        self.icl_examples, self.non_icl_examples = [], []
        self.icl_labels, self.non_icl_labels = [], []
        self.example_to_dataset_name = {}
        self.task_template = [
            {"role": "user", "content": "{task_prompt}{example}"},
            {"role": "assistant"},
        ]

        for dataset_name in dataset_names:
            num_dataset_samples = (
                self.num_samples // len(dataset_names) // self.num_repeats_per_example
            )
            num_icl_dataset_samples = self.num_icl_examples // len(dataset_names)
            if dataset_name == "brown":
                # Get sentences from Brown corpus
                examples = brown.sents()
                # Filter sentences to those with at least 8 words
                examples = [" ".join(sent) for sent in examples if len(sent) >= 8]
                icl_examples = random.sample(examples, num_icl_dataset_samples)
                non_icl_examples = []
                while len(non_icl_examples) < num_dataset_samples:
                    example = random.choice(examples)
                    if example not in icl_examples and example not in non_icl_examples:
                        non_icl_examples.append(example)
                        self.example_to_dataset_name[example] = dataset_name
            elif dataset_name == "gsm8k":
                examples = load_dataset("openai/gsm8k", "main")[self.data_split]
                prompt = "Answer the question based on the context provided. Respond with the answer as a number and nothing else.\n\n"
                icl_idxs = random.sample(range(len(examples)), num_icl_dataset_samples)
                icl_examples = [
                    f"{examples[i]['question']}\n\n{prompt}" for i in icl_idxs
                ]
                icl_labels = [examples[i]["answer"] for i in icl_idxs]
                non_icl_examples = [
                    f"{examples[i]['question']}\n\n{prompt}"
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
                non_icl_labels = [
                    examples[i]["answer"]
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
            elif dataset_name == "imdb":
                examples = load_dataset("stanfordnlp/imdb")[self.data_split]
                prompt = "What is the sentiment of this review? Respond with one of pos or neg and nothing else.\n\n"
                icl_idxs = random.sample(range(len(examples)), num_icl_dataset_samples)
                icl_examples = [
                    f"{examples[i]['text'][:4000]}\n\n{prompt}" for i in icl_idxs
                ]
                icl_labels = [
                    "neg" if examples[i]["label"] == 0 else "pos" for i in icl_idxs
                ]
                non_icl_examples = [
                    f"{examples[i]['text'][:4000]}\n\n{prompt}"
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
                non_icl_labels = [
                    "neg" if examples[i]["label"] == 0 else "pos"
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
            elif dataset_name == "counterfact":
                prompt = ""

                def format_example(example):
                    ex_prompt = example["requested_rewrite"]["prompt"].replace(
                        "{}", example["requested_rewrite"]["subject"]
                    )
                    correct_answer = example["requested_rewrite"]["target_true"]["str"]
                    incorrect_answer = example["requested_rewrite"]["target_new"]["str"]
                    # scramble the answers
                    answers = [correct_answer, incorrect_answer]
                    random.shuffle(answers)
                    ex_prompt += f"\n\nRespond with one of {answers[0]} or {answers[1]} and nothing else."
                    return ex_prompt

                examples = load_dataset("azhx/counterfact")[self.data_split]
                icl_idxs = random.sample(range(len(examples)), num_icl_dataset_samples)
                icl_examples = [format_example(examples[i]) for i in icl_idxs]
                icl_labels = [
                    examples[i]["requested_rewrite"]["target_true"]["str"]
                    for i in icl_idxs
                ]
                non_icl_examples = [
                    format_example(examples[i])
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
                non_icl_labels = [
                    examples[i]["requested_rewrite"]["target_true"]["str"]
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
            elif dataset_name == "orca_math":
                examples = load_dataset("microsoft/orca-math-word-problems-200k")[
                    "train"
                ]
                # make a train/test split
                if self.data_split == "train":
                    example_idxs = range(int(len(examples) * 0.8))
                else:
                    example_idxs = range(int(len(examples) * 0.8), len(examples))
                prompt = "Solve the following math problem. Respond with the solution as a number and nothing else.\n\n"
                icl_idxs = random.sample(example_idxs, num_icl_dataset_samples)
                icl_examples = [examples[i]["question"] for i in icl_idxs]
                icl_labels = [examples[i]["answer"] for i in icl_idxs]
                non_icl_examples = [
                    examples[i]["question"] for i in example_idxs if i not in icl_idxs
                ]
                non_icl_labels = [
                    examples[i]["answer"] for i in example_idxs if i not in icl_idxs
                ]
            elif dataset_name == "superglue_rte":
                examples = load_dataset("super_glue", "rte")[self.data_split]
                prompt = "Does the premise entail the hypothesis? Respond with yes or no and nothing else.\n\n"
                icl_idxs = random.sample(range(len(examples)), num_icl_dataset_samples)
                icl_examples = [
                    "Premise: "
                    + examples[i]["premise"]
                    + "\nHypothesis: "
                    + examples[i]["hypothesis"]
                    for i in icl_idxs
                ]
                icl_labels = [examples[i]["label"] for i in icl_idxs]
                non_icl_examples = [
                    "Premise: "
                    + examples[i]["premise"]
                    + "\nHypothesis: "
                    + examples[i]["hypothesis"]
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
                non_icl_labels = [
                    examples[i]["label"]
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
            elif dataset_name in [
                "multi_digit_multiply",
                "multi_digit_multiply_with_hints",
                "multi_digit_multiply_with_hints_selffilter",
                "multi_digit_multiply_with_hints_basefilter",
            ]:
                prompt = "Respond with the answer and nothing else.\n\n"
                with open(
                    os.path.join(
                        "data/",
                        dataset_name,
                        f"{self.data_split}.jsonl",
                    ),
                    "r",
                ) as f:
                    examples = [json.loads(line) for line in f]
                icl_idxs = random.sample(range(len(examples)), num_icl_dataset_samples)
                icl_examples = []
                icl_labels = []
                for i in icl_idxs:
                    problem = examples[i]["problem"]
                    if "multi_digit_multiply_with_hints" in dataset_name:
                        hint = examples[i]["hint"]
                        problem = f"# TODO check model answers {hint} (this part should not be shown to model)\n{problem}"
                    icl_examples.append(problem)
                    icl_labels.append(examples[i]["correct_answer"])

                non_icl_examples = []
                non_icl_labels = []
                for i in range(len(examples)):
                    if i not in icl_idxs:
                        problem = examples[i]["problem"]
                        if "multi_digit_multiply_with_hints" in dataset_name:
                            hint = examples[i]["hint"]
                            problem = f"# TODO check model answers {hint} (this part should not be shown to model)\n{problem}"
                        non_icl_examples.append(problem)
                        non_icl_labels.append(examples[i]["correct_answer"])
            elif dataset_name == "gender_bias":
                examples = []
                if self.data_split == "train":
                    data_split_name = "train"
                else:
                    data_split_name = "valid"
                save_path = os.path.join(
                    "data/obituary/",
                    f"cont_ds_{data_split_name}.jsonl",
                )
                answer_map = {
                    "male": "he",
                    "female": "she",
                    "neutral": "they",
                }
                with open(save_path, "r") as f:
                    prompts = set()
                    lines = f.readlines()
                    for line in tqdm(lines, f"Loading {self.data_split} data"):
                        line = json.loads(line)
                        prompt = self.predictor_tokenizer.decode(
                            line["source_input_ids"][10:-5]
                        )
                        if prompt not in prompts:
                            prompts.add(prompt)
                            examples.append(
                                {
                                    "prompt": self.predictor_tokenizer.decode(
                                        line["source_input_ids"][10:-5]
                                    ),
                                    # "prompt": line["prompt"]["prefix"],
                                    "answer": answer_map[line["source_gender"]],
                                    # max(
                                    #     line["all_probs"].items(), key=lambda x: x[1]
                                    # )[0],
                                    # "token_ranges_per_region": line[
                                    #     "token_ranges_per_region"
                                    # ],
                                }
                            )
                    print(f"Loaded {len(examples)} examples for {self.data_split}")
                prompt = "My friend left behind three children, for whom"
                self.task_template = [
                    {"role": "user", "content": "{example}"},
                    {
                        "role": "assistant",
                        "content": "{task_prompt}",
                    },
                ]
                icl_idxs = random.sample(range(len(examples)), num_icl_dataset_samples)
                icl_examples = [examples[i]["prompt"] for i in icl_idxs]
                non_icl_examples = [
                    examples[i]["prompt"]
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
                icl_labels = [examples[i]["answer"] for i in icl_idxs]
                non_icl_labels = [
                    examples[i]["answer"]
                    for i in range(len(examples))
                    if i not in icl_idxs
                ]
            elif dataset_name == "fineweb":
                # Load arrow datasets
                examples = load_dataset(
                    "HuggingFaceFW/fineweb",
                    "sample-10BT",
                    split=self.data_split,
                )
                raise NotImplementedError("loading FineWeb here is not supported; use the pretrain dataset")

                example_idxs = range(len(examples))

                icl_idxs = random.sample(list(example_idxs), num_icl_dataset_samples)

                prompt = ""

                # Adjust these field names based on your arrow schema
                icl_examples = [
                    examples[i]["text"] for i in icl_idxs
                ]  # or whatever field contains the text
                icl_labels = [
                    examples[i]["label"] for i in icl_idxs
                ]  # or whatever field contains the label

                non_icl_examples = [
                    examples[i]["text"] for i in example_idxs if i not in icl_idxs
                ]
                non_icl_labels = [
                    examples[i]["label"] for i in example_idxs if i not in icl_idxs
                ]
            elif dataset_name == "ultrachat":
                # Load arrow datasets
                examples = load_dataset_auto(
                    os.path.join(
                        self.question_config["label_path"], f"{self.data_split}.parquet"
                    )
                )
                icl_examples = []
                icl_labels = []

                non_icl_examples = []
                non_icl_labels = []
                for i in tqdm(range(len(examples)), f"Loading {self.data_split} data"):
                    if len(non_icl_examples) < num_dataset_samples:
                        non_icl_examples.append(examples[i]["example"])
                        non_icl_labels.append(examples[i]["generated_label"])
                prompt = ""
            elif dataset_name.startswith("causal"):
                # (
                #     prompt,
                #     icl_examples,
                #     icl_labels,
                #     non_icl_examples,
                #     non_icl_labels,
                # ) = load_causal_dataset(dataset_name, self.data_split, num_icl_dataset_samples)
                # Load intervention data
                (
                    non_icl_examples,
                    non_icl_labels,
                    icl_examples,
                    icl_labels,
                    self.example_to_dataset_name,
                ) = self.load_intervention_data(self.question_config, self.data_split)
                prompt = ""
            else:
                raise ValueError(f"Dataset {dataset_name} not supported")

            non_icl_examples = non_icl_examples[:num_dataset_samples]
            for example in non_icl_examples:
                self.example_to_dataset_name[example] = dataset_name
            self.icl_examples.extend(icl_examples)
            self.non_icl_examples.extend(non_icl_examples)
            if dataset_name != "brown":
                non_icl_labels = non_icl_labels[:num_dataset_samples]
                self.icl_labels.extend(icl_labels)
                self.non_icl_labels.extend(non_icl_labels)

                assert len(icl_examples) == len(icl_labels)
                assert len(non_icl_examples) == len(non_icl_labels)

            self.task_prompt = prompt

        # assert len(self.icl_examples) == len(self.icl_labels)
        # assert len(self.non_icl_examples) == len(self.non_icl_labels)

    def load_intervention_data(self, question_config, data_split):
        raise NotImplementedError("Subclass must implement this method")

    def __len__(self):
        return len(self.non_icl_examples) * self.num_repeats_per_example

    def get_example_idx(self, encoded_input_ids, example_tokens):
        for i in range(len(encoded_input_ids) - len(example_tokens) + 1):
            try:
                if (
                    encoded_input_ids[i : i + len(example_tokens)] == example_tokens
                ).all():
                    return i
            except:
                raise
        raise ValueError(f"Example not found in encoded input ids")

    def get_last_user_turn_start_idx(self, encoded_input_ids, example, example_tokens):
        """
        TODO should not be the *first* user turn (last user turn?)
        """
        if not isinstance(self.target_tokenizer, LlamaTokenizer):
            raise NotImplementedError("only Llama tokenizers are supported here")
        # find first user turn
        user_turn_tokens = self.target_tokenizer(
            "<|start_header_id|>user<|end_header_id|>\n\n", return_tensors="pt"
        ).input_ids[
            0, 1:
        ]  # remove bos token
        user_turn_idx = None
        for i in range(len(encoded_input_ids) - len(user_turn_tokens)):
            try:
                if (
                    encoded_input_ids[i : i + len(user_turn_tokens)] == user_turn_tokens
                ).all():
                    user_turn_idx = i
                    # break
            except:
                raise
        try:
            sentence_offset = user_turn_idx + len(user_turn_tokens)
        except:
            raise

        try:
            # may have whitespace differences
            assert self.target_tokenizer.decode(
                encoded_input_ids[
                    sentence_offset : sentence_offset + len(example_tokens)
                ],
                skip_special_tokens=True,
            ).strip().replace(" ", "") == example.strip().replace(" ", "")
            # assert self.tokenizer.decode(encoded_input.input_ids[0][lm_answer_offset], skip_special_tokens=True).strip() == lm_answer.strip()
        except:
            raise
        return sentence_offset

    def make_prompt(
        self,
        tokenizer,
        messages_so_far: list[dict[str, str]],
        example: str | list[str],
    ):
        return_str = isinstance(example, str)
        if return_str:
            example = [example]
        prompt = [*messages_so_far]
        prompts = [copy.deepcopy(prompt) for _ in example]
        for i, ex in enumerate(example):
            for turn in self.task_template:
                prompts[i].append(
                    {
                        "role": turn["role"],
                        "content": turn.get("content", "").format(
                            task_prompt=self.task_prompt, example=ex
                        ),
                    }
                )
        prompt = prompts
        chat_without_answer = [self.format_messages(tokenizer, p) for p in prompts]
        inputs = tokenizer(
            chat_without_answer,
            return_tensors="pt",
            padding=True,
        )
        example_offsets = []
        for i, ex in enumerate(example):
            example_token_ids = tokenizer(
                ex, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
            example_start_idx = self.get_example_idx(
                inputs.input_ids[i], example_token_ids
            )
            example_offsets.append(
                [example_start_idx, example_start_idx + len(example_token_ids)]
            )
        if return_str:
            return {
                "prompt": prompt[0],
                "chat_without_answer": chat_without_answer[0],
                "input_ids": inputs.input_ids,
                "attn_mask": inputs.attention_mask,
                "example_offset": example_offsets[0],
            }
        else:
            return {
                "prompt": prompt,
                "chat_without_answer": chat_without_answer,
                "input_ids": inputs.input_ids,
                "attn_mask": inputs.attention_mask,
                "example_offset": example_offsets,
            }

    def get_lm_answer(
        self,
        tokenizer,
        prompt,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=5,
        **kwargs,
    ):
        """
        Get the answer from the model.
        If input_ids and attention_mask are provided, use them to get the answer.
        Otherwise, format the prompt and get the answer.
        If prompt is a list, get the answer for each prompt.

        Args:
            prompt: list of messages or single message
            input_ids: torch.Tensor of shape (batch_size, seq_len). batch_size = 1 if prompt is a single message
            attention_mask: torch.Tensor of shape (batch_size, seq_len). batch_size = 1 if prompt is a single message
            max_new_tokens: int
            **kwargs: additional arguments for the model.generate method

        Returns:
            lm_answers: list of strings or single string
        """
        if input_ids is None:
            chat_without_answer = self.format_messages(tokenizer, prompt)
            inputs = tokenizer(
                chat_without_answer,
                return_tensors="pt",
                device=self.target_model.device,
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
        outputs = self.target_model.generate(
            input_ids=input_ids.to(self.target_model.device),
            attention_mask=attention_mask.to(self.target_model.device),
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=None,
            top_p=None,
            do_sample=False,
            **kwargs,
        )

        def postprocess_lm_answer_single(lm_answer, input_ids, prompt):
            # add answer to chat
            if prompt[-1]["role"] == "user":
                prompt_with_answer = prompt + [
                    {"role": "assistant", "content": lm_answer}
                ]
            else:
                prompt_with_answer = copy.deepcopy(prompt)
                prompt_with_answer[-1]["content"] += lm_answer
            return {
                "answer": lm_answer,
                "answer_offset": len(input_ids[0]),
                "prompt_with_answer": prompt_with_answer,
            }

        if isinstance(prompt[0], list):
            lm_answers = [
                tokenizer.decode(
                    outputs[i][len(input_ids[i]) :], skip_special_tokens=True
                )
                for i in range(len(prompt))
            ]
            lm_answers = [
                postprocess_lm_answer_single(
                    lm_answer, input_ids[i].unsqueeze(0), prompt[i]
                )
                for i, lm_answer in enumerate(lm_answers)
            ]
            return {
                "answer": [lm_answer["answer"] for lm_answer in lm_answers],
                "answer_offset": [
                    lm_answer["answer_offset"] for lm_answer in lm_answers
                ],
                "prompt_with_answer": [
                    lm_answer["prompt_with_answer"] for lm_answer in lm_answers
                ],
            }
        else:
            lm_answer = tokenizer.decode(
                outputs[0][len(input_ids[0]) :], skip_special_tokens=True
            )
            return postprocess_lm_answer_single(lm_answer, input_ids, prompt)

    def load_question_config(self):
        # Create weighted list of question types for sampling
        self.question_types = []

        for qtype, config in self.question_config.get("question_types", {}).items():
            self.question_types.append(qtype)

    def get_labels(
        self, example, messages_so_far, label=None, question_type=None, idx=None
    ):
        """
        Last message should not have a period after the expected label.
        Return the prompt, prompt without answer, expected label, evaluation type, and other features.

        Should return the same set of features as the base dataset (can put extra features in extra_args)
        """

    def format_messages(self, tokenizer, messages):
        return format_messages(tokenizer, messages)

    def get_single_item(self, idx):
        # Single item
        try:
            example = self.non_icl_examples[idx // self.num_repeats_per_example]
            label = self.non_icl_labels[idx // self.num_repeats_per_example]
        except:
            raise

        # Start with an empty prompt
        prompt = []

        # Add ICL examples
        for icl_example, icl_label in zip(self.icl_examples, self.icl_labels):
            icl_example_labels = self.get_labels(
                icl_example,
                prompt,
                label=icl_label,
            )
            prompt = icl_example_labels.prompt

        # Get the formatted example
        if self.question_types:
            question_type = self.question_types[idx % self.num_repeats_per_example]
        else:
            question_type = None
        example_labels = self.get_labels(
            example,
            prompt,
            label=label,
            question_type=question_type,
            idx=idx,
        )
        return {
            "example": example,
            "prompt": example_labels.prompt,
            "prompt_without_answer": example_labels.prompt_without_answer,
            "expected_label": example_labels.expected_label,
            "evaluation_type": example_labels.evaluation_type,
            "task": example_labels.task,
            "dataset_name": example_labels.dataset_name,
            "question_type": example_labels.question_type,
            "extra_args": example_labels.extra_args,
        }

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Construct features dictionary
            features = self.get_single_item(idx)
            return features
        else:
            if hasattr(self, "batch_processing"):
                self.batch_processing(idx)
            # Batch of items
            batch_features = {}
            for i in idx:
                item_features = self[i]  # Use the single item logic

                # Initialize batch features if needed
                if not batch_features:
                    for key in item_features.keys():
                        batch_features[key] = []

                # Add features to batch
                for key, value in item_features.items():
                    if key in batch_features:
                        batch_features[key].append(value)
                    else:
                        # Handle case where a feature exists in some items but not others
                        batch_features[key] = [None] * (
                            len(batch_features.get("example", []))
                        )
                        batch_features[key].append(value)

            return batch_features

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
                return None, prompt_without_answer
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
                expected_label_start_idx = expected_label.index(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
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


class BaseDataCollator:
    """
    Data collator that computes token representations and similarity on the fly.
    """

    def __init__(self, predictor_tokenizer, target_tokenizer, question_config=None):
        self.predictor_tokenizer = predictor_tokenizer
        self.target_tokenizer = target_tokenizer
        self.question_config = question_config or {}
        # Set max_length for tokenization to prevent OOM - default to model_max_length or 2048
        self.max_length = question_config.get("max_sequence_length", None) if question_config else None
        if self.max_length is None:
            # Use a safe default that's less than most model limits
            self.max_length = min(
                getattr(predictor_tokenizer, "model_max_length", 2048),
                2048  # Cap at 2048 to prevent OOM
            )

    def pad_left_to_max_len(self, arr, pad_token_id, max_len, dim=-1):
        """
        Pads to the left of the tensor.
        """
        # assert self.predictor_tokenizer.pad_token_id == pad_token_id
        if arr.shape[dim] < max_len:
            pad_len = max_len - arr.shape[dim]
            pad_shape = list(arr.shape)
            pad_shape[dim] = pad_len
            pad_tensor = torch.full(
                pad_shape,
                pad_token_id,
                dtype=arr.dtype,
            )
            return torch.cat([pad_tensor, arr], dim=dim)
        else:
            # Create slice objects for all dimensions
            slices = [slice(None)] * arr.ndim
            # Set the slice for the specified dimension to cut to max_len
            slices[dim] = slice(max_len)
            return arr[tuple(slices)]

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        all_example_keys = set()
        for example in features:
            for key in example.keys():
                batch[key] = []
                all_example_keys.add(key)

        # Format prompts for the model
        formatted_str_inputs = []
        formatted_str_inputs_without_answer = []
        formatted_str_labels = []
        max_formatted_str_prompt_len = 0
        arr_inputs = []
        arr_inputs_without_answer = []
        arr_labels = []
        max_arr_prompt_len = 0
        formatted_str_idxs = []
        arr_idxs = []
        for i in range(len(features)):
            prompt = features[i]["prompt"]
            prompt_without_answer = features[i]["prompt_without_answer"]

            # Format for the model
            if isinstance(prompt, torch.Tensor):
                arr_idxs.append(i)
                arr_inputs.append(prompt)
                max_arr_prompt_len = max(max_arr_prompt_len, len(prompt))
                arr_inputs_without_answer.append(prompt_without_answer)
                arr_labels.append(features[i]["expected_label"])
            else:
                formatted_str_idxs.append(i)
                try:
                    formatted_input_with_answer = format_messages(
                        self.predictor_tokenizer,
                        prompt,
                        remove_final_assistant_eot=False,
                        add_final_assistant_header=False,
                    )
                except:
                    raise
                formatted_input_without_answer = format_messages(
                    self.predictor_tokenizer,
                    prompt_without_answer,
                    remove_final_assistant_eot=True,
                    add_final_assistant_header=False,
                )
                formatted_str_inputs.append(formatted_input_with_answer)
                formatted_str_inputs_without_answer.append(
                    formatted_input_without_answer
                )
                formatted_str_labels.append(features[i]["expected_label"])

        batch_idxs_order = formatted_str_idxs + arr_idxs

        max_len = max_arr_prompt_len
        # max(input_ids.shape[1], max_arr_prompt_len)
        if formatted_str_inputs:
            # Tokenize inputs with max_length to prevent OOM
            tokenized_str_inputs = self.predictor_tokenizer(
                formatted_str_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                padding_side="left",
            )
            tokenized_str_inputs_without_answer = self.predictor_tokenizer(
                formatted_str_inputs_without_answer,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                padding_side="left",
            )

            # Get input_ids from tokenized inputs
            input_ids = tokenized_str_inputs["input_ids"]
            input_ids_without_answer = tokenized_str_inputs_without_answer["input_ids"]
            max_len = max(max_len, input_ids.shape[1])

            # pad to max len
            input_ids = self.pad_left_to_max_len(
                input_ids, self.predictor_tokenizer.pad_token_id, max_len
            )
            input_ids_without_answer = self.pad_left_to_max_len(
                input_ids_without_answer, self.predictor_tokenizer.pad_token_id, max_len
            )
            attention_mask = self.pad_left_to_max_len(
                tokenized_str_inputs["attention_mask"],
                0,
                max_len,
            )
            attention_mask_without_answer = self.pad_left_to_max_len(
                tokenized_str_inputs_without_answer["attention_mask"],
                0,
                max_len,
            )

            # Create labels for the expected next tokens
            labels = input_ids.clone()

            # Mask out all tokens except the ones we want to predict
            for i in range(len(labels)):
                # Predict only the answer part
                expected_label = formatted_str_labels[i]
                """
                # TODO figure out whitespacing
                for t in range(1, labels.shape[1]):
                    if (
                        self.predictor_tokenizer.decode(labels[i, -t:]).strip()
                        == expected_label.strip()
                    ):
                        break
                if not (
                    self.predictor_tokenizer.decode(labels[i, -t:]).strip()
                    == expected_label.strip()
                ):
                    t = len(self.predictor_tokenizer.tokenize(expected_label))
                labels[i, :-t] = -100
                # mask out everything but the answer part
                # """
                for t in range(1, labels.shape[1]):
                    if self.predictor_tokenizer.decode(
                        labels[i, -t:], skip_special_tokens=True
                    ).strip().replace(" ", "") == expected_label.strip().replace(
                        " ", ""
                    ):
                        labels[i, :-t] = -100
                        break
                if not (
                    self.predictor_tokenizer.decode(labels[i][labels[i] >= 0])
                    .strip()
                    .replace(" ", "")
                    .startswith(expected_label.strip().replace(" ", ""))
                ):
                    for t in range(labels.shape[1]):
                        if "".join(
                            self.predictor_tokenizer.decode(
                                labels[i, :t], skip_special_tokens=True
                            ).split()
                        ) == "".join(
                            self.predictor_tokenizer.decode(
                                input_ids_without_answer[i], skip_special_tokens=True
                            ).split()
                        ):
                            break
                    # t = attention_mask_without_answer[i].sum()
                    t = len(self.predictor_tokenizer.tokenize(expected_label))
                    labels[i, :t] = -100
                # """
                if (
                    not self.predictor_tokenizer.decode(labels[i][labels[i] >= 0])
                    .strip()
                    .replace(" ", "")
                    .startswith(expected_label.strip().replace(" ", ""))
                ):
                    print("Label mismatch: ", expected_label)
                    # Fall back to masking based on tokenized label length
                    t = len(self.predictor_tokenizer.tokenize(expected_label))
                    labels[i, :-t] = -100
        else:
            input_ids = torch.tensor([], dtype=torch.long)
            input_ids_without_answer = torch.tensor([], dtype=torch.long)
            attention_mask = torch.tensor([], dtype=torch.long)
            attention_mask_without_answer = torch.tensor([], dtype=torch.long)
            labels = torch.tensor([], dtype=torch.long)
            activations_mask = torch.tensor([], dtype=torch.bool)
            input_ids_without_answer = torch.tensor([], dtype=torch.long)
            attention_mask_without_answer = torch.tensor([], dtype=torch.long)

        activations_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        # If we have tensor inputs to concatenate, pad them to match the tokenized inputs
        if arr_inputs:
            # Pad arr_inputs to match the sequence length of tokenized inputs
            padded_arr_inputs = []
            padded_arr_attention_masks = []
            padded_arr_inputs_without_answer = []
            padded_arr_attention_masks_without_answer = []
            padded_arr_labels = []
            for arr_input, arr_input_without_answer, arr_label in zip(
                arr_inputs, arr_inputs_without_answer, arr_labels
            ):
                padded_input = self.pad_left_to_max_len(
                    arr_input, self.predictor_tokenizer.pad_token_id, max_len
                )
                padded_arr_inputs.append(padded_input)

                padded_attention_mask = torch.zeros(max_len, dtype=torch.bool)
                # Set attention mask to 1 for actual tokens (at the end)
                padded_attention_mask[-arr_input_without_answer.shape[0] :] = 1
                padded_arr_attention_masks.append(padded_attention_mask)

                padded_input_without_answer = self.pad_left_to_max_len(
                    arr_input_without_answer,
                    self.predictor_tokenizer.pad_token_id,
                    max_len,
                )
                padded_arr_inputs_without_answer.append(padded_input_without_answer)

                padded_attention_mask_without_answer = torch.zeros(
                    max_len, dtype=torch.bool
                )
                # Set attention mask to 1 for actual tokens (at the end)
                padded_attention_mask_without_answer[
                    -arr_input_without_answer.shape[0] :
                ] = 1
                padded_arr_attention_masks_without_answer.append(
                    padded_attention_mask_without_answer
                )

                padded_label = self.pad_left_to_max_len(arr_label, -100, max_len)
                padded_arr_labels.append(padded_label)

            # Stack and concatenate
            arr_inputs_tensor = torch.stack(padded_arr_inputs).to(dtype=torch.long)
            arr_inputs_without_answer_tensor = torch.stack(
                padded_arr_inputs_without_answer
            ).to(dtype=torch.long)
            arr_labels_tensor = torch.stack(padded_arr_labels)

            input_ids = torch.cat([input_ids, arr_inputs_tensor], dim=0)
            input_ids_without_answer = torch.cat(
                [input_ids_without_answer, arr_inputs_without_answer_tensor], dim=0
            )
            labels = torch.cat([labels, arr_labels_tensor], dim=0)
            activations_mask = torch.cat(
                [
                    activations_mask,
                    torch.ones(arr_labels_tensor.shape[0], dtype=torch.bool),
                ],
                dim=0,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.stack(padded_arr_attention_masks)], dim=0
            )
            attention_mask_without_answer = torch.cat(
                [
                    attention_mask_without_answer,
                    torch.stack(padded_arr_inputs_without_answer),
                ],
                dim=0,
            )

        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        batch["input_ids_without_answer"] = input_ids_without_answer
        batch["attention_mask_without_answer"] = attention_mask_without_answer

        # add other features
        for i in batch_idxs_order:
            example = features[i]
            for key in all_example_keys:
                if key not in example:
                    batch[key].append(None)
                else:
                    batch[key].append(example[key])

        batch["activations_mask"] = activations_mask

        return batch
