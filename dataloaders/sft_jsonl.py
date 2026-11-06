"""
SFT-style JSONL co-training task with assistant-only loss masking.

Streams a local JSONL file whose rows look like:
    {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], "source":"..."}

For each conversation, applies the predictor tokenizer's chat template twice
(full conversation vs. prefix with add_generation_prompt=True) to identify the
assistant token span, then yields one example per conversation with labels
masked to -100 on everything except the final assistant turn's content tokens.

Loops infinitely over the file (re-shuffled each epoch with seed + epoch_idx)
because the source set is small (~10K rows) and the mixed schedule asks for
many more samples than that.

Usage:
    train:
      tasks:
        sft_jsonl:
          data_files: /path/to/messages.jsonl
          max_seq_len: 8192
          seed: 0
"""

import json
import random
from typing import Dict, List

import torch
from torch.utils.data import IterableDataset

from dataloaders.base_dataset import BaseDataCollator


class SftJsonlDataset(IterableDataset):
    def __init__(
        self,
        split,
        predictor_model,
        target_model,
        predictor_tokenizer,
        target_tokenizer,
        config,
        **kwargs,
    ):
        self.data_split = split
        self.predictor_tokenizer = predictor_tokenizer
        self.config = config
        self.debug = kwargs.get("debug", False)

        data_files = config.get("data_files")
        if not data_files:
            raise ValueError("sft_jsonl task requires `data_files` in config")
        self.data_files = data_files

        self.max_seq_len = int(config.get("max_seq_len", 8192))
        self.seed = int(config.get("seed", 0)) + (0 if split == "train" else 1)

        self.num_samples = kwargs.get("num_samples") or config.get("num_samples")
        if self.debug:
            self.num_samples = min(self.num_samples or 100, 100)

        if predictor_tokenizer.chat_template is None:
            raise ValueError(
                "sft_jsonl needs a tokenizer with a chat_template; got None."
            )

        # Load all rows into memory once. ~10K SFT rows is trivial.
        self._rows = self._load_rows(self.data_files)
        if not self._rows:
            raise ValueError(f"No rows loaded from {self.data_files}")

    @staticmethod
    def _load_rows(path) -> List[Dict]:
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        raise TypeError("SftJsonlDataset has no length unless num_samples is set")

    def _build_example(self, row: Dict):
        """Tokenize one conversation and build the assistant-only label mask.

        Returns None if the conversation is unusable (no assistant turn, exceeds
        max_seq_len, or template gives a degenerate prefix).
        """
        messages = row.get("messages")
        if not messages or messages[-1].get("role") != "assistant":
            return None

        prefix_messages = messages[:-1]

        full_ids = self.predictor_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prefix_ids = self.predictor_tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        if (
            not full_ids
            or not prefix_ids
            or len(prefix_ids) >= len(full_ids)
        ):
            return None

        if len(full_ids) > self.max_seq_len:
            return None

        # Sanity: the prefix should be an actual prefix of the full sequence.
        # If the templates disagree, fall back to None so the row is skipped.
        if list(full_ids[: len(prefix_ids)]) != list(prefix_ids):
            return None

        full_ids_t = torch.tensor(full_ids, dtype=torch.long)
        prefix_ids_t = torch.tensor(prefix_ids, dtype=torch.long)
        labels = full_ids_t.clone()
        labels[: len(prefix_ids)] = -100

        return {
            "prompt": full_ids_t,
            "labels": labels,
            "prompt_without_answer": prefix_ids_t,
            "expected_label": "",
            "question_type": "sft",
            "task": "sft_jsonl",
            "dataset_name": row.get("source", "sft_jsonl"),
            "extra_args": {},
        }

    def __iter__(self):
        emitted = 0
        epoch = 0
        dropped = 0
        while True:
            rng = random.Random(self.seed + epoch)
            order = list(range(len(self._rows)))
            rng.shuffle(order)
            for idx in order:
                example = self._build_example(self._rows[idx])
                if example is None:
                    dropped += 1
                    continue
                yield example
                emitted += 1
                if self.num_samples is not None and emitted >= self.num_samples:
                    return
            epoch += 1
            # Lightweight signal so we can verify drop rates in logs without
            # spamming. Only fires once per epoch boundary.
            if epoch == 1 and dropped > 0:
                print(
                    f"[sft_jsonl] dropped {dropped} rows in first epoch "
                    f"(out of {len(self._rows)})",
                    flush=True,
                )


class SftJsonlDataCollator(BaseDataCollator):
    """Pads input_ids, labels, and prompt_without_answer to max-in-batch.

    Padding tokens are masked with -100 in `labels` (in addition to the
    user/system masking the dataset already applied), so loss is computed only
    on assistant-content tokens.
    """

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_id = self.predictor_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.predictor_tokenizer.eos_token_id

        prompts = [f["prompt"] for f in features]
        labels_list = [f["labels"] for f in features]
        prefixes = [f["prompt_without_answer"] for f in features]

        max_len = max(p.shape[0] for p in prompts)
        max_prefix_len = max(p.shape[0] for p in prefixes)

        batch_size = len(features)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for i, (p, lab) in enumerate(zip(prompts, labels_list)):
            input_ids[i, : p.shape[0]] = p
            labels[i, : lab.shape[0]] = lab

        attention_mask = (input_ids != pad_id).long()

        input_ids_without_answer = torch.full(
            (batch_size, max_prefix_len), pad_id, dtype=torch.long
        )
        for i, p in enumerate(prefixes):
            input_ids_without_answer[i, : p.shape[0]] = p
        attention_mask_without_answer = (input_ids_without_answer != pad_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_ids_without_answer": input_ids_without_answer,
            "attention_mask_without_answer": attention_mask_without_answer,
            "activations_mask": torch.zeros(batch_size, dtype=torch.bool),
            "question_type": [f.get("question_type", "sft") for f in features],
            "task": [f.get("task", "sft_jsonl") for f in features],
            "dataset_name": [f.get("dataset_name", "") for f in features],
            "expected_label": [f.get("expected_label", "") for f in features],
            "extra_args": [f.get("extra_args", {}) for f in features],
        }
