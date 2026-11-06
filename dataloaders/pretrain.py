"""
Plain-text next-token-prediction task for pretrain-style training (e.g. fineweb).

Streams documents from a HuggingFace dataset, tokenizes them, and packs them
into fixed-length chunks. Each chunk is trained with full-sequence LM loss
(no prompt/answer masking).

Usage:
    train:
      tasks:
        pretrain:
          dataset_name: HuggingFaceFW/fineweb
          dataset_config: sample-10BT
          seq_len: 2048
          text_field: text
          seed: 0

The dataset uses the predictor tokenizer passed in by train.py.
"""

from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from dataloaders.base_dataset import BaseDataCollator


class PretrainDataset(IterableDataset):
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

        self.dataset_name = config.get("dataset_name", "HuggingFaceFW/fineweb")
        self.dataset_config = config.get("dataset_config", "sample-10BT")
        self.hf_split = config.get("hf_split", "train")
        self.text_field = config.get("text_field", "text")
        self.seq_len = int(config.get("seq_len", 2048))
        self.seed = int(config.get("seed", 0)) + (0 if split == "train" else 1)
        self.shuffle_buffer = int(config.get("shuffle_buffer", 10000))

        self.num_samples = kwargs.get("num_samples") or config.get("num_samples")
        if self.debug:
            self.num_samples = min(self.num_samples or 100, 100)

        eos = predictor_tokenizer.eos_token_id
        if eos is None:
            raise ValueError(
                "predictor_tokenizer has no eos_token_id; required for pretrain packing"
            )
        self.eos_token_id = eos

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        raise TypeError("PretrainDataset has no length unless num_samples is set")

    def _stream_tokens(self):
        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.hf_split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        for row in ds:
            text = row.get(self.text_field)
            if not text:
                continue
            ids = self.predictor_tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.eos_token_id)
            for tok in ids:
                yield tok

    def __iter__(self):
        buf: List[int] = []
        emitted = 0
        for tok in self._stream_tokens():
            buf.append(tok)
            if len(buf) >= self.seq_len:
                chunk = torch.tensor(buf[: self.seq_len], dtype=torch.long)
                buf = buf[self.seq_len :]
                yield {
                    "prompt": chunk,
                    "prompt_without_answer": chunk[:1],
                    "expected_label": "",
                    "question_type": "lm",
                    "task": "pretrain",
                    "dataset_name": self.dataset_name,
                    "extra_args": {},
                }
                emitted += 1
                if self.num_samples is not None and emitted >= self.num_samples:
                    return


class PretrainDataCollator(BaseDataCollator):
    """Full-sequence LM loss collator. Ignores the label-matching path in BaseDataCollator."""

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_id = self.predictor_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.predictor_tokenizer.eos_token_id

        prompts = [f["prompt"] for f in features]
        max_len = max(p.shape[0] for p in prompts)

        input_ids = torch.full(
            (len(prompts), max_len), pad_id, dtype=torch.long
        )
        for i, p in enumerate(prompts):
            input_ids[i, : p.shape[0]] = p

        attention_mask = (input_ids != pad_id).long()
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100

        batch_size = len(features)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_ids_without_answer": input_ids,
            "attention_mask_without_answer": attention_mask,
            "activations_mask": torch.zeros(batch_size, dtype=torch.bool),
            "question_type": [f.get("question_type", "lm") for f in features],
            "task": [f.get("task", "pretrain") for f in features],
            "dataset_name": [f.get("dataset_name", "") for f in features],
            "expected_label": [f.get("expected_label", "") for f in features],
            "extra_args": [f.get("extra_args", {}) for f in features],
        }
