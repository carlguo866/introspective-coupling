"""
Dispatching collator for ``MixedDataset`` in iterable mode.

Each item yielded by ``MixedDataset.__iter__`` carries a ``collator_kind``
field naming the sub-task it came from. Because ``MixedDataset`` guarantees
batch homogeneity (all items in a batch share the same ``collator_kind``),
this collator simply looks up the native collator for the batch's kind and
delegates.
"""

from typing import Dict, List

from dataloaders.base_dataset import BaseDataCollator
from dataloaders.causal import CausalDataCollator
from dataloaders.pretrain import PretrainDataCollator
from dataloaders.sft_jsonl import SftJsonlDataCollator


# Map collator_kind → native collator class. Extend as new sub-tasks join.
_KIND_TO_COLLATOR_CLS = {
    "causal": CausalDataCollator,
    "pretrain": PretrainDataCollator,
    "sft_jsonl": SftJsonlDataCollator,
}


class MixedDataCollator(BaseDataCollator):
    def __init__(self, predictor_tokenizer, target_tokenizer, question_config=None):
        super().__init__(predictor_tokenizer, target_tokenizer)
        self.question_config = question_config or {}

        # Build one native collator per kind so we can dispatch in O(1).
        self._collators = {}
        for kind, cls in _KIND_TO_COLLATOR_CLS.items():
            if cls is CausalDataCollator:
                # CausalDataCollator needs the question_config (intervention_type etc.).
                sub_cfg = self._extract_task_config(kind)
                self._collators[kind] = cls(
                    predictor_tokenizer=predictor_tokenizer,
                    target_tokenizer=target_tokenizer,
                    question_config=sub_cfg,
                )
            else:
                self._collators[kind] = cls(
                    predictor_tokenizer=predictor_tokenizer,
                    target_tokenizer=target_tokenizer,
                )

    def _extract_task_config(self, kind: str) -> Dict:
        # When ``MixedDataset`` is used, the question_config passed to the
        # collator is the full ``config["train"]`` block. Pull out the sub-task
        # config so ``CausalDataCollator`` sees intervention_type etc.
        tasks = (self.question_config or {}).get("tasks", {})
        return tasks.get(kind, self.question_config)

    def __call__(self, features: List[Dict]) -> Dict:
        if not features:
            raise ValueError("MixedDataCollator received an empty batch.")

        kinds = {f.get("collator_kind") for f in features}
        if len(kinds) != 1 or None in kinds:
            raise ValueError(
                "MixedDataCollator expects a homogeneous batch tagged with "
                f"collator_kind; got {kinds}. Check MixedDataset schedule/"
                "batch_size alignment."
            )
        kind = kinds.pop()

        collator = self._collators.get(kind)
        if collator is None:
            raise KeyError(
                f"No native collator registered for collator_kind={kind!r}. "
                "Add it to _KIND_TO_COLLATOR_CLS in dataloaders/mixed_collator.py."
            )

        # Strip the tag so sub-collators don't see it.
        for f in features:
            f.pop("collator_kind", None)

        return collator(features)
