from .base_dataset import BaseDataCollator, BaseDataset
from .causal import CausalDataset, CausalDataCollator
from .from_file import FromFileDataset
from .mixed_collator import MixedDataCollator
from .mixed_dataset import MixedDataset
from .pretrain import PretrainDataCollator, PretrainDataset
from .registry import DatasetRegistry
from .sft_jsonl import SftJsonlDataCollator, SftJsonlDataset

__all__ = [
    "BaseDataset",
    "BaseDataCollator",
    "MixedDataset",
    "DatasetRegistry",
    "FromFileDataset",
    "PretrainDataset",
    "PretrainDataCollator",
    "CausalDataset",
    "CausalDataCollator",
    "SftJsonlDataset",
    "SftJsonlDataCollator",
    "MixedDataCollator",
]


# Update registry with actual classes
DatasetRegistry.register("mixed", MixedDataset)
DatasetRegistry.register("from_file", FromFileDataset)
DatasetRegistry.register("causal", CausalDataset)
DatasetRegistry.register("pretrain", PretrainDataset)
DatasetRegistry.register("sft_jsonl", SftJsonlDataset)

# Create datasets mapping from registry
TASK_DATASET_MAPPING = DatasetRegistry.get_all()
TASK_DATALOADER_MAPPING = {
    "causal": CausalDataCollator,
    "pretrain": PretrainDataCollator,
    "sft_jsonl": SftJsonlDataCollator,
    "mixed": MixedDataCollator,
}
