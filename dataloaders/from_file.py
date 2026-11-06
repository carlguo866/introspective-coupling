import json

from dataloaders.base_dataset import BaseDataset


class FromFileDataset(BaseDataset):
    def __init__(self, split, examples, num_samples=None, **kwargs):
        self.data_split = split
        self.examples = examples
        self.num_samples = num_samples
        if num_samples is not None:
            self.examples = self.examples[:num_samples]

    @classmethod
    def from_file(cls, split, file_path, task, num_samples=None, **kwargs):
        data = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
                    continue
                data.append(example)
        return cls(split, data, num_samples, **kwargs)

    def __len__(self):
        return len(self.examples)

    def get_single_item(self, idx):
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    def __next__(self):
        return next(self.examples)
