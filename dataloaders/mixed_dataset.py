import math
import random

from torch.utils.data import IterableDataset

from dataloaders.base_dataset import BaseDataset
from dataloaders.registry import DatasetRegistry


class MixedDataset(BaseDataset, IterableDataset):
    """
    Dataset that mixes multiple task types together.

    Two modes:

    - **Iterable mode** (auto-selected when every sub-dataset is an
      ``IterableDataset``): builds a batch-homogeneous schedule at init time
      and in ``__iter__`` pulls from each sub-dataset's iterator in schedule
      order. Each yielded item is tagged with ``collator_kind`` so the
      dispatching ``MixedDataCollator`` can route it to the right native
      collator. Batches are guaranteed to contain a single task (no
      heterogeneous collation needed).

    - **Map-style mode** (when at least one sub-dataset is map-style): keeps
      the original per-index weighted allocation + ICL prepending path. Used
      by existing feature-descriptor / intervention mixes.
    """

    def __init__(
        self,
        split,
        predictor_model,
        target_model,
        predictor_tokenizer,
        target_tokenizer,
        config,
        num_icl_examples=0,
        **kwargs,
    ):
        super().__init__(
            split,
            predictor_model,
            target_model,
            predictor_tokenizer,
            target_tokenizer,
            config,
            num_icl_examples,
            do_load_datasets=False,
            **kwargs,
        )

        # Get task dataset mapping from registry
        self.task_dataset_mapping = DatasetRegistry.get_all()

        # Create datasets for each task type
        self.datasets = {}
        self.task_weights = {}
        total_weight = sum(
            task_config.get("weight", 1.0) for task_config in config["tasks"].values()
        )

        self.batch_size = int(config.get("batch_size", 1))
        self.num_samples = int(config.get("num_samples", 1000))

        # First pass: compute normalized weights
        raw_weights = {}
        for task_name, task_config in config["tasks"].items():
            if not task_config.get("enabled", True):
                continue
            raw_weights[task_name] = task_config.get("weight", 1.0) / total_weight
        for task_name, w in raw_weights.items():
            self.task_weights[task_name] = w

        # Decide mode: iterable if every (enabled) sub-task's registered class
        # is an IterableDataset subclass. We decide before instantiation so we
        # can size each sub-dataset exactly.
        self.all_iterable = all(
            issubclass(self.task_dataset_mapping[task_name], IterableDataset)
            for task_name in self.task_weights
        )

        if self.all_iterable:
            # Allocate batches first, then instantiate each sub-dataset with
            # the exact number of items it will be asked for. This keeps each
            # iterator from exhausting early or over-producing.
            num_batches = self.num_samples // max(self.batch_size, 1)
            per_task_batches = {
                task: int(num_batches * w) for task, w in self.task_weights.items()
            }
            # Assign any remainder batches to tasks in descending-weight order.
            assigned = sum(per_task_batches.values())
            remainder = num_batches - assigned
            for task in sorted(
                self.task_weights, key=self.task_weights.get, reverse=True
            ):
                if remainder <= 0:
                    break
                per_task_batches[task] += 1
                remainder -= 1

            # Build a shuffled schedule of per-batch tasks, then expand to per-item.
            batch_tasks = []
            for task, count in per_task_batches.items():
                batch_tasks.extend([task] * count)
            rng = random.Random(int(config.get("mixed_seed", 0)))
            rng.shuffle(batch_tasks)
            self.schedule = []
            for task in batch_tasks:
                self.schedule.extend([task] * self.batch_size)

            # Size each sub-dataset to exactly the items we'll pull.
            per_task_num_samples = {
                task: per_task_batches[task] * self.batch_size
                for task in self.task_weights
            }

            # KL regularization settings live at the root of the train config
            # (set by train.py from the YAML root). Propagate them into each
            # sub-task's config so CausalDataset etc. see them — otherwise
            # mixed-mode runs silently fall back to plain CE.
            kl_keys = (
                "use_kl_regularization",
                "kl_topk",
                "kl_temperature",
                "use_freeform_target",
                "teacher_logits_path",
            )
            kl_inherited = {k: config[k] for k in kl_keys if k in config}

            # Instantiate sub-datasets
            for task_name, task_config in config["tasks"].items():
                if not task_config.get("enabled", True):
                    continue
                dataset_class = self.task_dataset_mapping[task_name]
                for k, v in kl_inherited.items():
                    task_config.setdefault(k, v)
                self.datasets[task_name] = dataset_class(
                    split,
                    predictor_model,
                    target_model,
                    predictor_tokenizer,
                    target_tokenizer,
                    task_config,
                    num_icl_examples=0,
                    num_samples=per_task_num_samples[task_name],
                    **kwargs,
                )

            # Iterable mode does not support ICL prepending (sub-iterables
            # already format their own items).
            self.icl_examples = []
            self.icl_labels = []
            self.icl_tasks = []
            self.task_assignments = self.schedule
            self.idx_mapping = {}
        else:
            # Map-style mixing (original behavior).
            kl_keys = (
                "use_kl_regularization",
                "kl_topk",
                "kl_temperature",
                "use_freeform_target",
                "teacher_logits_path",
            )
            kl_inherited = {k: config[k] for k in kl_keys if k in config}
            for task_name, task_config in config["tasks"].items():
                if not task_config.get("enabled", True):
                    continue
                dataset_class = self.task_dataset_mapping[task_name]
                for k, v in kl_inherited.items():
                    task_config.setdefault(k, v)
                self.datasets[task_name] = dataset_class(
                    split,
                    predictor_model,
                    target_model,
                    predictor_tokenizer,
                    target_tokenizer,
                    task_config,
                    num_icl_examples=0,
                    num_samples=math.ceil(
                        self.num_samples * self.task_weights[task_name]
                    )
                    + num_icl_examples,
                    **kwargs,
                )

            self.icl_examples = []
            self.icl_labels = []
            self.icl_tasks = []
            self.icl_counters = []

            dataset_counters = {task: 0 for task in self.datasets.keys()}
            _, icl_idx_mapping = self.allocate_tasks(num_icl_examples, dataset_counters)
            for task, sub_idx in icl_idx_mapping.values():
                if sub_idx >= len(self.datasets[task]):
                    continue
                dataset = self.datasets[task]
                self.icl_examples.append(dataset[sub_idx]["example"])
                self.icl_labels.append(dataset[sub_idx]["expected_label"])
                self.icl_tasks.append((task, sub_idx))

            self.task_assignments, self.idx_mapping = self.allocate_tasks(
                self.num_samples, dataset_counters
            )

    def allocate_tasks(self, num_samples, dataset_counters=None):
        """
        Allocate tasks based on weights and create index mapping.
        Returns:
        1. A list of task names with length num_samples
        2. A mapping from dataset indices to (task, sub_idx) pairs
        """
        task_assignments = []
        idx_mapping = {}
        task_names = list(self.datasets.keys())

        # Initialize counters for each task
        if dataset_counters is None:
            dataset_counters = {task: 0 for task in task_names}

        # Calculate available samples for each task
        task_counts = {
            task: len(self.datasets[task]) - dataset_counters[task]
            for task in task_names
        }

        # Allocate based on weights
        for i in range(num_samples):
            possible_task_names = [task for task in task_names if task_counts[task] > 0]
            if len(possible_task_names) == 0:
                break
            possible_task_probs = [
                self.task_weights[task] for task in possible_task_names
            ]
            task = random.choices(
                possible_task_names, weights=possible_task_probs, k=1
            )[0]

            # Store the task assignment and mapping
            task_assignments.append(task)
            idx_mapping[i] = (task, dataset_counters[task])

            # Update counters
            dataset_counters[task] += 1
            task_counts[task] -= 1

        return task_assignments, idx_mapping

    def __len__(self):
        return len(self.task_assignments)

    def __iter__(self):
        """
        Iterable-mode entry point. HF Trainer picks this up because
        ``MixedDataset`` inherits from ``IterableDataset``. Only exercised
        when ``self.all_iterable`` is True.
        """
        if not self.all_iterable:
            # Fallback: iterate via map-style __getitem__
            for i in range(len(self)):
                yield self.get_single_item(i)
            return

        iterators = {task: iter(ds) for task, ds in self.datasets.items()}
        for task in self.schedule:
            it = iterators[task]
            try:
                item = next(it)
            except StopIteration:
                return
            # Tag for the dispatching collator. Keep intact whatever fields the
            # sub-dataset produced (CausalDataset / PretrainDataset both yield
            # fully-formatted training dicts).
            item["collator_kind"] = task
            yield item

    def batch_processing(self, idxs):
        # batch up subdataset items
        dataset_to_idxs = {}
        for idx in idxs:
            task, sub_idx = self.idx_mapping[idx]  # // self.num_repeats_per_example]
            dataset = self.datasets[task]
            # item = dataset[sub_idx // dataset.num_repeats_per_example]
            if task not in dataset_to_idxs:
                dataset_to_idxs[task] = []
            dataset_to_idxs[task].append(sub_idx)

        # run batch processing on sub-datasets if they exist
        for task, data_idxs in dataset_to_idxs.items():
            dataset = self.datasets[task]
            if hasattr(dataset, "batch_processing"):
                dataset.batch_processing(data_idxs)

    def get_single_item(self, idx):
        # Get the task and sub-dataset index for this item
        task, sub_idx = self.idx_mapping[idx]  # // self.num_repeats_per_example]
        dataset = self.datasets[task]
        item = dataset[sub_idx // dataset.num_repeats_per_example]

        # Get features from the sub-dataset at the specific index
        # features = dataset[sub_idx]

        # Start with an empty prompt
        prompt = []

        # Add our mixed ICL examples
        for icl_example, icl_label, icl_task in zip(
            self.icl_examples, self.icl_labels, self.icl_tasks
        ):
            # Get the appropriate dataset for this ICL example
            icl_dataset = self.datasets[icl_task[0]]
            # Use that dataset's get_labels method to format the example
            example_labels = icl_dataset.get_labels(icl_example, prompt, icl_label)
            prompt = example_labels.prompt

        # Now get the actual example using the target task's dataset
        example = item["example"]
        # label = features["expected_label"]
        # label = self.non_icl_labels[idx // self.num_repeats_per_example]
        # label = item["expected_label"]
        label = item["extra_args"].get("label", item["expected_label"])
        # label = dataset.non_icl_labels[sub_idx // dataset.num_repeats_per_example]

        if dataset.question_types:
            question_type = dataset.question_types[
                sub_idx % dataset.num_repeats_per_example
            ]
        else:
            question_type = None

        # Get the formatted example with our mixed ICL prompt
        example_labels = dataset.get_labels(
            example, prompt, label, question_type, idx=sub_idx
        )

        # Update the features
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
