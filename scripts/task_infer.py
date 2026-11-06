"""
Run inference on a task dataset with streaming save capabilities for very large datasets.
"""

import argparse
import glob
import os
import shutil
import tempfile

import vllm
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def get_single_item(idxs, non_icl_examples, vllm_wrapper, sampling_params):
    examples = [non_icl_examples[idx] for idx in idxs]
    prompts = [[{"role": "user", "content": example}] for example in examples]
    lm_outputs = vllm_wrapper.chat(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    answer = [lm_outputs[i].outputs[0].text for i in range(len(lm_outputs))]
    return examples, answer


def save_streaming_arrow(output_path, examples, generated_labels, original_labels, batch_size=1000):
    """Save data incrementally using Arrow format for memory efficiency."""
    # Create temporary directory for streaming
    temp_dir = tempfile.mkdtemp()

    try:
        # Define features for the dataset
        features = Features(
            {
                "example": Value("string"),
                "generated_label": Value("string"),
                "original_label": Value("string"),
            }
        )

        # Process in batches and save incrementally
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i : i + batch_size]
            batch_generated = generated_labels[i : i + batch_size]
            batch_original = original_labels[i : i + batch_size]

            # Create batch dataset
            batch_dataset = Dataset.from_dict(
                {
                    "example": batch_examples,
                    "generated_label": batch_generated,
                    "original_label": batch_original,
                },
                features=features,
            )

            # Save batch to temporary file
            batch_path = os.path.join(temp_dir, f"batch_{i//batch_size:06d}")
            batch_dataset.save_to_disk(batch_path)

        # Concatenate all batches
        batch_dirs = sorted([d for d in os.listdir(temp_dir) if d.startswith("batch_")])
        datasets = []
        for batch_dir in batch_dirs:
            batch_path = os.path.join(temp_dir, batch_dir)
            dataset = Dataset.load_from_disk(batch_path)
            datasets.append(dataset)

        # Concatenate and save final dataset
        final_dataset = concatenate_datasets(datasets)
        final_dataset.save_to_disk(output_path)

        print(f"Saved {len(examples)} examples to {output_path}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def load_streaming_arrow(output_path):
    """Load data saved with streaming arrow format."""
    return load_dataset_auto(output_path)


def save_incremental_arrow(
    output_path,
    examples,
    generated_labels,
    original_labels,
    batch_num=0,
    start_idx=0,
    end_idx=None,
):
    """Save a single batch incrementally - for true streaming without concatenation."""
    # Create batch dataset
    batch_dataset = Dataset.from_dict(
        {
            "example": examples,
            "generated_label": generated_labels,
            "original_label": original_labels,
        }
    )

    # Save to numbered batch file
    batch_path = output_path.replace(".parquet", f"_batch_{batch_num:06d}")
    if start_idx > 0:
        batch_path += f"_s{start_idx:06d}"
    if end_idx is not None:
        batch_path += f"_e{end_idx:06d}"
    batch_path += ".parquet"
    batch_dataset.save_to_disk(batch_path)
    print(f"Saved batch {batch_num} with {len(examples)} examples to {batch_path}")
    return batch_path


def load_dataset_auto(output_path):
    """
    Automatically detect and load data regardless of how it was saved.
    Handles both single files and batch files automatically.
    """
    import os

    # Check if the main file exists
    if os.path.exists(output_path):
        print(f"Loading single file: {output_path}")
        return load_from_disk(output_path)

    # Check for batch files
    batch_pattern = output_path.replace(".parquet", "_batch_*.parquet")
    batch_files = sorted(glob.glob(batch_pattern))

    if batch_files:
        print(f"Found {len(batch_files)} batch files, loading incrementally...")
        datasets = []
        for batch_file in batch_files:
            dataset = load_from_disk(batch_file)
            datasets.append(dataset)

        # Concatenate all batches
        final_dataset = concatenate_datasets(datasets)
        print(
            f"Successfully loaded {len(final_dataset)} examples from {len(batch_files)} batch files"
        )
        return final_dataset

    # If nothing found, raise error
    raise FileNotFoundError(
        f"No data files found for: {output_path}\n"
        f"Checked for:\n"
        f"  - Single file: {output_path}\n"
        f"  - Batch files: {batch_pattern}"
    )


def load_incremental_arrow(output_path_pattern):
    """Load all batch files and concatenate them."""
    return load_dataset_auto(output_path_pattern)


def main(args):
    dataset_name = args.dataset_name
    data_split = args.data_split
    model_name = args.model_name
    batch_size = args.batch_size
    args.max_tokens
    output_path = args.output_path
    save_batch_size = args.save_batch_size
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        cache_dir="checkpoints/",
    )
    start_idx = args.start_idx
    end_idx = args.end_idx

    if dataset_name == "ultrachat":
        # Load arrow datasets
        # 774k
        examples = load_dataset(
            "stingning/ultrachat",
            split="train",
        )
        total_examples = len(examples)
        if data_split == "test":
            # 7.7k
            split_example_idxs = range(int(total_examples * 0.99), total_examples)
        else:
            # 766.3k
            split_example_idxs = range(int(total_examples * 0.99))

        split_example_idxs = (
            split_example_idxs[start_idx:end_idx]
            if end_idx is not None
            else split_example_idxs[start_idx:]
        )

        non_icl_examples = []
        non_icl_labels = []
        for i in tqdm(split_example_idxs, f"Loading {data_split} data"):
            non_icl_examples.append(examples[i]["data"][0])
            non_icl_labels.append(examples[i]["data"][1])

    vllm_wrapper = vllm.LLM(
        model=model_name,
        max_model_len=min(tokenizer.model_max_length, 100000),
    )

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=128)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # For very large datasets, use streaming save
    if len(non_icl_examples) > save_batch_size:  # Threshold for streaming
        print(
            f"Large dataset detected ({len(non_icl_examples)} examples), using incremental save..."
        )

        # Initialize storage for current batch
        current_examples = []
        current_generated_labels = []
        current_original_labels = []
        batch_num = 0

        for i in tqdm(
            range(0, len(non_icl_examples), batch_size),
            desc=f"Computing {data_split} task labels",
        ):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            examples, labels = get_single_item(
                list(range(i, min(i + batch_size, len(non_icl_examples)))),
                non_icl_examples,
                vllm_wrapper,
                sampling_params,
            )

            current_examples.extend(examples)
            current_generated_labels.extend(labels)
            current_original_labels.extend(non_icl_labels[i : i + len(examples)])

            # Save batch when we reach the save batch size
            if len(current_examples) >= save_batch_size:
                if output_path.endswith(".parquet") or output_path.endswith(".arrow"):
                    # Use incremental save for Arrow format
                    save_incremental_arrow(
                        output_path,
                        current_examples,
                        current_generated_labels,
                        current_original_labels,
                        batch_num,
                    )
                    batch_num += 1

                # Reset current batch
                current_examples = []
                current_generated_labels = []
                current_original_labels = []

        # Save remaining examples
        if current_examples:
            if output_path.endswith(".parquet") or output_path.endswith(".arrow"):
                save_incremental_arrow(
                    output_path,
                    current_examples,
                    current_generated_labels,
                    current_original_labels,
                    batch_num,
                )

        print(f"Saved {batch_num + 1} batches. Use load_incremental_arrow() to load all batches.")

    else:
        # For smaller datasets, use regular approach
        all_examples = []
        all_generated_labels = []

        for i in tqdm(
            range(0, len(non_icl_examples), batch_size),
            desc=f"Computing {data_split} task labels",
        ):
            examples, labels = get_single_item(
                list(range(i, min(i + batch_size, len(non_icl_examples)))),
                non_icl_examples,
                vllm_wrapper,
                sampling_params,
            )
            all_examples.extend(examples)
            all_generated_labels.extend(labels)

        # Save using regular method
        if output_path.endswith(".parquet") or output_path.endswith(".arrow"):
            dataset = Dataset.from_dict(
                {
                    "example": all_examples,
                    "generated_label": all_generated_labels,
                    "original_label": non_icl_labels[: len(all_examples)],
                }
            )
            dataset.save_to_disk(output_path)
            print(f"Saved {len(all_examples)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="ultrachat")
    parser.add_argument("--data_split", type=str, required=True, default="test")
    parser.add_argument(
        "--model_name", type=str, required=True, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for saved labels (.parquet, .arrow)",
    )
    # 14.1M
    parser.add_argument(
        "--save_batch_size", type=int, default=100000, help="Batch size for saving (for streaming)"
    )
    args = parser.parse_args()
    main(args)
