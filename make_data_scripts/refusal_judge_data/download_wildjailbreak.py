"""Download the WildJailbreak vanilla harmful subset to a local JSONL file."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download WildJailbreak vanilla harmful prompts")
    parser.add_argument("--output", type=str, default="data/wildjailbreak_harmful.jsonl")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("allenai/wildjailbreak", "train", split="train", streaming=True)

    count = 0
    with open(output_path, "w") as f:
        for example in ds:
            if example["data_type"] != "vanilla_harmful":
                continue
            record = {
                "prompt": example["vanilla"],
                "category": "harmful",
                "category_text": "harmful",
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Saved {count} vanilla harmful examples to {output_path}")


if __name__ == "__main__":
    main()
