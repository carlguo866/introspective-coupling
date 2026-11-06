"""Generate responses from LLMs on the FalseReject dataset using vLLM."""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 outputs."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate responses on FalseReject dataset")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, default="AmazonScience/FalseReject",
                        help="HuggingFace dataset name or local JSONL path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--system_prompt_mode", type=str, default="none",
                        help="System prompt mode: none, or a key in the system prompts config")
    parser.add_argument("--system_prompts_file", type=str,
                        default=str(Path(__file__).parent / "config" / "system_prompts.json"),
                        help="Path to system prompts JSON config file")
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        help="Filter to only these category_text values")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit to first N examples after filtering")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--random_variants", type=str, default=None,
                        help="Comma-separated list of mode keys; one is picked uniformly at random per example. Overrides --system_prompt_mode.")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="Seed for random variant selection (deterministic across runs).")
    args = parser.parse_args()

    # Load dataset
    if args.dataset.endswith((".jsonl", ".json")):
        with open(args.dataset) as f:
            if args.dataset.endswith(".jsonl"):
                dataset = [json.loads(line) for line in f]
            else:
                dataset = json.load(f)
    else:
        ds = load_dataset(args.dataset)
        dataset = list(ds[args.split])

    if args.categories:
        dataset = [e for e in dataset if e.get("category_text", e.get("category", "")) in args.categories]
        print(f"Filtered to categories: {args.categories}")
    if args.max_examples and len(dataset) > args.max_examples:
        dataset = dataset[:args.max_examples]
        print(f"Limited to first {args.max_examples} examples")
    print(f"Loaded {len(dataset)} examples from {args.dataset} [{args.split}]")

    # Load system prompts config if needed
    prompts_lookup = {}
    random_modes = None
    if args.random_variants:
        random_modes = [m.strip() for m in args.random_variants.split(",") if m.strip()]
        rng = random.Random(args.random_seed)
        with open(args.system_prompts_file) as f:
            sp_config = json.load(f)
        # prompts_lookup[cat_text] = {mode: prompt}
        for cat_text, prompts in sp_config["category_prompts"].items():
            prompts_lookup[cat_text] = {m: prompts[m] for m in random_modes if m in prompts}
        fallback_prompts = {m: sp_config["category_prompts"]["Others"][m] for m in random_modes}
        print(f"Loaded random-variant prompts for {len(prompts_lookup)} categories "
              f"(variants={random_modes}, seed={args.random_seed})")
    elif args.system_prompt_mode != "none":
        with open(args.system_prompts_file) as f:
            sp_config = json.load(f)
        for cat_text, prompts in sp_config["category_prompts"].items():
            prompts_lookup[cat_text] = prompts[args.system_prompt_mode]
        print(f"Loaded {len(prompts_lookup)} category system prompts (mode={args.system_prompt_mode})")
        fallback_prompt = sp_config["category_prompts"]["Others"][args.system_prompt_mode]

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1.0,
    )

    # Build chat messages
    all_messages = []
    chosen_variants = []  # for random mode: which variant was picked per example
    for example in dataset:
        messages = []
        chosen_variant = None
        if random_modes is not None:
            cat_text = example.get("category_text", "")
            per_cat = prompts_lookup.get(cat_text, fallback_prompts)
            chosen_variant = rng.choice(random_modes)
            sys_prompt = per_cat.get(chosen_variant, fallback_prompts[chosen_variant])
            messages.append({"role": "system", "content": sys_prompt})
        elif args.system_prompt_mode != "none":
            cat_text = example.get("category_text", "")
            sys_prompt = prompts_lookup.get(cat_text)
            if sys_prompt is None:
                print(f"WARNING: No system prompt for category '{cat_text}', using 'Others' fallback")
                sys_prompt = fallback_prompt
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": example["prompt"]})
        all_messages.append(messages)
        chosen_variants.append(chosen_variant)

    # Qwen3 thinking mode handling
    chat_template_kwargs = {}
    if "qwen3" in args.model.lower():
        chat_template_kwargs["enable_thinking"] = False
        print("Qwen3 detected: setting enable_thinking=False")

    # Batch generate
    print(f"Generating responses with {args.model}...")
    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
    )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for example, output, chosen_variant in zip(dataset, outputs, chosen_variants):
            response_text = output.outputs[0].text
            # Strip thinking tags if present
            response_text = strip_thinking_tags(response_text)

            cat_text = example.get("category_text", "")
            if random_modes is not None:
                per_cat = prompts_lookup.get(cat_text, fallback_prompts)
                sys_prompt_used = per_cat.get(chosen_variant, fallback_prompts[chosen_variant])
                mode_label = chosen_variant
            elif args.system_prompt_mode != "none":
                sys_prompt_used = prompts_lookup.get(cat_text, "")
                mode_label = args.system_prompt_mode
            else:
                sys_prompt_used = ""
                mode_label = "none"
            result = {
                "prompt": example["prompt"],
                "category": example.get("category"),
                "category_text": cat_text,
                "model": args.model,
                "system_prompt_mode": mode_label,
                "system_prompt": sys_prompt_used,
                "response": response_text,
                "finish_reason": output.outputs[0].finish_reason,
            }
            f.write(json.dumps(result) + "\n")

    print(f"Saved {len(outputs)} responses to {args.output}")


if __name__ == "__main__":
    main()
