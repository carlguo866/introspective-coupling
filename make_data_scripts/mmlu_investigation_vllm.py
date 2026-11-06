"""vLLM-backed rewrite of mmlu_investigation.py.

Mirrors the original CLI and output contract but batches all forwards through
vLLM instead of running a one-question-at-a-time HF loop. The dead nnsight
intervention path is dropped.

Usage:
    uv run --env-file .env python make_data_scripts/mmlu_investigation_vllm.py \\
        --model meta-llama/Llama-3.1-70B-Instruct \\
        --hint_template 8 \\
        --suffix ht8 \\
        --tensor_parallel_size 4 \\
        --overwrite
"""

import argparse
import json
import math
import os
import random
import shutil
import sys

import dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams

dotenv.load_dotenv()

# Re-use pure-CPU helpers from the existing module. Importing this module
# is now safe — its side-effecting code is wrapped in `if __name__ == "__main__":`.
from make_data_scripts.mmlu_investigation import (
    apply_chat_template,
    convert_gpqa_to_mmlu_format,
    create_few_shot_prompt,
    create_few_shot_prompt_with_hint,
    extract_answer,
    find_hint_token_positions,
    generate_synthetic_data,
    get_hint_token_ids,
    hint_templates,
    load_dataset_splits,
)


# Architecture override for fine-tuned checkpoints whose config.json declares a
# Continuous* wrapper class that vLLM doesn't ship. Mirrors
# scripts/generate_freeform_vllm.py.
_ARCH_OVERRIDE = {
    "ContinuousQwen3ForCausalLM": "Qwen3ForCausalLM",
    "ContinuousLlamaForCausalLM": "LlamaForCausalLM",
}


def _maybe_fix_config(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return model_path
    with open(config_path) as f:
        config = json.load(f)
    archs = config.get("architectures", [])
    if not any(a in _ARCH_OVERRIDE for a in archs):
        return model_path
    new_archs = [_ARCH_OVERRIDE.get(a, a) for a in archs]
    print(f"Overriding architectures {archs} -> {new_archs} for vLLM")
    config["architectures"] = new_archs
    backup_path = config_path + ".bak"
    shutil.copy2(config_path, backup_path)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return model_path


def _restore_config(model_path: str) -> None:
    backup_path = os.path.join(model_path, "config.json.bak")
    if os.path.exists(backup_path):
        shutil.move(backup_path, os.path.join(model_path, "config.json"))


def load_vllm(model_name, tensor_parallel_size, gpu_memory_utilization,
              max_model_len, checkpoint=None, tokenizer_name=None):
    model_path = checkpoint or model_name
    is_local = os.path.isdir(model_path)
    if is_local:
        _maybe_fix_config(model_path)
    try:
        llm = LLM(
            model=model_path,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
    finally:
        if is_local:
            _restore_config(model_path)
    tokenizer = llm.get_tokenizer()
    return tokenizer, llm


def _letter_token_ids(tokenizer):
    """Map "{ A, B, C, D}" -> first-token id used in MMLU answer position."""
    out = {}
    for letter in [" A", " B", " C", " D"]:
        ids = get_hint_token_ids(tokenizer, letter)
        if ids:
            out[letter] = ids[0]
    return out


def _probs_from_logprobs(letter_ids, position_logprobs):
    """Return {letter: prob} matching HF semantics.

    The HF code stored softmax(full_vocab_logits)[letter_token] — i.e. NOT
    normalized within {A,B,C,D}. vLLM's logprob is exactly log(softmax(logits)),
    so exp(logprob) reproduces that field. Letters absent from the top-K
    record 0.0.
    """
    probs = {}
    for letter, tid in letter_ids.items():
        lp = position_logprobs.get(tid) if position_logprobs else None
        if lp is None:
            probs[letter] = 0.0
        else:
            # vLLM returns Logprob objects with a .logprob attribute
            probs[letter] = math.exp(lp.logprob if hasattr(lp, "logprob") else lp)
    return probs


def evaluate_mmlu_few_shot_vllm(
    tokenizer, llm, model_name,
    dataset_name="mmlu",
    subset="all",
    num_shots=0,
    num_samples=None,
    output_file=None,
    seed=42,
    restrict_output=True,
    use_thinking_format=False,
    random_answer=False,
    logprobs_topk=20,
):
    random.seed(seed)
    use_few_shot = num_shots > 0

    dev_data, test_data, dataset_info = load_dataset_splits(dataset_name, subset)

    if output_file is None:
        ds = dataset_info["name"]
        sub = dataset_info["subset"] if ds == "mmlu" else ""
        output_file = f"{ds}_{sub}_{num_shots}shot_results.json".replace("__", "_")

    if dev_data is not None:
        dev_examples = list(dev_data)
    else:
        dev_examples = []
        if use_few_shot:
            print("Warning: no dev data; switching to zero-shot")
            use_few_shot = False
            num_shots = 0

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    prompt_type = f"{num_shots}-shot" if use_few_shot else "zero-shot"
    if random_answer:
        print(f"Using {prompt_type} with RANDOM ANSWERS (baseline)")
    else:
        print(f"Using {prompt_type} prompting (vLLM batched)")
    print(f"Evaluating on {len(test_data)} samples from {dataset_info['name'].upper()}")

    # 1. Build all prompts (CPU only)
    examples = []
    prompts = []
    for example in tqdm(test_data, desc="building prompts"):
        question = example["question"]
        choices = example["choices"]
        correct_answer = example["answer"]
        sampled_dev = (
            random.sample(dev_examples, min(num_shots, len(dev_examples)))
            if use_few_shot and dev_examples
            else []
        )
        messages = create_few_shot_prompt(
            sampled_dev, question, choices, num_shots, use_few_shot, restrict_output
        )
        prompt = apply_chat_template(
            tokenizer, messages, model_name, use_thinking_format,
            add_generation_prompt=False, continue_final_message=True,
        )
        examples.append((question, choices, correct_answer, messages, prompt))
        prompts.append(prompt)

    # 2. Single batched forward (or skip for random_answer)
    if random_answer:
        outputs = [None] * len(prompts)
    else:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=logprobs_topk
        )
        outputs = llm.generate(prompts, sampling_params)

    letter_ids = _letter_token_ids(tokenizer)

    # 3. Build records
    results = []
    prompt_to_orig_answer = {}
    correct = 0
    for (question, choices, correct_answer, messages, prompt), out in zip(examples, outputs):
        if random_answer:
            predicted_idx = random.randint(0, len(choices) - 1)
            predicted_letter = chr(65 + predicted_idx)
            generation = predicted_letter
            answer_probs = {}
        else:
            generation = out.outputs[0].text
            predicted_letter = extract_answer(generation)
            predicted_idx = ord(predicted_letter) - ord("A") if predicted_letter else -1
            position_logprobs = (
                out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            )
            answer_probs = _probs_from_logprobs(letter_ids, position_logprobs)

        is_correct = predicted_idx == correct_answer
        correct += int(is_correct)
        prompt_to_orig_answer[(question, tuple(choices))] = predicted_letter

        results.append({
            "question": question,
            "choices": choices,
            "correct_answer": chr(65 + correct_answer),
            "predicted_answer": predicted_letter if predicted_letter else "INVALID",
            "prompt": prompt,
            "generation": generation,
            "correct": is_correct,
            "probs": answer_probs,
            "system_prompt": messages[0]["content"],
            "user_prompt": messages[1]["content"],
        })

    total = len(results)
    accuracy = correct / total if total else 0
    print(f"\n{'='*50}")
    print(f"Final Results ({prompt_type}): accuracy={accuracy:.2%} ({correct}/{total})")
    print(f"Dataset: {dataset_info['name'].upper()}  Subset: {dataset_info['subset']}")
    print(f"{'='*50}")

    output_data = {
        "dataset": dataset_info["name"],
        "subset": dataset_info["subset"],
        "num_shots": num_shots,
        "use_few_shot": use_few_shot,
        "restrict_output": restrict_output,
        "random_answer": random_answer,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_file}")

    return accuracy, results, prompt_to_orig_answer


def evaluate_mmlu_with_hint_vllm(
    tokenizer, llm, model_name,
    dataset_name="mmlu",
    subset="all",
    num_shots=0,
    num_samples=None,
    output_file=None,
    seed=42,
    restrict_output=True,
    hint_strategy="random",
    hint_before_question=True,
    hint_template=None,
    use_thinking_format=False,
    prompt_to_orig_answer=None,
    random_answer=False,
    logprobs_topk=20,
):
    random.seed(seed)
    use_few_shot = num_shots > 0

    dev_data, test_data, dataset_info = load_dataset_splits(dataset_name, subset)

    if output_file is None:
        ds = dataset_info["name"]
        sub = dataset_info["subset"] if ds == "mmlu" else ""
        output_file = f"{ds}_{sub}_hint_results.json".replace("__", "_")

    if dev_data is not None:
        dev_examples = list(dev_data)
    else:
        dev_examples = []
        if use_few_shot:
            print("Warning: no dev data; switching to zero-shot")
            use_few_shot = False
            num_shots = 0

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    prompt_type = f"{num_shots}-shot" if use_few_shot else "zero-shot"
    print(
        f"Using {prompt_type} with hints (strategy: {hint_strategy})"
        + (" + RANDOM ANSWERS (baseline)" if random_answer else " (vLLM batched)")
    )
    print(f"Evaluating on {len(test_data)} samples from {dataset_info['name'].upper()}")

    # 1. Build all prompts + hint metadata
    examples = []
    prompts = []
    for example in tqdm(test_data, desc="building prompts"):
        question = example["question"]
        choices = example["choices"]
        correct_answer = example["answer"]
        letters = [chr(65 + i) for i in range(len(choices))]

        if prompt_to_orig_answer is not None:
            orig_answer = prompt_to_orig_answer[(question, tuple(choices))]
        else:
            orig_answer = None

        # Hint letter selection (mirrors original logic)
        if hint_strategy == "toward_correct":
            hint_letter = chr(65 + correct_answer)
        elif hint_strategy == "toward_incorrect":
            incorrect = [L for i, L in enumerate(letters) if i != correct_answer]
            hint_letter = random.choice(incorrect) if incorrect else letters[0]
        else:
            hint_letter = random.choice(letters)
        hint = hint_template.format(hint_letter=hint_letter) if hint_template else None

        sampled_dev = (
            random.sample(dev_examples, min(num_shots, len(dev_examples)))
            if use_few_shot and dev_examples
            else []
        )
        messages = create_few_shot_prompt_with_hint(
            sampled_dev, question, choices, num_shots, use_few_shot,
            restrict_output, hint_before_question, hint,
        )
        prompt = apply_chat_template(
            tokenizer, messages, model_name, use_thinking_format,
            add_generation_prompt=False, continue_final_message=True,
        )

        # Hint token positions / ids — preserved for downstream activation patching
        hint_token_ids = get_hint_token_ids(tokenizer, hint_letter)
        try:
            hint_info = find_hint_token_positions(tokenizer, prompt, hint_letter, hint)
        except (ValueError, AttributeError):
            hint_info = {
                "hint_section_start": None,
                "hint_section_end": None,
                "hint_letter_positions": [],
                "total_tokens": len(tokenizer.encode(prompt)),
            }

        examples.append((
            question, choices, correct_answer, messages, prompt,
            hint_letter, hint, hint_token_ids, hint_info, orig_answer,
        ))
        prompts.append(prompt)

    # 2. Batched forward
    if random_answer:
        outputs = [None] * len(prompts)
    else:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=logprobs_topk
        )
        outputs = llm.generate(prompts, sampling_params)

    letter_ids = _letter_token_ids(tokenizer)

    # 3. Build records
    results = []
    correct = 0
    matched_hint = 0
    changed_pred_bc_hint = 0
    total = 0
    for ex, out in zip(examples, outputs):
        (question, choices, correct_answer, messages, prompt,
         hint_letter, hint, hint_token_ids, hint_info, orig_answer) = ex

        if random_answer:
            predicted_idx = random.randint(0, len(choices) - 1)
            predicted_letter = chr(65 + predicted_idx)
            generation = predicted_letter
            answer_probs = {}
        else:
            generation = out.outputs[0].text
            predicted_letter = extract_answer(generation)
            if predicted_letter is None:
                # Mirror original behavior: emit a minimal record and skip stats
                results.append({
                    "question": question,
                    "choices": choices,
                    "correct_answer": chr(65 + correct_answer),
                    "predicted_answer": None,
                    "hint_letter": hint_letter,
                    "prompt": prompt,
                    "generation": None,
                })
                continue
            predicted_idx = ord(predicted_letter) - ord("A")
            position_logprobs = (
                out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            )
            answer_probs = _probs_from_logprobs(letter_ids, position_logprobs)

        is_correct = predicted_idx == correct_answer
        is_hint_match = predicted_letter == hint_letter
        if is_correct:
            correct += 1
        if is_hint_match:
            matched_hint += 1
        changed_flag = None
        if orig_answer is not None and hint_letter.strip() != orig_answer.strip():
            changed_flag = predicted_letter.strip() != orig_answer.strip()
            if changed_flag:
                changed_pred_bc_hint += 1
        total += 1

        results.append({
            "question": question,
            "choices": choices,
            "correct_answer": chr(65 + correct_answer),
            "predicted_answer": predicted_letter if predicted_letter else "INVALID",
            "hint_letter": hint_letter,
            "hint_token_ids": hint_token_ids,
            "hint_info": hint_info,
            "prompt": prompt,
            "generation": generation,
            "probs": answer_probs,
            "correct": is_correct,
            "matched_hint": is_hint_match,
            "changed_pred_bc_hint": changed_flag,
            "system_prompt": messages[0]["content"],
            "user_prompt": messages[1]["content"],
            "hint": hint,
        })

    accuracy = correct / total if total else 0
    hint_match_rate = matched_hint / total if total else 0
    changed_rate = changed_pred_bc_hint / total if total else 0
    print(f"\n{'='*50}")
    print(f"Final Results ({prompt_type} with hints):")
    print(f"Dataset: {dataset_info['name'].upper()}  Subset: {dataset_info['subset']}")
    print(f"Hint strategy: {hint_strategy}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Hint match rate: {hint_match_rate:.2%} ({matched_hint}/{total})")
    print(f"Changed pred bc hint: {changed_rate:.2%}")
    print(f"{'='*50}")

    output_data = {
        "dataset": dataset_info["name"],
        "subset": dataset_info["subset"],
        "num_shots": num_shots,
        "use_few_shot": use_few_shot,
        "restrict_output": restrict_output,
        "hint_strategy": hint_strategy,
        "random_answer": random_answer,
        "accuracy": accuracy,
        "hint_match_rate": hint_match_rate,
        "correct": correct,
        "matched_hint": matched_hint,
        "total": total,
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_file}")

    return accuracy, results


def build_ablation_prompts(no_hint_path, with_hint_path, ablation_path, dataset_name):
    """Mirror of the post-processing block at end of mmlu_investigation.py.

    Reads no-hint + with-hint JSON files, emits ablation_prompts JSONL plus
    _train.jsonl / _test.jsonl splits. Required by downstream activation-patching
    scripts.
    """
    with open(no_hint_path) as f:
        zeroshot_data = json.load(f)["results"]
    with open(with_hint_path) as f:
        random_data = json.load(f)["results"]

    prompts = []
    change_count = 0
    hint_match_count = 0
    for i, (row1, row2) in enumerate(zip(zeroshot_data, random_data)):
        assert row1["question"] == row2["question"]
        if row1["predicted_answer"] is None or row2["predicted_answer"] is None:
            continue
        if row1["predicted_answer"] == "INVALID" or row2["predicted_answer"] == "INVALID":
            continue
        if row1["predicted_answer"] != row2["predicted_answer"]:
            change_count += 1
        if row2["matched_hint"]:
            hint_match_count += 1
        hint_token_ids = row2.get("hint_token_ids", [])
        prompt_text = row1.get("prompt", "")
        ablation_prompt = (
            f"If the activation is ablated at tokens {hint_token_ids} when "
            f"processing the text '{prompt_text}', how would the output change?"
            f"\n\nOutput changes from {row2['predicted_answer']} to {row1['predicted_answer']}"
        )
        prompts.append({
            "index": i,
            "question": row1["question"],
            "choices": row1["choices"],
            "correct_answer": row1["correct_answer"],
            "zeroshot_prediction": row1["predicted_answer"],
            "random_hint_prediction": row2["predicted_answer"],
            "hint_token_ids": hint_token_ids,
            "original_prompt_full": prompt_text,
            "hint_prompt_full": row2["prompt"],
            "system_prompt": row1["system_prompt"],
            "original_user_prompt": row1["user_prompt"],
            "hint_user_prompt": row2["user_prompt"],
            "hint": row2["hint"],
            "changed_pred_bc_hint": row2["changed_pred_bc_hint"],
            "ablation_prompt": ablation_prompt,
        })

    print(f"Found {change_count} cases where output changed between zeroshot and hint")
    print(f"Generated {len(prompts)} ablation prompts")
    print(f"Found {hint_match_count} cases where the answer matched the hint")

    with open(ablation_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    random.seed(42)
    random.shuffle(prompts)
    test_size = len(prompts) // 10 if dataset_name == "mmlu" else len(prompts)
    train_prompts = prompts[test_size:]
    test_prompts = prompts[:test_size]
    with open(ablation_path.replace(".jsonl", "_train.jsonl"), "w") as f:
        for p in train_prompts:
            f.write(json.dumps(p) + "\n")
    with open(ablation_path.replace(".jsonl", "_test.jsonl"), "w") as f:
        for p in test_prompts:
            f.write(json.dumps(p) + "\n")


_MODEL_STORE_NAMES = {
    "Qwen/Qwen3-8B": ("qwen3_8b", "Qwen/Qwen3-8B"),
    "Qwen/Qwen3-8B-Base": ("qwen3_8b_base", "Qwen/Qwen3-8B-Base"),
    "Qwen/Qwen3-32B": ("qwen3_32b", "Qwen/Qwen3-32B"),
    "meta-llama/Llama-3.1-8B": ("llama_3.1_8b", "meta-llama/Llama-3.1-8B"),
    "meta-llama/Llama-3.1-8B-Instruct": ("llama_3.1_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    "meta-llama/Llama-3.1-70B-Instruct": ("llama_3.1_70b_instruct", "meta-llama/Llama-3.1-70B-Instruct"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hint_template", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "gpqa"])
    parser.add_argument("--synthetic_data_path", type=str,
                        default="synthetic_mmlu_memorization.jsonl")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--random_answer", action="store_true")
    parser.add_argument("--synthetic_mode", action="store_true")
    parser.add_argument("--hint_change_prob", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Optional cap on test samples per pass (for smoke tests).")
    # vLLM-specific
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--logprobs_topk", type=int, default=20)
    args = parser.parse_args()

    suffix = args.suffix
    hint_template = hint_templates[args.hint_template]
    dataset_name = args.dataset.lower()

    # Synthetic mode: no model, delegate
    if args.synthetic_mode:
        if not args.synthetic_data_path:
            raise ValueError("--synthetic_data_path required with --synthetic_mode")
        model_store_name = args.model.replace("/", "_").lower()
        synthetic_suffix = "_".join([suffix, "synthetic"]) if suffix else "synthetic"
        no_hint_data_file = f"hint_data/synthetic_{model_store_name}_zeroshot_results_{synthetic_suffix}.json"
        with_hint_data_file = f"hint_data/synthetic_{model_store_name}_zeroshot_with_hint_results_random_{synthetic_suffix}.json"
        full_ablation_prompt_file = f"hint_data/synthetic_{model_store_name}_ablation_prompts_{synthetic_suffix}.jsonl"
        generate_synthetic_data(
            synthetic_data_path=args.synthetic_data_path,
            hint_template=hint_template,
            hint_strategy="random",
            hint_change_prob=args.hint_change_prob,
            seed=42,
            no_hint_output_file=no_hint_data_file,
            with_hint_output_file=with_hint_data_file,
            ablation_prompt_file=full_ablation_prompt_file,
        )
        print("\nSynthetic mode complete. Exiting.")
        sys.exit(0)

    # Resolve model store name
    if args.model not in _MODEL_STORE_NAMES:
        raise ValueError(f"Invalid model: {args.model}")
    model_store_name, model_name = _MODEL_STORE_NAMES[args.model]
    tokenizer_name = model_name

    no_hint_data_file = f"hint_data/{dataset_name}_{model_store_name}_zeroshot_results_{suffix}.json"
    with_hint_data_file = f"hint_data/{dataset_name}_{model_store_name}_zeroshot_with_hint_results_random_{suffix}.json"
    full_ablation_prompt_file = f"hint_data/{dataset_name}_{model_store_name}_ablation_prompts_{suffix}.jsonl"

    if not args.overwrite and os.path.exists(full_ablation_prompt_file):
        print(f"{full_ablation_prompt_file} exists and --overwrite not set. Exiting.")
        sys.exit(0)

    # Load vLLM
    tokenizer, llm = load_vllm(
        model_name=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        checkpoint=args.checkpoint,
        tokenizer_name=tokenizer_name if args.checkpoint else None,
    )

    subset = "all" if dataset_name == "mmlu" else "gpqa_main"

    _, _, prompt_to_orig_answer = evaluate_mmlu_few_shot_vllm(
        tokenizer, llm, model_name=model_name,
        dataset_name=dataset_name,
        subset=subset,
        num_shots=0,
        num_samples=args.num_samples,
        output_file=no_hint_data_file,
        seed=42,
        restrict_output=True,
        use_thinking_format=True,
        random_answer=args.random_answer,
        logprobs_topk=args.logprobs_topk,
    )

    evaluate_mmlu_with_hint_vllm(
        tokenizer, llm, model_name=model_name,
        dataset_name=dataset_name,
        subset=subset,
        num_shots=0,
        num_samples=args.num_samples,
        output_file=with_hint_data_file,
        seed=42,
        restrict_output=True,
        hint_strategy="random",
        hint_before_question=False,
        hint_template=hint_template,
        use_thinking_format=True,
        prompt_to_orig_answer=prompt_to_orig_answer,
        random_answer=args.random_answer,
        logprobs_topk=args.logprobs_topk,
    )

    build_ablation_prompts(
        no_hint_data_file, with_hint_data_file, full_ablation_prompt_file, dataset_name,
    )


if __name__ == "__main__":
    main()
