"""Precompute top-k teacher logits for KL regularization.

Two modes:

1. **single-token (legacy)**: stores top-k logits at the *last* position only.
   Used for tasks where the assistant response is a 1-2 token answer key
   (e.g. MMLU hint letter, Yes/No judge verdict). Detected from the JSONL by
   the presence of ``hint_prompt_full`` / ``with_system_prompt_full``.

2. **freeform**: stores top-k logits at *every* assistant-response token,
   shape ``[T, k]`` per sample. Used when the regularization target is a full
   model rollout (e.g. ``freeform_response_with_sys`` written by
   ``recompute_labels_freeform_judge``). Triggered by
   ``--mode freeform``; requires ``freeform_response_with_sys`` and
   ``freeform_response_no_sys`` fields in each JSONL row, plus
   ``original_user_prompt`` / ``user_prompt`` and ``system_prompt``.

Usage:
    # single-token (existing)
    uv run --env-file .env scripts/precompute_teacher_logits.py \
        --model_path Qwen/Qwen3-8B \
        --hint_path hint_data/mmlu_qwen3_8b_ablation_prompts_train_shuffled.jsonl \
        --topk 128

    # freeform (new)
    uv run --env-file .env scripts/precompute_teacher_logits.py \
        --model_path Qwen/Qwen3-8B \
        --hint_path falsereject_data/combined_qwen3_8b_system_prompt_train_judge_balanced.jsonl \
        --mode freeform --topk 128
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_chat_text(tokenizer, user_prompt: str, system_prompt: str,
                     add_generation_prompt: bool, assistant_response: str | None = None,
                     enable_thinking: bool | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    # Only forward enable_thinking when explicitly set so non-Qwen templates
    # (which don't accept the kwarg) are unaffected.
    extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **extra
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **extra
    )


@torch.no_grad()
def _per_token_topk(
    model, tokenizer, user_prompt: str, system_prompt: str,
    response: str, topk: int,
    enable_thinking: bool | None = None,
    return_response_ids: bool = False,
):
    """Compute per-position top-k teacher logits over the response span.

    Returns (topk_idx[T, k], topk_vals[T, k]) on CPU, or None on alignment
    failure (prompt prefix doesn't match full sequence). When
    ``return_response_ids`` is True, returns (topk_idx, topk_vals,
    response_ids[T]) — the exact token ids the T teacher rows supervise, so a
    consuming dataset can label precisely those positions (front-aligned KL).
    """
    prompt_text = _build_chat_text(
        tokenizer, user_prompt, system_prompt, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = _build_chat_text(
        tokenizer, user_prompt, system_prompt,
        add_generation_prompt=False, assistant_response=response,
        enable_thinking=enable_thinking,
    )

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids

    p_len = prompt_ids.shape[1]
    f_len = full_ids.shape[1]
    if f_len <= p_len:
        return None
    if not torch.equal(full_ids[0, :p_len], prompt_ids[0]):
        return None

    full_ids = full_ids.to(model.device)
    logits = model(input_ids=full_ids).logits[0]  # [f_len, vocab]

    # Align the cached span with the runtime answer-mask rule
    # (dataloaders/base_dataset.py:1054-1061): take the smallest suffix
    # whose decoded form (with special tokens stripped) matches the raw
    # response string. This drops chat-template wrappers like Qwen3's
    # `<think>\n\n</think>\n\n` preamble that sit inside [p_len, f_len)
    # but outside the student's labeled positions.
    t_max = f_len - p_len
    target = response.strip().replace(" ", "")
    t_match = None
    for t in range(1, t_max + 1):
        cand = tokenizer.decode(
            full_ids[0, f_len - t:f_len], skip_special_tokens=True
        ).strip().replace(" ", "")
        if cand == target:
            t_match = t
            break
    if t_match is None:
        fallback_t = len(tokenizer.tokenize(response))
        t_match = max(1, min(fallback_t, t_max))
        if not getattr(_per_token_topk, "_warned_fallback", False):
            print(
                f"[precompute] WARN: answer-span match failed; using "
                f"tokenize-length fallback t={t_match} (t_max={t_max}). "
                f"Suppressing further warnings.",
                flush=True,
            )
            _per_token_topk._warned_fallback = True

    # Predictions for the last t_match tokens are produced by indices
    # f_len - 1 - t_match .. f_len - 2 (standard next-token shift).
    response_logits = logits[f_len - 1 - t_match:f_len - 1, :].float()
    topk_vals, topk_idx = response_logits.topk(topk, dim=-1)
    if return_response_ids:
        response_ids = full_ids[0, f_len - t_match:f_len].cpu()
        return topk_idx.cpu(), topk_vals.cpu(), response_ids
    return topk_idx.cpu(), topk_vals.cpu()


@torch.no_grad()
def _generate_responses(
    model, tokenizer, user_prompts: list, max_new_tokens: int,
    enable_thinking: bool | None = None,
) -> list:
    """Greedy-decode reference completions for a batch of user prompts.

    Decoder-only batched generation requires LEFT padding; we restore the
    tokenizer's padding side afterwards. Returns a list of completion strings
    (stripped of the prompt and special tokens), aligned with user_prompts.
    """
    prompt_texts = [
        _build_chat_text(
            tokenizer, p, system_prompt="", add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for p in user_prompts
    ]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev_side
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
    )
    gen_ids = out[:, enc["input_ids"].shape[1]:]
    return [
        tokenizer.decode(row, skip_special_tokens=True).strip()
        for row in gen_ids
    ]


def _iter_wildchat_prompts(dataset_name: str, num_prompts: int, tokenizer,
                           max_prompt_tokens: int):
    """Stream WildChat, yield (index, user_prompt) for English single-turn,
    non-toxic prompts under the token budget. Assigns a stable 0..N-1 index."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train", streaming=True)
    kept = 0
    for row in ds:
        if kept >= num_prompts:
            break
        if row.get("language") != "English":
            continue
        if row.get("turn") not in (1, "1"):
            continue
        if row.get("toxic"):
            continue
        conv = row.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        user_prompt = (conv[0].get("content") or "").strip()
        if not user_prompt:
            continue
        n_tok = len(tokenizer(user_prompt).input_ids)
        if n_tok == 0 or n_tok > max_prompt_tokens:
            continue
        yield kept, user_prompt
        kept += 1


def _run_wildchat(args):
    """Generate greedy reference completions for WildChat prompts and store
    per-token top-k teacher logits + the exact scored token-id span.

    Writes a chat JSONL (one row per prompt: index, messages[user, assistant],
    source) and a .pt keyed by index:
        { index: {"teacher": (idx[T,k], vals[T,k]), "response_ids": LongTensor[T]} }
    """
    enable_thinking = True if args.enable_thinking else False

    model_slug = args.model_path.rstrip("/").split("/")[-1]
    if args.wildchat_jsonl_out is None:
        args.wildchat_jsonl_out = (
            f"data/wildchat_kl/wildchat_{model_slug}_{args.wildchat_num_prompts}.jsonl"
        )
    if args.output_path is None:
        args.output_path = (
            f"{args.wildchat_jsonl_out}.teacher_logits_wildchat_top{args.topk}.pt"
        )

    if os.path.exists(args.output_path):
        print(f"Output file already exists: {args.output_path}")
        print("Delete it first if you want to recompute.")
        return

    os.makedirs(os.path.dirname(args.wildchat_jsonl_out) or ".", exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, cache_dir=args.cache_dir,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    print(
        f"Streaming {args.wildchat_dataset} for {args.wildchat_num_prompts} prompts "
        f"(enable_thinking={enable_thinking}, max_new_tokens={args.max_new_tokens})"
    )

    results = {}
    skipped = 0
    written = 0
    jsonl_f = open(args.wildchat_jsonl_out, "w")

    def _flush_batch(batch):
        nonlocal skipped, written
        if not batch:
            return
        idxs = [b[0] for b in batch]
        prompts_b = [b[1] for b in batch]
        responses = _generate_responses(
            model, tokenizer, prompts_b, args.max_new_tokens,
            enable_thinking=enable_thinking,
        )
        # Scoring is one teacher-forced forward per sample (cheap relative to
        # the 512-step generation, which is now amortized across the batch).
        for index, user_prompt, response in zip(idxs, prompts_b, responses):
            if not response:
                skipped += 1
                continue
            scored = _per_token_topk(
                model, tokenizer, user_prompt, system_prompt="", response=response,
                topk=args.topk, enable_thinking=enable_thinking,
                return_response_ids=True,
            )
            if scored is None:
                skipped += 1
                continue
            topk_idx, topk_vals, response_ids = scored
            results[index] = {
                "teacher": (topk_idx, topk_vals),
                "response_ids": response_ids,
            }
            jsonl_f.write(json.dumps({
                "index": index,
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response},
                ],
                "source": "wildchat",
            }) + "\n")
            written += 1
        jsonl_f.flush()

    try:
        prompts = _iter_wildchat_prompts(
            args.wildchat_dataset, args.wildchat_num_prompts, tokenizer,
            args.max_prompt_tokens,
        )
        batch = []
        for index, user_prompt in tqdm(prompts, total=args.wildchat_num_prompts,
                                       desc="WildChat teacher logits"):
            batch.append((index, user_prompt))
            if len(batch) >= args.gen_batch_size:
                _flush_batch(batch)
                batch = []
                if written and written % 2000 < args.gen_batch_size:
                    print(f"  Checkpoint at {written} kept, saving partial...")
                    torch.save(results, args.output_path + ".partial")
        _flush_batch(batch)
    finally:
        jsonl_f.close()

    print(f"Saving {len(results)} entries to {args.output_path}"
          + (f" (skipped {skipped})" if skipped else ""))
    torch.save(results, args.output_path)
    partial_path = args.output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"Wrote {written} rows to {args.wildchat_jsonl_out}")
    print(f"Teacher cache: {args.output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--hint_path", type=str, default=None,
                        help="Input JSONL (single/freeform modes). Not used in "
                             "wildchat mode, which streams prompts from HF.")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "freeform", "wildchat"],
        help="single = topk at last position only (legacy); freeform = per-token "
             "topk over the assistant response span; wildchat = generate a greedy "
             "reference completion per streamed WildChat prompt, then per-token topk "
             "over it (single completion, no with/no pair).",
    )
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output .pt path. Defaults to "
                             "<hint_path>.teacher_logits_top<k>.pt (single), "
                             "<hint_path>.teacher_logits_freeform_top<k>.pt (freeform), "
                             "or <wildchat_jsonl_out>.teacher_logits_wildchat_top<k>.pt.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (currently only 1 supported)")
    # wildchat-mode options
    parser.add_argument("--wildchat_dataset", type=str, default="allenai/WildChat-1M",
                        help="HF dataset id streamed for wildchat mode.")
    parser.add_argument("--wildchat_num_prompts", type=int, default=16000,
                        help="Number of WildChat prompts to keep.")
    parser.add_argument("--wildchat_jsonl_out", type=str, default=None,
                        help="Output chat JSONL written in wildchat mode. Defaults "
                             "to data/wildchat_kl/wildchat_<model>_<N>.jsonl.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Greedy generation length for wildchat completions.")
    parser.add_argument("--gen_batch_size", type=int, default=32,
                        help="Batch size for wildchat greedy generation (the "
                             "throughput lever; scoring stays per-sample).")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024,
                        help="Skip WildChat prompts longer than this many tokens.")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3 thinking in wildchat gen/scoring "
                             "(default OFF — keep consistent with the dataset).")
    args = parser.parse_args()

    if args.mode == "wildchat":
        _run_wildchat(args)
        return

    if not args.hint_path:
        raise ValueError("--hint_path is required for single/freeform modes.")

    if args.output_path is None:
        suffix = (
            f"teacher_logits_freeform_top{args.topk}"
            if args.mode == "freeform"
            else f"teacher_logits_top{args.topk}"
        )
        args.output_path = f"{args.hint_path}.{suffix}.pt"

    if os.path.exists(args.output_path):
        print(f"Output file already exists: {args.output_path}")
        print("Delete it first if you want to recompute.")
        return

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, cache_dir=args.cache_dir
    )
    model_kwargs = dict(
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    print(f"Loading data: {args.hint_path}")
    with open(args.hint_path) as f:
        lines = f.readlines()
    print(f"Total samples: {len(lines)}")

    first_sample = json.loads(lines[0])

    if args.mode == "freeform":
        # Two layouts supported:
        #   (a) freeform_response_with_sys / freeform_response_no_sys
        #       (written by recompute_labels_freeform_judge)
        #   (b) with_system_prediction / no_system_prediction containing the
        #       freeform model rollout directly (this is the case for the
        #       judge-balanced dataset where the verdicts live in
        #       random_hint_prediction / zeroshot_prediction).
        if "freeform_response_with_sys" in first_sample:
            with_resp_key = "freeform_response_with_sys"
            no_resp_key = "freeform_response_no_sys"
            print("Detected freeform-judge format (freeform_response_* fields)")
        elif "with_system_prediction" in first_sample:
            with_resp_key = "with_system_prediction"
            no_resp_key = "no_system_prediction"
            print("Detected judge-balanced format (with_system_prediction holds freeform)")
        else:
            raise ValueError(
                "freeform mode requires 'freeform_response_with_sys' or "
                "'with_system_prediction' fields."
            )
        save_with, save_without = "with_hint", "no_hint"
    else:
        if "hint_prompt_full" in first_sample:
            with_key, without_key = "hint_prompt_full", "original_prompt_full"
            print("Detected MMLU hint format")
        elif "with_system_prompt_full" in first_sample:
            with_key, without_key = "with_system_prompt_full", "no_system_prompt_full"
            print("Detected system_prompt ablation format")
        else:
            raise ValueError(
                f"Unknown data format. Expected 'hint_prompt_full' or "
                f"'with_system_prompt_full'. Found keys: {sorted(first_sample.keys())}"
            )
        save_with, save_without = "with_hint", "no_hint"

    results = {}
    skipped = 0
    for idx, line in enumerate(tqdm(lines, desc="Precomputing teacher logits")):
        sample = json.loads(line)
        sample_key = sample["index"]

        if args.mode == "freeform":
            user_prompt = sample.get("original_user_prompt") or sample.get("user_prompt", "")
            system_prompt = sample.get("system_prompt", "")
            with_resp = sample.get(with_resp_key)
            no_resp = sample.get(no_resp_key)
            if (
                not user_prompt or with_resp is None or no_resp is None
                or sample.get("judge_skip", False)
            ):
                skipped += 1
                continue
            with_pair = _per_token_topk(
                model, tokenizer, user_prompt, system_prompt, with_resp, args.topk,
            )
            no_pair = _per_token_topk(
                model, tokenizer, user_prompt, "", no_resp, args.topk,
            )
            if with_pair is None or no_pair is None:
                skipped += 1
                continue
            results[sample_key] = {save_with: with_pair, save_without: no_pair}
        else:
            with torch.no_grad():
                tokens = tokenizer(sample[with_key], return_tensors="pt")
                tokens = {k: v.to(model.device) for k, v in tokens.items()}
                with_logits = model(**tokens).logits[0, -1, :].float()
                with_topk_vals, with_topk_idx = with_logits.topk(args.topk)

                tokens = tokenizer(sample[without_key], return_tensors="pt")
                tokens = {k: v.to(model.device) for k, v in tokens.items()}
                without_logits = model(**tokens).logits[0, -1, :].float()
                without_topk_vals, without_topk_idx = without_logits.topk(args.topk)

            results[sample_key] = {
                save_with: (with_topk_idx.cpu(), with_topk_vals.cpu()),
                save_without: (without_topk_idx.cpu(), without_topk_vals.cpu()),
            }

        if (idx + 1) % 5000 == 0:
            print(f"  Checkpoint at {idx + 1} samples, saving...")
            torch.save(results, args.output_path + ".partial")

    print(f"Saving {len(results)} entries to {args.output_path}"
          + (f" (skipped {skipped})" if skipped else ""))
    torch.save(results, args.output_path)

    partial_path = args.output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
