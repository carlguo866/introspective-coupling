#%%
import argparse
import json
import re
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# --- Config ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="pre_hint",
                        help="Run identifier for output filenames")
    parser.add_argument("--job_id", type=str, default=None,
                        help="Job ID for output filenames (random if unset)")
    parser.add_argument("--max_pairs_per_hint", type=int, default=10,
                        help="Max pairs per hint letter (<=0 means all)")
    parser.add_argument("--num_questions", type=int, default=1399,
                        help="Number of question triples to consider")
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="Path to causal_predictions.json (orig or self)")
    parser.add_argument("--require_all_correct", action="store_true",
                        help="Only keep triples where exact_match=True on all three question_types")
    parser.add_argument("--denom_threshold", type=float, default=1e-6,
                        help="Drop a pair if any (variant, direction) has "
                             "|source_diff - target_diff| <= this. Default 1e-6 "
                             "matches the original 1e-6 inner-loop guard "
                             "(no upfront pair filtering). Use ~1.0 to drop "
                             "denom-collapse pairs that explode the normalized "
                             "logit-diff ratio.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="HF-format checkpoint dir (or hub ID) for the predictor model.")
    parser.add_argument("--output_dir", type=str, default="interp/output",
                        help="Base output directory")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache dir (defaults to HF_HOME)")
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
RUN_NAME = args.run_id
JOB_ID = args.job_id
if JOB_ID is None:
    random_suffix = random.randint(1000000, 9999999)
    JOB_ID = f"random_{random_suffix}"
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_DIR = OUTPUT_DIR / RUN_NAME / JOB_ID
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = args.predictions_path
CHECKPOINT_DIR = args.checkpoint_dir
CACHE_DIR = args.cache_dir
NUM_QUESTIONS = args.num_questions
MAX_PAIRS_PER_HINT = args.max_pairs_per_hint
REQUIRE_ALL_CORRECT = args.require_all_correct
DENOM_THRESHOLD = args.denom_threshold

VARIANTS = ["orig_match", "explanation"]
DIRECTIONS = ["changed_into_unchanged", "unchanged_into_changed"]

#%% --- 1. Load data and group by hint ---
print("Loading predictions...")
all_entries = []
with open(PREDICTIONS_PATH) as f:
    for line in f:
        all_entries.append(json.loads(line))


def get_sample_info(triple_idx):
    """Extract sample info from a triple index, including all three prompt types."""
    i = triple_idx * 3
    if i + 2 >= len(all_entries):
        return None
    triple = {e["question_type"]: e for e in all_entries[i:i+3]}
    if "generative_explanation" not in triple or "orig_match" not in triple or "change_match" not in triple:
        return None
    ea = triple["orig_match"]["extra_args"]
    gen = triple["generative_explanation"]
    orig = ea["original_response"].strip()
    ablated = ea["ablated_response"].strip()
    hint_match = re.search(r"the answer is likely ([A-D])", ea["hint"])
    hint_letter = hint_match.group(1) if hint_match else None
    orig_letter = re.search(r"([A-D])", orig)
    ablated_letter = re.search(r"([A-D])", ablated)
    return {
        "idx": triple_idx,
        "orig": orig,
        "ablated": ablated,
        "orig_letter": orig_letter.group(1) if orig_letter else None,
        "ablated_letter": ablated_letter.group(1) if ablated_letter else None,
        "hint": hint_letter,
        "changed": orig != ablated,
        "gen_correct": gen.get("exact_match", False),
        "orig_match_correct": triple["orig_match"].get("exact_match", False),
        "change_match_correct": triple["change_match"].get("exact_match", False),
        "prompts": {
            "orig_match": triple["orig_match"]["prompt_without_answer"],
            "change_match": triple["change_match"]["prompt_without_answer"],
            "generative_explanation": triple["generative_explanation"]["prompt_without_answer"],
        },
    }


# Group first NUM_QUESTIONS triples by hint letter and changed/unchanged
by_hint = defaultdict(lambda: {"changed": [], "unchanged": []})
for idx in range(NUM_QUESTIONS):
    info = get_sample_info(idx)
    if info is None or info["hint"] is None:
        continue
    if not info["gen_correct"]:
        continue
    if REQUIRE_ALL_CORRECT and not (info["orig_match_correct"] and info["change_match_correct"]):
        continue
    if info["orig_letter"] is None or info["ablated_letter"] is None:
        continue
    key = "changed" if info["changed"] else "unchanged"
    by_hint[info["hint"]][key].append(info)

print(f"\nGrouping (first {NUM_QUESTIONS} questions, gen_correct only):")
for hint in sorted(by_hint):
    c = len(by_hint[hint]["changed"])
    u = len(by_hint[hint]["unchanged"])
    print(f"  Hint {hint}: {c} changed, {u} unchanged")

# Create pairs: for each hint, pair changed[i] with unchanged[i]
pairs = []
for hint in sorted(by_hint):
    changed_list = by_hint[hint]["changed"]
    unchanged_list = by_hint[hint]["unchanged"]
    # Pair prompts whose cue-included answers differ (M(x) != M(x')): each changed prompt takes the next
    # unused unchanged prompt that answers a different letter, so the behavior logit difference is defined.
    cap = MAX_PAIRS_PER_HINT if MAX_PAIRS_PER_HINT is not None and MAX_PAIRS_PER_HINT > 0 else None
    used = set()
    for c in changed_list:
        if cap is not None and sum(1 for p_ in pairs if p_[0]["hint"] == hint) >= cap:
            break
        for j, u in enumerate(unchanged_list):
            if j not in used and u["orig_letter"] != c["orig_letter"]:
                used.add(j)
                pairs.append((c, u))
                break

print(f"\nTotal pairs: {len(pairs)}")

#%% --- 2. Load model ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, cache_dir=CACHE_DIR)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
model.eval()

nnsight_model = LanguageModel(model, tokenizer=tokenizer)
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Model loaded: {num_layers} layers, hidden_size={hidden_size}")

# Metric token IDs
# For orig_match: answer letters
answer_token_ids = {}
for letter in "ABCD":
    toks = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
    answer_token_ids[letter] = toks[0]
    print(f"  Token for ' {letter}': {toks[0]} -> {repr(tokenizer.decode([toks[0]]))}")

# For explanation variant
output_token_id = tokenizer(" output", add_special_tokens=False)["input_ids"][0]
most_token_id = tokenizer(" most", add_special_tokens=False)["input_ids"][0]
print(f"  Token for ' output': {output_token_id} -> {repr(tokenizer.decode([output_token_id]))}")
print(f"  Token for ' most': {most_token_id} -> {repr(tokenizer.decode([most_token_id]))}")

IM_END = "<|im_end|>"

#%% --- 3. Prompt preparation and region identification ---


def prepare_prompt(sample, variant):
    """Prepare prompt text for a given variant.

    For orig_match: apply template, strip trailing <|im_end|>
      so the last token is ":" from "Answer:" and logits predict the answer letter.
    For explanation: apply template, append "The"
      so the last token is "The" and logits predict " output" or " most".
    """
    if variant == "explanation":
        messages = [dict(m) for m in sample["prompts"]["generative_explanation"]]
        # The meta-question ends with an empty assistant turn; rendering it as-is closes the turn, and "The"
        # would land outside the reply. Put "The" inside the assistant turn and drop the closing tag, as
        # orig_match ends at "Answer:".
        if messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
            messages[-1]["content"] = "The"
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ).rstrip()
            if text.endswith(IM_END):
                text = text[:-len(IM_END)]
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            text += "The"
    else:
        messages = sample["prompts"][variant]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Strip trailing <|im_end|> so the prompt ends at "Answer:"
        text = text.rstrip()
        if text.endswith(IM_END):
            text = text[:-len(IM_END)]
    return text


def compute_suffix_offsets(sample):
    """Compute tokens-from-end offset for the post-hint boundary per variant.

    The patching window of NUM_TOKENS ends at the hint's last token.
    Suffix = number of tokens AFTER the hint to the end of the prompt.
    """
    # Tokenize all three needed variants
    om_text = prepare_prompt(sample, "orig_match")
    cm_text = prepare_prompt(sample, "change_match")
    ge_text = prepare_prompt(sample, "explanation")
    om_ids = tokenizer(om_text, add_special_tokens=False)["input_ids"]
    cm_ids = tokenizer(cm_text, add_special_tokens=False)["input_ids"]
    ge_ids = tokenizer(ge_text, add_special_tokens=False)["input_ids"]

    # hint tokens = difference between orig_match and change_match
    hint_tokens = len(om_ids) - len(cm_ids)

    # Count matching tokens from the END of change_match and orig_match
    # These are the template tokens after the user content (shared suffix = "Answer:" etc.)
    template_overhead = 0
    for i in range(1, min(len(cm_ids), len(om_ids)) + 1):
        if cm_ids[-i] == om_ids[-i]:
            template_overhead += 1
        else:
            break

    # For explanation: compare from the START with orig_match to find divergence
    # They share: system + question + choices + hint, then diverge at meta-question
    prefix_match = 0
    for i in range(min(len(ge_ids), len(om_ids))):
        if ge_ids[i] == om_ids[i]:
            prefix_match += 1
        else:
            break

    # The hint's final "." can merge with the meta-question's "\n\n" into one token (".\n\n") in the
    # explanation prompt, so the shared prefix stops one token short of the hint end and the explanation
    # window would sit one token left of the orig_match window. Count that merged token as the hint's last
    # token so both windows cover the same hint tokens.
    if prefix_match < min(len(ge_ids), len(om_ids)):
        ge_tok = tokenizer.decode([ge_ids[prefix_match]])
        om_tok = tokenizer.decode([om_ids[prefix_match]])
        if om_tok and ge_tok.startswith(om_tok) and om_tok.strip():
            prefix_match += 1

    # explanation suffix = tokens after the shared prefix (hint end)
    explanation_suffix = len(ge_ids) - prefix_match

    return {
        # For orig_match: suffix is just the template tokens AFTER the hint
        "orig_match": template_overhead,
        # For explanation: suffix is everything after the shared prefix (meta-question + template + "The")
        "explanation": explanation_suffix,
        "hint_tokens": hint_tokens,
        "token_counts": {
            "orig_match": len(om_ids),
            "change_match": len(cm_ids),
            "explanation": len(ge_ids),
        },
    }


def get_metric_token_ids(variant, source_sample, target_sample):
    """Return metric token IDs as {source_tid, target_tid} for a given variant."""
    if variant == "explanation":
        if source_sample["changed"]:
            return {"source": most_token_id, "target": output_token_id}
        else:
            return {"source": output_token_id, "target": most_token_id}
    elif variant == "orig_match":
        return {
            "source": answer_token_ids[source_sample["orig_letter"]],
            "target": answer_token_ids[target_sample["orig_letter"]],
        }


def get_logit_diffs(logits, metric_tids):
    """Get logits for metric tokens at the last position."""
    last_logits = logits[0, -1, :]
    return {name: last_logits[tid].item() for name, tid in metric_tids.items()}


# Compute hint token counts to determine patching window (hint tokens only)
hint_token_counts = []
for cs, us in pairs:
    offsets = compute_suffix_offsets(cs)
    hint_token_counts.append(offsets["hint_tokens"])
NUM_TOKENS = max(hint_token_counts)
print(f"Hint token counts: min={min(hint_token_counts)}, max={max(hint_token_counts)}, using max={NUM_TOKENS}")

#%% --- 4. Patching functions ---


def run_patching_for_variant(source_sample, target_sample, variant, suffix_offsets,
                             desc=None):
    """Run activation patching for one (source, target) pair on one prompt variant.

    Patches at NUM_TOKENS positions ending at the hint's last token (includes hint).
    Right-aligned: rightmost position = last hint token.

    Returns: dict with 'matrix' (num_layers, NUM_TOKENS), 'source_diff', 'target_diff'.
    """
    device = next(model.parameters()).device

    # Prepare and tokenize
    text_src = prepare_prompt(source_sample, variant)
    text_tgt = prepare_prompt(target_sample, variant)

    # Compute unpadded lengths to determine available tokens
    src_len = len(tokenizer(text_src, add_special_tokens=False)["input_ids"])
    tgt_len = len(tokenizer(text_tgt, add_special_tokens=False)["input_ids"])
    suffix = suffix_offsets[variant]
    src_available = src_len - suffix
    tgt_available = tgt_len - suffix
    hint_tokens = suffix_offsets["hint_tokens"]
    actual_tokens = min(hint_tokens, src_available, tgt_available)

    # Left-pad together for aligned indexing from the right
    # Run each prompt unpadded: the patching window is indexed from the right, so no shared padding is
    # needed, and the traces pass no attention mask, so left padding would be attended to and shift the
    # shorter prompt's activations and logits.
    ids_src = tokenizer(text_src, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids_tgt = tokenizer(text_tgt, return_tensors="pt", add_special_tokens=False)["input_ids"]

    # Patching region slice (right-aligned: rightmost = last hint token)
    region_end = -suffix if suffix > 0 else None
    region_start = -(actual_tokens + suffix) if suffix > 0 else -actual_tokens

    # Metric token IDs
    metric_tids = get_metric_token_ids(variant, source_sample, target_sample)

    # Step 1: Extract source hidden states at patching positions
    clean_hidden_states = {}
    with torch.no_grad():
        with nnsight_model.trace() as tracer:
            with tracer.invoke(ids_src.to(device)):
                for layer_idx in range(num_layers):
                    clean_hidden_states[layer_idx] = (
                        nnsight_model.model.layers[layer_idx].output[
                            :, region_start:region_end, :
                        ].save()
                    )
                source_logits_raw = nnsight_model.output.logits.save()

    source_logits = get_logit_diffs(source_logits_raw, metric_tids)
    source_diff = source_logits["source"] - source_logits["target"]
    del source_logits_raw
    for k in clean_hidden_states:
        clean_hidden_states[k] = clean_hidden_states[k].cpu()
    torch.cuda.empty_cache()

    # Step 2: Target baseline
    with torch.no_grad():
        with nnsight_model.trace() as tracer:
            with tracer.invoke(ids_tgt.to(device)):
                target_logits_raw = nnsight_model.output.logits.save()

    target_logits = get_logit_diffs(target_logits_raw, metric_tids)
    target_diff = target_logits["source"] - target_logits["target"]
    del target_logits_raw
    torch.cuda.empty_cache()

    denom = source_diff - target_diff

    # Early-return if denom is too small: skip the (expensive) patching loop.
    if abs(denom) <= DENOM_THRESHOLD:
        clean_hidden_states.clear()
        torch.cuda.empty_cache()
        return {
            "matrix": None,
            "patched_diff_matrix": None,
            "source_diff": source_diff,
            "target_diff": target_diff,
            "denom": denom,
            "is_valid": False,
        }

    # Step 3: Per layer x per token patching
    raw_matrix = np.zeros((num_layers, actual_tokens))
    raw_patched_diff_matrix = np.zeros((num_layers, actual_tokens))
    pbar = tqdm(total=num_layers * actual_tokens, desc=desc or "Patching",
                leave=False, unit="patch")

    for layer_idx in range(num_layers):
        layer_hidden = clean_hidden_states[layer_idx].to(device)

        for tok_pos in range(actual_tokens):
            with torch.no_grad():
                with nnsight_model.trace() as tracer:
                    with tracer.invoke(ids_tgt.to(device)):
                        abs_offset = region_start + tok_pos
                        nnsight_model.model.layers[layer_idx].output[
                            :, abs_offset, :
                        ] = layer_hidden[:, tok_pos, :]
                        patched_logits_raw = nnsight_model.output.logits.save()

            patched = get_logit_diffs(patched_logits_raw, metric_tids)
            del patched_logits_raw

            patched_diff = patched["source"] - patched["target"]
            raw_patched_diff_matrix[layer_idx, tok_pos] = patched_diff
            raw_matrix[layer_idx, tok_pos] = (
                (patched_diff - target_diff) / denom
            )

            pbar.update(1)

        del layer_hidden
        torch.cuda.empty_cache()

    pbar.close()

    # Left-pad to NUM_TOKENS if needed (right-align: content at the right, zeros on the left)
    if actual_tokens < NUM_TOKENS:
        patching_matrix = np.zeros((num_layers, NUM_TOKENS))
        patching_matrix[:, NUM_TOKENS - actual_tokens:] = raw_matrix
        patched_diff_matrix = np.zeros((num_layers, NUM_TOKENS))
        patched_diff_matrix[:, NUM_TOKENS - actual_tokens:] = raw_patched_diff_matrix
    else:
        patching_matrix = raw_matrix
        patched_diff_matrix = raw_patched_diff_matrix

    return {
        "matrix": patching_matrix,
        "patched_diff_matrix": patched_diff_matrix,
        "source_diff": source_diff,
        "target_diff": target_diff,
        "denom": denom,
        "is_valid": True,
    }


#%% --- 5. Sanity checks on first pair ---
print("\n--- Sanity check (pair 0) ---")
rep_changed, rep_unchanged = pairs[0]
suffix_offsets = compute_suffix_offsets(rep_changed)
print(f"  Suffix offsets: orig_match={suffix_offsets['orig_match']}, "
      f"explanation={suffix_offsets['explanation']}")
print(f"  Hint tokens: {suffix_offsets['hint_tokens']}")
print(f"  Token counts: {suffix_offsets['token_counts']}")

hint_toks = suffix_offsets["hint_tokens"]
for variant in VARIANTS:
    text = prepare_prompt(rep_changed, variant)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    suffix = suffix_offsets[variant]
    region_end = -suffix if suffix > 0 else None
    region_start = -(hint_toks + suffix) if suffix > 0 else -hint_toks
    if region_end:
        region_tokens = ids[region_start:region_end]
    else:
        region_tokens = ids[region_start:]
    decoded = tokenizer.decode(region_tokens)
    print(f"\n  {variant} patching region ({len(region_tokens)} tokens):")
    print(f"    {decoded[:300]}...")

print(f"\n  Changed: idx={rep_changed['idx']}, orig={rep_changed['orig_letter']}, "
      f"ablated={rep_changed['ablated_letter']}")
print(f"  Unchanged: idx={rep_unchanged['idx']}, orig={rep_unchanged['orig_letter']}, "
      f"ablated={rep_unchanged['ablated_letter']}")

#%% --- 6. Run patching across all pairs ---
print(f"\nRunning activation patching for {len(pairs)} pairs "
      f"(2 variants x 2 directions each)...")

# Collect per-pair matrices: all_matrices[variant][direction] = list of (num_layers, NUM_TOKENS)
all_matrices = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}
# Un-normalized patched_diff per (pair, layer, token) — lets us reconstruct
# any normalized variant post-hoc (e.g. with a different denom guard).
all_patched_diff_matrices = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}
# Per-pair denom diagnostics, aligned with all_matrices.
all_source_diffs = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}
all_target_diffs = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}
all_denoms = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}
n_dropped_denom = 0
denom_drop_log = []  # list of (pair_idx, hint, failing variant/direction, denom)

for pair_idx, (cs, us) in enumerate(pairs):
    hint = cs["hint"]
    print(f"\n--- Pair {pair_idx+1}/{len(pairs)} (hint={hint}) ---")

    pair_suffix_offsets = compute_suffix_offsets(cs)

    pair_results = {v: {} for v in VARIANTS}
    pair_failed = None
    for variant in VARIANTS:
        for direction, source, target in [
            ("changed_into_unchanged", cs, us),
            ("unchanged_into_changed", us, cs),
        ]:
            result = run_patching_for_variant(
                source, target, variant, pair_suffix_offsets,
                desc=f"P{pair_idx+1}/{len(pairs)} {variant}/{direction[:7]}",
            )
            pair_results[variant][direction] = result
            if not result["is_valid"] and pair_failed is None:
                pair_failed = (variant, direction,
                               result["source_diff"] - result["target_diff"])

    if pair_failed is not None:
        v_f, d_f, denom_f = pair_failed
        print(f"  -> DROP pair (denom guard {DENOM_THRESHOLD:g}): "
              f"{v_f}/{d_f} denom={denom_f:.4f}")
        n_dropped_denom += 1
        denom_drop_log.append({
            "pair_idx": pair_idx, "hint": hint,
            "failed_variant": v_f, "failed_direction": d_f,
            "denom": float(denom_f),
        })
        continue

    for variant in VARIANTS:
        for direction in DIRECTIONS:
            r = pair_results[variant][direction]
            all_matrices[variant][direction].append(r["matrix"])
            all_patched_diff_matrices[variant][direction].append(
                r["patched_diff_matrix"])
            all_source_diffs[variant][direction].append(float(r["source_diff"]))
            all_target_diffs[variant][direction].append(float(r["target_diff"]))
            all_denoms[variant][direction].append(float(r["denom"]))

print(f"\nCollected {len(all_matrices['orig_match']['changed_into_unchanged'])} "
      f"valid pairs (dropped {n_dropped_denom} via denom guard "
      f"{DENOM_THRESHOLD:g}).")

#%% --- 7. Per-cell correlation across pairs ---
print(f"\n{'='*60}")
print("Per-(layer, token) correlation across pairs: orig_match vs explanation")

# Get token labels from pair 0
rep_text = prepare_prompt(pairs[0][0], "orig_match")
rep_ids = tokenizer(rep_text, add_special_tokens=False)["input_ids"]
rep_suffix = compute_suffix_offsets(pairs[0][0])["orig_match"]
if rep_suffix > 0:
    region_token_ids = rep_ids[-(NUM_TOKENS + rep_suffix):-rep_suffix]
else:
    region_token_ids = rep_ids[-NUM_TOKENS:]
token_labels = [tokenizer.decode([t]) for t in region_token_ids]

correlation_results = {}
for direction in DIRECTIONS:
    # Stack into (n_pairs, num_layers, NUM_TOKENS)
    om_stack = np.array(all_matrices["orig_match"][direction])
    ex_stack = np.array(all_matrices["explanation"][direction])
    n_pairs = om_stack.shape[0]

    # Per-cell correlation: for each (layer, tok), correlate across pairs
    corr_matrix = np.zeros((num_layers, NUM_TOKENS))
    pval_matrix = np.zeros((num_layers, NUM_TOKENS))
    for layer_idx in range(num_layers):
        for tok_pos in range(NUM_TOKENS):
            om_vals = om_stack[:, layer_idx, tok_pos]
            ex_vals = ex_stack[:, layer_idx, tok_pos]
            if np.std(om_vals) < 1e-8 or np.std(ex_vals) < 1e-8:
                corr_matrix[layer_idx, tok_pos] = 0.0
                pval_matrix[layer_idx, tok_pos] = 1.0
            else:
                r, p = pearsonr(om_vals, ex_vals)
                corr_matrix[layer_idx, tok_pos] = r
                pval_matrix[layer_idx, tok_pos] = p

    # Also compute mean heatmaps for reference
    om_mean = om_stack.mean(axis=0)
    ex_mean = ex_stack.mean(axis=0)

    # Overall correlation (flatten mean matrices)
    om_flat = om_mean.flatten()
    ex_flat = ex_mean.flatten()
    if np.std(om_flat) > 1e-8 and np.std(ex_flat) > 1e-8:
        overall_r, overall_p = pearsonr(om_flat, ex_flat)
    else:
        overall_r, overall_p = 0.0, 1.0

    # Summary stats
    sig_cells = pval_matrix < 0.05
    pos_sig = sig_cells & (corr_matrix > 0)
    neg_sig = sig_cells & (corr_matrix < 0)

    correlation_results[direction] = {
        "corr_matrix": corr_matrix,
        "pval_matrix": pval_matrix,
        "om_mean": om_mean,
        "ex_mean": ex_mean,
        "overall_mean_r": float(overall_r),
        "overall_mean_p": float(overall_p),
        "n_pairs": n_pairs,
        "n_sig_positive": int(pos_sig.sum()),
        "n_sig_negative": int(neg_sig.sum()),
        "n_sig_total": int(sig_cells.sum()),
        "n_cells": int(num_layers * NUM_TOKENS),
        "mean_corr": float(corr_matrix.mean()),
        "median_corr": float(np.median(corr_matrix)),
    }

    print(f"\n  {direction} (n={n_pairs} pairs):")
    print(f"    Mean per-cell r={corr_matrix.mean():.4f}, median={np.median(corr_matrix):.4f}")
    print(f"    Significant cells (p<0.05): {sig_cells.sum()}/{num_layers * NUM_TOKENS} "
          f"({sig_cells.mean()*100:.1f}%)")
    print(f"      Positive: {pos_sig.sum()}, Negative: {neg_sig.sum()}")
    print(f"    Overall mean-matrix correlation: r={overall_r:.4f} (p={overall_p:.2e})")

#%% --- 8. Plotting ---

VARIANT_TITLES = {
    "orig_match": "Orig Match (with hint) — answer letter",
    "explanation": "Explanation — logit(most) vs logit(output)",
}

METRIC_LABELS = {
    "orig_match": "logit(src_orig) - logit(tgt_orig)",
    "explanation": "logit('most') - logit('output')",
}

DIRECTION_LABELS = {
    "changed_into_unchanged": "CHANGED→UNCHANGED",
    "unchanged_into_changed": "UNCHANGED→CHANGED",
}

xtick_labels = [
    repr(t).strip("'").replace("\\n", r"\n") for t in token_labels
]


def plot_correlation_heatmap(corr_matrix, pval_matrix, direction, n_pairs, save_path):
    """Plot the per-(layer, token) correlation matrix across pairs."""
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    vmax = max(abs(corr_matrix.min()), abs(corr_matrix.max()), 0.3)
    im = ax.imshow(
        corr_matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest",
        origin="upper", vmin=-vmax, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="Pearson r (across pairs)")

    ax.set_xlabel("Token Position (hint only)")
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6, fontfamily="monospace")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(0, num_layers, 4))
    ax.set_yticklabels([str(i) for i in range(0, num_layers, 4)])

    n_sig = (pval_matrix < 0.05).sum()
    n_total = num_layers * NUM_TOKENS
    ax.set_title(f"Per-cell correlation (orig_match vs explanation) | "
                 f"{DIRECTION_LABELS[direction]}\n"
                 f"n={n_pairs} pairs, {n_sig}/{n_total} cells p<0.05, "
                 f"mean r={corr_matrix.mean():.3f}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")


def plot_mean_heatmaps_side_by_side(om_mean, ex_mean, direction, n_pairs, save_path):
    """Plot mean orig_match and explanation heatmaps side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 8), constrained_layout=True)

    vmax = max(
        abs(om_mean.min()), abs(om_mean.max()),
        abs(ex_mean.min()), abs(ex_mean.max()),
    )
    if vmax < 1e-6:
        vmax = 1.0

    for ax, matrix, variant in [(ax1, om_mean, "orig_match"), (ax2, ex_mean, "explanation")]:
        im = ax.imshow(
            matrix, aspect="auto", cmap="RdBu", interpolation="nearest",
            origin="upper", vmin=-vmax, vmax=vmax,
        )
        ax.set_xlabel("Token Position (hint only)")
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6, fontfamily="monospace")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(0, num_layers, 4))
        ax.set_yticklabels([str(i) for i in range(0, num_layers, 4)])
        ax.set_title(f"{VARIANT_TITLES[variant]}\n{METRIC_LABELS[variant]}")

    fig.colorbar(im, ax=[ax1, ax2], label="Mean Normalized Logit Diff (0=no effect, 1=full shift)",
                 shrink=0.8)
    fig.suptitle(f"Mean over {n_pairs} pairs | {DIRECTION_LABELS[direction]}",
                 fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved mean heatmaps to {save_path}")


def plot_scatter_means(om_mean, ex_mean, direction, overall_r, save_path):
    """Scatter plot of mean orig_match vs mean explanation per (layer, pos)."""
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    all_x = om_mean.flatten()
    all_y = ex_mean.flatten()
    layer_colors = np.repeat(np.arange(num_layers), om_mean.shape[1])

    sc = ax.scatter(all_x, all_y, c=layer_colors, cmap="viridis",
                    s=10, alpha=0.5, vmin=0, vmax=num_layers - 1)
    cbar = fig.colorbar(sc, ax=ax, label="Layer", shrink=0.8)
    cbar.set_ticks(range(0, num_layers, 4))

    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
              abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=1)
    ax.axhline(0, color="gray", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("orig_match mean normalized logit diff")
    ax.set_ylabel("explanation mean normalized logit diff")
    ax.set_title(f"Mean across pairs | {DIRECTION_LABELS[direction]}\n"
                 f"Pearson r={overall_r:.4f}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter to {save_path}")


# Generate all plots
for direction in DIRECTIONS:
    res = correlation_results[direction]

    plot_correlation_heatmap(
        res["corr_matrix"], res["pval_matrix"], direction, res["n_pairs"],
        str(PLOT_OUTPUT_DIR / f"corr_heatmap_{direction}.png"),
    )
    plot_mean_heatmaps_side_by_side(
        res["om_mean"], res["ex_mean"], direction, res["n_pairs"],
        str(PLOT_OUTPUT_DIR / f"mean_heatmaps_{direction}.png"),
    )
    plot_scatter_means(
        res["om_mean"], res["ex_mean"], direction, res["overall_mean_r"],
        str(PLOT_OUTPUT_DIR / f"scatter_means_{direction}.png"),
    )

#%% --- 9. Save results ---
# Save raw matrices per pair plus per-pair source/target diffs for downstream
# denom analysis.
save_dict = {}
for variant in VARIANTS:
    for direction in DIRECTIONS:
        # Normalized matrix used by all existing replot scripts: shape (n_pairs, num_layers, NUM_TOKENS).
        save_dict[f"{variant}_{direction}"] = np.array(all_matrices[variant][direction])
        # Un-normalized patched_diff per cell — multiply through and you can
        # rebuild the normalized matrix with a different denom guard:
        #   norm = (patched_diff - target_diff) / (source_diff - target_diff)
        save_dict[f"{variant}_{direction}_patched_diff"] = np.array(
            all_patched_diff_matrices[variant][direction])
        # Per-pair scalars: source_diff = source's logit(src) - logit(tgt) on
        # source's prompt; target_diff = same on target's prompt;
        # denom = source_diff - target_diff (pre-saved for convenience).
        save_dict[f"{variant}_{direction}_source_diff"] = np.array(
            all_source_diffs[variant][direction])
        save_dict[f"{variant}_{direction}_target_diff"] = np.array(
            all_target_diffs[variant][direction])
        save_dict[f"{variant}_{direction}_denom"] = np.array(
            all_denoms[variant][direction])
np.savez(str(PLOT_OUTPUT_DIR / "patching_results.npz"), **save_dict)
print(f"\nSaved raw matrices to {PLOT_OUTPUT_DIR / 'patching_results.npz'}")

with open(PLOT_OUTPUT_DIR / "denom_drop_log.json", "w") as f:
    json.dump({
        "denom_threshold": DENOM_THRESHOLD,
        "n_input_pairs": len(pairs),
        "n_kept": len(all_matrices["orig_match"]["changed_into_unchanged"]),
        "n_dropped": n_dropped_denom,
        "dropped": denom_drop_log,
    }, f, indent=2)
print(f"Saved denom drop log to {PLOT_OUTPUT_DIR / 'denom_drop_log.json'}")

# Save correlation matrices
corr_save = {}
for direction in DIRECTIONS:
    corr_save[f"corr_{direction}"] = correlation_results[direction]["corr_matrix"]
    corr_save[f"pval_{direction}"] = correlation_results[direction]["pval_matrix"]
    corr_save[f"om_mean_{direction}"] = correlation_results[direction]["om_mean"]
    corr_save[f"ex_mean_{direction}"] = correlation_results[direction]["ex_mean"]
np.savez(str(PLOT_OUTPUT_DIR / "correlation_matrices.npz"), **corr_save)
print(f"Saved correlation matrices to {PLOT_OUTPUT_DIR / 'correlation_matrices.npz'}")

# Save summary stats as JSON
stats_json = {}
for direction in DIRECTIONS:
    res = correlation_results[direction]
    stats_json[direction] = {k: v for k, v in res.items()
                             if not isinstance(v, np.ndarray)}
with open(str(PLOT_OUTPUT_DIR / "correlation_stats.json"), "w") as f:
    json.dump(stats_json, f, indent=2)
print(f"Saved correlation stats to {PLOT_OUTPUT_DIR / 'correlation_stats.json'}")

# Save token labels
with open(str(PLOT_OUTPUT_DIR / "token_labels.json"), "w") as f:
    json.dump(token_labels, f)

print(f"\nAll outputs saved to {PLOT_OUTPUT_DIR}")
# %%
