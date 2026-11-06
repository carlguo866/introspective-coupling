#%%
"""Activation patching at the last ~10 tokens of the question (pre-hint).

Three variants are patched at the SAME underlying tokens (the question tail,
which is shared content across all three prompts):
  - orig_match    : prompt with hint, ends at "Answer:"; reads answer-letter logit
  - change_match  : prompt without hint, ends at "Answer:"; reads answer-letter logit
  - explanation   : prompt with hint + meta + "The"; reads logit(' most') vs logit(' output')

Cross-variant per-(layer, token) correlations across pairs are computed for
all three pairings: orig_match vs explanation, change_match vs explanation,
orig_match vs change_match.
"""
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
from scipy.stats import pearsonr

# --- Config ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="pre_question",
                        help="Run identifier for output filenames")
    parser.add_argument("--job_id", type=str, default=None,
                        help="Job ID for output filenames (random if unset)")
    parser.add_argument("--max_pairs_per_hint", type=int, default=10,
                        help="Max pairs per hint letter (None=all)")
    parser.add_argument("--num_questions", type=int, default=1399,
                        help="Number of question triples to consider")
    parser.add_argument("--num_tokens", type=int, default=10,
                        help="Width of the patching window (last N question tokens)")
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="Path to causal_predictions.json (orig or self)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="HF-format checkpoint dir (or hub ID) for the predictor model")
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
NUM_TOKENS = args.num_tokens

VARIANTS = ["orig_match", "change_match", "explanation"]
DIRECTIONS = ["changed_into_unchanged", "unchanged_into_changed"]
PAIRINGS = [
    ("orig_match", "explanation"),
    ("change_match", "explanation"),
    ("orig_match", "change_match"),
]

#%% --- 1. Load data and group by hint ---
print("Loading predictions...")
all_entries = []
with open(PREDICTIONS_PATH) as f:
    for line in f:
        all_entries.append(json.loads(line))


def get_sample_info(triple_idx):
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
        "prompts": {
            "orig_match": triple["orig_match"]["prompt_without_answer"],
            "change_match": triple["change_match"]["prompt_without_answer"],
            "generative_explanation": triple["generative_explanation"]["prompt_without_answer"],
        },
    }


by_hint = defaultdict(lambda: {"changed": [], "unchanged": []})
for idx in range(NUM_QUESTIONS):
    info = get_sample_info(idx)
    if info is None or info["hint"] is None:
        continue
    if not info["gen_correct"]:
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

pairs = []
for hint in sorted(by_hint):
    changed_list = by_hint[hint]["changed"]
    unchanged_list = by_hint[hint]["unchanged"]
    # Pair each changed prompt with the next unused unchanged prompt whose cue-included answer differs,
    # so the two prompts give different answer letters and the logit-difference metric is defined.
    cap = MAX_PAIRS_PER_HINT if MAX_PAIRS_PER_HINT is not None and MAX_PAIRS_PER_HINT > 0 else None
    used = set()
    n_hint = 0
    for c in changed_list:
        if cap is not None and n_hint >= cap:
            break
        for j, u in enumerate(unchanged_list):
            if j not in used and u["orig_letter"] != c["orig_letter"]:
                used.add(j)
                pairs.append((c, u))
                n_hint += 1
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

answer_token_ids = {}
for letter in "ABCD":
    toks = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
    answer_token_ids[letter] = toks[0]
    print(f"  Token for ' {letter}': {toks[0]} -> {repr(tokenizer.decode([toks[0]]))}")

output_token_id = tokenizer(" output", add_special_tokens=False)["input_ids"][0]
most_token_id = tokenizer(" most", add_special_tokens=False)["input_ids"][0]
print(f"  Token for ' output': {output_token_id} -> {repr(tokenizer.decode([output_token_id]))}")
print(f"  Token for ' most': {most_token_id} -> {repr(tokenizer.decode([most_token_id]))}")

IM_END = "<|im_end|>"

#%% --- 3. Prompt preparation and region identification ---


def prepare_prompt(sample, variant):
    """Render the prompt text for the given variant."""
    if variant == "explanation":
        messages = [dict(m) for m in sample["prompts"]["generative_explanation"]]
        # Put "The" inside the (empty) assistant turn and drop the closing tag, so the
        # prompt continues the reply rather than starting after <|im_end|>.
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
    else:  # orig_match or change_match: strip trailing <|im_end|> so prompt ends at "Answer:"
        messages = sample["prompts"][variant]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text = text.rstrip()
        if text.endswith(IM_END):
            text = text[:-len(IM_END)]
    return text


def compute_question_offsets(sample):
    """Compute, per variant, the number of trailing tokens AFTER the end of the
    shared question content. The patching window of NUM_TOKENS ends right at
    the last question token, i.e. region = ids[-(NUM_TOKENS+offset) : -offset].

    The shared question prefix is the longest common prefix between
    orig_match and change_match (which diverges where orig_match injects the
    hint and change_match jumps straight to the Answer: template).
    """
    om_text = prepare_prompt(sample, "orig_match")
    cm_text = prepare_prompt(sample, "change_match")
    ge_text = prepare_prompt(sample, "explanation")
    om_ids = tokenizer(om_text, add_special_tokens=False)["input_ids"]
    cm_ids = tokenizer(cm_text, add_special_tokens=False)["input_ids"]
    ge_ids = tokenizer(ge_text, add_special_tokens=False)["input_ids"]

    # Longest common prefix between orig_match and change_match.
    Q = 0
    for a, b in zip(om_ids, cm_ids):
        if a == b:
            Q += 1
        else:
            break
    return {
        "orig_match": len(om_ids) - Q,
        "change_match": len(cm_ids) - Q,
        "explanation": len(ge_ids) - Q,
        "Q": Q,
        # The explanation prompt must contain the same Q question tokens, or its window is misaligned.
        "explanation_aligned": ge_ids[:Q] == om_ids[:Q],
        "token_counts": {
            "orig_match": len(om_ids),
            "change_match": len(cm_ids),
            "explanation": len(ge_ids),
        },
    }


def get_metric_token_ids(variant, source_sample, target_sample):
    if variant == "explanation":
        if source_sample["changed"]:
            return {"source": most_token_id, "target": output_token_id}
        else:
            return {"source": output_token_id, "target": most_token_id}
    else:  # orig_match or change_match
        return {
            "source": answer_token_ids[source_sample["orig_letter"]],
            "target": answer_token_ids[target_sample["orig_letter"]],
        }


def get_logit_diffs(logits, metric_tids):
    last_logits = logits[0, -1, :]
    return {name: last_logits[tid].item() for name, tid in metric_tids.items()}


# Filter out pairs that don't have enough question tokens (very unlikely on MMLU).
filtered_pairs = []
for cs, us in pairs:
    cs_off = compute_question_offsets(cs)
    us_off = compute_question_offsets(us)
    if cs_off["Q"] < NUM_TOKENS or us_off["Q"] < NUM_TOKENS:
        print(f"  Skipping pair (idx={cs['idx']},{us['idx']}): Q < {NUM_TOKENS}")
        continue
    if not (cs_off["explanation_aligned"] and us_off["explanation_aligned"]):
        print(f"  Skipping pair (idx={cs['idx']},{us['idx']}): explanation prompt does not share the question prefix")
        continue
    filtered_pairs.append((cs, us))
pairs = filtered_pairs
print(f"After Q>={NUM_TOKENS} filter: {len(pairs)} pairs")

#%% --- 4. Patching ---


def run_patching_for_variant(source_sample, target_sample, variant, src_offsets, tgt_offsets,
                             desc=None):
    """Run activation patching for one (source, target) pair on one variant.

    Patches at NUM_TOKENS positions ending at the last question token, indexed
    from the right of each (unpadded) prompt.
    """
    device = next(model.parameters()).device

    text_src = prepare_prompt(source_sample, variant)
    text_tgt = prepare_prompt(target_sample, variant)

    src_suffix = src_offsets[variant]
    tgt_suffix = tgt_offsets[variant]

    # Run each prompt unpadded: the traces pass no attention mask, so padding would be attended to.
    ids_src = tokenizer(text_src, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids_tgt = tokenizer(text_tgt, return_tensors="pt", add_special_tokens=False)["input_ids"]

    # Per-side region: right-aligned with each side's own suffix offset.
    src_region_end = -src_suffix if src_suffix > 0 else None
    src_region_start = -(NUM_TOKENS + src_suffix)
    tgt_region_end = -tgt_suffix if tgt_suffix > 0 else None
    tgt_region_start = -(NUM_TOKENS + tgt_suffix)

    metric_tids = get_metric_token_ids(variant, source_sample, target_sample)

    # Step 1: extract source hidden states at its patching region
    clean_hidden_states = {}
    with torch.no_grad():
        with nnsight_model.trace() as tracer:
            with tracer.invoke(ids_src.to(device)):
                for layer_idx in range(num_layers):
                    clean_hidden_states[layer_idx] = (
                        nnsight_model.model.layers[layer_idx].output[
                            :, src_region_start:src_region_end, :
                        ].save()
                    )
                source_logits_raw = nnsight_model.output.logits.save()

    source_logits = get_logit_diffs(source_logits_raw, metric_tids)
    source_diff = source_logits["source"] - source_logits["target"]
    del source_logits_raw
    for k in clean_hidden_states:
        clean_hidden_states[k] = clean_hidden_states[k].cpu()
    torch.cuda.empty_cache()

    # Step 2: target baseline
    with torch.no_grad():
        with nnsight_model.trace() as tracer:
            with tracer.invoke(ids_tgt.to(device)):
                target_logits_raw = nnsight_model.output.logits.save()

    target_logits = get_logit_diffs(target_logits_raw, metric_tids)
    target_diff = target_logits["source"] - target_logits["target"]
    del target_logits_raw
    torch.cuda.empty_cache()

    denom = source_diff - target_diff

    # Step 3: per-(layer, token) patching
    patching_matrix = np.zeros((num_layers, NUM_TOKENS))
    pbar = tqdm(total=num_layers * NUM_TOKENS, desc=desc or "Patching",
                leave=False, unit="patch")

    for layer_idx in range(num_layers):
        layer_hidden = clean_hidden_states[layer_idx].to(device)

        for tok_pos in range(NUM_TOKENS):
            with torch.no_grad():
                with nnsight_model.trace() as tracer:
                    with tracer.invoke(ids_tgt.to(device)):
                        abs_offset = tgt_region_start + tok_pos
                        nnsight_model.model.layers[layer_idx].output[
                            :, abs_offset, :
                        ] = layer_hidden[:, tok_pos, :]
                        patched_logits_raw = nnsight_model.output.logits.save()

            patched = get_logit_diffs(patched_logits_raw, metric_tids)
            del patched_logits_raw

            patched_diff = patched["source"] - patched["target"]
            if abs(denom) > 1e-6:
                patching_matrix[layer_idx, tok_pos] = (
                    (patched_diff - target_diff) / denom
                )
            else:
                patching_matrix[layer_idx, tok_pos] = 0.0

            pbar.update(1)

        del layer_hidden
        torch.cuda.empty_cache()

    pbar.close()

    return {
        "matrix": patching_matrix,
        "source_diff": source_diff,
        "target_diff": target_diff,
    }


#%% --- 5. Sanity check on first pair ---
print("\n--- Sanity check (pair 0) ---")
rep_changed, rep_unchanged = pairs[0]
rep_offsets = compute_question_offsets(rep_changed)
print(f"  Q (shared question prefix): {rep_offsets['Q']}")
print(f"  Suffix offsets: {{ {', '.join(f'{k}={rep_offsets[k]}' for k in VARIANTS)} }}")
print(f"  Token counts: {rep_offsets['token_counts']}")

for variant in VARIANTS:
    text = prepare_prompt(rep_changed, variant)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    suffix = rep_offsets[variant]
    region_end = -suffix if suffix > 0 else None
    region_start = -(NUM_TOKENS + suffix)
    region_tokens = ids[region_start:region_end] if region_end else ids[region_start:]
    decoded = tokenizer.decode(region_tokens)
    print(f"\n  {variant} patching region ({len(region_tokens)} tokens): {repr(decoded)}")

print(f"\n  Changed: idx={rep_changed['idx']}, orig={rep_changed['orig_letter']}, "
      f"ablated={rep_changed['ablated_letter']}")
print(f"  Unchanged: idx={rep_unchanged['idx']}, orig={rep_unchanged['orig_letter']}, "
      f"ablated={rep_unchanged['ablated_letter']}")

#%% --- 6. Run patching across all pairs ---
print(f"\nRunning activation patching for {len(pairs)} pairs "
      f"(3 variants x 2 directions each)...")

all_matrices = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}
all_source_diffs = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}
all_target_diffs = {v: {d: [] for d in DIRECTIONS} for v in VARIANTS}

for pair_idx, (cs, us) in enumerate(pairs):
    hint = cs["hint"]
    print(f"\n--- Pair {pair_idx+1}/{len(pairs)} (hint={hint}) ---")

    cs_offsets = compute_question_offsets(cs)
    us_offsets = compute_question_offsets(us)

    for variant in VARIANTS:
        for direction, source, target, src_off, tgt_off in [
            ("changed_into_unchanged", cs, us, cs_offsets, us_offsets),
            ("unchanged_into_changed", us, cs, us_offsets, cs_offsets),
        ]:
            result = run_patching_for_variant(
                source, target, variant, src_off, tgt_off,
                desc=f"P{pair_idx+1}/{len(pairs)} {variant}/{direction[:7]}",
            )
            all_matrices[variant][direction].append(result["matrix"])
            all_source_diffs[variant][direction].append(result["source_diff"])
            all_target_diffs[variant][direction].append(result["target_diff"])

print(f"\nCollected {len(pairs)} pairs.")

#%% --- 7. Per-cell correlations across pairs ---
print(f"\n{'='*60}")
print("Per-(layer, token) correlation across pairs (3 pairings)")

# Token labels: take the last NUM_TOKENS question tokens from pair 0's orig_match,
# which by construction match the question tail of all variants.
rep_text = prepare_prompt(pairs[0][0], "orig_match")
rep_ids = tokenizer(rep_text, add_special_tokens=False)["input_ids"]
rep_suffix = compute_question_offsets(pairs[0][0])["orig_match"]
region_token_ids = rep_ids[-(NUM_TOKENS + rep_suffix):-rep_suffix] if rep_suffix > 0 else rep_ids[-NUM_TOKENS:]
token_labels = [tokenizer.decode([t]) for t in region_token_ids]

# Per-variant mean matrices (used by triptych and by scatter)
mean_matrices = {v: {} for v in VARIANTS}
for variant in VARIANTS:
    for direction in DIRECTIONS:
        stack = np.array(all_matrices[variant][direction])
        mean_matrices[variant][direction] = stack.mean(axis=0)

correlation_results = {p: {} for p in PAIRINGS}
for variant_a, variant_b in PAIRINGS:
    for direction in DIRECTIONS:
        a_stack = np.array(all_matrices[variant_a][direction])
        b_stack = np.array(all_matrices[variant_b][direction])
        n_pairs = a_stack.shape[0]

        corr_matrix = np.zeros((num_layers, NUM_TOKENS))
        pval_matrix = np.zeros((num_layers, NUM_TOKENS))
        for layer_idx in range(num_layers):
            for tok_pos in range(NUM_TOKENS):
                a_vals = a_stack[:, layer_idx, tok_pos]
                b_vals = b_stack[:, layer_idx, tok_pos]
                if np.std(a_vals) < 1e-8 or np.std(b_vals) < 1e-8:
                    corr_matrix[layer_idx, tok_pos] = 0.0
                    pval_matrix[layer_idx, tok_pos] = 1.0
                else:
                    r, p = pearsonr(a_vals, b_vals)
                    corr_matrix[layer_idx, tok_pos] = r
                    pval_matrix[layer_idx, tok_pos] = p

        a_mean = mean_matrices[variant_a][direction]
        b_mean = mean_matrices[variant_b][direction]
        a_flat, b_flat = a_mean.flatten(), b_mean.flatten()
        if np.std(a_flat) > 1e-8 and np.std(b_flat) > 1e-8:
            overall_r, overall_p = pearsonr(a_flat, b_flat)
        else:
            overall_r, overall_p = 0.0, 1.0

        sig_cells = pval_matrix < 0.05
        pos_sig = sig_cells & (corr_matrix > 0)
        neg_sig = sig_cells & (corr_matrix < 0)

        correlation_results[(variant_a, variant_b)][direction] = {
            "corr_matrix": corr_matrix,
            "pval_matrix": pval_matrix,
            "a_mean": a_mean,
            "b_mean": b_mean,
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

        tag = f"{variant_a}_vs_{variant_b}"
        print(f"\n  {tag} | {direction} (n={n_pairs}):")
        print(f"    Mean per-cell r={corr_matrix.mean():.4f}, median={np.median(corr_matrix):.4f}")
        print(f"    Significant cells (p<0.05): {sig_cells.sum()}/{num_layers * NUM_TOKENS} "
              f"({sig_cells.mean()*100:.1f}%)")
        print(f"      Positive: {pos_sig.sum()}, Negative: {neg_sig.sum()}")
        print(f"    Overall mean-matrix correlation: r={overall_r:.4f} (p={overall_p:.2e})")

#%% --- 8. Plotting ---

VARIANT_TITLES = {
    "orig_match": "Orig Match (with hint) — answer letter",
    "change_match": "Change Match (no hint) — answer letter",
    "explanation": "Explanation — logit(' most') vs logit(' output')",
}

METRIC_LABELS = {
    "orig_match": "logit(src_orig) - logit(tgt_orig)",
    "change_match": "logit(src_orig) - logit(tgt_orig)",
    "explanation": "logit(' most') - logit(' output')",
}

DIRECTION_LABELS = {
    "changed_into_unchanged": "CHANGED→UNCHANGED",
    "unchanged_into_changed": "UNCHANGED→CHANGED",
}

xtick_labels = [
    repr(t).strip("'").replace("\\n", r"\n") for t in token_labels
]


def plot_correlation_heatmap(corr_matrix, pval_matrix, pairing, direction, n_pairs, save_path):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    vmax = max(abs(corr_matrix.min()), abs(corr_matrix.max()), 0.3)
    im = ax.imshow(
        corr_matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest",
        origin="upper", vmin=-vmax, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label="Pearson r (across pairs)")

    ax.set_xlabel("Token Position (last 10 question tokens)")
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(0, num_layers, 4))
    ax.set_yticklabels([str(i) for i in range(0, num_layers, 4)])

    n_sig = (pval_matrix < 0.05).sum()
    n_total = num_layers * NUM_TOKENS
    a, b = pairing
    ax.set_title(f"Per-cell correlation ({a} vs {b}) | "
                 f"{DIRECTION_LABELS[direction]}\n"
                 f"n={n_pairs} pairs, {n_sig}/{n_total} cells p<0.05, "
                 f"mean r={corr_matrix.mean():.3f}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to {save_path}")


def plot_mean_heatmaps_triptych(direction, n_pairs, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(30, 8), constrained_layout=True)

    vmax = 0.0
    for v in VARIANTS:
        m = mean_matrices[v][direction]
        vmax = max(vmax, abs(m.min()), abs(m.max()))
    if vmax < 1e-6:
        vmax = 1.0

    last_im = None
    for ax, variant in zip(axes, VARIANTS):
        matrix = mean_matrices[variant][direction]
        im = ax.imshow(
            matrix, aspect="auto", cmap="RdBu", interpolation="nearest",
            origin="upper", vmin=-vmax, vmax=vmax,
        )
        last_im = im
        ax.set_xlabel("Token Position (last 10 question tokens)")
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8, fontfamily="monospace")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(0, num_layers, 4))
        ax.set_yticklabels([str(i) for i in range(0, num_layers, 4)])
        ax.set_title(f"{VARIANT_TITLES[variant]}\n{METRIC_LABELS[variant]}")

    fig.colorbar(last_im, ax=list(axes),
                 label="Mean Normalized Logit Diff (0=no effect, 1=full shift)",
                 shrink=0.8)
    fig.suptitle(f"Mean over {n_pairs} pairs | {DIRECTION_LABELS[direction]}",
                 fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved mean heatmap triptych to {save_path}")


def plot_scatter_means(a_mean, b_mean, pairing, direction, overall_r, save_path):
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    all_x = a_mean.flatten()
    all_y = b_mean.flatten()
    layer_colors = np.repeat(np.arange(num_layers), a_mean.shape[1])

    sc = ax.scatter(all_x, all_y, c=layer_colors, cmap="viridis",
                    s=10, alpha=0.5, vmin=0, vmax=num_layers - 1)
    cbar = fig.colorbar(sc, ax=ax, label="Layer", shrink=0.8)
    cbar.set_ticks(range(0, num_layers, 4))

    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
              abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=1)
    ax.axhline(0, color="gray", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", alpha=0.3, linewidth=0.5)

    a, b = pairing
    ax.set_xlabel(f"{a} mean normalized logit diff")
    ax.set_ylabel(f"{b} mean normalized logit diff")
    ax.set_title(f"Mean across pairs | {DIRECTION_LABELS[direction]}\n"
                 f"Pearson r={overall_r:.4f}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter to {save_path}")


for direction in DIRECTIONS:
    n_pairs_dir = len(all_matrices[VARIANTS[0]][direction])
    plot_mean_heatmaps_triptych(
        direction, n_pairs_dir,
        str(PLOT_OUTPUT_DIR / f"mean_heatmaps_{direction}.png"),
    )
    for pairing in PAIRINGS:
        res = correlation_results[pairing][direction]
        a, b = pairing
        tag = f"{a}_vs_{b}"
        plot_correlation_heatmap(
            res["corr_matrix"], res["pval_matrix"], pairing, direction, res["n_pairs"],
            str(PLOT_OUTPUT_DIR / f"corr_heatmap_{tag}_{direction}.png"),
        )
        plot_scatter_means(
            res["a_mean"], res["b_mean"], pairing, direction, res["overall_mean_r"],
            str(PLOT_OUTPUT_DIR / f"scatter_{tag}_{direction}.png"),
        )

#%% --- 9. Save results ---
save_dict = {}
for variant in VARIANTS:
    for direction in DIRECTIONS:
        save_dict[f"{variant}_{direction}"] = np.array(all_matrices[variant][direction])
        save_dict[f"source_diff_{variant}_{direction}"] = np.array(all_source_diffs[variant][direction])
        save_dict[f"target_diff_{variant}_{direction}"] = np.array(all_target_diffs[variant][direction])
np.savez(str(PLOT_OUTPUT_DIR / "patching_results.npz"), **save_dict)
print(f"\nSaved raw matrices + source/target diffs to {PLOT_OUTPUT_DIR / 'patching_results.npz'}")

corr_save = {}
for pairing in PAIRINGS:
    a, b = pairing
    tag = f"{a}_vs_{b}"
    for direction in DIRECTIONS:
        res = correlation_results[pairing][direction]
        corr_save[f"corr_{tag}_{direction}"] = res["corr_matrix"]
        corr_save[f"pval_{tag}_{direction}"] = res["pval_matrix"]
for variant in VARIANTS:
    for direction in DIRECTIONS:
        corr_save[f"mean_{variant}_{direction}"] = mean_matrices[variant][direction]
np.savez(str(PLOT_OUTPUT_DIR / "correlation_matrices.npz"), **corr_save)
print(f"Saved correlation matrices to {PLOT_OUTPUT_DIR / 'correlation_matrices.npz'}")

stats_json = {}
for pairing in PAIRINGS:
    a, b = pairing
    tag = f"{a}_vs_{b}"
    stats_json[tag] = {}
    for direction in DIRECTIONS:
        res = correlation_results[pairing][direction]
        stats_json[tag][direction] = {k: v for k, v in res.items()
                                      if not isinstance(v, np.ndarray)}
with open(str(PLOT_OUTPUT_DIR / "correlation_stats.json"), "w") as f:
    json.dump(stats_json, f, indent=2)
print(f"Saved correlation stats to {PLOT_OUTPUT_DIR / 'correlation_stats.json'}")

with open(str(PLOT_OUTPUT_DIR / "token_labels.json"), "w") as f:
    json.dump(token_labels, f)

print(f"\nAll outputs saved to {PLOT_OUTPUT_DIR}")
# %%
