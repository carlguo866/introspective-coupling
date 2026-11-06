"""Llama-3.1-8B-Instruct explanation labels on the Hint-MMLU test set (paper Fig 6 scoring).

Takes a Qwen3-8B label set baked by scripts/make_test_data.py with the base model as target and
rewrites every generative_explanation label from Llama's answers on the same question and hint (same
`index` in hint_data/mmlu_llama_3.1_8b_instruct_ablation_prompts_ht8.jsonl), in the loader's two
explanation formats. Rows whose question is missing from the Llama file are dropped. The output can be
passed to scripts/score_explanations.py as --labels llama=<out>.

    python scripts/llama_label_set.py --qwen_labels results/orig_hint/labels/causal_data.jsonl \
        --out results/llama_labels/causal_data.jsonl
"""
import argparse
import json
import os

QWEN_TEST = "hint_data/mmlu_qwen3_8b_ablation_prompts_ht8_test.jsonl"
LLAMA = "hint_data/mmlu_llama_3.1_8b_instruct_ablation_prompts_ht8.jsonl"
UNCHANGED = "The output would remain unchanged from <<<Answer: {}>>>."
CHANGED = "The most likely output would change to <<<Answer: {}>>>."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_user = {r["hint_user_prompt"].strip(): r for r in map(json.loads, open(QWEN_TEST))}
    llama = {r["index"]: r for r in map(json.loads, open(LLAMA))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    kept = dropped = agree = 0
    with open(a.out, "w") as f:
        for r in map(json.loads, open(a.qwen_labels)):
            if r["question_type"] != "generative_explanation":
                continue
            user = r["prompt_without_answer"][1]["content"].split("\n\nIf the hint were removed")[0].strip()
            q = by_user.get(user)
            l = llama.get(q["index"]) if q else None
            if l is None or l["question"] != q["question"] or l["hint"] != q["hint"]:
                dropped += 1
                continue
            hinted, plain = l["random_hint_prediction"], l["zeroshot_prediction"]
            lab = UNCHANGED.format(plain) if hinted == plain else CHANGED.format(plain)
            agree += lab == r["expected_label"]
            r["expected_label"] = lab
            r["extra_args"] = dict(r["extra_args"], original_response=f"Answer: {hinted}",
                                   ablated_response=f"Answer: {plain}")
            f.write(json.dumps(r) + "\n")
            kept += 1
    print(f"kept {kept}, dropped {dropped}; Llama label == Qwen base label on {agree / max(kept, 1):.3f}")


if __name__ == "__main__":
    main()
