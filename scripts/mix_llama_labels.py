"""Cross-model explanation labels for paper Fig 6 (§4.3).

For a fraction alpha of Qwen3-8B ht8 rows (exact count, fixed seed), the
explanation target is taken from Llama-3.1-8B-Instruct's answers on the same
question (same `index`, same hint): we set `gen_random_hint_prediction` /
`gen_zeroshot_prediction`, which dataloaders/causal.py uses for
generative_explanation rows only. orig_match / change_match keep Qwen's answers
and the row keeps Qwen's `index`, so the Qwen teacher-logit file is reused
unchanged for KL regularization toward Qwen.

Rows whose question is missing from the Llama file (filtered as invalid there)
keep the Qwen label and are reported.

Usage:
    python scripts/mix_llama_labels.py --alpha 0.5 --split train
"""
import argparse
import json
import random

QWEN = "hint_data/mmlu_qwen3_8b_ablation_prompts_ht8{split}.jsonl"
LLAMA = "hint_data/mmlu_llama_3.1_8b_instruct_ablation_prompts_ht8.jsonl"  # full file: covers both splits
OUT = "hint_data/mmlu_mixed_qwen_llama_gen_alpha{tag}_ht8{split}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = str(a.alpha).replace(".", "p")
    sfx = f"_{a.split}"

    llama = {}
    for line in open(LLAMA):
        r = json.loads(line)
        llama[r["index"]] = r
    rows = [json.loads(l) for l in open(QWEN.format(split=sfx))]

    eligible = [i for i, r in enumerate(rows)
                if r["index"] in llama
                and llama[r["index"]]["question"] == r["question"]
                and llama[r["index"]]["hint"] == r["hint"]]
    n_mix = round(a.alpha * len(rows))
    assert n_mix <= len(eligible), (n_mix, len(eligible))
    rng = random.Random(a.seed)
    chosen = set(rng.sample(eligible, n_mix))

    agree = 0
    with open(OUT.format(tag=tag, split=sfx), "w") as f:
        for i, r in enumerate(rows):
            if i in chosen:
                l = llama[r["index"]]
                r["gen_random_hint_prediction"] = l["random_hint_prediction"]
                r["gen_zeroshot_prediction"] = l["zeroshot_prediction"]
                r["gen_label_source"] = "llama"
            else:
                r["gen_label_source"] = "qwen"
            q_label = (r["random_hint_prediction"], r["zeroshot_prediction"])
            g_label = (r.get("gen_random_hint_prediction", q_label[0]), r.get("gen_zeroshot_prediction", q_label[1]))
            agree += q_label == g_label
            f.write(json.dumps(r) + "\n")

    # Explanation agreement between the two base models on rows where both exist (paper: 53%).
    both = [r for r in rows if r["index"] in llama and llama[r["index"]]["question"] == r["question"]
            and llama[r["index"]]["hint"] == r["hint"]]
    def expl(x):
        # same string the loader builds: unchanged-from-<answer> or change-to-<cue-removed answer>
        same = x["random_hint_prediction"] == x["zeroshot_prediction"]
        return ("unchanged:" if same else "change:") + x["zeroshot_prediction"]
    base_agree = sum(expl(r) == expl(llama[r["index"]]) for r in both) / max(len(both), 1)
    print(f"split={a.split} alpha={a.alpha} rows={len(rows)} eligible={len(eligible)} mixed={n_mix} "
          f"gen label == Qwen label on {agree / len(rows):.3f}; Qwen-vs-Llama explanation agreement on shared rows {base_agree:.3f}")


if __name__ == "__main__":
    main()
