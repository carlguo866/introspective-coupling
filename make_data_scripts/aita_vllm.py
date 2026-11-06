"""Build the AITA system-prompt-ablation dataset (paper Section 3.1 and App. B.4).

Each Reddit r/AmItheAsshole post from ``OsamaBsher/AITA-Reddit-Dataset`` is
shown to Qwen3-8B twice, greedy, thinking off, with the assistant turn
prefilled with "Answer:":

  * with the cue C, a system prompt:
      "Read the following situation in a post on Reddit and determine if the
       person is the asshole."
  * without it (no system message at all).

The verdict (YTA or NTA) is read from the next tokens. A row's
``changed_pred`` is True when the two verdicts differ.

Prompts are rendered from the training config's own question templates
(``orig_match`` = with system prompt, ``change_match`` = system prompt removed)
through ``dataloaders.base_dataset.format_messages``, so the stored
``with_system_prompt_full`` / ``no_system_prompt_full`` strings are byte-equal
to what ``CausalDataset.get_labels_aita`` builds at training time. The stored
``user_prompt`` is therefore the bare post (title, blank line, body): the
config templates append the instruction line ("Output only YTA ...") for the
behavior questions and the meta question for ``generative_explanation``.

Outputs (under ``--out_dir``, default ``aita_data/``):
  aita_<model>_system_prompt_all.jsonl            every scored post that passed the filters
  aita_<model>_system_prompt_train_balanced.jsonl train, 50/50 changed vs unchanged
  aita_<model>_system_prompt_test_large.jsonl     test, 50/50 over the no-system-prompt verdict
  aita_<model>_system_prompt_test_balanced.jsonl  nested half-size subset of test_large
  aita_<model>_system_prompt_stats.json           counts, change rates, verdict distributions

Train and test come from disjoint post pools (a hash of the Reddit post id).
Posts are dropped when any of the three rendered training prompts would exceed
the length limit CausalDataset applies (see ``--length_limit``), and when
either verdict is not exactly YTA or NTA.

Usage:
    uv run python make_data_scripts/aita_vllm.py --model Qwen/Qwen3-8B
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from dataloaders.base_dataset import format_messages

SYSTEM_PROMPT = (
    "Read the following situation in a post on Reddit and determine if the person is the asshole."
)
VERDICTS = ("YTA", "NTA")
QUESTION_TYPES = ("orig_match", "change_match", "generative_explanation")
# Without the system prompt Qwen3-8B often writes the verdict in markdown bold
# (" **YTA**"); accept that form and store the bare verdict (the raw text stays
# in *_generation). The evaluator's _normalize_aita_label reads both forms.
_VERDICT_RE = re.compile(r"^\s*(?:\*\*|__)?\s*(YTA|NTA)(?![A-Za-z])")

_MODEL_STORE_NAMES = {
    "Qwen/Qwen3-8B": "qwen3_8b",
    "Qwen/Qwen3-32B": "qwen3_32b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama_3.1_8b_instruct",
}


def load_config_templates(config_path, length_limit="effective"):
    """Return ({question_type: message template}, max_prompt_length, max_sequence_length)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    causal = cfg["train"]["tasks"]["causal"]
    qts = causal["question_types"]
    missing = [qt for qt in QUESTION_TYPES if qt not in qts]
    if missing:
        raise ValueError(f"{config_path} lacks question types {missing}; use the KL config")
    templates = {}
    for qt in QUESTION_TYPES:
        prompts = qts[qt]["prompts"]
        assert len(prompts) == 1, f"{qt}: expected one prompt template, got {len(prompts)}"
        templates[qt] = prompts[0]["messages"]
    # train.py hands CausalDataset only cfg["train"]["tasks"]["causal"], which has
    # no max_prompt_length, so the dataset applies its default of 1024 even
    # though cfg["train"] declares 2048. Rows over the effective limit would be
    # skipped silently at training and eval time, so "effective" is the default.
    if length_limit == "effective":
        max_prompt = causal.get("max_prompt_length", 1024)
        max_seq = causal.get("max_sequence_length", 1024)
    else:
        max_prompt = cfg["train"].get("max_prompt_length", 1024)
        max_seq = cfg["train"].get("max_sequence_length", 1024)
    print(f"Length limit ({length_limit}): max_prompt_length={max_prompt} "
          f"max_sequence_length={max_seq}; declared in cfg['train']: "
          f"{cfg['train'].get('max_prompt_length')}/{cfg['train'].get('max_sequence_length')}")
    return templates, max_prompt, max_seq


def fill_template(template, args):
    """Mirror CausalDataset.fill_in_prompt_args (prompt part) + get_labels_aita's
    removal of empty system messages."""
    out = []
    for m in template:
        content = m.get("content", "")
        fmt = {k: v for k, v in args.items() if "{" + k + "}" in content}
        if fmt:
            content = content.format(**fmt)
        out.append({"role": m["role"], "content": content})
    return [m for m in out if not (m["role"] == "system" and not m["content"].strip())]


def with_answer(messages, answer):
    """Mirror fill_in_prompt_args: append the label to a trailing assistant turn."""
    msgs = [dict(m) for m in messages]
    if msgs[-1]["role"] == "assistant":
        msgs[-1]["content"] = msgs[-1]["content"] + answer
    else:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def explanation_label(with_v, no_v):
    """Mirror CausalDataset._format_causal_change("Answer: "+with, "Answer: "+no)."""
    a, b = "Answer: " + with_v, "Answer: " + no_v
    if a == b:
        return f"The output would remain unchanged from <<<{a}>>>."
    return f"The most likely output would change to <<<{b}>>>."


def post_text(row):
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title}\n\n{text}" if title else text


def is_test_post(post_id, test_percent):
    return int(hashlib.md5(post_id.encode()).hexdigest(), 16) % 100 < test_percent


def parse_verdict(generation):
    m = _VERDICT_RE.match(generation or "")
    return m.group(1) if m else None


def first_token_prob(logprobs_list, token_id):
    if not logprobs_list:
        return None
    lp = logprobs_list[0].get(token_id)
    if lp is None:
        return 0.0
    return math.exp(lp.logprob if hasattr(lp, "logprob") else lp)


def candidate_posts(seed, min_words):
    ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
    print(f"Loaded {len(ds)} raw posts")
    seen_ids, seen_text = set(), set()
    posts = []
    drop = Counter()
    for row in ds:
        pid = row.get("id")
        text = (row.get("text") or "").strip()
        if not pid or pid in seen_ids:
            drop["duplicate_id"] += 1
            continue
        if text.lower() in ("", "[removed]", "[deleted]", "removed", "deleted"):
            drop["removed_or_empty"] += 1
            continue
        if len(text.split()) < min_words:
            drop["too_short"] += 1
            continue
        post = post_text(row)
        key = re.sub(r"\s+", " ", post.lower())
        if key in seen_text:
            drop["duplicate_text"] += 1
            continue
        seen_ids.add(pid)
        seen_text.add(key)
        posts.append({
            "id": pid,
            "title": (row.get("title") or "").strip(),
            "post": post,
            "reddit_verdict": row.get("verdict"),
            "reddit_score": row.get("score"),
        })
    random.Random(seed).shuffle(posts)
    print(f"{len(posts)} candidate posts after filtering; dropped {dict(drop)}")
    return posts, dict(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--config", default="configs/qwen_qwen_aita_kl_regularized.yaml",
                    help="Config whose question templates define the stored prompts.")
    ap.add_argument("--out_dir", default="aita_data")
    ap.add_argument("--n_train", type=int, default=12000,
                    help="Train rows, half changed_pred=True and half False.")
    ap.add_argument("--n_test", type=int, default=2000,
                    help="test_large rows, half no-system-prompt YTA and half NTA.")
    ap.add_argument("--n_test_small", type=int, default=1000,
                    help="test_balanced rows (nested subset of test_large, same balancing).")
    ap.add_argument("--test_percent", type=int, default=15,
                    help="Percent of posts (by id hash) reserved for the test pool.")
    ap.add_argument("--chunk_size", type=int, default=20000)
    ap.add_argument("--max_posts", type=int, default=200000,
                    help="Stop scoring after this many posts even if quotas are unmet.")
    ap.add_argument("--length_limit", choices=["effective", "declared"], default="effective",
                    help="effective = the limit CausalDataset applies as train.py builds it "
                         "(1024 for the AITA configs); declared = cfg['train'] values (2048).")
    ap.add_argument("--min_words", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: score only this many posts.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported late so --help works without a GPU

    store = _MODEL_STORE_NAMES.get(args.model, args.model.split("/")[-1].lower())
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"aita_{store}_system_prompt")

    templates, max_prompt, max_seq = load_config_templates(args.config, args.length_limit)
    print(f"Templates from {args.config}")
    for qt, t in templates.items():
        print(f"  {qt}: {json.dumps(t)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    y_id = tok.encode(" Y", add_special_tokens=False)
    n_id = tok.encode(" N", add_special_tokens=False)
    assert len(y_id) == 1 and len(n_id) == 1
    y_id, n_id = y_id[0], n_id[0]

    def render(qt, post, answer=None):
        sys_prompt = "" if qt == "change_match" else SYSTEM_PROMPT
        msgs = fill_template(templates[qt], {"system_prompt": sys_prompt, "user_prompt": post})
        if answer is None:
            return format_messages(tok, msgs, remove_final_assistant_eot=True,
                                   add_final_assistant_header=False)
        return tok.apply_chat_template(with_answer(msgs, answer), tokenize=False)

    def ntok(texts):
        return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]

    # The stored behavior prompts must equal what the hint pipeline would feed
    # the model (apply_chat_template with enable_thinking=False and
    # continue_final_message=True): check on a dummy post.
    for qt, sys_prompt in (("orig_match", SYSTEM_PROMPT), ("change_match", "")):
        msgs = fill_template(templates[qt], {"system_prompt": sys_prompt, "user_prompt": "POST"})
        a = render(qt, "POST")
        b = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False,
                                    continue_final_message=True)
        assert a == b, (a, b)
        assert a.endswith("</think>\n\nAnswer:"), a[-60:]

    posts, drop_counts = candidate_posts(args.seed, args.min_words)
    if args.limit:
        posts = posts[: args.limit]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, seed=args.seed)
    sp = SamplingParams(temperature=0.0, max_tokens=4, logprobs=20)

    half_train = args.n_train // 2
    half_test = args.n_test // 2
    scored = []  # every post that passed all filters, in scoring order
    counts = Counter()
    unparsed = Counter()  # raw generations that were not YTA/NTA
    pos = 0
    while pos < len(posts) and pos < args.max_posts:
        chunk = posts[pos: pos + args.chunk_size]
        pos += len(chunk)
        # Length pre-filter on the longest behavior prompt (with system prompt).
        with_full = [render("orig_match", p["post"]) for p in chunk]
        lens = ntok(with_full)
        keep = [i for i, n in enumerate(lens) if n + 16 <= max_prompt]
        counts["too_long_prefilter"] += len(chunk) - len(keep)
        chunk = [chunk[i] for i in keep]
        with_full = [with_full[i] for i in keep]
        no_full = [render("change_match", p["post"]) for p in chunk]

        outs = llm.generate(with_full + no_full, sp)
        w_outs, n_outs = outs[: len(chunk)], outs[len(chunk):]

        new_rows = []
        for p, wf, nf, wo, no in zip(chunk, with_full, no_full, w_outs, n_outs):
            w_gen, n_gen = wo.outputs[0].text, no.outputs[0].text
            w_v, n_v = parse_verdict(w_gen), parse_verdict(n_gen)
            if w_v is None or n_v is None:
                counts["unparsed_verdict"] += 1
                unparsed[("with: " + w_gen) if w_v is None else ("no: " + n_gen)] += 1
                continue
            new_rows.append({
                "id": p["id"],
                "title": p["title"],
                "reddit_verdict": p["reddit_verdict"],
                "reddit_score": p["reddit_score"],
                "pool": "test" if is_test_post(p["id"], args.test_percent) else "train",
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": p["post"],
                "with_system_prompt_full": wf,
                "no_system_prompt_full": nf,
                "with_system_prediction": w_v,
                "no_system_prediction": n_v,
                "with_system_generation": w_gen,
                "no_system_generation": n_gen,
                "with_system_p_first_Y": first_token_prob(wo.outputs[0].logprobs, y_id),
                "with_system_p_first_N": first_token_prob(wo.outputs[0].logprobs, n_id),
                "no_system_p_first_Y": first_token_prob(no.outputs[0].logprobs, y_id),
                "no_system_p_first_N": first_token_prob(no.outputs[0].logprobs, n_id),
                "changed_pred": w_v != n_v,
            })

        # Exact length check mirroring CausalDataset.__iter__: the chat-templated
        # prompt with its answer, plus the label tokens again, must fit
        # max_prompt_length / max_sequence_length for every question type.
        texts, labels = [], []
        for r in new_rows:
            post, w_v, n_v = r["user_prompt"], r["with_system_prediction"], r["no_system_prediction"]
            for qt, lab in (("orig_match", " " + w_v), ("change_match", " " + n_v),
                            ("generative_explanation", explanation_label(w_v, n_v))):
                texts.append(render(qt, post, lab))
                labels.append(lab)
        full_lens, lab_lens = ntok(texts), ntok(labels)
        for j, r in enumerate(new_rows):
            ok = all(
                full_lens[3 * j + k] <= max_prompt
                and full_lens[3 * j + k] + lab_lens[3 * j + k] <= max_seq
                for k in range(3)
            )
            if not ok:
                counts["too_long_exact"] += 1
                continue
            scored.append(r)

        tr = [r for r in scored if r["pool"] == "train"]
        te = [r for r in scored if r["pool"] == "test"]
        tr_c = sum(r["changed_pred"] for r in tr)
        te_y = sum(r["no_system_prediction"] == "YTA" for r in te)
        print(f"[scored {pos} posts] kept={len(scored)} train_pool={len(tr)} (changed {tr_c}) "
              f"test_pool={len(te)} (no-sys YTA {te_y}) drops={dict(counts)} "
              f"top_unparsed={unparsed.most_common(5)}", flush=True)
        if (tr_c >= half_train and len(tr) - tr_c >= half_train
                and te_y >= half_test and len(te) - te_y >= half_test):
            break

    for i, r in enumerate(scored):
        r["index"] = i
        # Hint-schema aliases (same convention as the refusal AITA-format rows
        # in make_data_scripts/refusal_judge_data/judge_refusals.py).
        r["random_hint_prediction"] = r["with_system_prediction"]
        r["zeroshot_prediction"] = r["no_system_prediction"]
        r["changed_pred_bc_hint"] = r["changed_pred"]
        # Unique example key, in the style of the Hint-MMLU ablation_prompt.
        r["ablation_prompt"] = (
            f"If the system prompt is removed when processing the text "
            f"'{r['no_system_prompt_full']}', how would the output change?\n\n"
            f"Output changes from {r['with_system_prediction']} to {r['no_system_prediction']}"
        )

    rng = random.Random(args.seed)
    train_pool = [r for r in scored if r["pool"] == "train"]
    test_pool = [r for r in scored if r["pool"] == "test"]

    # Train: balanced 50/50 on changed_pred (the change/unchange verdict).
    changed = [r for r in train_pool if r["changed_pred"]]
    unchanged = [r for r in train_pool if not r["changed_pred"]]
    k = min(half_train, len(changed), len(unchanged))
    train = rng.sample(changed, k) + rng.sample(unchanged, k)
    rng.shuffle(train)

    # Test: balanced 50/50 on the no-system-prompt verdict (paper App. B.4).
    by_v = {v: [r for r in test_pool if r["no_system_prediction"] == v] for v in VERDICTS}
    for v in VERDICTS:
        rng.shuffle(by_v[v])
    kt = min(half_test, *(len(by_v[v]) for v in VERDICTS))
    test_large = by_v["YTA"][:kt] + by_v["NTA"][:kt]
    rng.shuffle(test_large)
    ks = min(args.n_test_small // 2, kt)
    test_small = by_v["YTA"][:ks] + by_v["NTA"][:ks]
    rng.shuffle(test_small)

    def write(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows)} rows to {path}")

    def describe(rows):
        n = len(rows)
        if not n:
            return {"n": 0}
        joint = Counter(f"{r['with_system_prediction']}->{r['no_system_prediction']}" for r in rows)
        return {
            "n": n,
            "change_rate": sum(r["changed_pred"] for r in rows) / n,
            "with_system_verdicts": dict(Counter(r["with_system_prediction"] for r in rows)),
            "no_system_verdicts": dict(Counter(r["no_system_prediction"] for r in rows)),
            "with_to_no_transitions": dict(joint),
            "reddit_verdicts": dict(Counter(r["reddit_verdict"] for r in rows)),
            "with_system_bold_rate": sum("*" in r["with_system_generation"] for r in rows) / n,
            "no_system_bold_rate": sum("*" in r["no_system_generation"] for r in rows) / n,
        }

    write(f"{prefix}_all.jsonl", scored)
    write(f"{prefix}_train_balanced.jsonl", train)
    write(f"{prefix}_test_large.jsonl", test_large)
    write(f"{prefix}_test_balanced.jsonl", test_small)

    stats = {
        "model": args.model,
        "config": args.config,
        "system_prompt": SYSTEM_PROMPT,
        "length_limit": {"mode": args.length_limit, "max_prompt_length": max_prompt,
                         "max_sequence_length": max_seq},
        "posts_scored": pos,
        "candidate_filter_drops": drop_counts,
        "scoring_drops": dict(counts),
        "unparsed_generations_top20": unparsed.most_common(20),
        "all_scored": describe(scored),
        "train_pool": describe(train_pool),
        "test_pool": describe(test_pool),
        "train_balanced": describe(train),
        "test_large": describe(test_large),
        "test_balanced": describe(test_small),
        "train_balancing": "50/50 on changed_pred (with-system verdict != no-system verdict)",
        "test_balancing": "50/50 on no_system_prediction (YTA vs NTA), natural change rate within",
    }
    with open(f"{prefix}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
