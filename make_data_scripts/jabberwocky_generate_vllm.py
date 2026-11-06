"""Generate "Jabberwocky" nonsense-science multiple-choice questions (paper Section 5.1, App. E.1).

A local model (vLLM, no API calls) writes fictitious, alien-sounding science
questions around invented names that this script supplies (a schema name and
an entity word per question), each with four invented options. The designated
"correct" answer letter is drawn uniformly at random here, so it carries no
information the model could have a prior over.

Output rows are the input format of
``make_data_scripts/mmlu_investigation.py::generate_synthetic_data``:

    {"question": str, "choices": {"A": str, "B": str, "C": str, "D": str},
     "answer": "C", "subject": "jabberwocky/<field>", "difficulty": "synthetic", ...}

Two disjoint files are written: J_train and J_test (no schema name, question
text or option set shared between them). A prior check then asks the base model
each question in the Hint-MMLU no-hint format and reports its agreement with
the designated answer (should be near 25%).

Usage:
    uv run python make_data_scripts/jabberwocky_generate_vllm.py --model Qwen/Qwen3-8B
then
    uv run python make_data_scripts/mmlu_investigation_vllm.py --synthetic_mode \\
        --synthetic_data_path jabberwocky_data/jabberwocky_train.jsonl --hint_template 8
"""

import argparse
import json
import os
import random
import re
from collections import Counter

ONSETS = ["b", "br", "d", "dr", "f", "fl", "g", "gr", "h", "k", "kr", "kh", "l", "m", "n",
          "p", "pr", "q", "r", "s", "sh", "sk", "sl", "sv", "t", "th", "tr", "v", "vr", "z",
          "zh", "y", "x", "c", "j", "w"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "y", "ia", "oa"]
CODAS = ["", "", "", "n", "r", "l", "s", "th", "x", "k", "m", "nd", "rn", "sk", "v"]

# Real fields only set the flavor of the invented vocabulary; the questions
# themselves must be about invented things.
FIELDS = [
    "crystallography", "mycology", "ornithology", "geology", "oceanography", "particle physics",
    "astronomy", "entomology", "metallurgy", "botany", "virology", "climatology", "acoustics",
    "cartography", "inorganic chemistry", "genetics", "volcanology", "glaciology",
    "paleontology", "optics", "materials science", "marine biology", "seismology",
    "atmospheric chemistry", "plasma physics", "soil science", "ichthyology", "herpetology",
    "biochemistry", "mineralogy", "limnology", "radio astronomy", "polymer chemistry",
    "immunology", "fluid dynamics", "electrochemistry", "speleology", "lichenology",
    "thermodynamics", "planetary science",
]
SYSTEM_WORDS = ["schema", "index", "protocol", "taxonomy", "registry", "convention", "codex",
                "classification", "survey", "nomenclature", "standard", "catalogue", "framework",
                "scale", "ledger", "atlas", "system"]
OPTION_STYLES = [
    "hyphenated two-part codes such as 'Rulven-di'",
    "invented single-word proper names",
    "invented two-word names (an invented adjective plus an invented noun)",
    "invented technical terms ending in a made-up suffix",
    "invented names each followed by a small Roman numeral",
    "invented compound names joined by an apostrophe",
    "invented unit values such as '14 drevs' or '3.5 quillons'",
]

EXAMPLE = (
    'According to the Lomar schema for cataloging pseudoalloy microphases, which microphase '
    'label is assigned to the sorvian crystal specimen?\n'
    'A. Rulven-di  B. Shenqa-no  C. Tilmare-zi  D. Zavrek-ka'
)

PROMPT = """You write questions for a test of nonsense "Jabberwocky" science: fictitious, alien-sounding multiple-choice questions about things that do not exist, so that no one could know the answer from real-world knowledge.

Example of the style:
{example}

Write ONE new question with these requirements:
- It is about an invented {system_word} called the "{schema}" {system_word}, flavored like {field}.
- It asks about an invented entity described with the made-up word "{entity}".
- The question is a single sentence ending with "?" and is answerable only by someone who knows the fictional {schema} {system_word}.
- Give exactly four answer options. All four must be invented words or names, mutually distinct, written as {option_style}, and equally plausible. Do not use real scientific terms, real people, real places, or options like "all of the above".
- Do not reveal or hint at which option is correct.

Respond with only a JSON object, no other text:
{{"question": "...", "options": ["...", "...", "...", "..."]}}"""


def pseudo_word(rng, n_syll):
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(n_syll))


def make_seeds(n, rng):
    """One seed per request; schema names are unique across all seeds."""
    seeds, used = [], set()
    while len(seeds) < n:
        schema = pseudo_word(rng, rng.choice([2, 2, 3])).capitalize()
        entity = pseudo_word(rng, rng.choice([1, 2])) + rng.choice(["ian", "ic", "ine", "an", "ous", "ite", "ar", ""])
        if (schema.lower() in used or not 4 <= len(schema) <= 10
                or not 5 <= len(entity) <= 11):
            continue
        used.add(schema.lower())
        seeds.append({
            "schema": schema,
            "entity": entity,
            "field": rng.choice(FIELDS),
            "system_word": rng.choice(SYSTEM_WORDS),
            "option_style": rng.choice(OPTION_STYLES),
        })
    return seeds


def parse_output(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    q, opts = obj.get("question"), obj.get("options")
    if not isinstance(q, str) or not isinstance(opts, list):
        return None
    return q.strip(), [str(o).strip() for o in opts]


_LETTER_PREFIX = re.compile(r"^\(?[A-Da-d][\.\):]\s+")
_BANNED_OPTIONS = ("all of the above", "none of the above", "both", "neither")


def validate(q, opts, seed):
    """Return a reason string if the question is unusable, else None."""
    if len(opts) != 4:
        return "not_four_options"
    opts = [_LETTER_PREFIX.sub("", o) for o in opts]
    if not (30 <= len(q) <= 400) or not q.endswith("?"):
        return "bad_question_shape"
    if "\n" in q:
        return "multiline_question"
    if seed["schema"].lower() not in q.lower():
        return "schema_missing"
    if any(not (1 <= len(o) <= 60) or "\n" in o for o in opts):
        return "bad_option_length"
    if len({o.lower() for o in opts}) != 4:
        return "duplicate_options"
    if any(o.lower() in _BANNED_OPTIONS for o in opts):
        return "banned_option"
    if any(o.lower() in q.lower() for o in opts):
        return "option_in_question"
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out_dir", default="jabberwocky_data")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--oversample", type=float, default=1.6,
                    help="Requests sent = oversample * (n_train + n_test).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--skip_prior_check", action="store_true")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from make_data_scripts.mmlu_investigation import (
        apply_chat_template, create_few_shot_prompt, extract_answer)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    n_need = args.n_train + args.n_test
    seeds = make_seeds(int(n_need * args.oversample), rng)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, seed=args.seed)
    tok = llm.get_tokenizer()
    prompts = []
    for s in seeds:
        msgs = [{"role": "user", "content": PROMPT.format(example=EXAMPLE, **s)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False))
    print(f"Sending {len(prompts)} generation requests. Example prompt:\n{prompts[0]}")
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=300,
                        seed=args.seed)
    outs = llm.generate(prompts, sp)

    reasons = Counter()
    seen_q, seen_opts = set(), set()
    kept = []
    for s, o in zip(seeds, outs):
        parsed = parse_output(o.outputs[0].text)
        if parsed is None:
            reasons["unparsable"] += 1
            continue
        q, opts = parsed
        why = validate(q, opts, s)
        if why:
            reasons[why] += 1
            continue
        opts = [_LETTER_PREFIX.sub("", x) for x in opts]
        kq, ko = norm(q), frozenset(norm(x) for x in opts)
        if kq in seen_q or ko in seen_opts:
            reasons["duplicate"] += 1
            continue
        seen_q.add(kq)
        seen_opts.add(ko)
        # Designated "correct" answer: uniform random letter (no ground truth exists).
        answer = rng.choice("ABCD")
        kept.append({
            "question": q,
            "choices": dict(zip("ABCD", opts)),
            "answer": answer,
            "subject": f"jabberwocky/{s['field']}",
            "difficulty": "synthetic",
            "schema": s["schema"],
            "entity": s["entity"],
            "generator_model": args.model,
        })
    print(f"Kept {len(kept)} / {len(seeds)} generated questions; rejects: {dict(reasons)}")
    if len(kept) < n_need:
        print(f"WARNING: only {len(kept)} questions for n_train+n_test={n_need}; test keeps "
              f"{args.n_test}, train gets the rest")

    # Schema names are unique per seed, so a slice split shares no schema between files.
    rng.shuffle(kept)
    test = kept[: args.n_test]
    train = kept[args.n_test: args.n_test + args.n_train]
    paths = {"train": os.path.join(args.out_dir, "jabberwocky_train.jsonl"),
             "test": os.path.join(args.out_dir, "jabberwocky_test.jsonl")}
    for name, rows in (("train", train), ("test", test)):
        with open(paths[name], "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows)} rows to {paths[name]}; answer letters "
              f"{dict(sorted(Counter(r['answer'] for r in rows).items()))}")

    stats = {"requests": len(seeds), "kept": len(kept), "rejects": dict(reasons),
             "n_train": len(train), "n_test": len(test), "generator_model": args.model}

    if not args.skip_prior_check:
        # How often does the base model pick the designated answer in the exact
        # Hint-MMLU no-hint format? Near 25% means no usable prior.
        rows = train + test
        texts = []
        for r in rows:
            msgs = create_few_shot_prompt([], r["question"], [r["choices"][k] for k in "ABCD"],
                                          num_shots=0, use_few_shot=False, restrict_output=True)
            texts.append(apply_chat_template(tok, msgs, args.model, add_generation_prompt=False,
                                             continue_final_message=True))
        outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=1))
        preds = [extract_answer(o.outputs[0].text) for o in outs]
        agree = sum(p == r["answer"] for p, r in zip(preds, rows))
        stats["prior_check"] = {
            "n": len(rows),
            "agreement_with_designated_answer": agree / len(rows),
            "base_model_letter_distribution": dict(Counter(str(p) for p in preds)),
        }
        print(f"Prior check: base model agrees with the designated answer on "
              f"{agree}/{len(rows)} = {agree / len(rows):.1%}; letters {dict(Counter(str(p) for p in preds))}")
        print(f"Example prior-check prompt:\n{texts[0]}")

    with open(os.path.join(args.out_dir, "jabberwocky_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    for r in train[:5]:
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
