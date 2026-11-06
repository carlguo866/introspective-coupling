"""Build the llmlat SFT datasets consumed by the *_llmlat_*_mixed configs.

One step, no intermediate format-rewrite files:

  llmlat_chosen.jsonl            — every LLM-LAT/harmful-dataset row, the
                                   dataset's own `chosen` refusal completion as
                                   the assistant turn.
  llmlat_chosen_comply_mix.jsonl — 50/50 mix of the above + short benign-prompt
                                   compliance responses drawn from the raw
                                   GPT-5-mini rewrites (rewrite_fr_train.py).
                                   Requires --rewrites; skipped otherwise.

The compliance half applies, in memory, the same conservative "clean" filter
that produced the original gpt5mini_rewrites_fr_train_clean.jsonl (drop
rewriter meta-refusals, empty rewrites, and stance-flip-suspect rows), then
keeps rows whose `chosen_template_id` starts with `comply_`, subsamples with a
fixed seed to match the LLM-LAT row count, concatenates, and shuffles.

Output schema matches dataloaders/sft_jsonl.py: one JSON object per line with
`messages` (user + assistant) and `source`.

Usage:
    # refusal-only dataset
    python build_llmlat_sft.py --out_dir data/

    # both datasets
    python build_llmlat_sft.py --out_dir data/ \
        --rewrites rewrites/gpt5mini_rewrites_fr_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# Clean-filter heuristic (verbatim from falsereject's rewrite_analysis.py):
# a real refusal rewrite should start with a refusal-shaped phrase; a real
# compliance rewrite shouldn't.
REFUSAL_STARTS = re.compile(
    r"^\s*(i can['’]?t|i cannot|i['’]m not able|i will not|i won['’]?t|this request|sorry|i['’]m sorry|"
    r"i decline|i am unable|i am not able|this falls outside|i['’]m not the right)",
    re.IGNORECASE,
)


def stance_flip_flag(row: dict) -> str:
    """Return '' if the rewrite looks stance-consistent, else a short reason."""
    if row["rewriter_refused"] or not row.get("rewritten_response"):
        return ""
    wg = row["wildguard_response_refusal"]
    rw = row["rewritten_response"].strip()
    starts_refusal = bool(REFUSAL_STARTS.match(rw))
    if wg == "refusal" and not starts_refusal:
        return "refusal→looks_like_compliance"
    if wg == "compliance" and starts_refusal:
        return "compliance→looks_like_refusal"
    return ""


def is_clean(row: dict) -> bool:
    if row.get("rewriter_refused"):
        return False
    if not (row.get("rewritten_response") or "").strip():
        return False
    return not stance_flip_flag(row)


def build_llmlat_rows(limit: int | None) -> list[dict]:
    ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
    print(f"Loaded LLM-LAT/harmful-dataset train split: {len(ds)} rows")
    rows: list[dict] = []
    n_skipped = 0
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        chosen = (row.get("chosen") or "").strip()
        if not prompt or not chosen:
            n_skipped += 1
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ],
            "source": "llmlat",
        })
        if limit is not None and len(rows) >= limit:
            break
    print(f"LLM-LAT refusal rows: {len(rows)} (skipped {n_skipped} empty)")
    return rows


def build_comply_pool(rewrites_path: Path) -> list[dict]:
    n_raw = n_dirty = 0
    pool: list[dict] = []
    with open(rewrites_path) as f:
        for line in f:
            if not line.strip():
                continue
            n_raw += 1
            row = json.loads(line)
            if not is_clean(row):
                n_dirty += 1
                continue
            template = (row.get("chosen_template_id") or "")
            if not template.startswith("comply_"):
                continue
            prompt = (row.get("prompt") or "").strip()
            rewritten = (row.get("rewritten_response") or "").strip()
            if not prompt or not rewritten:
                continue
            pool.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rewritten},
                ],
                "source": "comply_gpt5mini",
            })
    print(f"Loaded {n_raw} rewrite rows from {rewrites_path} "
          f"({n_dirty} dropped by clean filter)")
    print(f"Compliance pool after filter: {len(pool)} rows")
    return pool


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("."))
    parser.add_argument("--rewrites", type=Path, default=None,
                        help="Raw GPT-5-mini FR-train rewrites (rewrite_fr_train.py output). "
                             "If given, also build llmlat_chosen_comply_mix.jsonl.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for compliance subsample + mix shuffle")
    parser.add_argument("--limit", type=int, default=None,
                        help="Emit at most N LLM-LAT rows (smoke test).")
    args = parser.parse_args()

    llmlat_rows = build_llmlat_rows(args.limit)
    chosen_path = args.out_dir / "llmlat_chosen.jsonl"
    write_jsonl(llmlat_rows, chosen_path)
    print(f"Wrote {len(llmlat_rows)} rows to {chosen_path}")

    if args.rewrites is None:
        print("No --rewrites given; skipping llmlat_chosen_comply_mix.jsonl")
        return

    rng = random.Random(args.seed)
    comply_pool = build_comply_pool(args.rewrites)

    target_n = len(llmlat_rows)
    if len(comply_pool) < target_n:
        raise RuntimeError(
            f"Compliance pool ({len(comply_pool)}) smaller than LLM-LAT count ({target_n})"
        )
    comply_sample = rng.sample(comply_pool, target_n)
    print(f"Subsampled compliance to {len(comply_sample)} rows (seed={args.seed})")

    mixed = llmlat_rows + comply_sample
    rng.shuffle(mixed)

    mix_path = args.out_dir / "llmlat_chosen_comply_mix.jsonl"
    write_jsonl(mixed, mix_path)
    src_counts = Counter(r.get("source") for r in mixed)
    print(f"Wrote {len(mixed)} rows to {mix_path}")
    print(f"Source breakdown: {dict(src_counts)}")


if __name__ == "__main__":
    main()
