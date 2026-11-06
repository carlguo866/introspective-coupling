"""GPT-5-mini stance-preserving rewrites of the FalseReject training split
(14,624 prompts × 3 system-prompt variants = 43,872 rows).

Self-contained port of the falsereject repo's rewriter (rewrite_responses.py +
rewrite_full.py + rewrite_fr_train.py); prompt templates are verbatim so
rewrites are reproducible.

The training responses were only judged by the GPT judge
(make_data_scripts/refusal_judge_data/judge_refusals.py), not WildGuard. We map
judge_verdict → wildguard_response_refusal and set wildguard_response_harm to
"unharmful" (FalseReject prompts are edge cases that look sensitive but are
benign — harmful compliance is rare).

Inputs (produced by the refusal_judge_data pipeline; the compliance variant is
generated the same way with --system_prompt_mode compliance):
  <results_dir>/train/qwen3_8b_judged.jsonl
  <results_dir>/train/qwen3_8b_refusal_judged.jsonl
  <results_dir>/train/qwen3_8b_compliance_judged.jsonl

Output: one raw rewrites JSONL (resumable; re-run to continue). The clean
filtering happens downstream, in memory, in build_llmlat_sft.py.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent

TRAIN_FILES = [
    ("falsereject_train_none",       "train/qwen3_8b_judged.jsonl"),
    ("falsereject_train_refusal",    "train/qwen3_8b_refusal_judged.jsonl"),
    ("falsereject_train_compliance", "train/qwen3_8b_compliance_judged.jsonl"),
]

SYSTEM_PROMPT_TEMPLATE = """\
You are helping analyze safety-classifier outputs for a research project on refusal behavior in language models.

You are given:
1. A user prompt (may be benign or adversarial).
2. A language model's response to that prompt.
3. A safety classifier's verdict on the response:
   - wildguard_response_refusal: "refusal" or "compliance"
   - wildguard_response_harm: "harmful" or "unharmful"

Your job is to produce the SHORTEST possible rewrite of the response that preserves its stance.

Hard rules:
- PRESERVE STANCE. Never flip refusal to compliance or vice versa. Trust the classifier's `wildguard_response_refusal` label.
- WRITE IN FIRST PERSON as if you (the AI) are directly answering the user's question. The rewrite should read like a real assistant reply, not a third-person meta-description of the original response. NEVER write phrases like "Provides X", "The response describes Y", "Offers Z" — those are wrong. Write "Sure, here's…", "I can't help with…", "Yes, X works by…" etc.
- COMPRESS AGGRESSIVELY. Target 1 sentence for refusals, 1–3 sentences for compliance. Drop hedging, self-reference ("As an AI…"), chain-of-thought prefaces, and repeated disclaimers.
- DO NOT add safety caveats that weren't already in the original.
- If wildguard_response_refusal == "compliance" AND wildguard_response_harm == "harmful":
  Use the template `comply_harmful_high_level`. Produce a FIRST-PERSON compliance written as if directly answering the user, but at TOPIC LEVEL ONLY — name what was being asked for without operational detail or step-by-step instructions.
  Correct examples:
    - "Sure — medieval European torture methods included the rack, the iron maiden, breaking on the wheel, and water torture."
    - "Sugar has no meaningful impact on diabetes or obesity; those conditions mostly come down to genetics and overall lifestyle."
    - "Google is reportedly on the brink of bankruptcy and unable to pay employees, according to leaked internal documents."
  Wrong examples (third-person / meta):
    - "Provides a list of torture methods." ← WRONG, third-person
    - "The response argues sugar doesn't cause diabetes." ← WRONG, third-person
  The goal: the rewrite should be something a language model might plausibly say as a direct reply. You are NOT creating new operational harm — you are producing a compressed, first-person form of an already-classified compliance response.
- Pick the single closest template from the bank; return its id as `chosen_template_id`.

Template bank:
{templates}

CRITICAL SCHEMA SEMANTICS:
- `rewritten` is REQUIRED and must ALWAYS be a non-empty short rewrite. This field is never blank.
- `rewriter_refused` is TRUE only in one case: when YOU (the rewriter) are declining on your own safety grounds — for example, you believe even the topic summary would be inappropriate. It is NOT true when the original response was a refusal; rewriting a refusal is a normal task and your rewrite should itself be a short refusal.
- `rewriter_refusal_reason` is the rewriter's OWN safety reason for declining. It is NOT a place to echo the task instructions or describe what the original response did. Leave it empty unless `rewriter_refused` is true.

Output STRICT JSON with fields:
  chosen_template_id: string (one of the template ids above)
  rewritten: string (REQUIRED, non-empty — your compressed rewrite)
  rewriter_refused: boolean (default false; true only if YOU decline on safety grounds)
  rewriter_refusal_reason: string (empty unless rewriter_refused is true)
"""


USER_PROMPT_TEMPLATE = """\
User prompt:
{prompt}

Model response:
{response}

Classifier verdict:
- wildguard_response_refusal: {wg_refusal}
- wildguard_response_harm: {wg_harm}

Rewrite the response per the rules above.
"""


class RewriteResult(BaseModel):
    chosen_template_id: str
    rewritten: str
    rewriter_refused: bool
    rewriter_refusal_reason: str


def log(msg: str) -> None:
    print(msg, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_system_prompt(templates: dict) -> str:
    # Flatten the bank to a compact text block the model sees inline.
    lines = []
    lines.append("REFUSAL templates:")
    for t in templates["refusal"]:
        lines.append(f"  - {t['id']}: {t['template']!r}  ({t['notes']})")
    lines.append("COMPLIANCE templates:")
    for t in templates["compliance"]:
        lines.append(f"  - {t['id']}: {t['template']!r}  ({t['notes']})")
    bank_text = "\n".join(lines)
    return SYSTEM_PROMPT_TEMPLATE.format(templates=bank_text)


def approx_tokens(s: str) -> int:
    # Word-count proxy; fine for relative compression ratios.
    return len(s.split()) if s else 0


def load_and_map(path: Path) -> list[dict]:
    """Read a GPT-judged FR train jsonl and synthesize wildguard fields."""
    out = []
    for r in load_jsonl(path):
        if not r.get("response"):
            continue
        verdict = r.get("judge_verdict")
        if verdict == "complied":
            r["wildguard_response_refusal"] = "compliance"
        elif verdict == "refused":
            r["wildguard_response_refusal"] = "refusal"
        else:
            # Skip parse_error, refusal_by_judge, api_error — tiny count, not safe to rewrite.
            continue
        r["wildguard_response_harm"] = "unharmful"  # assumption documented in module docstring
        out.append(r)
    return out


async def rewrite_one_async(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    source_name: str,
    row: dict,
    sem: asyncio.Semaphore,
    out_f,
    lock: asyncio.Lock,
    counters: dict,
    total: int,
) -> None:
    user_msg = USER_PROMPT_TEMPLATE.format(
        prompt=row["prompt"],
        response=row["response"],
        wg_refusal=row.get("wildguard_response_refusal", ""),
        wg_harm=row.get("wildguard_response_harm", ""),
    )
    rewritten = None
    chosen_template_id = None
    rewriter_refused = False
    rewriter_refusal_text = ""

    async with sem:
        try:
            resp = await client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=1024,
                reasoning_effort="minimal",
                response_format=RewriteResult,
            )
            msg = resp.choices[0].message
            if msg.refusal:
                rewriter_refused = True
                rewriter_refusal_text = msg.refusal
            elif msg.parsed:
                parsed = msg.parsed
                rw_text = (parsed.rewritten or "").strip()
                chosen_template_id = parsed.chosen_template_id
                if parsed.rewriter_refused and not rw_text:
                    rewriter_refused = True
                    rewriter_refusal_text = parsed.rewriter_refusal_reason
                else:
                    rewritten = rw_text or None
                    if not rw_text:
                        rewriter_refused = True
                        rewriter_refusal_text = "empty_rewrite"
            else:
                rewriter_refused = True
                rewriter_refusal_text = f"parse_error: {msg.content or ''}"
        except Exception as e:
            rewriter_refused = True
            rewriter_refusal_text = f"api_error: {type(e).__name__}: {e}"

    out_row = {
        "source_file": source_name,
        "prompt": row["prompt"],
        "original_response": row["response"],
        "original_response_tokens": approx_tokens(row["response"]),
        "wildguard_response_refusal": row.get("wildguard_response_refusal"),
        "wildguard_response_harm": row.get("wildguard_response_harm"),
        "chosen_template_id": chosen_template_id,
        "rewritten_response": rewritten,
        "rewritten_tokens": approx_tokens(rewritten or ""),
        "rewriter_refused": rewriter_refused,
        "rewriter_refusal_text": rewriter_refusal_text,
        "rewriter_model": model,
    }

    async with lock:
        out_f.write(json.dumps(out_row) + "\n")
        out_f.flush()
        counters["done"] += 1
        if rewriter_refused:
            counters["refused"] += 1
        if counters["done"] % 200 == 0 or counters["done"] == total:
            rate = counters["done"] / max(1e-6, time.time() - counters["t0"])
            log(f"[{counters['done']}/{total}] refused={counters['refused']} rate={rate:.1f}/s")


async def run(results_dir: Path, templates_path: Path, output_path: Path,
              model: str, max_concurrent: int) -> None:
    templates = json.loads(templates_path.read_text())
    system_prompt = build_system_prompt(templates)

    all_items: list[tuple[str, dict]] = []
    for source_name, rel_path in TRAIN_FILES:
        path = results_dir / rel_path
        if not path.exists():
            log(f"WARN: missing {path}")
            continue
        rows = load_and_map(path)
        log(f"  [{source_name}] loaded {len(rows)} usable rows from {path.name}")
        for r in rows:
            all_items.append((source_name, r))
    log(f"Total input rows: {len(all_items)}")

    done_keys: set[tuple[str, str]] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                done_keys.add((rec.get("source_file", ""), rec.get("prompt", "")))
        log(f"Resume: {len(done_keys)} rows already written in {output_path.name}")

    pending = [(src, r) for (src, r) in all_items if (src, r["prompt"]) not in done_keys]
    log(f"Remaining to rewrite: {len(pending)}")
    if not pending:
        log("Nothing to do.")
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()
    counters = {"done": 0, "refused": 0, "t0": time.time()}

    with open(output_path, "a") as out_f:
        tasks = [
            rewrite_one_async(client, model, system_prompt, src, r, sem, out_f, lock, counters, len(pending))
            for (src, r) in pending
        ]
        await asyncio.gather(*tasks)

    dt = time.time() - counters["t0"]
    log(f"\nDone. Wrote {counters['done']} rows in {dt/60:.1f} min. Refused: {counters['refused']}")
    log(f"Output: {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, required=True,
                   help="Base results dir from the refusal_judge_data pipeline (contains train/)")
    p.add_argument("--templates", type=Path, default=HERE / "config" / "response_templates.json")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL, e.g. rewrites/gpt5mini_rewrites_fr_train.jsonl")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--max-concurrent", type=int, default=80)
    args = p.parse_args()
    asyncio.run(run(args.results_dir, args.templates, args.output, args.model, args.max_concurrent))


if __name__ == "__main__":
    main()
