"""Multi-token behavior targets + per-token teacher logits for AITA KL regularization.

Regularizing only the first token after "Answer:" is not enough for AITA: Qwen3-8B often answers in bold
(" **YTA**"), so the first token is " **" and the verdict tokens after it are unconstrained; trained models
then degenerate into " ** ** ** ...". This script records the starting model's own greedy reply after "Answer:" for
both conditions (with / without the system prompt) and its top-k logits at every reply token, so KL can cover the
whole verdict (configs/qwen_qwen_aita_kl_regularized.yaml, dataloaders/causal.py `aita_freeform_target`).

Writes
  <out_jsonl>: the input rows plus aita_reply_with_sys / aita_reply_no_sys (decoded reply, leading space kept)
  <out_jsonl>.teacher_logits_freeform_top<k>.pt: {index: {"with_hint": (idx[T,k], vals[T,k]), "no_hint": (...)}}

    python make_data_scripts/aita_freeform_teacher.py --in aita_data/aita_qwen3_8b_system_prompt_train_balanced.jsonl \
        --out aita_data/aita_qwen3_8b_system_prompt_train_balanced_freeform.jsonl
"""
import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    rows = [json.loads(l) for l in open(a.inp)]
    jobs = [(i, side) for i in range(len(rows)) for side in ("with", "no")]
    replies, teacher = {}, {}
    for b in tqdm(range(0, len(jobs), a.batch_size)):
        batch = jobs[b:b + a.batch_size]
        texts = [rows[i]["with_system_prompt_full" if side == "with" else "no_system_prompt_full"] for i, side in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, output_logits=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        logits = torch.stack(out.logits, dim=1)  # [B, steps, vocab]
        for j, (i, side) in enumerate(batch):
            ids = gen[j].tolist()
            T = next((t for t, x in enumerate(ids) if x in stop_ids), len(ids))
            T = max(T, 1)
            vals, idx = logits[j, :T].float().topk(a.topk, dim=-1)
            replies[(i, side)] = tok.decode(ids[:T])
            teacher.setdefault(rows[i]["index"], {})["with_hint" if side == "with" else "no_hint"] = (idx.cpu(), vals.cpu())

    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            r["aita_reply_with_sys"] = replies[(i, "with")]
            r["aita_reply_no_sys"] = replies[(i, "no")]
            f.write(json.dumps(r) + "\n")
    pt = f"{a.out}.teacher_logits_freeform_top{a.topk}.pt"
    torch.save(teacher, pt)
    from collections import Counter
    c = Counter(replies.values())
    print(f"wrote {len(rows)} rows -> {a.out}; teacher logits -> {pt}; most common replies: {c.most_common(8)}")


if __name__ == "__main__":
    main()
