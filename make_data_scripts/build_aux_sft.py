"""Auxiliary SFT corpora for paper Fig 8 (Hint-MMLU + WildChat, Hint-MMLU + warm replies).

Writes `{"messages": [...], "source": ...}` JSONL for dataloaders/sft_jsonl.py
(loss on the final assistant turn only).

  wildchat: English, first-turn conversations from allenai/WildChat-1M with the
            original ChatGPT reply; non-toxic, non-redacted rows only.
  warm:     datasets/transformed_conversations_1.json (ShareGPT-Vicuna
            conversations whose assistant turns were rewritten to be warm and
            empathetic). Every assistant turn becomes one row: the conversation
            prefix up to and including that turn, so all 4,297 rewritten turns
            are supervised.

Usage:
    python make_data_scripts/build_aux_sft.py wildchat --n 40000 --out data/wildchat_first_turn_sft.jsonl
    python make_data_scripts/build_aux_sft.py warm --out data/warm_sft.jsonl
"""
import argparse
import json
import os


def build_wildchat(n, out, max_chars):
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    kept = seen = 0
    with open(out, "w") as f:
        for ex in ds:
            seen += 1
            if ex.get("language") != "English" or ex.get("toxic") or ex.get("redacted"):
                continue
            conv = ex.get("conversation") or []
            if len(conv) < 2 or conv[0].get("role") != "user" or conv[1].get("role") != "assistant":
                continue
            u, a = conv[0]["content"].strip(), conv[1]["content"].strip()
            if not u or not a or len(u) + len(a) > max_chars:
                continue
            f.write(json.dumps({"messages": [{"role": "user", "content": u},
                                             {"role": "assistant", "content": a}],
                                "source": f"wildchat:{ex.get('conversation_hash', seen)}"}) + "\n")
            kept += 1
            if kept >= n:
                break
    print(f"wildchat: kept {kept} of {seen} streamed -> {out}")


def build_warm(out, src):
    data = json.load(open(src))
    role = {"human": "user", "gpt": "assistant"}
    rows = 0
    with open(out, "w") as f:
        for c in data:
            msgs = []
            for m in c["conversations"]:
                r = role.get(m["from"])
                if r is None:
                    continue
                msgs.append({"role": r, "content": m["value"]})
                if r == "assistant" and len(msgs) >= 2 and msgs[-2]["role"] == "user":
                    f.write(json.dumps({"messages": list(msgs), "source": f"warm:{c['id']}"}) + "\n")
                    rows += 1
    print(f"warm: {len(data)} conversations -> {rows} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["wildchat", "warm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--max_chars", type=int, default=16000)
    ap.add_argument("--src", default="datasets/transformed_conversations_1.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.which == "wildchat":
        build_wildchat(a.n, a.out, a.max_chars)
    else:
        build_warm(a.out, a.src)
