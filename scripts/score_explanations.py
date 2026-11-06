"""Score a trained explainer against several label sets (paper §2.3 metrics).

Inputs
  --pred   evaluate.py output (<out>/pred/causal_predictions.json, JSONL): the
           explainer's generated explanation for each meta-question. Generations
           are deterministic and depend only on the prompt, so one prediction file
           is scored against every label set by joining on the example key.
  --labels NAME=PATH (repeatable): causal_data.jsonl label sets from
           scripts/make_test_data.py, e.g. self=<ckpt labels> orig=<base labels>.
  --behavior A,B (repeatable): Behavior EM between label sets A and B
           (both cue-included and cue-ablated answers must match, Eq. 4).

Explanation EM = normalized exact match of the generated explanation to the label
set's explanation (strip, lowercase, strip "."), as in
evaluator/causal_exact_match_eval.py. Also reports Self-vs-Orig on the subset where
the two label sets agree / disagree, and change / unchanged F1.
Writes a JSON summary with --out.
"""
import argparse
import json
import re
from collections import defaultdict


def norm(s):
    return (s or "").strip().lower().strip(".")


def is_changed(expl):
    e = norm(expl)
    if "remain unchanged" in e or "would not change" in e:
        return False
    if "would change" in e or "change to" in e:
        return True
    return None


def key_of(rec):
    # The rendered meta-question is unique per example (extra_args.input is not:
    # synthetic Jabberwocky rows share 16 ablation_prompt strings).
    pw = rec.get("prompt_without_answer")
    return pw if isinstance(pw, str) else json.dumps(pw, sort_keys=True)


def load_labels(path):
    out = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        qt = r.get("question_type")
        out[key_of(r)][qt] = r.get("expected_label")
        if qt == "generative_explanation":
            ea = r.get("extra_args") or {}
            # behaviour behind this explanation label: cue-included / cue-ablated answers
            out[key_of(r)]["_beh"] = (ea.get("original_response"), ea.get("ablated_response"))
    return out


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", action="append", default=[])
    ap.add_argument("--behavior", action="append", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sets = {}
    for spec in a.labels:
        name, path = spec.split("=", 1)
        sets[name] = load_labels(path)

    preds = {}
    for line in open(a.pred):
        r = json.loads(line)
        if r.get("question_type") != "generative_explanation":
            continue
        preds[key_of(r)] = r.get("predicted_label")

    summary = {"n_predictions": len(preds), "explanation_em": {}, "f1": {}, "behavior_em": {}}
    for name, lab in sets.items():
        hit = tot = 0
        tp = fp = fn = utp = ufp = ufn = 0
        for k, p in preds.items():
            exp = lab.get(k, {}).get("generative_explanation")
            if exp is None:
                continue
            tot += 1
            hit += norm(p) == norm(exp)
            pc, ec = is_changed(p), is_changed(exp)
            if ec is True:
                tp += pc is True
                fn += pc is not True
            elif pc is True:
                fp += 1
            if ec is False:
                utp += pc is False
                ufn += pc is not False
            elif pc is False:
                ufp += 1
        summary["explanation_em"][name] = {"em": hit / tot if tot else None, "n": tot}
        summary["f1"][name] = {"change_f1": f1(tp, fp, fn), "unchange_f1": f1(utp, ufp, ufn)}

    names = list(sets)
    if "self" in sets and "orig" in sets:
        agree = defaultdict(lambda: [0, 0])
        for k, p in preds.items():
            s = sets["self"].get(k, {}).get("generative_explanation")
            o = sets["orig"].get(k, {}).get("generative_explanation")
            if s is None or o is None:
                continue
            sub = "agree" if norm(s) == norm(o) else "disagree"
            agree[sub][0] += 1
            if sub == "disagree":
                agree["disagree_self"][0] += norm(p) == norm(s)
                agree["disagree_orig"][0] += norm(p) == norm(o)
            else:
                agree["agree_hit"][0] += norm(p) == norm(s)
        n_all = agree["agree"][0] + agree["disagree"][0]
        summary["self_vs_orig_split"] = {
            "n": n_all,
            "agree_frac": agree["agree"][0] / n_all if n_all else None,
            "agree_subset_em": agree["agree_hit"][0] / agree["agree"][0] if agree["agree"][0] else None,
            "disagree_subset_self_em": agree["disagree_self"][0] / agree["disagree"][0] if agree["disagree"][0] else None,
            "disagree_subset_orig_em": agree["disagree_orig"][0] / agree["disagree"][0] if agree["disagree"][0] else None,
        }

    # Subset where the trained model's own answers are well-formed verdicts/letters (e.g. AITA replies that
    # degenerate into " ** ** **" give a self label no explanation can match).
    if "self" in sets:
        wf = re.compile(r"^Answer: [A-Za-z]+$")
        ok = [k for k in preds if (sets["self"].get(k, {}).get("_beh") or (None, None))[0] is not None
              and all(wf.match((x or "").strip()) for x in sets["self"][k]["_beh"])]
        summary["well_formed_self_subset"] = {"n": len(ok)}
        for name, lab in sets.items():
            hits = [norm(preds[k]) == norm(lab[k]["generative_explanation"]) for k in ok
                    if lab.get(k, {}).get("generative_explanation") is not None]
            summary["well_formed_self_subset"][name] = sum(hits) / len(hits) if hits else None

    for spec in a.behavior:
        x, y = spec.split(",")
        both = inc = abl = n = 0
        for k in set(sets[x]) & set(sets[y]):
            bx, by = sets[x][k].get("_beh"), sets[y][k].get("_beh")
            if not bx or not by or None in bx or None in by:
                continue
            n += 1
            i = norm(bx[0]) == norm(by[0])
            c = norm(bx[1]) == norm(by[1])
            inc += i
            abl += c
            both += i and c
        summary["behavior_em"][f"{x}~{y}"] = {
            "both": both / n if n else None, "cue_included": inc / n if n else None,
            "cue_ablated": abl / n if n else None, "n": n}

    print(json.dumps(summary, indent=1))
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
