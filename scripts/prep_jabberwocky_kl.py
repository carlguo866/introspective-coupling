"""Fig 7 KL variant: make Jabberwocky rows safe to mix with KL-regularized Hint-MMLU rows.

- Offset `index` (J_train +1,000,000, J_test +2,000,000) so the teacher-logit
  cache (keyed by index) can never return an MMLU question's logits.
- Set teacher_logits_with_hint / teacher_logits_no_hint to null: the loader then
  skips the on-demand base-model logit computation (which would pull nonsense-
  question behavior toward the base model) and the loss falls back to per-row
  cross-entropy on the designated answers (model/continuous_base.py).
"""
import json
import sys

for src, dst, off in [
    ("hint_data/synthetic_qwen_qwen3-8b_ablation_prompts_synthetic.jsonl",
     "hint_data/jabberwocky_train_kl_ht8.jsonl", 1_000_000),
    ("hint_data/synthetic_qwen_qwen3-8b_ablation_prompts_jtest_synthetic.jsonl",
     "hint_data/jabberwocky_test_kl_ht8.jsonl", 2_000_000),
]:
    n = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            r = json.loads(line)
            r["index"] = int(r["index"]) + off
            r["teacher_logits_with_hint"] = None
            r["teacher_logits_no_hint"] = None
            g.write(json.dumps(r) + "\n")
            n += 1
    print(dst, n)
