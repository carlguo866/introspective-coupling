# Refusal data

The train and test files used for the paper's refusal experiments.

| File | Rows | Contents |
|---|---|---|
| `combined_qwen3_8b_system_prompt_train_judge_natural.jsonl.gz` | 23,436 (FalseReject 14,624, WildJailbreak 8,812) | training data |
| `combined_qwen3_8b_system_prompt_test_judge_natural.jsonl.gz` | 2,374 (1,187 from each source) | test data; its labels are the Orig labels |

Each row holds one user prompt, a refusal-inducing system prompt, Qwen3-8B's reply with the system prompt
(`with_system_prediction`) and without it (`no_system_prediction`), and the judge's reading of each reply, where `Yes`
means refused: `random_hint_prediction` (with the system prompt) and `zeroshot_prediction` (without it). `changed_pred`
is true when the two readings differ. `index` keys the teacher-logit file. Replies were judged by gpt-5-mini at minimal
reasoning with the prompt in `make_data_scripts/refusal_judge_data/judge_refusals.py`.

```bash
bash data/refusal/unpack.sh   # -> falsereject_data/, checksums verified
python scripts/precompute_teacher_logits.py --model_path Qwen/Qwen3-8B --topk 128 --mode freeform \
    --hint_path falsereject_data/combined_qwen3_8b_system_prompt_train_judge_natural.jsonl
python train.py --config_path configs/qwen_qwen_refusal_combined_judge_freeform_natural.yaml
```

Regenerating the data with `make_data_scripts/refusal_judge_data/` gives different labels, because the judge is not
deterministic.

The train and test files were judged in separate runs, and the FalseReject test labels are stricter than the training
labels:

| FalseReject replies judged refused | Without system prompt | With system prompt |
|---|---|---|
| Training file | 15.4% | 48.0% |
| Test file | 31.9% | 56.9% |

Keep this in mind when comparing Self and Orig on this test set. Self labels are re-judged at evaluation time, so the
stricter test labels can widen the gap between Self and Orig.
