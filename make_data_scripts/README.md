# Building the data

Run from the repo root (`uv run --env-file .env python ...`). Each dataset is independent.

## Hint-MMLU

```bash
python make_data_scripts/mmlu_investigation_vllm.py --model Qwen/Qwen3-8B --hint_template 8 --suffix ht8
python scripts/precompute_teacher_logits.py --model_path Qwen/Qwen3-8B --topk 128 \
    --hint_path hint_data/mmlu_qwen3_8b_ablation_prompts_ht8_train.jsonl
```

Writes `hint_data/mmlu_qwen3_8b_ablation_prompts_ht8_{train,test}.jsonl` and the teacher logits used by the KL term.

## AITA

```bash
python make_data_scripts/aita_vllm.py --model Qwen/Qwen3-8B --out_dir aita_data
python make_data_scripts/aita_freeform_teacher.py \
    --in aita_data/aita_qwen3_8b_system_prompt_train_balanced.jsonl \
    --out aita_data/aita_qwen3_8b_system_prompt_train_balanced_freeform.jsonl
```

The second step records the base model's whole verdict reply and its per-token logits.

## Refusal

The paper's files ship in `data/refusal/`; unpack them with `bash data/refusal/unpack.sh` (see
[its README](../data/refusal/README.md)). To build new data instead, follow [`refusal_judge_data/`](refusal_judge_data/README.md).
Teacher logits use `scripts/precompute_teacher_logits.py --mode freeform`.

## Jabberwocky

```bash
python make_data_scripts/jabberwocky_generate_vllm.py --model Qwen/Qwen3-8B --n_train 6000 --n_test 1000 --out_dir jabberwocky_data
python make_data_scripts/mmlu_investigation_vllm.py --synthetic_mode --model Qwen/Qwen3-8B --hint_template 8 \
    --synthetic_data_path jabberwocky_data/jabberwocky_train.jsonl
python make_data_scripts/mmlu_investigation_vllm.py --synthetic_mode --model Qwen/Qwen3-8B --hint_template 8 \
    --suffix jtest --synthetic_data_path jabberwocky_data/jabberwocky_test.jsonl
python scripts/prep_jabberwocky_kl.py
```

## Llama explanation labels

```bash
python make_data_scripts/mmlu_investigation_vllm.py --model meta-llama/Llama-3.1-8B-Instruct --hint_template 8 --suffix ht8
python scripts/mix_llama_labels.py --alpha 0.5 --split train   # and --alpha 1.0
```

Requires the Hint-MMLU data above.

## Auxiliary SFT corpora

```bash
python make_data_scripts/build_aux_sft.py wildchat --n 40000 --out data/wildchat_first_turn_sft.jsonl
python make_data_scripts/build_aux_sft.py warm --out data/warm_sft.jsonl   # from datasets/transformed_conversations_1.json
```

The LLM-LAT corpora for the refusal mixes are built by [`llmlat_sft_data/`](llmlat_sft_data/README.md).

## Formats

- **Hint data:** chat `messages` with and without the hint, and the answer under each.
- **Refusal data:** prompt `messages` and the judge's verdict on the reply with and without the system prompt.
- **Auxiliary data:** plain SFT chat `messages`, mixed in at the ratio set in the config.
