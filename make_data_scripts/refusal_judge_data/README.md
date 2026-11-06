# Refusal judge-verdict data pipeline

Builds the judge-verdict train/test JSONLs consumed by the `refusal_combined` configs
(`data/combined_qwen3_8b_system_prompt_{train,test}_judge_natural.jsonl`), from public
sources: [AmazonScience/FalseReject](https://huggingface.co/datasets/AmazonScience/FalseReject)
and [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak).

## Pipeline overview

For each prompt we generate two Qwen3-8B rollouts — one under a refusal-inducing
system prompt, one with no system prompt — judge each rollout as `refused`/`complied`
with an external LLM judge, and pack the paired verdicts into the JSONL schema that
`dataloaders/causal.py` consumes (`random_hint_prediction` = with-sys verdict,
`zeroshot_prediction` = no-sys verdict, `changed_pred` = the two differ).

All generation is greedy (`temperature 0.0`, `max_tokens 1024`, Qwen3 thinking
disabled). The judge is OpenAI `gpt-5-mini` (needs `OPENAI_API_KEY`).

## Step 1 — FalseReject rollouts

Run four times: {train, test} × {no system prompt, refusal system prompt}.
System prompts are per-category, from `config/system_prompts.json` (mode `refusal`).

```bash
# test split, no system prompt
python generate_responses.py \
    --model Qwen/Qwen3-8B \
    --dataset AmazonScience/FalseReject --split test \
    --output results/qwen3_8b_responses.jsonl \
    --max_tokens 1024 --temperature 0.0 --trust_remote_code

# test split, refusal system prompt
python generate_responses.py \
    --model Qwen/Qwen3-8B \
    --dataset AmazonScience/FalseReject --split test \
    --output results/qwen3_8b_refusal_responses.jsonl \
    --max_tokens 1024 --temperature 0.0 --trust_remote_code \
    --system_prompt_mode refusal \
    --system_prompts_file config/system_prompts.json

# train split: same two commands with --split train and outputs under results/train/
```

## Step 2 — WildJailbreak rollouts

Prompts are the `vanilla_harmful` subset of allenai/wildjailbreak, capped at the
first 10,000. The refusal system prompt comes from
`config/or_bench_system_prompts.json` (every record has `category_text: harmful`,
so the `harmful.refusal` prompt is used).

```bash
python download_wildjailbreak.py --output data/wildjailbreak_harmful.jsonl

# no system prompt
python generate_responses.py \
    --model Qwen/Qwen3-8B \
    --dataset data/wildjailbreak_harmful.jsonl \
    --output results/wildjailbreak/qwen3_8b_responses.jsonl \
    --max_tokens 1024 --temperature 0.0 --max_model_len 4096 \
    --trust_remote_code --max_examples 10000

# refusal system prompt
python generate_responses.py \
    --model Qwen/Qwen3-8B \
    --dataset data/wildjailbreak_harmful.jsonl \
    --output results/wildjailbreak/qwen3_8b_refusal_responses.jsonl \
    --max_tokens 1024 --temperature 0.0 --max_model_len 4096 \
    --trust_remote_code --max_examples 10000 \
    --system_prompt_mode refusal \
    --system_prompts_file config/or_bench_system_prompts.json
```

## Step 3 — Judge verdicts

For every `*_responses.jsonl`, write the matching `*_judged.jsonl`
(adds `judge_verdict: refused|complied`). Resumable; safe to re-run.

```bash
for f in results/qwen3_8b{,_refusal}_responses.jsonl \
         results/train/qwen3_8b{,_refusal}_responses.jsonl \
         results/wildjailbreak/qwen3_8b{,_refusal}_responses.jsonl; do
    python judge_refusals.py --input "$f" \
        --output "${f/_responses.jsonl/_judged.jsonl}" \
        --model gpt-5-mini --max_concurrent 200
done
```

`judge_wildguard.py` is an alternative local judge (allenai/wildguard via vLLM, no
OpenAI dependency); to train on its labels, map `wildguard_response_refusal`
(`refusal`/`compliance`) into `judge_verdict` (`refused`/`complied`) before Step 4.

## Step 4 — Convert to training format

Combines FalseReject + WildJailbreak pairs. FalseReject train/test follow the HF
dataset splits; the single WildJailbreak pool is split deterministically
(`--wj_split_seed 42`, `--wj_test_size 1187` — use the same values for both
invocations so the splits stay disjoint). `--balance natural` (default) preserves
the natural class distribution.

```bash
python judge_refusals.py --emit_training_data --split train \
    --model qwen3_8b \
    --falsereject_dir results \
    --wildjailbreak_dir results/wildjailbreak \
    --output_dir ../../data
python judge_refusals.py --emit_training_data --split test \
    --model qwen3_8b \
    --falsereject_dir results \
    --wildjailbreak_dir results/wildjailbreak \
    --output_dir ../../data
```

Note `--falsereject_dir` points at the *base* results dir: `--split train` reads
its `train/` subdirectory, `--split test` reads the dir itself.

Downstream, `scripts/precompute_teacher_logits.py --mode freeform` derives the
`.teacher_logits_freeform_top128.pt` sidecar from the train JSONL.
