# llmlat SFT-mix data pipeline

Builds the auxiliary SFT datasets consumed by the `*_llmlat_*_mixed` configs
(via `dataloaders/sft_jsonl.py`):

- `llmlat_chosen.jsonl` — used by
  `qwen_qwen_refusal_combined_judge_freeform_natural_llmlat_only_mixed.yaml`
- `llmlat_chosen_comply_mix.jsonl` — used by
  `qwen_qwen_refusal_combined_judge_freeform_natural_llmlat_comply_mixed.yaml`

## Two steps

### 1. `rewrite_fr_train.py` — GPT-5-mini compliance rewrites (OpenAI API)

Only needed for the comply mix. Takes the judged FalseReject *train* rollouts
from the `refusal_judge_data` pipeline (all three system-prompt variants:
none / refusal / compliance — the compliance variant is generated with
`generate_responses.py --system_prompt_mode compliance`) and asks `gpt-5-mini`
for the shortest stance-preserving first-person rewrite of each response,
anchored to the template bank in `config/response_templates.json` (resumable).

```bash
python rewrite_fr_train.py \
    --results_dir <refusal_judge_results_dir> \
    --output rewrites/gpt5mini_rewrites_fr_train.jsonl
```

Judge verdicts are mapped to `wildguard_response_refusal` labels and
`wildguard_response_harm` is fixed to `"unharmful"` (FalseReject prompts are
benign edge cases).

### 2. `build_llmlat_sft.py` — final datasets, one step

No intermediate format-rewrite files: the "clean" filtering of the rewrites
(drop rewriter meta-refusals, empty rewrites, stance-flip-suspect rows) happens
in memory here.

```bash
# refusal-only dataset (public HF data only, no API needed)
python build_llmlat_sft.py --out_dir <out>

# both datasets
python build_llmlat_sft.py --out_dir <out> \
    --rewrites rewrites/gpt5mini_rewrites_fr_train.jsonl
```

- Refusal half: every row of
  [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset),
  pairing `prompt` with the dataset's own `chosen` refusal completion.
- Compliance half: clean rewrites with `chosen_template_id` starting `comply_`,
  subsampled with `--seed 0` to match the LLM-LAT row count, concatenated and
  shuffled with the same seed.
