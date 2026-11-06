# Configs

One YAML file per experiment, passed to `train.py --config_path` and `evaluate.py --config`. All use Qwen3-8B as both
target and predictor. Build the data each one reads with [`make_data_scripts/`](../make_data_scripts/README.md).

| Experiment | Config | Flags |
|---|---|---|
| Hint-MMLU | `qwen_qwen_hint_kl_regularized.yaml` | `--regularization_weight <λ>` (0 = unregularized) |
| Online relabeling | `qwen_qwen_hint_kl_online_relabel.yaml` | `--online_relabel_nu <ν>` |
| AITA | `qwen_qwen_aita_kl_regularized.yaml` | `--regularization_weight <λ>` |
| Refusal | `qwen_qwen_refusal_combined_judge_freeform_natural.yaml` | `--regularization_weight <λ>` |
| Cross-model labels | `qwen_qwen_hint_kl_llama_labels_alpha{0p5,1p0}.yaml` | |
| Jabberwocky, behavior-only | `qwen_qwen_hint_kl_jabberwocky_behavior.yaml`; held-out eval: `qwen_qwen_hint_kl_jabberwocky_eval_jtest.yaml` | |
| Behavior drift, Hint-MMLU | `qwen_qwen_hint_kl_{wildchat,warm}_mixed.yaml` | |
| Behavior drift, refusal | `qwen_qwen_refusal_combined_judge_freeform_natural_{fineweb,llmlat_only}_mixed.yaml` | |

Each config stops at the step its results are reported at (`max_steps`: 8,000; AITA 12,000; Jabberwocky runs its full
two epochs with the Jabberwocky rows upsampled 5x). Paths (`output_dir`, `hint_path`, ...) are relative to the repo root.
