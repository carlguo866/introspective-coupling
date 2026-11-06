# Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision

NeurIPS 2026 · [Paper (arXiv:2606.32038)](https://arxiv.org/abs/2606.32038)

Code for the paper's experiments on training language models to produce *counterfactual
explanations* of their own behavior, and measuring whether the trained explainer is
introspectively coupled to the model it explains.

The pipeline trains a **predictor** model to answer questions about how a **target**
model's behavior changes under an intervention (e.g. removing a hint from the prompt),
and evaluates explanation accuracy against labels derived from the target's actual
behavior (`orig`) vs. the trained model's own behavior (`self`).

## Setup

Requires Python 3.12 (not 3.12.5) and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Create a `.env` with `PYTHONPATH=.`, `HF_HOME` (model cache) and, for the refusal judge, `OPENAI_API_KEY`. Run
every command below from the repo root as `uv run --env-file .env python ...`.

## Repository layout

```
train.py               # training entry point (config-driven)
evaluate.py            # evaluation entry point (writes <task>_predictions.json)
configs/               # YAML configs for every experiment in the paper
dataloaders/           # causal / mixed / judge-verdict / sft datasets
evaluator/             # exact-match & mixed-type evaluators, eval callback
model/                 # predictor wrappers (full FT + LoRA, intervention support)
interp/                # activation patching (mechanistic analysis)
scripts/               # teacher-logit precompute, eval label-set baking, scoring
make_data_scripts/     # data pipelines (see its README)
datasets/              # shipped auxiliary data (warm transformed conversations)
data/, hint_data/, checkpoints/   # expected locations for data & outputs (only data/refusal/ is shipped)
```

## Workflow

1. **Build the data and teacher logits:** see [`make_data_scripts/README.md`](make_data_scripts/README.md).
2. **Train:**

   ```bash
   python train.py --config_path configs/<config>.yaml [--regularization_weight <λ>]
   ```

3. **Bake label sets:** run once with the base model as target (Orig) and once with the trained checkpoint (Self).

   ```bash
   python scripts/make_test_data.py --config configs/<config>.yaml --task causal --recompute_labels \
       --target_model_path <base model or checkpoint> --output_dir results/<labels>
   ```

4. **Generate explanations and score them** against both label sets:

   ```bash
   python evaluate.py --config configs/<config>.yaml --task causal --data_path results/<self>/causal_data.jsonl \
       --target_model_path <checkpoint> --model_path <checkpoint> --output_dir results/<run> --batch_size 16
   python scripts/score_explanations.py --pred results/<run>/causal_predictions.json \
       --labels self=results/<self>/causal_data.jsonl --labels orig=results/<orig>/causal_data.jsonl --behavior self,orig
   ```

[`configs/README.md`](configs/README.md) maps each experiment to its config. Paths in configs are repo-relative.

## Reproducing the main results

Qwen3-8B is both target and predictor (full fine-tuning, one GPU per run, one seed). Each config stops at the step its
scores are reported at: 8,000, AITA 12,000, and Jabberwocky after two epochs.

| Result | Config and flags |
|---|---|
| Self above Orig with regularization, not without | `qwen_qwen_{hint,aita}_kl_regularized`, `qwen_qwen_refusal_combined_judge_freeform_natural`; `--regularization_weight` 1 and 0 |
| Regularization-weight threshold | hint KL config; `--regularization_weight` in {1e-4, 1e-3, 5e-3, 0.1, 1} |
| Online relabeling threshold | `qwen_qwen_hint_kl_online_relabel`; `--online_relabel_nu` in {0.5, 0.6, 0.7, 0.9} |
| Cross-model labels | `qwen_qwen_hint_kl_llama_labels_alpha{0p5,1p0}`; also score against `scripts/llama_label_set.py` labels |
| Behavior trained without explanation labels | `qwen_qwen_hint_kl_jabberwocky_behavior` (J_test: `..._eval_jtest`) |
| Explanations follow behavior drift | `qwen_qwen_hint_kl_{wildchat,warm}_mixed`, `qwen_qwen_refusal_..._{fineweb,llmlat_only}_mixed` |

Controls (Jabberwocky, behavior drift): the plain regularized model explains the trained model, i.e. `evaluate.py` with
the trained checkpoint as `--target_model_path` and the plain model as `--model_path`. Refusal Self labels need the LLM
judge (`OPENAI_API_KEY`). Expect differences of a few points between runs.

## Activation patching (mechanistic analysis)

Patch activations at the hint (or the question tail before it) and correlate the effect on the answer with the effect
on the explanation, across layers and tokens. Correlations depend on which question pairs are patched.

```bash
uv run --env-file .env python interp/activation_patching_pre_hint.py \
    --predictions_path <eval_output>/causal_predictions.json \
    --checkpoint_dir <trained checkpoint> --output_dir interp/output
uv run --env-file .env python interp/activation_patching_pre_question.py ...   # question tail before the hint
```

Outputs `patching_results.npz` + per-pair plots under `interp/output/<run_id>/<job_id>/`.

## Citation

```bibtex
@inproceedings{
guo2026introspective,
title={Introspective Coupling: Self-Explanation Training Tracks Behavioral Change Despite Fixed Supervision},
author={Zifan Carl Guo and Laura Ruis and Jacob Andreas and Belinda Z. Li},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=GgruvAw5m1}
}
```
