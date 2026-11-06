import argparse
import json
import os

import yaml
from model.utils import load_models
from tqdm import tqdm
from train import TASK_DATASET_MAPPING
from transformers import AutoTokenizer


def process_example(
    example,
    f,
):
    try:
        json.dump(example, f)
        f.write("\n")
    except Exception as e:
        raise
        print(e)
        print(example)
        raise e


def main(args, config):
    if args.cache_dir is not None:
        config["cache_dir"] = args.cache_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.target_tokenizer_path is None:
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            padding_side="left",
            cache_dir=config.get("cache_dir", None),
        )
    else:
        target_tokenizer = AutoTokenizer.from_pretrained(
            args.target_tokenizer_path,
            padding_side="left",
            cache_dir=config.get("cache_dir", None),
        )
    target_tokenizer.pad_token = target_tokenizer.eos_token
    if args.predictor_tokenizer_path is None:
        predictor_tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"],
            padding_side="left",
            cache_dir=config.get("cache_dir", None),
        )
    else:
        predictor_tokenizer = AutoTokenizer.from_pretrained(
            args.predictor_tokenizer_path,
            padding_side="left",
            cache_dir=config.get("cache_dir", None),
        )
    predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
    wrapped_model, target_model, _ = load_models(
        predictor_model_path=config["model_path"],
        target_model_path=args.target_model_path,
        cache_dir=config.get("cache_dir", None),
        train_self=False,
        use_bf16=config["train"].get("bf16", True),
        batch_size=config["test"]["batch_size"],
        use_embed_proj=config.get("use_embed_proj", False),
        embed_proj_path=config.get("embed_proj_path", None),
        predictor_model_type=config.get("predictor_model_type", None),
    )

    explainer_llm, simulator = None, None

    # Load the dataset
    train_feature_descriptor = None
    if args.task == "activation_explain":
        # computing features from save_path
        dataset = TASK_DATASET_MAPPING[args.task](
            "train",
            wrapped_model,
            target_model,
            predictor_tokenizer,
            target_tokenizer,
            config=config["train"]["tasks"][args.task],
            num_samples=config["train"]["num_samples"],
            explainer_llm=explainer_llm,
            simulator=simulator,
            model_name=config["model_path"],
            model_cache_dir=config.get("cache_dir", None),
        )
        train_feature_descriptor = dataset.feature_descriptor

    dataset = TASK_DATASET_MAPPING[args.task](
        "test",
        wrapped_model,
        target_model,
        predictor_tokenizer,
        target_tokenizer,
        config=config["test" if not args.train_split else "train"]["tasks"][args.task],
        feature_descriptor=train_feature_descriptor,
        model_name=config["model_path"],
        model_cache_dir=config.get("cache_dir", None),
    )

    task_cfg = config["test" if not args.train_split else "train"]["tasks"][args.task]
    if (
        args.recompute_labels
        and task_cfg.get("recompute_method") == "freeform_judge"
        and hasattr(dataset, "non_icl_labels")
    ):
        from dataloaders.freeform_judge import recompute_labels_freeform_judge
        cache_dir = args.judge_cache_dir or task_cfg.get("judge_cache_dir")
        cache_path = (
            os.path.join(cache_dir, f"{args.task}_judge_cache.jsonl")
            if cache_dir
            else None
        )
        recompute_labels_freeform_judge(
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            labels=dataset.non_icl_labels,
            max_new_tokens=task_cfg.get("freeform_max_new_tokens", 256),
            judge_model=task_cfg.get("judge_model", "gpt-5-mini"),
            max_concurrent=task_cfg.get("judge_max_concurrent", 20),
            cache_path=cache_path,
            batch_size=task_cfg.get("freeform_batch_size", 8),
        )

    save_file = os.path.join(args.output_dir, f"{args.task}_data.jsonl")
    print("Saving predictions to", save_file)

    with open(save_file, "w") as f:
        if dataset[0]:
            for i in tqdm(range(len(dataset))):
                example = dataset[i]
                process_example(example, f)
        else:
            for example in tqdm(dataset):
                process_example(example, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument(
        "--target_tokenizer_path", type=str, required=False, default="Qwen/Qwen3-8B"
    )
    parser.add_argument(
        "--predictor_tokenizer_path", type=str, required=False, default="Qwen/Qwen3-8B"
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_split", action="store_true", default=False)
    parser.add_argument("--recompute_labels", action="store_true", default=False)
    parser.add_argument("--judge_cache_dir", type=str, required=False, default=None,
                        help="Override judge_cache_dir from config (use per-checkpoint dirs for self-eval)")
    parser.add_argument("--cache_dir", type=str, required=False, default=None)
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.recompute_labels:
        config["test"]["tasks"][args.task]["self_consistency"] = True
    main(args, config)
