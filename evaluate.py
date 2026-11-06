import argparse
import os

import yaml
from dataloaders import BaseDataCollator, FromFileDataset, TASK_DATALOADER_MAPPING
from evaluator import EVALUATOR_MAPPING
from model.utils import load_models
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_model(args, config):
    model, target_model, _ = load_models(
        predictor_model_path=args.model_path,
        target_model_path=args.target_model_path,
        cache_dir=config.get("cache_dir", None),
        train_self=False,
        use_bf16=config["train"].get("bf16", True),
        batch_size=config["test"]["batch_size"],
        use_embed_proj=config.get("use_embed_proj", False),
        embed_proj_path=config.get("embed_proj_path", None),
        predictor_model_type=config.get("predictor_model_type", config["model_path"]),
    )
    return model, target_model


def main(args):
    # Load the config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.target_tokenizer_path:
        target_tokenizer_path = args.target_model_path
    else:
        target_tokenizer_path = args.target_tokenizer_path
    # Load the model
    target_tokenizer = AutoTokenizer.from_pretrained(
        target_tokenizer_path,
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    target_tokenizer.pad_token = target_tokenizer.eos_token
    target_tokenizer.pad_token_id = target_tokenizer.eos_token_id
    model_tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    model_tokenizer.pad_token = model_tokenizer.eos_token
    model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
    if args.debug:
        num_samples = 100
    else:
        num_samples = None
    dataset = FromFileDataset.from_file("test", args.data_path, args.task, num_samples)
    model, target_model = load_model(args, config)
    data_collator = TASK_DATALOADER_MAPPING.get(args.task, BaseDataCollator)(
        predictor_tokenizer=model_tokenizer,
        target_tokenizer=target_tokenizer,
        question_config=config["test"]["tasks"][args.task],
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=data_collator
    )

    # Load the evaluator
    evaluator = EVALUATOR_MAPPING[
        config["test"]["tasks"][args.task]["evaluation_type"]
    ](
        config["test"]["tasks"][args.task],
        model,
        model_tokenizer,
        dataloader,
        parse_response_func=(
            # parse_activation_explain_response
            # if args.task == "activation_explain"
            # else None
            None
        ),
        fast_eval=args.fast_eval,
        cap_at_100=args.cap_at_100,
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        verbose=not args.quiet,
    )

    save_file = os.path.join(args.output_dir, f"{args.task}_predictions.json")
    print("Saving predictions to", save_file)
    open(save_file, "w").close()

    # Evaluate the model + Stream predictions to file
    predictions = evaluator.evaluate(save_file=save_file)

    # print metrics
    evaluator.print_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--target_tokenizer_path", type=str, required=False, default=None
    )
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast_eval", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose per-instance output during evaluation")
    parser.add_argument("--cap_at_100", action="store_true", help="Stop evaluation after ~100 samples (matches training-time eval behavior)")
    args = parser.parse_args()

    main(args)
