"""Online relabeling of explanation targets (paper §4.2, Figure 5).

At every training batch, each `generative_explanation` row's target is rebuilt
from the model's CURRENT behavior: we read the model's argmax answer letter on
the cue-included prompt (`orig_match_prompt`) and on the cue-ablated prompt
(`change_match_prompt`), both stashed on gen rows by dataloaders/causal.py.

With probability `nu` the target is the faithful self-explanation of those two
answers. Otherwise it is a coherent wrong explanation: the cue-ablated answer is
replaced by a uniformly drawn different letter, and the string is formatted the
same way (so it reads "unchanged" iff the drawn letter equals the cue-included
answer). Behavior rows (orig_match / change_match) are untouched, so their KL
regularization toward the frozen base model is unchanged.

Wraps the task collator. The HF Trainer here is single-process with
`dataloader_num_workers=0`, so collation runs in the training process and can
call the model directly (accelerate may prefetch one batch, so labels can be one
optimizer step stale).
"""

import copy
import random

import torch

LETTERS = ("A", "B", "C", "D")


def format_change(orig_letter: str, change_letter: str) -> str:
    """Same string as dataloaders/causal.py::_format_causal_change for hint rows."""
    orig = "Answer: " + orig_letter
    change = "Answer: " + change_letter
    if orig == change:
        return f"The output would remain unchanged from <<<{orig}>>>."
    return f"The most likely output would change to <<<{change}>>>."


class OnlineRelabelCollator:
    def __init__(self, base_collator, model, tokenizer, nu: float, seed: int = 0,
                 flip_explanation_direction: bool = False):
        assert 0.0 <= nu <= 1.0
        self.base = base_collator
        self.model = model
        self.tokenizer = tokenizer
        self.nu = nu
        self.rng = random.Random(seed)
        self.flip = flip_explanation_direction
        self._calls = 0
        self.stats = {"rows": 0, "relabeled_self": 0, "relabeled_noise": 0,
                      "skipped_invalid": 0, "agree_with_m0_label": 0}

    def __getattr__(self, name):
        # Forward attributes (e.g. question_config) to the wrapped collator.
        return getattr(self.base, name)

    @torch.no_grad()
    def _letters(self, prompts):
        device = next(self.model.parameters()).device
        tok = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=False, padding_side="left").to(device)
        was_training = self.model.training
        saved_out = getattr(self.model, "last_outputs", None)
        saved_in = getattr(self.model, "last_inputs", None)
        self.model.eval()
        try:
            out = self.model(input_ids=tok["input_ids"],
                             attention_mask=tok["attention_mask"], return_dict=True)
            # left padding -> the last position is the next-token position
            ids = out.logits[:, -1, :].argmax(dim=-1).tolist()
        finally:
            self.model.last_outputs = saved_out
            self.model.last_inputs = saved_in
            if was_training:
                self.model.train()
        return [self.tokenizer.decode([i]).strip() for i in ids]

    def __call__(self, features):
        gen_idx = [
            i for i, f in enumerate(features)
            if f.get("question_type") == "generative_explanation"
            and isinstance(f.get("extra_args"), dict)
            and f["extra_args"].get("orig_match_prompt")
            and f["extra_args"].get("change_match_prompt")
        ]
        if gen_idx:
            prompts = []
            for i in gen_idx:
                ea = features[i]["extra_args"]
                prompts += [ea["orig_match_prompt"], ea["change_match_prompt"]]
            letters = self._letters(prompts)
            for k, i in enumerate(gen_idx):
                o, c = letters[2 * k], letters[2 * k + 1]
                self.stats["rows"] += 1
                if o not in LETTERS or c not in LETTERS:
                    self.stats["skipped_invalid"] += 1
                    continue
                if self.rng.random() < self.nu:
                    self.stats["relabeled_self"] += 1
                    c_used = c
                else:
                    self.stats["relabeled_noise"] += 1
                    c_used = self.rng.choice([x for x in LETTERS if x != c])
                if self.flip:
                    new = format_change(c_used, o)
                else:
                    new = format_change(o, c_used)
                if self.tokenizer.chat_template is None:
                    new = " " + new
                f = dict(features[i])
                old = f["expected_label"]
                if old == new:
                    self.stats["agree_with_m0_label"] += 1
                prompt = copy.deepcopy(f["prompt"])
                content = prompt[-1]["content"]
                assert content.endswith(old), (content[-200:], old)
                prompt[-1]["content"] = content[: len(content) - len(old)] + new
                f["prompt"] = prompt
                f["expected_label"] = new
                features[i] = f
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls % 200 == 1:
            print(f"[online_relabel] nu={self.nu} calls={self._calls} stats={self.stats}", flush=True)
        return self.base(features)
