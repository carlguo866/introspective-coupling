import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from safetensors.torch import save_file as safe_save_file
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class ContinuousCausalLMOutputWithPast(CausalLMOutputWithPast):
    text_loss: Optional[torch.Tensor] = None
    causal_loss: Optional[torch.Tensor] = None
    regularization_loss: Optional[torch.Tensor] = None
    kl_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    top1_match_by_qt: Optional[Dict[str, Tuple[int, int]]] = None


def fixed_cross_entropy_batchwise(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    normalize_by: str = "seq_len",
    **kwargs,
) -> torch.Tensor:
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction="none"
    )
    if normalize_by == "supervised":
        # Average over supervised positions only — keeps tasks with few
        # supervised tokens (e.g. HINT — 1 letter) on the same scale as
        # longer-supervised tasks (e.g. explanation generation).
        valid = (target != ignore_index).to(loss.dtype)
        denom = valid.sum(-1).clamp(min=1.0)
        return (loss * valid).sum(-1) / denom
    if normalize_by != "seq_len":
        raise ValueError(f"normalize_by must be 'seq_len' or 'supervised', got {normalize_by!r}")
    return loss.mean(-1)


def ForCausalLMLossBatchwise(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    shift_labels = shift_labels.to(logits.device)
    logits = logits.permute(0, 2, 1)
    loss = fixed_cross_entropy_batchwise(
        logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss


def top1_match_batchwise(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
):
    """Per-sample (correct, total) lists for student argmax vs gt label.

    Mirrors the next-token shift used by ForCausalLMLossBatchwise so the
    counted positions match those that contribute to the CE loss.
    """
    if shift_labels is None:
        shift_labels = nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    shift_labels = shift_labels.to(logits.device)
    student_top1 = logits.argmax(dim=-1)
    mask = shift_labels != ignore_index
    correct_per_sample = ((student_top1 == shift_labels) & mask).sum(dim=-1).tolist()
    total_per_sample = mask.sum(dim=-1).tolist()
    return correct_per_sample, total_per_sample


def kl_divergence_loss_topk(
    student_logits: torch.Tensor,
    teacher_logits_list: list,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ignore_index: int = -100,
    track_top1: bool = False,
):
    """Compute forward KL divergence KL(teacher || student) using top-k teacher logits.

    Args:
        student_logits: [batch, seq_len, vocab_size] student model logits
        teacher_logits_list: list of (topk_indices[k], topk_values[k]) tuples per sample,
                            or None for samples without teacher logits
        labels: [batch, seq_len] for identifying answer token positions
        temperature: temperature scaling for both teacher and student
        ignore_index: label value to ignore

    Returns:
        [batch_size] tensor of per-sample KL divergence losses
    """
    batch_size = student_logits.size(0)
    device = student_logits.device

    # Shift labels by 1 to align with next-token prediction (same as CE)
    shifted_labels = nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].to(device)

    per_sample_losses = torch.zeros(batch_size, device=device)
    correct_per_sample = [0] * batch_size
    total_per_sample = [0] * batch_size
    n_skip_none = 0
    n_skip_no_answer = 0
    n_skip_shape = 0
    n_used = 0
    n_ce_fallback = 0
    last_shape_mismatch = None

    for i in range(batch_size):
        if teacher_logits_list[i] is None:
            # No cached teacher for this sample — fall back to per-sample CE at
            # answer positions so the sample still contributes a gradient.
            answer_mask = shifted_labels[i] != ignore_index
            if not answer_mask.any():
                n_skip_none += 1
                continue
            answer_positions = answer_mask.nonzero(as_tuple=True)[0]
            student_at_pos = student_logits[i, answer_positions, :].float()
            target_at_pos = shifted_labels[i, answer_positions].to(torch.long)
            per_sample_losses[i] = nn.functional.cross_entropy(
                student_at_pos, target_at_pos, reduction="mean"
            )
            if track_top1:
                student_top1 = student_at_pos.argmax(dim=-1)
                correct_per_sample[i] = (student_top1 == target_at_pos).sum().item()
                total_per_sample[i] = int(answer_positions.numel())
            n_ce_fallback += 1
            continue

        topk_indices, topk_values = teacher_logits_list[i]
        topk_indices = topk_indices.to(device)
        topk_values = topk_values.to(device).float()

        # Find answer positions (where labels != ignore_index)
        answer_mask = shifted_labels[i] != ignore_index
        if not answer_mask.any():
            n_skip_no_answer += 1
            continue

        answer_positions = answer_mask.nonzero(as_tuple=True)[0]
        num_pos = len(answer_positions)

        # Teacher topk may be 1D [k] (legacy: one teacher distribution at the
        # first semantic answer position — for hint tasks, predicting the
        # letter after "Answer:") or 2D [T, k] (per-position distributions,
        # e.g. for freeform-rollout regularization).
        #
        # Align to student answer positions from the START: the cached teacher
        # at index j supervises answer_positions[j]. Extra student positions
        # (trailing chat-template tokens like <|im_end|>, \n for hint) are
        # dropped — they get no KL supervision, which is correct because the
        # cache has no teacher distribution for them. Broadcasting the letter
        # distribution onto those positions (the old 1D path) distorts the
        # student massively — see KL_LOSS_ISSUE.txt.
        if topk_values.dim() == 1:
            teacher_vals_full = topk_values.unsqueeze(0)      # [1, k]
            teacher_idx_full = topk_indices.unsqueeze(0)      # [1, k]
        else:
            teacher_vals_full = topk_values                   # [T_cache, k]
            teacher_idx_full = topk_indices                   # [T_cache, k]

        T_cache = teacher_vals_full.size(0)
        T_use = min(T_cache, num_pos)
        if T_cache != num_pos:
            # Record as diagnostic, but don't skip — truncation is the fix.
            n_skip_shape += 1
            last_shape_mismatch = (T_cache, num_pos)

        answer_positions_used = answer_positions[:T_use]
        teacher_at_pos = teacher_vals_full[:T_use] / temperature
        topk_indices_expanded = teacher_idx_full[:T_use]

        # Get student logits at the aligned subset: [T_use, vocab_size]
        student_at_pos = student_logits[i, answer_positions_used, :].float() / temperature
        student_log_probs = nn.functional.log_softmax(student_at_pos, dim=-1)

        teacher_probs = nn.functional.softmax(teacher_at_pos, dim=-1)
        student_log_probs_at_topk = student_log_probs.gather(1, topk_indices_expanded)

        # KL(teacher || student) = sum(teacher_prob * (log(teacher_prob) - student_log_prob))
        teacher_log_probs = torch.log(teacher_probs + 1e-10)
        kl_per_pos = (teacher_probs * (teacher_log_probs - student_log_probs_at_topk)).sum(dim=-1)

        per_sample_losses[i] = kl_per_pos.mean()

        # Top-1 agreement: student argmax over full vocab vs teacher argmax
        # (teacher argmax restricted to its cached top-k, which is the only
        # information available — sufficient because teacher top-1 is always
        # within its own top-k by definition).
        if track_top1:
            student_top1 = student_at_pos.argmax(dim=-1)
            teacher_top1_local = teacher_at_pos.argmax(dim=-1)
            teacher_top1 = topk_indices_expanded.gather(
                1, teacher_top1_local.unsqueeze(1)
            ).squeeze(1)
            correct_per_sample[i] = (student_top1 == teacher_top1).sum().item()
            total_per_sample[i] = int(T_use)
        n_used += 1

    # One-time diagnostic so we can see the skip/truncation distribution in
    # the training log. `shape_truncated` counts samples where cache_T !=
    # student_num_pos and we aligned on min(T_cache, num_pos) — typical for
    # the 1D-cache hint path where T_cache=1 and num_pos=3.
    if not getattr(kl_divergence_loss_topk, "_diag_done", False):
        print(
            f"[KL DIAG] batch={batch_size} used={n_used} ce_fallback={n_ce_fallback} "
            f"skip_none={n_skip_none} skip_no_answer={n_skip_no_answer} "
            f"shape_truncated={n_skip_shape} "
            f"last_shape_mismatch(teacher_T,student_num_pos)={last_shape_mismatch}",
            flush=True,
        )
        kl_divergence_loss_topk._diag_done = True

    if n_used == 0 and n_ce_fallback == 0:
        # Every sample was skipped — per_sample_losses is a non-grad zero tensor
        # and would break backward. Connect it to student_logits via a zero
        # multiplier so grad flows. Value stays zero.
        per_sample_losses = per_sample_losses + (student_logits * 0).sum()

    return per_sample_losses, correct_per_sample, total_per_sample


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    if num_items_in_batch is not None:
        num_items_in_batch = num_items_in_batch.to(logits.device)
    loss = fixed_cross_entropy(
        logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss


class ContinuousCausalLMBase:
    """
    Base class for the causal language model wrappers that provides shared
    loss computation (text/causal CE, KL regularization, question-type
    weighting, top-1 match tracking) for all model families.
    """

    def __init__(
        self,
        batch_size: int = 24,
        subject_embed_dim: int | None = None,
        use_embed_proj: bool = True,
        target_model_layers: int | None = None,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.debug = False

        # KL regularization parameters (set via train.py after model creation)
        self.use_kl_regularization = kwargs.get("use_kl_regularization", False)
        self.kl_temperature = kwargs.get("kl_temperature", 1.0)
        # CE loss normalization: "seq_len" (legacy: divide by seq_len) or
        # "supervised" (divide by num supervised tokens). Set via train.py.
        self.loss_normalize_by = kwargs.get("loss_normalize_by", "seq_len")

        # Embed projection for different subject dimensions
        if use_embed_proj or (
            subject_embed_dim is not None
            and subject_embed_dim != self.config.hidden_size
        ):
            if target_model_layers is None:
                target_model_layers = self.config.num_hidden_layers
            # W_subject -> W_predictor
            self.embed_projs = nn.ModuleList(
                [
                    nn.Linear(
                        subject_embed_dim,
                        self.config.hidden_size,
                        bias=False,
                    )
                    for _ in range(target_model_layers)
                ]
            )
            for embed_proj in self.embed_projs:
                embed_proj.requires_grad_(True)  # type: ignore
        else:
            self.embed_projs = None

    def train(self, is_training: bool = True):
        super().train(is_training)
        if is_training:
            self._loss_function = ForCausalLMLoss
            self.is_training = True
        else:
            self._loss_function = ForCausalLMLossBatchwise
            self.is_training = False
        return self

    def eval(self):
        super().eval()
        self._loss_function = ForCausalLMLossBatchwise
        return self

    def _apply_question_type_weights(
        self,
        losses: torch.Tensor,
        question_types: List[str],
        question_config: dict,
    ) -> torch.Tensor:
        """
        Apply question type weights to individual losses.

        Args:
            losses: Tensor of individual losses [batch_size]
            question_types: List of question types for each batch item
            question_config: Question configuration containing weights

        Returns:
            Weighted losses tensor
        """
        if question_types is None or question_config is None:
            return losses

        # Apply weights to each loss based on its question type
        weighted_losses = losses.clone()
        for i, question_type in enumerate(question_types):
            if i < len(losses) and question_type in question_config.get(
                "question_types", {}
            ):
                weight = question_config["question_types"][question_type].get(
                    "weight", 1.0
                )
                weighted_losses[i] = losses[i] * weight

        return weighted_losses

    def _get_layer_activations(self, input_ids, attention_mask, target_layer):
        """
        Get model activations at a specific layer by running partial forward pass.
        Must be implemented by each model subclass (Llama, Gemma, Qwen).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            target_layer: Layer index to extract activations from

        Returns:
            Layer activations [batch_size, seq_len, hidden_size]
        """
        raise NotImplementedError(
            "Each model subclass must implement _get_layer_activations"
        )

    def _register_intervention_hooks(
        self,
        intervention_mask,
        intervention_layers,
        intervention_positions,
        intervention_vectors,
    ):
        """
        Register forward hooks to apply causal interventions during model forward pass.

        Args:
            intervention_mask: Boolean tensor indicating which batch items need interventions
            intervention_layers: List of layer indices for each batch item
            intervention_positions: List of token positions for each batch item
            intervention_vectors: List of intervention vector info for each batch item

        Returns:
            List of hook handles that can be removed later
        """
        hooks = []

        # Group interventions by layer for efficiency
        layer_to_interventions = {}
        for batch_idx in range(len(intervention_mask)):
            if (
                intervention_mask[batch_idx]
                and intervention_vectors[batch_idx] is not None
            ):
                layer = intervention_layers[batch_idx]
                if layer not in layer_to_interventions:
                    layer_to_interventions[layer] = []
                layer_to_interventions[layer].append(
                    {
                        "batch_idx": batch_idx,
                        "positions": intervention_positions[batch_idx],
                        "vector_info": intervention_vectors[batch_idx],
                    }
                )

        # Register hooks for each layer that needs interventions
        for layer_idx, interventions in layer_to_interventions.items():

            def make_hook(layer_idx, interventions):
                def intervention_hook(module, input, output):
                    # output is the hidden states tensor [batch_size, seq_len, hidden_size]
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Apply interventions for each batch item
                    for intervention in interventions:
                        batch_idx = intervention["batch_idx"]
                        positions = intervention["positions"]
                        vector_info = intervention["vector_info"]

                        # Apply intervention using the helper method
                        intervention_info = vector_info.copy()
                        intervention_info["positions"] = positions

                        # Apply intervention to this batch item's hidden states
                        hidden_states[batch_idx : batch_idx + 1] = (
                            self._apply_causal_intervention(
                                hidden_states[batch_idx : batch_idx + 1],
                                intervention_info,
                                layer_idx,
                            )
                        )

                    # Return modified output in the same format
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states

                return intervention_hook

            # Register the hook on the appropriate layer
            layer_module = self.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(
                make_hook(layer_idx, interventions)
            )
            hooks.append(hook)

        return hooks

    def _apply_causal_intervention(
        self, hidden_states: torch.Tensor, intervention_info: dict, layer_idx: int
    ) -> torch.Tensor:
        """
        Apply causal intervention to hidden states at specified layer and positions.

        Args:
            hidden_states: Layer hidden states [batch_size, seq_len, hidden_size]
            intervention_info: Dict containing intervention details
            layer_idx: Current layer index

        Returns:
            Modified hidden states with intervention applied
        """
        if intervention_info is None or intervention_info.get("layer") != layer_idx:
            return hidden_states

        # No remaining intervention type modifies hidden states here.
        return hidden_states.clone()

    @property
    def loss_function(self):
        """Get the appropriate loss function based on training mode."""
        return getattr(self, "_loss_function", ForCausalLMLoss)

    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        model = super().from_pretrained(model_path, *args, **kwargs)
        model.load_embed_proj(model_path)
        return model

    def save_embed_proj(self, save_directory: str, safe_serialization: bool = True):
        """Save embed_proj weights to a file."""
        if self.embed_projs is not None:
            # Save all layers instead of just the first one
            all_embed_proj_states = {}
            for i, embed_proj in enumerate(self.embed_projs):
                # Flatten the state dict for each layer
                layer_state_dict = embed_proj.state_dict()
                for key, value in layer_state_dict.items():
                    all_embed_proj_states[f"layer_{i}.{key}"] = value

            if safe_serialization:
                embed_proj_path = os.path.join(save_directory, "embed_proj.safetensors")
                safe_save_file(
                    all_embed_proj_states,
                    embed_proj_path,
                    metadata={"format": "pt"},
                )
            else:
                embed_proj_path = os.path.join(save_directory, "embed_proj.pt")
                torch.save(all_embed_proj_states, embed_proj_path)

            print(
                f"Saved embed_proj weights for {len(self.embed_projs)} layers to {embed_proj_path}"
            )
            return True
        return False

    def load_embed_proj(self, load_directory: str):
        """Load embed_proj weights from a file."""
        if self.embed_projs is not None:
            embed_proj_safetensors_path = os.path.join(
                load_directory, "embed_proj.safetensors"
            )
            embed_proj_pt_path = os.path.join(load_directory, "embed_proj.pt")

            if os.path.exists(embed_proj_safetensors_path):
                all_embed_proj_states = load_file(embed_proj_safetensors_path)
                # Load all layers
                for i, embed_proj in enumerate(self.embed_projs):
                    # Reconstruct the state dict for this layer
                    layer_state_dict = {}
                    layer_prefix = f"layer_{i}."
                    for key, value in all_embed_proj_states.items():
                        if key.startswith(layer_prefix):
                            # Remove the layer prefix to get the original key
                            original_key = key[len(layer_prefix) :]
                            layer_state_dict[original_key] = value

                    if layer_state_dict:
                        embed_proj.load_state_dict(layer_state_dict)
                    else:
                        print(f"Warning: No weights found for layer {i}")
                print(
                    f"Loaded embed_proj weights for {len(self.embed_projs)} layers from {embed_proj_safetensors_path}"
                )
                return True
            elif os.path.exists(embed_proj_pt_path):
                all_embed_proj_states = torch.load(embed_proj_pt_path)
                # Load all layers
                for i, embed_proj in enumerate(self.embed_projs):
                    # Reconstruct the state dict for this layer
                    layer_state_dict = {}
                    layer_prefix = f"layer_{i}."
                    for key, value in all_embed_proj_states.items():
                        if key.startswith(layer_prefix):
                            # Remove the layer prefix to get the original key
                            original_key = key[len(layer_prefix) :]
                            layer_state_dict[original_key] = value

                    if layer_state_dict:
                        embed_proj.load_state_dict(layer_state_dict)
                    else:
                        print(f"Warning: No weights found for layer {i}")
                print(
                    f"Loaded embed_proj weights for {len(self.embed_projs)} layers from {embed_proj_pt_path}"
                )
                return True
        return False

    def _get_model_hidden_states(self, outputs):
        """Get hidden states from model outputs - handles different model types."""
        # Llama uses outputs[0], while Gemma/Qwen use outputs.last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        else:
            return outputs[0]

    def _apply_logit_postprocessing(self, logits):
        """Apply model-specific logit postprocessing (e.g., Gemma's softcapping)."""
        # Gemma-specific softcapping
        if (
            hasattr(self.config, "final_logit_softcapping")
            and self.config.final_logit_softcapping is not None
        ):
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits

    def shared_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,  # Llama parameter name
        logits_to_keep=None,  # Gemma/Qwen parameter name
        activations_mask=None,
        debug=False,
        extra_args=None,
        # Causal intervention parameters
        intervention_mask=None,
        intervention_layers=None,
        intervention_positions=None,
        intervention_vectors=None,
        teacher_logits_list=None,
        **kwargs,
    ):
        """Shared forward method implementation for all continuous models."""
        # Normalize parameter names - use logits_to_keep consistently
        if logits_to_keep is None:
            logits_to_keep = num_logits_to_keep

        if "num_items_in_batch" in kwargs:
            # kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"].item()
            del kwargs["num_items_in_batch"]

        # Forward the model-level CE normalization setting into every
        # ForCausalLMLossBatchwise call below via **kwargs.
        kwargs.setdefault("normalize_by", getattr(self, "loss_normalize_by", "seq_len"))

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inputs_embeds = self.model.embed_tokens(input_ids)

        # Check if we need to apply causal interventions
        has_interventions = (
            intervention_mask is not None
            and intervention_mask.any()
            and intervention_layers is not None
            and intervention_vectors is not None
        )

        # Apply intervention hooks if needed
        intervention_hooks = []
        if has_interventions:
            intervention_hooks = self._register_intervention_hooks(
                intervention_mask,
                intervention_layers,
                intervention_positions,
                intervention_vectors,
            )

        # Run the model
        try:
            outputs = self.model(
                input_ids=None if inputs_embeds is not None else input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        except Exception as e:
            import sys
            import traceback
            print(f"Error in model forward pass: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            if sys.stdin.isatty():
                # Interactive — drop to pdb for debugging.
                raise
            raise e
        finally:
            # Always remove hooks to prevent memory leaks
            for hook in intervention_hooks:
                hook.remove()

        # Get hidden states (handles different model types)
        last_hidden_states = self._get_model_hidden_states(outputs)
        # layers, batch, seq_len, hidden_size
        # hidden_states = torch.stack(
        #     [state.to("cpu") for state in outputs.hidden_states],
        #     dim=0,
        # )

        # Compute logits with proper slicing
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        if slice_indices is not None and not isinstance(slice_indices, slice):
            if slice_indices.max() > last_hidden_states.shape[1]:
                raise ValueError("logits_to_keep index exceeds the sequence length")
        logits = self.lm_head(last_hidden_states[:, slice_indices, :])

        # Apply model-specific postprocessing (e.g., Gemma softcapping)
        logits = self._apply_logit_postprocessing(logits)


        # Extract question type weights from batch for loss weighting
        question_type_weights = kwargs.get("question_type_weights", None)

        # Compute losses with activations_mask filtering
        loss = None
        text_loss = None
        causal_loss = None
        regularization_loss = None
        text_kl_loss = None
        causal_kl_loss = None
        kl_loss = None
        # Per-sample top-1 match counts (student argmax vs supervision target).
        # Gated by self.log_top1_match (YAML/CLI). Filled by each loss branch
        # using its sub-batch's helper return; later aggregated by
        # kwargs["question_type"] for wandb logging.
        log_top1_match = getattr(self, "log_top1_match", False)
        batch_size_full = logits.size(0)
        batch_correct = [0] * batch_size_full
        batch_total = [0] * batch_size_full
        if labels is not None:
            # Filter inputs/labels based on activations_mask
            if activations_mask is not None:
                # Get indices for text examples (activations_mask=False)
                text_mask = ~activations_mask

                # Compute regular loss only for text examples
                # Use sum() > 0 instead of .any() for Dynamo compatibility
                if text_mask.any():
                    try:
                        text_logits = logits[text_mask.to(logits.device)]
                    except Exception as e:
                        raise
                        raise e
                    text_labels = labels[text_mask.to(labels.device)].to(torch.long)
                    if text_logits.size(0) > 0:
                        # Use KL divergence if teacher logits available, else CE
                        text_used_kl = False
                        text_sub_correct = None
                        text_sub_total = None
                        if (
                            self.use_kl_regularization
                            and teacher_logits_list is not None
                        ):
                            text_teacher = [
                                teacher_logits_list[i]
                                for i, m in enumerate(text_mask)
                                if m
                            ]
                            if any(t is not None for t in text_teacher):
                                (
                                    text_losses_per_sample,
                                    text_sub_correct,
                                    text_sub_total,
                                ) = kl_divergence_loss_topk(
                                    text_logits,
                                    text_teacher,
                                    text_labels,
                                    temperature=self.kl_temperature,
                                    track_top1=log_top1_match,
                                )
                                text_used_kl = True
                            else:
                                text_losses_per_sample = ForCausalLMLossBatchwise(
                                    logits=text_logits,
                                    labels=text_labels,
                                    vocab_size=self.config.vocab_size,
                                    **kwargs,
                                )
                                if log_top1_match:
                                    text_sub_correct, text_sub_total = top1_match_batchwise(
                                        text_logits, text_labels
                                    )
                        else:
                            text_losses_per_sample = ForCausalLMLossBatchwise(
                                logits=text_logits,
                                labels=text_labels,
                                vocab_size=self.config.vocab_size,
                                **kwargs,
                            )
                            if log_top1_match:
                                text_sub_correct, text_sub_total = top1_match_batchwise(
                                    text_logits, text_labels
                                )

                        # Scatter sub-batch top-1 stats back to full-batch indices
                        if text_sub_correct is not None:
                            text_indices = text_mask.nonzero(as_tuple=True)[0].tolist()
                            for j, idx in enumerate(text_indices):
                                batch_correct[idx] = text_sub_correct[j]
                                batch_total[idx] = text_sub_total[j]

                        # Apply question type weights if available
                        if question_type_weights is not None:
                            text_weights = question_type_weights[text_mask].to(
                                text_losses_per_sample.device
                            )
                            weighted_text_losses = text_losses_per_sample * text_weights
                            text_loss = weighted_text_losses.mean()
                        else:
                            text_loss = text_losses_per_sample.mean()

                        if text_used_kl:
                            text_kl_loss = text_loss

                        loss = text_loss

                # Check if we have causal intervention examples (change_match)
                intervention_batch_mask = torch.zeros_like(activations_mask)
                if (
                    hasattr(kwargs, "get")
                    and kwargs.get("intervention_mask") is not None
                ):
                    intervention_batch_mask = kwargs["intervention_mask"]
                elif intervention_mask is not None:
                    intervention_batch_mask = intervention_mask

                # For causal intervention examples, apply standard language modeling loss
                if intervention_batch_mask.any():
                    causal_logits = logits[intervention_batch_mask.to(logits.device)]
                    causal_labels = labels[
                        intervention_batch_mask.to(labels.device)
                    ].to(torch.long)
                    if causal_logits.size(0) > 0:
                        # Use KL divergence if teacher logits available, else CE
                        causal_used_kl = False
                        causal_sub_correct = None
                        causal_sub_total = None
                        if (
                            self.use_kl_regularization
                            and teacher_logits_list is not None
                        ):
                            causal_teacher = [
                                teacher_logits_list[i]
                                for i, m in enumerate(intervention_batch_mask)
                                if m
                            ]
                            if any(t is not None for t in causal_teacher):
                                (
                                    causal_losses_per_sample,
                                    causal_sub_correct,
                                    causal_sub_total,
                                ) = kl_divergence_loss_topk(
                                    causal_logits,
                                    causal_teacher,
                                    causal_labels,
                                    temperature=self.kl_temperature,
                                    track_top1=log_top1_match,
                                )
                                causal_used_kl = True
                            else:
                                causal_losses_per_sample = ForCausalLMLossBatchwise(
                                    logits=causal_logits,
                                    labels=causal_labels,
                                    vocab_size=self.config.vocab_size,
                                    **kwargs,
                                )
                                if log_top1_match:
                                    causal_sub_correct, causal_sub_total = top1_match_batchwise(
                                        causal_logits, causal_labels
                                    )
                        else:
                            causal_losses_per_sample = ForCausalLMLossBatchwise(
                                logits=causal_logits,
                                labels=causal_labels,
                                vocab_size=self.config.vocab_size,
                                **kwargs,
                            )
                            if log_top1_match:
                                causal_sub_correct, causal_sub_total = top1_match_batchwise(
                                    causal_logits, causal_labels
                                )

                        # Scatter sub-batch top-1 stats back to full-batch indices
                        if causal_sub_correct is not None:
                            causal_indices = intervention_batch_mask.nonzero(
                                as_tuple=True
                            )[0].tolist()
                            for j, idx in enumerate(causal_indices):
                                batch_correct[idx] = causal_sub_correct[j]
                                batch_total[idx] = causal_sub_total[j]

                        # Apply question type weights if available
                        if question_type_weights is not None:
                            causal_weights = question_type_weights[
                                intervention_batch_mask
                            ].to(causal_losses_per_sample.device)
                            weighted_causal_losses = (
                                causal_losses_per_sample * causal_weights
                            )
                            causal_loss = weighted_causal_losses.mean()
                        else:
                            causal_loss = causal_losses_per_sample.mean()

                        if causal_used_kl:
                            causal_kl_loss = causal_loss

                        # Track intervention regularization loss separately
                        regularization_loss = causal_loss
                        if text_loss is not None:
                            regularization_loss = regularization_loss + text_loss

                        if loss is not None:
                            loss += causal_loss
                        else:
                            loss = causal_loss
            else:
                # Original behavior when no activations_mask is provided
                # Use batchwise loss to get per-sample losses for weighting
                losses_per_sample = ForCausalLMLossBatchwise(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    **kwargs,
                )
                if log_top1_match:
                    fb_correct, fb_total = top1_match_batchwise(logits, labels)
                    for idx in range(batch_size_full):
                        batch_correct[idx] = fb_correct[idx]
                        batch_total[idx] = fb_total[idx]

                # Apply question type weights if available
                if question_type_weights is not None:
                    weights = question_type_weights.to(losses_per_sample.device)
                    weighted_losses = losses_per_sample * weights
                    loss = weighted_losses.mean()
                else:
                    loss = losses_per_sample.mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        kl_components = [x for x in (text_kl_loss, causal_kl_loss) if x is not None]
        if kl_components:
            kl_loss = sum(kl_components) / len(kl_components)

        # When KL is used as the training regularizer but no causal-intervention
        # branch fires (e.g. judge_verdict, refusal_compliance_reg), the
        # `regularization_loss` assignment above never runs. Surface the KL
        # component here so it shows up in LossLoggingCallback.
        if regularization_loss is None and kl_loss is not None:
            regularization_loss = kl_loss

        # Aggregate per-sample top-1 match counts by question_type for wandb.
        # `question_type` arrives as a top-level batch list via kwargs (set by
        # BaseDataCollator from each example's "question_type" field).
        top1_match_by_qt = None
        question_types_list = kwargs.get("question_type", None)
        if (
            log_top1_match
            and question_types_list is not None
            and any(t > 0 for t in batch_total)
        ):
            top1_match_by_qt = {}
            for i, qt in enumerate(question_types_list):
                if i >= len(batch_total) or batch_total[i] == 0 or qt is None:
                    continue
                # Per-sample exact match: 1 iff every supervision token matches.
                sample_correct = 1 if batch_correct[i] == batch_total[i] else 0
                c_acc, t_acc = top1_match_by_qt.get(qt, (0, 0))
                top1_match_by_qt[qt] = (
                    c_acc + sample_correct,
                    t_acc + 1,
                )
            if not top1_match_by_qt:
                top1_match_by_qt = None

        # Create the output object
        output_obj = ContinuousCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            text_loss=text_loss,
            causal_loss=causal_loss,
            regularization_loss=regularization_loss,
            kl_loss=kl_loss,
            top1_match_by_qt=top1_match_by_qt,
        )

        # Save outputs for logging callback
        self.last_outputs = output_obj

        # Stash per-row inputs needed by the rollout-consistency callback (gated
        # by `log_explanation_behavior_match` to avoid memory cost when off).
        # For orig/change rows we precompute the predicted token id at the
        # supervision (answer letter) position; for gen_explanation rows we
        # carry the prompt-only tensors so the callback can run autoregressive
        # generate and (if needed) on-the-fly forwards on stashed orig/change
        # prompts to fill any orphans.
        if getattr(self, "log_explanation_behavior_match", False) and labels is not None:
            shifted_labels = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
            pred_token_ids: List[Optional[int]] = []
            with torch.no_grad():
                for i in range(logits.size(0)):
                    sup_pos = (shifted_labels[i] != -100).nonzero(as_tuple=True)[0]
                    if sup_pos.numel() == 0:
                        pred_token_ids.append(None)
                    else:
                        # First supervised position is the answer letter for
                        # orig/change rows (single-token supervision).
                        pred_token_ids.append(int(logits[i, sup_pos[0], :].argmax().item()))
            self.last_inputs = {
                "input_ids_without_answer": kwargs.get("input_ids_without_answer"),
                "attention_mask_without_answer": kwargs.get("attention_mask_without_answer"),
                "question_type": list(kwargs.get("question_type", [])),
                "extra_args": list(extra_args) if extra_args is not None else [],
                "pred_token_ids": pred_token_ids,
            }
        else:
            self.last_inputs = None

        return output_obj
