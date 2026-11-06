from typing import Optional, Tuple, Union

import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache

from .continuous_base import ContinuousCausalLMBase, ContinuousCausalLMOutputWithPast


class ContinuousLlama(ContinuousCausalLMBase, LlamaForCausalLM):
    """
    Extension of LlamaForCausalLM that allows for intervention in the hidden states
    and attention head outputs during the forward pass.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 24,
        subject_embed_dim: int | None = None,
        use_embed_proj: bool = False,
        target_model_layers: int | None = None,
        **kwargs,
    ):
        LlamaForCausalLM.__init__(self, *args, **kwargs)
        ContinuousCausalLMBase.__init__(
            self,
            batch_size=batch_size,
            subject_embed_dim=subject_embed_dim,
            use_embed_proj=use_embed_proj,
            target_model_layers=target_model_layers,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        activations_mask: Optional[torch.BoolTensor] = None,
        debug: bool = False,
        extra_args: Optional[list[dict]] = None,
        **kwargs,
    ) -> Union[Tuple, ContinuousCausalLMOutputWithPast]:
        """
        """
        return self.shared_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            activations_mask=activations_mask,
            debug=debug,
            extra_args=extra_args,
            **kwargs,
        )

    def _get_layer_activations(self, input_ids, attention_mask, target_layer):
        """
        Llama-specific implementation to get activations at a specific layer.
        Based on the HF Llama forward implementation.
        """
        # Get embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Setup cache and position handling (following HF Llama implementation)
        past_key_values = None
        use_cache = False

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        from transformers.models.llama.modeling_llama import create_causal_mask

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Create position embeddings
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # Run through layers up to target_layer
        for layer_idx in range(target_layer + 1):
            decoder_layer = self.model.layers[layer_idx]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        return hidden_states
