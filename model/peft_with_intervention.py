from typing import Any, Dict, List, Optional, Union

from peft import PeftModelForCausalLM


class PeftLlamaWithIntervention(PeftModelForCausalLM):
    """
    Extension of LlamaForCausalLM that allows for intervention in the hidden states
    during the forward pass.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 24,
        subject_embed_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.batch_size = batch_size
        self.subject_embed_dim = subject_embed_dim

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        return self.base_model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

    def train(self, is_training: bool = True):
        super().train(is_training)
        return self.base_model.train(is_training)

    def eval(self):
        super().eval()
        return self.base_model.eval()

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        """Override to save embed_proj weights alongside adapter weights."""
        # Call parent's save_pretrained
        super().save_pretrained(
            save_directory=save_directory,
            safe_serialization=safe_serialization,
            selected_adapters=selected_adapters,
            save_embedding_layers=True,
            is_main_process=is_main_process,
            **kwargs,
        )

        # Additionally save embed_proj if it exists (using base model's method)
        if is_main_process and hasattr(self.base_model, "save_embed_proj"):
            self.base_model.save_embed_proj(save_directory, safe_serialization)

    @classmethod
    def from_pretrained(
        cls,
        base_model,
        model_path,
        peft_config,
        adapter_name="default",
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=False,
        special_tokens=None,
        **kwargs,
    ):
        # Call parent class's from_pretrained
        model = super().from_pretrained(base_model, model_path)

        # Load embed_proj weights if the base model has the method
        if hasattr(model.base_model, "load_embed_proj"):
            model.base_model.load_embed_proj(model_path)

        # Convert to PeftLlamaWithIntervention
        new_model = cls(
            model.base_model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            special_tokens=special_tokens,
            **kwargs,
        )
        new_model.base_model = model.base_model
        return new_model
