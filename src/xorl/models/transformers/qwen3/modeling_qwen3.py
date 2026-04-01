from typing import Optional, Tuple, Unpack

import torch
from torch import nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.checkpoint_handlers.buffers import (
    detect_prequantized_block_fp8_checkpoint,
    detect_prequantized_checkpoint,
    get_prequantized_exclude_modules,
)
from xorl.models.layers import ACT2FN, RMSNorm, RotaryEmbedding
from xorl.models.layers.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    is_flash_attention,
    update_causal_mask,
)
from xorl.models.module_utils import GradientCheckpointingLayer
from xorl.models.outputs import BaseModelOutput, CausalLMOutput
from xorl.models.transformers.qwen3 import parallelize
from xorl.models.transformers.qwen3.checkpoint_handler import Qwen3CheckpointHandler
from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.utils import logging


logger = logging.get_logger(__name__)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self._use_fused_silu = config.hidden_act == "silu" and not getattr(config, "_activation_native", False)

    def unfuse_for_tp(self):
        """Replace fused gate_up_proj with separate gate_proj and up_proj for tensor parallelism."""
        device = self.gate_up_proj.weight.device
        dtype = self.gate_up_proj.weight.dtype
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device, dtype=dtype)
        del self.gate_up_proj

    def forward(self, x):
        if hasattr(self, "gate_up_proj"):
            if self._use_fused_silu:
                x = fused_silu_and_mul(self.gate_up_proj(x))
            else:
                gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
                x = self.act_fn(gate) * up
        else:
            x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)


class Qwen3Attention(MultiHeadAttention):
    """Qwen3 attention with per-layer sliding window control."""

    def _init_sliding_window(self, config):
        sliding_window = config.sliding_window
        if not (
            config.use_sliding_window
            and getattr(config, "sliding_window", None) is not None
            and self.layer_idx >= config.max_window_layers
        ):
            sliding_window = None
        return sliding_window


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.sliding_window and not is_flash_attention(
            config._attn_implementation
        ):  # diff with Llama is this warning
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3PreTrainedModel(XorlPreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3DecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, RotaryEmbedding):
            # Recompute inv_freq buffer from config (RotaryEmbedding has no reset_parameters)
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq = module.inv_freq

    def get_checkpoint_handler(self, **kwargs):
        # When unfused for TP, checkpoint keys (q_proj, k_proj, v_proj, gate_proj,
        # up_proj) already match the model's parameter names — no merging needed.
        if getattr(self, "_unfused_for_tp", False):
            return None

        weights_path = kwargs.get("weights_path", None)
        is_prequantized = detect_prequantized_checkpoint(weights_path)
        if not is_prequantized:
            is_prequantized = detect_prequantized_block_fp8_checkpoint(weights_path)

        # Use user-specified exclude_modules (stored by train.py) if available,
        # otherwise auto-detect from checkpoint config.
        exclude_modules = getattr(self, "_qlora_exclude_modules", None)
        if exclude_modules is None:
            exclude_modules = get_prequantized_exclude_modules(weights_path) if is_prequantized else set()

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        return Qwen3CheckpointHandler(
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            is_prequantized=is_prequantized,
            exclude_modules=exclude_modules,
            model=self if is_prequantized else None,
        )


class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Whether this attention impl handles causal masking internally (flash/sdpa)
        self._skip_causal_mask = is_flash_attention(config._attn_implementation)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[AttentionKwargs],
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # PP support: when embed_tokens is None, input is already hidden_states
        if self.embed_tokens is not None:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        else:
            # Middle/last PP stage: input_ids is actually hidden_states from previous stage
            hidden_states = input_ids if inputs_embeds is None else inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        if self._skip_causal_mask:
            causal_mask = None
        else:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            causal_mask = update_causal_mask(
                self.config._attn_implementation,
                attention_mask,
                hidden_states,
                cache_position,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
                output_attentions=output_attentions,
            )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # SP strategy handles slicing (sync: slice, async: keep full-length)
        ps = get_parallel_state()
        position_embeddings = get_cp_strategy(num_kv_heads=self.config.num_key_value_heads).prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_key_value_heads,
        )

        # decoder layers
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if decoder_layer is None:  # PP: pruned layer
                continue
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # PP support: norm may be None on non-last stages
        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(AttentionKwargs): ...


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def unfuse_for_tp(self):
        """Unfuse all fused projections for tensor parallelism compatibility."""
        parallelize.unfuse_for_tp(self)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_pp_module_config(self):
        """Return PP module config for pipeline_module_split."""
        return {
            "input_fqns": ["model.embed_tokens"],
            "layer_prefix": "model.layers",
            "output_fqns": ["model.norm", "lm_head"],
            "always_keep_fqns": ["model.rotary_emb"],
            "num_layers": self.config.num_hidden_layers,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state

        return CausalLMOutput(last_hidden_state=last_hidden_state)


ModelClass = Qwen3ForCausalLM

__all__ = ["Qwen3ForCausalLM", "Qwen3Model", "Qwen3PreTrainedModel"]
