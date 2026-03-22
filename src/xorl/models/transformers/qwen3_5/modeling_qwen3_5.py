from typing import Callable, Optional, Tuple, Unpack

import torch
from torch import nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.layers import ACT2FN, RotaryEmbedding
from xorl.models.layers.attention import (
    AttentionKwargs,
    is_flash_attention,
    update_causal_mask,
)
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.module_utils import GradientCheckpointingLayer
from xorl.models.outputs import BaseModelOutput, CausalLMOutput
from xorl.models.checkpoint_handlers.buffers import (
    detect_prequantized_checkpoint,
    detect_prequantized_block_fp8_checkpoint,
    get_prequantized_exclude_modules,
)
from xorl.models.transformers.qwen3_5_shared import (
    LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE,
    QWEN3_5_CHECKPOINT_CONVERSION_MAPPING,
    QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS,
    has_linear_attention_layers,
    qwen3_5_apply_rotary_pos_emb,
)
from xorl.models.transformers.qwen3_5.checkpoint_handler import Qwen3_5CheckpointHandler
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5 import parallelize
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.utils import logging


logger = logging.get_logger(__name__)


def _adapt_qwen3_5_config(config):
    if hasattr(config, "text_config"):
        return Qwen3_5Config.from_hf_config(config)
    if isinstance(config, Qwen3_5Config):
        return config
    if getattr(config, "model_type", None) in {"qwen3_5", "qwen3_5_text"}:
        return Qwen3_5Config.from_hf_config(config)
    return config


def _raise_if_ring_fla_unsupported(config: Qwen3_5Config, ps) -> None:
    if ps.ringattn_size > 1 and has_linear_attention_layers(config):
        logger.warning_once(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)
        raise ValueError(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self._use_fused_silu = config.hidden_act == "silu"

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


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = None
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._attn_gate: torch.Tensor | None = None

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        self._attn_gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = qwen3_5_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            interleaved=getattr(self.config, "mrope_interleaved", False),
        )
        return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        gate = self._attn_gate
        self._attn_gate = None
        if gate is None:
            raise RuntimeError("Qwen3.5 attention gate was not initialized before output projection.")
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)

    def _get_attention_fn(self) -> Callable:
        return ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

    def _attention_kwargs(self) -> dict:
        return dict(
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del position_ids, past_key_values
        attn_strategy = get_cp_strategy()
        query_states, key_states, value_states = attn_strategy.project_qkv(self, hidden_states, position_embeddings)
        attn_output = attn_strategy.compute_attention(self, query_states, key_states, value_states, attention_mask, **kwargs)
        attn_output = attn_strategy.project_output(self, attn_output)
        return attn_output, None


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "full_attention"
        self.self_attn = None
        self.linear_attn = None
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                expand_v=config.linear_value_head_dim / config.linear_key_head_dim,
                head_dim=config.linear_key_head_dim,
                num_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                mode="chunk",
                use_gate=config.attn_output_gate,
                use_short_conv=True,
                conv_size=config.linear_conv_kernel_dim,
                layer_idx=layer_idx,
                norm_eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = Qwen3_5Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if (
            config.sliding_window and not is_flash_attention(config._attn_implementation)
        ):
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

        if self.linear_attn is not None:
            linear_kwargs = {}
            if kwargs.get("cu_seq_lens_q") is not None:
                linear_kwargs["cu_seqlens"] = kwargs.get("cu_seq_lens_q")
            cp_context = build_linear_attention_cp_context(
                kwargs.get("cu_seq_lens_q"),
                conv1d_kernel_size=self.linear_attn.conv_size if self.linear_attn.use_short_conv else None,
            )
            if cp_context is not None:
                linear_kwargs["cp_context"] = cp_context
            linear_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
            if cp_context is not None:
                linear_mask = None
            hidden_states, self_attn_weights, _ = self.linear_attn(
                hidden_states=hidden_states,
                attention_mask=linear_mask,
                use_cache=False,
                **linear_kwargs,
            )
        else:
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3_5PreTrainedModel(XorlPreTrainedModel):
    config_class = Qwen3_5Config
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _checkpoint_conversion_mapping = QWEN3_5_CHECKPOINT_CONVERSION_MAPPING
    _checkpoint_skip_key_patterns = QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS

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
        elif isinstance(module, Qwen3_5RMSNorm):
            module.weight.data.zero_()
        elif isinstance(module, GatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, RotaryEmbedding):
            # Recompute inv_freq buffer from config (RotaryEmbedding has no reset_parameters)
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq = module.inv_freq

    def get_checkpoint_handler(self, **kwargs):
        # When unfused for TP, checkpoint keys (q_proj, k_proj, v_proj, gate_proj,
        # up_proj) already match the model's parameter names - no merging needed.
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
        return Qwen3_5CheckpointHandler(
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            linear_key_dim=self.config.linear_num_key_heads * self.config.linear_key_head_dim,
            linear_value_dim=self.config.linear_num_value_heads * self.config.linear_value_head_dim,
            skip_qkv_merge=True,
            is_prequantized=is_prequantized,
            exclude_modules=exclude_modules,
        )


class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    def __init__(self, config: Qwen3_5Config):
        config = _adapt_qwen3_5_config(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
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
        use_cache: bool | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[AttentionKwargs],
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.embed_tokens is not None:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        else:
            hidden_states = input_ids if inputs_embeds is None else inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        if use_cache is None:
            use_cache = False

        ps = get_parallel_state()
        _raise_if_ring_fla_unsupported(self.config, ps)

        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        causal_mask = update_causal_mask(
            self.config._attn_implementation,
            attention_mask,
            hidden_states,
            cache_position,
            sliding_window=None,
            is_training=self.training,
            output_attentions=output_attentions,
        )
        linear_attn_mask = attention_mask
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = get_cp_strategy().prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_key_value_heads,
        )

        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if decoder_layer is None:
                continue
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
        )


class Qwen3_5Model(Qwen3_5TextModel):
    pass


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config):
        config = _adapt_qwen3_5_config(config)
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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


class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM):
    """Text-only local implementation for HF Qwen3.5 wrapper configs."""


ModelClass = [Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration]

__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5Model",
    "Qwen3_5PreTrainedModel",
]
