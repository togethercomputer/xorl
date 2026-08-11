from typing import Callable, Literal, Optional, Tuple, Unpack

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
from xorl.models.layers import ACT2FN, RotaryEmbedding
from xorl.models.layers.attention import (
    AttentionKwargs,
    is_flash_attention,
    update_causal_mask,
)
from xorl.models.layers.attention.backend import get_attention_fn
from xorl.models.layers.fused_projection_lora import project_fused_linear_with_lora
from xorl.models.layers.normalization import (
    compiled_zero_centered_rms_norm,
    eager_zero_centered_rms_norm,
    fast_zero_centered_batch_invariant_residual_rms_norm,
    fast_zero_centered_batch_invariant_rms_norm,
    fast_zero_centered_families_v2_rms_norm,
    get_rmsnorm_mode,
    native_zero_centered_rms_norm,
    native_zero_centered_rms_norm_without_batch_invariant,
)
from xorl.models.module_utils import GradientCheckpointingLayer
from xorl.models.outputs import BaseModelOutput, CausalLMOutput
from xorl.models.transformers.qwen3_5 import parallelize
from xorl.models.transformers.qwen3_5.checkpoint_handler import Qwen3_5CheckpointHandler
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5_shared import (
    LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE,
    QWEN3_5_CHECKPOINT_CONVERSION_MAPPING,
    QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS,
    _apply_qwen35_gdn_exact,
    has_linear_attention_layers,
    qwen3_5_apply_rotary_pos_emb,
)
from xorl.ops.fused_silu_and_mul import fused_silu_and_mul
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.utils import logging


logger = logging.get_logger(__name__)


def _adapt_qwen3_5_config(config):
    exact_contract = bool(getattr(config, "_qwen35_exact_contract", False))
    rmsnorm_family = getattr(config, "_qwen35_rmsnorm_family", "v1")
    if hasattr(config, "text_config"):
        adapted = Qwen3_5Config.from_hf_config(config)
    elif isinstance(config, Qwen3_5Config):
        adapted = config
    elif getattr(config, "model_type", None) in {"qwen3_5", "qwen3_5_text"}:
        adapted = Qwen3_5Config.from_hf_config(config)
    else:
        adapted = config
    adapted._qwen35_exact_contract = exact_contract
    adapted._qwen35_rmsnorm_family = rmsnorm_family
    return adapted


def _raise_if_ring_fla_unsupported(config: Qwen3_5Config, ps) -> None:
    if ps.ringattn_size > 1 and has_linear_attention_layers(config):
        logger.warning_once(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)
        raise ValueError(LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE)


class Qwen3_5MLP(nn.Module):
    _supports_fused_gate_up_lora = True

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
            gate_up = project_fused_linear_with_lora(
                self,
                x,
                base_name="gate_up_proj",
                projection_names=("gate_proj", "up_proj"),
                projection_sizes=(self.intermediate_size, self.intermediate_size),
            )
            if self._use_fused_silu:
                x = fused_silu_and_mul(gate_up)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                x = self.act_fn(gate) * up
        else:
            x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)


class Qwen3_5RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        exact_contract: bool = False,
        rmsnorm_family: Literal["v1", "v2"] = "v1",
    ):
        super().__init__()
        if rmsnorm_family not in ("v1", "v2"):
            raise ValueError(f"Unsupported Qwen3.5 RMSNorm family: {rmsnorm_family!r}")
        if rmsnorm_family == "v2" and not exact_contract:
            raise RuntimeError("Qwen families-v2 RMSNorm is admitted only in the exact training lane.")
        self.eps = eps
        self.exact_contract = exact_contract
        self.rmsnorm_family = rmsnorm_family
        self.weight = nn.Parameter(torch.zeros(dim))
        self.mode = get_rmsnorm_mode()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        force_sglang_residual: bool = False,
    ):
        if self.exact_contract and self.rmsnorm_family == "v2":
            if self.mode != "sglang_fused":
                raise RuntimeError(
                    f"The Qwen families-v2 RMSNorm program requires rmsnorm_mode='sglang_fused'; got {self.mode!r}."
                )
            if residual is None:
                out = fast_zero_centered_families_v2_rms_norm(x, self.weight, self.eps)
                residual_out = None
            else:
                out, residual_out = fast_zero_centered_families_v2_rms_norm(
                    x,
                    self.weight,
                    self.eps,
                    residual=residual,
                )
            if residual_out is not None and prenorm:
                return out, residual_out
            return out

        residual_out: Optional[torch.Tensor] = None
        norm_input = x
        if residual is not None:
            residual_out = x + residual
            norm_input = residual_out

        if self.mode == "eager":
            out = eager_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        elif self.mode == "native":
            out = native_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        elif self.mode == "compile":
            out = compiled_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        elif self.mode in ("sglang", "sglang_fused"):
            # Qwen3.5 uses two serving norm families. Residual-tree norms
            # (layer>0 input / post-attention / final) use the BI-mean
            # composition, while qk norms and the layer-0 input norm use the
            # no-residual family-1 kernel.
            if residual_out is not None or force_sglang_residual:
                if self.exact_contract:
                    out = fast_zero_centered_batch_invariant_residual_rms_norm(norm_input, self.weight, self.eps)
                else:
                    out = native_zero_centered_rms_norm_without_batch_invariant(norm_input, self.weight, self.eps)
            elif self.mode == "sglang_fused" and self.exact_contract:
                out = fast_zero_centered_batch_invariant_rms_norm(norm_input, self.weight, self.eps)
            else:
                out = native_zero_centered_rms_norm(norm_input, self.weight, self.eps)
        else:
            raise NotImplementedError(f"Unsupported rmsnorm_mode for Qwen3.5 RMSNorm: {self.mode}")

        if residual_out is not None and prenorm:
            return out, residual_out
        return out


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
        exact_contract = bool(getattr(config, "_qwen35_exact_contract", False))
        rmsnorm_family = getattr(config, "_qwen35_rmsnorm_family", "v1")
        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, exact_contract=exact_contract, rmsnorm_family=rmsnorm_family
        )
        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, exact_contract=exact_contract, rmsnorm_family=rmsnorm_family
        )
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
            class_b=bool(getattr(self.config, "_rope_class_b", False)),
        )
        if getattr(self.config, "_attention_cast_bf16", False):
            query_states = query_states.to(torch.bfloat16)
            key_states = key_states.to(torch.bfloat16)
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
        return get_attention_fn(self.config._attn_implementation)

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
        attn_output = attn_strategy.compute_attention(
            self, query_states, key_states, value_states, attention_mask, **kwargs
        )
        attn_output = attn_strategy.project_output(self, attn_output)
        return attn_output, None


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
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
                exact_contract=bool(getattr(config, "_qwen35_exact_contract", False)),
            )
        else:
            self.self_attn = Qwen3_5Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3_5MLP(config)
        exact_contract = bool(getattr(config, "_qwen35_exact_contract", False))
        rmsnorm_family = getattr(config, "_qwen35_rmsnorm_family", "v1")
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            exact_contract=exact_contract,
            rmsnorm_family=rmsnorm_family,
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            exact_contract=exact_contract,
            rmsnorm_family=rmsnorm_family,
        )
        if config.sliding_window and not is_flash_attention(config._attn_implementation):
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

        hidden_states = self.input_layernorm(
            hidden_states,
            force_sglang_residual=self.layer_idx > 0 and self.input_layernorm.mode in ("sglang", "sglang_fused"),
        )

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
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual=residual, prenorm=True)
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
        unfused = getattr(self, "_unfused_for_tp", False)

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
            # Only the merges are skipped, never the handler: it also remaps the
            # GatedDeltaNet in_proj_qkv packing, regardless of how the MLP is stored.
            skip_gate_up_merge=unfused,
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
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            exact_contract=bool(getattr(config, "_qwen35_exact_contract", False)),
            rmsnorm_family=getattr(config, "_qwen35_rmsnorm_family", "v1"),
        )
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
        # output_hidden_states collects the per-layer residual-stream hidden states
        # (pre-final-norm), one entry per decoder layer in layer order. Used by the
        # multi-layer OPRD path; off by default (no collection, no extra memory).
        all_hidden_states = () if output_hidden_states else None

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

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.norm is not None:
            hidden_states = self.norm(
                hidden_states,
                force_sglang_residual=getattr(self.norm, "mode", None) in ("sglang", "sglang_fused"),
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
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

    def _apply_qwen35_gdn_exact(self) -> dict[str, int]:
        return _apply_qwen35_gdn_exact(self)

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

        return CausalLMOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM):
    """Text-only local implementation for HF Qwen3.5 wrapper configs."""


ModelClass = [Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration]

__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5Model",
    "Qwen3_5PreTrainedModel",
]
