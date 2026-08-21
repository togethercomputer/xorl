"""MiniMax M3 text model for xorl.

The HuggingFace checkpoint is a multimodal wrapper. This implementation owns
the language model tensors and exposes a tested text-only wrapper; image/video
inputs are rejected explicitly until the vision/projector stack is implemented.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Unpack

import torch
import torch.nn.functional as F
from torch import nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.layers import RotaryEmbedding
from xorl.models.layers.attention import AttentionKwargs, update_causal_mask
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.layers.moe import MoEBlock, MoEExperts
from xorl.models.layers.moe.router import _balanced_selected_experts, balanced_synthetic_routing
from xorl.models.module_utils import MoEGradientCheckpointingLayer
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.models.transformers.qwen3_5_shared import qwen3_5_apply_rotary_pos_emb
from xorl.ops.moe.activations import normalize_hidden_act

from . import parallelize
from .checkpoint_handler import MiniMaxM3CheckpointHandler
from .configuration_minimax_m3 import MiniMaxM3Config
from .msa_attention import minimax_msa_attention_forward


MINIMAX_M3_UNSUPPORTED_PARALLEL_MESSAGE = (
    "MiniMax M3 xorl support currently supports data/FSDP2 and expert parallelism only; "
    "tensor parallelism, pipeline parallelism, Ulysses, Ring, and lm-head TP are not supported yet."
)
_MULTIMODAL_KWARGS = {
    "pixel_values",
    "image_grid_thw",
    "images",
    "videos",
    "video_pixel_values",
    "video_grid_thw",
    "image_sizes",
}


def _adapt_minimax_m3_config(config):
    if isinstance(config, MiniMaxM3Config):
        return config
    if getattr(config, "model_type", None) in {"minimax_m3_vl", "xorl_minimax_m3"} or hasattr(config, "text_config"):
        return MiniMaxM3Config.from_hf_config(config)
    return config


def _raise_if_minimax_parallel_unsupported(ps=None) -> None:
    ps = ps or get_parallel_state()
    unsupported = (
        getattr(ps, "tp_size", 1) > 1
        or getattr(ps, "pp_size", 1) > 1
        or getattr(ps, "ringattn_size", 1) > 1
        or getattr(ps, "ulysses_size", 1) > 1
        or getattr(ps, "lm_head_tp_size", 1) > 1
    )
    if unsupported:
        raise ValueError(MINIMAX_M3_UNSUPPORTED_PARALLEL_MESSAGE)


def _reject_multimodal_kwargs(kwargs: dict) -> None:
    present = sorted(name for name in _MULTIMODAL_KWARGS if kwargs.get(name) is not None)
    if present:
        raise ValueError(
            "MiniMax M3 xorl support is text-only for now and rejects image/video inputs: " + ", ".join(present)
        )


def _reject_multimodal_tokens(config: MiniMaxM3Config, input_ids: torch.Tensor | None) -> None:
    if input_ids is None:
        return
    image_token = getattr(config, "image_token_index", None)
    video_token = getattr(config, "video_token_index", None)
    if image_token is not None and torch.any(input_ids == image_token):
        raise ValueError("MiniMax M3 xorl text-only mode rejects image token inputs.")
    if video_token is not None and torch.any(input_ids == video_token):
        raise ValueError("MiniMax M3 xorl text-only mode rejects video token inputs.")


def minimax_m3_swigluoai(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


class MiniMaxM3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_gemma_norm: bool = True):
        super().__init__()
        self.eps = eps
        self.use_gemma_norm = use_gemma_norm
        init = torch.zeros(dim) if use_gemma_norm else torch.ones(dim)
        self.weight = nn.Parameter(init)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
    ):
        residual_out: Optional[torch.Tensor] = None
        norm_input = x
        if residual is not None:
            residual_out = x + residual
            norm_input = residual_out

        variance = norm_input.float().pow(2).mean(dim=-1, keepdim=True)
        out = norm_input * torch.rsqrt(variance + self.eps).to(norm_input.dtype)
        weight = (1.0 + self.weight) if self.use_gemma_norm else self.weight
        out = out * weight
        if residual_out is not None and prenorm:
            return out, residual_out
        return out


class MiniMaxM3PerHeadRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, use_gemma_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.use_gemma_norm = use_gemma_norm
        init = torch.zeros(num_heads * head_dim) if use_gemma_norm else torch.ones(num_heads * head_dim)
        self.weight = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        weight = self.weight.view(self.num_heads, self.head_dim)
        if self.use_gemma_norm:
            weight = 1.0 + weight
        return out * weight


class MiniMaxM3MLP(nn.Module):
    def __init__(self, config: MiniMaxM3Config, intermediate_size: int | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.dense_intermediate_size
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(minimax_m3_swigluoai(gate, up, alpha=self.swiglu_alpha, limit=self.swiglu_limit))


class MiniMaxM3Router(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        use_routing_bias: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.use_routing_bias = use_routing_bias
        self.scoring_func = "sigmoid"
        self.synthetic_routing_mode = None

    def forward(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype,
        *,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(router_logits.float()).to(router_logits.dtype)
        if self.synthetic_routing_mode == "balanced":
            selected_experts = _balanced_selected_experts(router_logits, self.num_experts, self.top_k)
        elif expert_bias is not None:
            _, selected_experts = torch.topk(scores + expert_bias, self.top_k, dim=-1)
        else:
            _, selected_experts = torch.topk(scores, self.top_k, dim=-1)
        routing_weights = torch.gather(scores, dim=1, index=selected_experts)
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return routing_weights.to(input_dtype), selected_experts


class MiniMaxM3SparseMoeBlock(MoEBlock):
    def __init__(self, config: MiniMaxM3Config, moe_implementation: str = "native"):
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.intermediate_size,
            hidden_act="silu",
            norm_topk_prob=True,
            moe_implementation=moe_implementation,
            train_router=getattr(config, "train_router", False),
            record_routing_weights=getattr(config, "record_routing_weights", True),
            activation_native=getattr(config, "_activation_native", False),
            swiglu_limit=config.swiglu_limit,
        )
        self.config = config
        self.router = MiniMaxM3Router(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            use_routing_bias=config.use_routing_bias,
        )
        self.experts.hidden_act = normalize_hidden_act("clamped_swiglu")
        self.experts.swiglu_limit = float(config.swiglu_limit)
        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)
        self.experts.alltoall_combine_hidden_chunk_size = getattr(config, "_alltoall_combine_hidden_chunk_size", 0)
        if config.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=True)
        else:
            self.register_parameter("e_score_correction_bias", None)
        self.shared_experts = None
        if config.n_shared_experts > 0:
            self.shared_experts = MiniMaxM3MLP(
                config,
                intermediate_size=config.shared_intermediate_size * config.n_shared_experts,
            )

    def _regather_routing(self, router_logits, cached_experts, input_dtype):
        if self.router.synthetic_routing_mode == "balanced":
            routing_weights, _ = balanced_synthetic_routing(
                cached_experts.size(0),
                self.num_experts,
                cached_experts.size(1),
                router_logits.device,
                input_dtype,
            )
            return cached_experts, routing_weights

        scores = torch.sigmoid(router_logits.float()).to(router_logits.dtype)
        routing_weights = torch.gather(scores, 1, cached_experts)
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.config.routed_scaling_factor
        return cached_experts, routing_weights.to(input_dtype)

    def route(self, hidden_states: torch.Tensor):
        router_fp32 = getattr(self.config, "_router_fp32", False)
        if router_fp32 and not hasattr(self.gate, "fp8_block_size"):
            router_logits = F.linear(hidden_states.float(), self.gate.weight.float())
        else:
            router_logits = self.gate(hidden_states)

        from xorl.models.layers.moe.routing_replay import get_replay_stage  # noqa: PLC0415

        stage = get_replay_stage()
        replay = self._routing_replay
        if stage is not None and replay is not None:
            if stage == "record":
                with torch.no_grad():
                    _, selected_experts = self.router(
                        router_logits, hidden_states.dtype, expert_bias=self.e_score_correction_bias
                    )
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()
            selected_experts, routing_weights = self._regather_routing(
                router_logits, selected_experts, hidden_states.dtype
            )
            if self.record_routing_weights:
                if stage == "record":
                    replay.record_weights(routing_weights)
                elif stage == "replay_backward":
                    cached_weights = replay.pop_backward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(hidden_states.dtype)
                elif stage == "replay_forward":
                    cached_weights = replay.pop_forward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(hidden_states.dtype)
        else:
            routing_weights, selected_experts = self.router(
                router_logits, hidden_states.dtype, expert_bias=self.e_score_correction_bias
            )

        ep_dispatch = getattr(self.experts, "ep_dispatch", "alltoall")
        if self.train_router and ep_dispatch == "deepep":
            raise AssertionError(
                "train_router=True is not supported with ep_dispatch='deepep'. "
                "Set train_router=False or switch to ep_dispatch='alltoall'."
            )
        if not self.train_router:
            routing_weights = routing_weights.detach()
        return routing_weights, selected_experts, router_logits

    def _shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is None:
            return torch.zeros_like(hidden_states)
        return self.shared_experts(hidden_states.view(-1, hidden_states.size(-1))).view_as(hidden_states)

    def forward_experts_only(self, hidden_states, routing_weights, selected_experts):
        expert_output = super().forward_experts_only(hidden_states, routing_weights, selected_experts)
        return expert_output + self._shared_expert(hidden_states)

    def forward(self, hidden_states: torch.Tensor):
        expert_output, router_logits = super().forward(hidden_states)
        return expert_output + self._shared_expert(hidden_states), router_logits


MINIMAX_M3_MOE_CLASSES = {
    "eager": partial(MiniMaxM3SparseMoeBlock, moe_implementation="eager"),
    "triton": partial(MiniMaxM3SparseMoeBlock, moe_implementation="triton"),
    "native": partial(MiniMaxM3SparseMoeBlock, moe_implementation="native"),
    "quack": partial(MiniMaxM3SparseMoeBlock, moe_implementation="quack"),
}


class MiniMaxM3Attention(nn.Module):
    def __init__(self, config: MiniMaxM3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = None
        self.is_sparse_layer = bool(config.sparse_attention_freq[layer_idx])

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = MiniMaxM3PerHeadRMSNorm(
            self.num_heads, self.head_dim, eps=config.rms_norm_eps, use_gemma_norm=config.use_gemma_norm
        )
        self.k_norm = MiniMaxM3PerHeadRMSNorm(
            self.num_key_value_heads, self.head_dim, eps=config.rms_norm_eps, use_gemma_norm=config.use_gemma_norm
        )
        if self.is_sparse_layer:
            self.index_q_proj = nn.Linear(
                config.hidden_size,
                config.sparse_num_index_heads * config.sparse_index_dim,
                bias=config.attention_bias,
            )
            self.index_k_proj = nn.Linear(
                config.hidden_size,
                config.sparse_num_index_heads * config.sparse_index_dim,
                bias=config.attention_bias,
            )
            self.index_q_norm = MiniMaxM3PerHeadRMSNorm(
                config.sparse_num_index_heads,
                config.sparse_index_dim,
                eps=config.rms_norm_eps,
                use_gemma_norm=config.use_gemma_norm,
            )
            self.index_k_norm = MiniMaxM3PerHeadRMSNorm(
                config.sparse_num_index_heads,
                config.sparse_index_dim,
                eps=config.rms_norm_eps,
                use_gemma_norm=config.use_gemma_norm,
            )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        return qwen3_5_apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            fp32_single_round=bool(getattr(self.config, "_rope_fp32_single_round", False)),
        ) + (value_states,)

    def _project_index_qk(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        index_query = self.index_q_proj(hidden_states).view(
            *input_shape, self.config.sparse_num_index_heads, self.config.sparse_index_dim
        )
        index_key = self.index_k_proj(hidden_states).view(
            *input_shape, self.config.sparse_num_index_heads, self.config.sparse_index_dim
        )
        index_query = self.index_q_norm(index_query)
        index_key = self.index_k_norm(index_key)
        cos, sin = position_embeddings
        return qwen3_5_apply_rotary_pos_emb(
            index_query,
            index_key,
            cos,
            sin,
            fp32_single_round=bool(getattr(self.config, "_rope_fp32_single_round", False)),
        )

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        return self.o_proj(attn_output)

    def _get_attention_fn(self):
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
        if self.config._attn_implementation == "minimax_msa" and self.is_sparse_layer:
            query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)
            index_query, index_key = self._project_index_qk(hidden_states, position_embeddings)
            attn_output = minimax_msa_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                index_query,
                index_key,
                cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
                cu_seq_lens_k=kwargs.get("cu_seq_lens_k"),
                scaling=self.scaling,
                topk_blocks=self.config.sparse_topk_blocks,
                block_size=self.config.sparse_block_size,
                force_begin_blocks=self.config.sparse_init_block,
                force_end_blocks=self.config.sparse_local_block,
            )
            return self._project_output(attn_output), None

        attn_strategy = get_cp_strategy()
        query_states, key_states, value_states = attn_strategy.project_qkv(self, hidden_states, position_embeddings)
        attn_output = attn_strategy.compute_attention(
            self, query_states, key_states, value_states, attention_mask, **kwargs
        )
        return attn_strategy.project_output(self, attn_output), None


class MiniMaxM3DecoderLayer(MoEGradientCheckpointingLayer):
    def __init__(self, config: MiniMaxM3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniMaxM3Attention(config, layer_idx)
        self.input_layernorm = MiniMaxM3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, use_gemma_norm=config.use_gemma_norm
        )
        self.post_attention_layernorm = MiniMaxM3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, use_gemma_norm=config.use_gemma_norm
        )
        if config.moe_layer_freq[layer_idx]:
            moe_implementation = getattr(config, "_moe_implementation", "native")
            self.mlp = MINIMAX_M3_MOE_CLASSES[moe_implementation](config)
        else:
            self.mlp = MiniMaxM3MLP(config, intermediate_size=config.dense_intermediate_size)

    def _pre_mlp_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual=residual, prenorm=True)
        return hidden_states, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        del use_cache, output_attentions
        return self._moe_forward(
            hidden_states,
            output_router_logits=output_router_logits,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )


class MiniMaxM3PreTrainedModel(XorlPreTrainedModel):
    config_class = MiniMaxM3Config
    base_model_prefix = "model"
    _no_split_modules = ["MiniMaxM3DecoderLayer"]
    supports_tensor_parallelism = False
    _checkpoint_conversion_mapping = {
        r"^model\.language_model\.model\.": "model.",
        r"^language_model\.model\.": "model.",
        r"^model\.language_model\.lm_head\.": "lm_head.",
        r"^language_model\.lm_head\.": "lm_head.",
    }

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
        elif isinstance(module, MoEExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, (MiniMaxM3RMSNorm, MiniMaxM3PerHeadRMSNorm)):
            if module.use_gemma_norm:
                module.weight.data.zero_()
            else:
                module.weight.data.fill_(1.0)
        elif isinstance(module, RotaryEmbedding):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq = module.inv_freq

    def get_parallel_plan(self):
        return parallelize.get_ep_plan()

    def get_checkpoint_handler(self, **kwargs):
        ep_rank = kwargs.get("ep_rank", 0)
        ep_size = kwargs.get("ep_size", 1)
        is_broadcast = kwargs.get("is_broadcast", False)
        if is_broadcast:
            ep_rank, ep_size = 0, 1
        return MiniMaxM3CheckpointHandler(
            num_experts=self.config.num_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )


class MiniMaxM3Model(MiniMaxM3PreTrainedModel):
    def __init__(self, config: MiniMaxM3Config):
        config = _adapt_minimax_m3_config(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniMaxM3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, use_gemma_norm=config.use_gemma_norm)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> MoeModelOutput:
        _raise_if_minimax_parallel_unsupported()
        _reject_multimodal_kwargs(kwargs)
        _reject_multimodal_tokens(self.config, input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        causal_mask = update_causal_mask(
            "eager" if self.config._attn_implementation == "minimax_msa" else self.config._attn_implementation,
            attention_mask,
            hidden_states,
            cache_position,
            sliding_window=None,
            is_training=self.training,
            output_attentions=output_attentions,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = get_cp_strategy().prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_key_value_heads,
        )

        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers:
            _use_outer_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and self._gradient_checkpointing_method == "recompute_full_layer"
            )
            if _use_outer_checkpoint:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    None,
                    use_cache,
                    output_attentions,
                    output_router_logits,
                    position_embeddings,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if output_attentions:
                all_self_attns += (None,)
            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states
        return MoeModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            hidden_states=all_hidden_states,
        )


class MiniMaxM3SparseForCausalLM(MiniMaxM3PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        config = _adapt_minimax_m3_config(config)
        super().__init__(config)
        self.model = MiniMaxM3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_logits: bool = False,
        **kwargs,
    ) -> MoeCausalLMOutput:
        _reject_multimodal_kwargs(kwargs)
        output_router_logits = self.config.output_router_logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        logits = None
        loss = None
        if labels is not None or return_logits:
            logits = self.lm_head(outputs.last_hidden_state)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return MoeCausalLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
            hidden_states=outputs.hidden_states,
        )


class MiniMaxM3SparseForConditionalGeneration(MiniMaxM3SparseForCausalLM):
    """Text-only implementation for HF MiniMax-M3 multimodal wrapper configs."""


ModelClass = [MiniMaxM3SparseForConditionalGeneration, MiniMaxM3SparseForCausalLM]

__all__ = [
    "MiniMaxM3Config",
    "MiniMaxM3Model",
    "MiniMaxM3PreTrainedModel",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3SparseMoeBlock",
    "MINIMAX_M3_UNSUPPORTED_PARALLEL_MESSAGE",
    "minimax_m3_swigluoai",
]
