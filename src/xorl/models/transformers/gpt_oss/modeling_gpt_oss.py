"""GPT-OSS model implementation for the xorl framework.

GPT-OSS is a Mixture-of-Experts transformer with:
- Standard RMSNorm (not zero-centered)
- Custom YaRN-style RoPE with NTK-by-parts scaling
- GQA with learned attention sinks (per-head softmax bias)
- Alternating sliding-window / full attention layers (even / odd)
- SwiGLU activation with interleaved layout, clamping, and alpha=1.702
- Expert biases on both gate_up and down projections
- Router gate with bias
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple, Unpack

import torch
import torch.nn.functional as F
from torch import nn

from xorl.distributed.moe.deepep import sync_pending_combine
from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.base import XorlPreTrainedModel
from xorl.models.layers.attention import AttentionKwargs, update_causal_mask
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS
from xorl.models.layers.moe import MoEBlock
from xorl.models.layers.normalization import RMSNorm
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.models.transformers.gpt_oss import parallelize
from xorl.models.transformers.gpt_oss.checkpoint_handler import GptOssCheckpointHandler
from xorl.models.transformers.gpt_oss.configuration_gpt_oss import GptOssConfig
from xorl.utils import logging


logger = logging.get_logger(__name__)


_sinks_grad_hook_logged = False


def _scale_sinks_grad_for_ulysses(grad: torch.Tensor) -> torch.Tensor:
    global _sinks_grad_hook_logged
    ps = get_parallel_state()
    if ps.ulysses_enabled:
        if not _sinks_grad_hook_logged:
            logger.info(f"sinks grad hook active: scaling by ulysses_size={ps.ulysses_size}")
            _sinks_grad_hook_logged = True
        return grad * ps.ulysses_size
    return grad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapt_gpt_oss_config(config):
    """Convert an HF config to GptOssConfig if needed."""
    if isinstance(config, GptOssConfig):
        return config
    return GptOssConfig.from_hf_config(config)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings using half-dim split (not interleaved)."""
    x1, x2 = x.chunk(2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def gpt_oss_apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key.

    Args:
        query: ``[batch, seq, num_heads, head_dim]``
        key: ``[batch, seq, num_kv_heads, head_dim]``
        cos: ``[batch, seq, head_dim // 2]``
        sin: ``[batch, seq, head_dim // 2]``
    """
    # Unsqueeze for head dimension: [batch, seq, 1, head_dim // 2]
    cos = cos.unsqueeze(2).to(query.dtype)
    sin = sin.unsqueeze(2).to(query.dtype)
    return _apply_rotary_emb(query, cos, sin), _apply_rotary_emb(key, cos, sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for grouped-query attention."""
    if n_rep == 1:
        return hidden_states
    batch, seq, num_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, seq, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seq, num_kv_heads * n_rep, head_dim)


# ---------------------------------------------------------------------------
# Rotary Embedding
# ---------------------------------------------------------------------------


class GptOssRotaryEmbedding(nn.Module):
    """Custom YaRN-style RoPE with NTK-by-parts for GPT-OSS.

    See YaRN paper: https://arxiv.org/abs/2309.00071
    """

    def __init__(self, config: GptOssConfig, device=None):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        self.initial_context_length = config.initial_context_length
        self.scaling_factor = config.rope_scaling_factor
        self.ntk_alpha = config.rope_ntk_alpha
        self.ntk_beta = config.rope_ntk_beta

    def _compute_inv_freq_and_concentration(self, device: torch.device):
        freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.head_dim / 2
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return inv_freq, concentration

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(cos, sin)`` position embeddings.

        Args:
            x: Hidden states (used for device only).
            position_ids: ``[batch, seq_len]``.

        Returns:
            ``(cos, sin)`` each of shape ``[batch, seq_len, head_dim // 2]``.
        """
        inv_freq, concentration = self._compute_inv_freq_and_concentration(x.device)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class GptOssAttention(nn.Module):
    """GPT-OSS multi-head attention with learned sinks and alternating SWA.

    Attention sinks are per-head scalar biases that compete with real tokens
    in the softmax, allowing the model to "waste" attention weight rather than
    attending to specific tokens.  Sinks are applied in both the eager path
    (via explicit concat-then-softmax) and the ``flash_attention_3`` backend
    (via a custom autograd wrapper that post-multiplies by
    ``sigmoid(lse - sink)``; see ``flash_sink_attention.py``).  Other backends
    raise ``NotImplementedError`` — silently dropping sinks would corrupt
    GPT-OSS semantics.
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Alternating sliding window: even layers → SWA, odd layers → full
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None

        # Learned attention sinks (one per query head)
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        # Ulysses SP shards heads: only ulysses_rank=r writes grad to slice r,
        # but FSDP averages across the full dp×ulysses group. Scale by
        # ulysses_size so the post-reduce result is the correct dp-only average.
        self.sinks.register_hook(_scale_sinks_grad_for_ulysses)

        # Fused QKV projection (with bias)
        qkv_dim = config.head_dim * (config.num_attention_heads + 2 * config.num_key_value_heads)
        self.qkv_proj = nn.Linear(config.hidden_size, qkv_dim, bias=config.attention_bias)

        # Output projection (with bias)
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def _local_sinks(self) -> torch.Tensor:
        # Ulysses SP shards heads across ranks via all-to-all; the sinks
        # parameter is replicated, so slice it to match this rank's head shard.
        ps = get_parallel_state()
        if not ps.ulysses_enabled:
            return self.sinks
        sp = ps.ulysses_size
        heads_per_rank = self.sinks.size(0) // sp
        rank = ps.ulysses_rank
        return self.sinks[rank * heads_per_rank : (rank + 1) * heads_per_rank]

    # -- TP helpers ----------------------------------------------------------

    def unfuse_for_tp(self):
        """Split fused QKV projection into separate q/k/v for tensor parallelism."""
        device = self.qkv_proj.weight.device
        dtype = self.qkv_proj.weight.dtype
        has_bias = self.qkv_proj.bias is not None
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(self.config.hidden_size, q_dim, bias=has_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.config.hidden_size, kv_dim, bias=has_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.config.hidden_size, kv_dim, bias=has_bias, device=device, dtype=dtype)

        self.q_proj.weight.data.copy_(self.qkv_proj.weight.data[:q_dim])
        self.k_proj.weight.data.copy_(self.qkv_proj.weight.data[q_dim : q_dim + kv_dim])
        self.v_proj.weight.data.copy_(self.qkv_proj.weight.data[q_dim + kv_dim :])
        if has_bias:
            self.q_proj.bias.data.copy_(self.qkv_proj.bias.data[:q_dim])
            self.k_proj.bias.data.copy_(self.qkv_proj.bias.data[q_dim : q_dim + kv_dim])
            self.v_proj.bias.data.copy_(self.qkv_proj.bias.data[q_dim + kv_dim :])

        del self.qkv_proj

    # -- Attention strategy hooks -------------------------------------------

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if hasattr(self, "qkv_proj"):
            qkv = self.qkv_proj(hidden_states)
            q_dim = self.num_attention_heads * self.head_dim
            kv_dim = self.num_key_value_heads * self.head_dim
            query_states = qkv[..., :q_dim].view(hidden_shape)
            key_states = qkv[..., q_dim : q_dim + kv_dim].view(hidden_shape)
            value_states = qkv[..., q_dim + kv_dim :].view(hidden_shape)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = gpt_oss_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        return self.o_proj(attn_output)

    def _get_attention_fn(self) -> Callable:
        impl = getattr(self.config, "_attn_implementation", "eager")
        if impl == "eager":
            return self._attention_with_sinks
        if impl == "flash_attention_3":
            return self._flash_attention_3_with_sinks
        if impl == "flash_attention_4":
            raise NotImplementedError(
                "GPT-OSS attention sinks are not wired through the FA4 (CUTE) backend. Use flash_attention_3 or eager."
            )
        if impl in ATTENTION_FUNCTIONS:
            raise NotImplementedError(
                f"GPT-OSS attention sinks are not wired through the {impl!r} backend. Use flash_attention_3 or eager."
            )
        return ATTENTION_FUNCTIONS.get(impl, self._attention_with_sinks)

    def _flash_attention_3_with_sinks(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        softcap: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """FA3 forward with learned sinks applied via sigmoid(lse - sink).

        The installed FA3 build does not expose a native ``sinks=`` kwarg, so
        sinks are fused by a custom autograd wrapper (FA3 fwd + manual bwd).
        Supports both batched 4D and packed-varlen 3D paths.
        """
        from xorl.models.transformers.gpt_oss.flash_sink_attention import (  # noqa: PLC0415
            flash_attn_varlen_with_sink,
            flash_attn_with_sink,
        )

        causal = getattr(module, "is_causal", True)
        if sliding_window is not None:
            window_size = (sliding_window, 0 if causal else sliding_window)
        else:
            window_size = (-1, -1)

        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        cu_seq_lens_k = kwargs.get("cu_seq_lens_k", None)

        if cu_seq_lens_q is not None and cu_seq_lens_k is not None:
            # Packed / varlen path — flatten batch dim (packing collator uses B=1).
            cu_seq_lens_q = cu_seq_lens_q.to(torch.int32)
            cu_seq_lens_k = cu_seq_lens_k.to(torch.int32)
            max_length_q = kwargs.get("max_length_q")
            max_length_k = kwargs.get("max_length_k")

            def _flatten(x):
                return x.squeeze(0) if x.size(0) == 1 else x.reshape(-1, x.size(-2), x.size(-1))

            q_varlen = _flatten(query)
            k_varlen = _flatten(key)
            v_varlen = _flatten(value)

            out = flash_attn_varlen_with_sink(
                q_varlen,
                k_varlen,
                v_varlen,
                self._local_sinks(),
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_k,
                max_seqlen_q=max_length_q,
                max_seqlen_k=max_length_k,
                causal=causal,
                window_size=window_size,
                softmax_scale=scaling,
                softcap=softcap if softcap is not None else 0.0,
            )
            out = out.unsqueeze(0)
        else:
            out = flash_attn_with_sink(
                query,
                key,
                value,
                self._local_sinks(),
                causal=causal,
                window_size=window_size,
                softmax_scale=scaling,
                softcap=softcap if softcap is not None else 0.0,
            )
        return out, None

    def _attention_kwargs(self) -> dict:
        return dict(
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

    # -- Eager attention with sinks -----------------------------------------

    def _attention_with_sinks(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        sliding_window: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Eager attention that injects learned sink logits into softmax.

        Args:
            query: ``[batch, seq, num_heads, head_dim]``
            key: ``[batch, seq, num_kv_heads, head_dim]``
            value: ``[batch, seq, num_kv_heads, head_dim]``
            attention_mask: ``[batch, 1, seq, seq]`` or *None*
            sliding_window: Per-layer sliding window size or *None*.
        """
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)

        # -> [batch, heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling

        # Causal mask — trim to actual key length (framework mask may
        # include an extra column for KV-cache bookkeeping).
        if attention_mask is not None:
            causal_mask = attention_mask
            mk = causal_mask.shape[-1]
            k_len = attn_weights.shape[-1]
            if mk != k_len:
                causal_mask = causal_mask[..., :k_len]
            attn_weights = attn_weights + causal_mask

        # Sliding window mask (lower triangle beyond window)
        if sliding_window is not None and sliding_window > 0:
            seq_len = attn_weights.shape[-1]
            sw_mask = torch.tril(
                attn_weights.new_ones(seq_len, seq_len, dtype=torch.bool),
                diagonal=-sliding_window,
            )
            attn_weights = attn_weights.masked_fill(sw_mask, float("-inf"))

        # Attention sinks: [num_heads] -> [1, num_heads, 1, 1] -> [B, H, S, 1]
        sinks = self._local_sinks().to(attn_weights.dtype)
        sink_logits = sinks.view(1, -1, 1, 1).expand(attn_weights.shape[0], -1, attn_weights.shape[2], 1)
        attn_weights = torch.cat([attn_weights, sink_logits], dim=-1)

        # Subtract max for numerical stability (matches HF implementation)
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
        # Softmax over (seq + 1) then drop the sink column
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
        attn_weights = attn_weights[..., :-1]

        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del position_ids, past_key_values
        attn_strategy = get_cp_strategy()
        query_states, key_states, value_states = attn_strategy.project_qkv(self, hidden_states, position_embeddings)
        attn_output = attn_strategy.compute_attention(
            self, query_states, key_states, value_states, attention_mask, **kwargs
        )
        attn_output = attn_strategy.project_output(self, attn_output)
        return attn_output, None


# ---------------------------------------------------------------------------
# MoE Block
# ---------------------------------------------------------------------------


class GptOssMoEBlock(MoEBlock):
    """GPT-OSS MoE with expert biases and clamped SwiGLU activation.

    The checkpoint handler deinterleaves the original interleaved gate/up
    layout into the standard concatenated ``[gate | up]`` format used by xorl
    MoE backends.  Per-expert biases (``gate_up_bias``, ``down_bias``) and the
    ``"clamped_swiglu"`` activation kind are threaded through the shared
    ``hidden_act`` dispatch (``SUPPORTED_HIDDEN_ACTS``).  This supports both
    single-GPU and Expert Parallel (EP) execution.
    """

    _SUPPORTED_MOE_IMPLEMENTATIONS = {"eager", "native"}

    def __init__(self, config: GptOssConfig, moe_implementation="triton"):
        if moe_implementation not in self._SUPPORTED_MOE_IMPLEMENTATIONS:
            raise NotImplementedError(
                f"GPT-OSS requires per-expert biases (gate_up_bias, down_bias) and a "
                f"clamped SwiGLU activation, which the {moe_implementation!r} MoE backend "
                f"does not currently thread through. Supported backends: "
                f"{sorted(self._SUPPORTED_MOE_IMPLEMENTATIONS)}. "
                f"Running with {moe_implementation!r} would silently drop the biases and "
                f"use a plain SiLU SwiGLU, producing incorrect outputs."
            )
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=moe_implementation,
            train_router=getattr(config, "train_router", False),
            activation_native=getattr(config, "_activation_native", False),
        )
        self.config = config
        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)

        # GPT-OSS uses a clamped SwiGLU rather than the default SiLU SwiGLU.
        # The activation is threaded through ``hidden_act`` so that all MoE
        # backends that support it (currently ``eager`` and ``native``) dispatch
        # on the string. Triton/quack raise NotImplementedError at entry.
        self.experts.hidden_act = "clamped_swiglu"

        # GPT-OSS router gate has bias (base MoEBlock creates bias=False)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=True)

        # Expert biases — registered on ``self.experts`` so that the
        # checkpoint handler finds them at
        # ``model.layers.*.mlp.experts.gate_up_bias`` / ``down_bias``.
        self.experts.gate_up_bias = nn.Parameter(torch.zeros(config.num_experts, 2 * config.moe_intermediate_size))
        self.experts.down_bias = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Routing — topk on raw logits, then softmax on top-k values
        # (matches the original OSS and HF implementations).
        router_logits = self.gate(hidden_states_flat)
        top_values, selected_experts = torch.topk(router_logits, k=self.top_k, dim=-1, sorted=True)
        routing_weights = F.softmax(top_values, dim=1, dtype=top_values.dtype)

        if not self.train_router:
            routing_weights = routing_weights.detach()

        if self.moe_implementation == "eager":
            final = self._eager_forward(hidden_states_flat, routing_weights, selected_experts)
        else:
            final = self.experts(hidden_states_flat, routing_weights, selected_experts)
        final = final.view(batch_size, sequence_length, hidden_dim)
        return final, router_logits


GPT_OSS_MOE_CLASSES = {
    "eager": partial(GptOssMoEBlock, moe_implementation="eager"),
    "native": partial(GptOssMoEBlock, moe_implementation="native"),
}


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, layer_idx)
        moe_implementation = getattr(config, "_moe_implementation", "triton")
        self.mlp = GPT_OSS_MOE_CLASSES[moe_implementation](config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
    ) -> Tuple[torch.FloatTensor, ...]:
        # Pre-norm → attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Pre-norm → MLP (fused residual add via prenorm)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual=residual, prenorm=True)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        sync_pending_combine()
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------


class GptOssPreTrainedModel(XorlPreTrainedModel):
    config_class = GptOssConfig
    base_model_prefix = "model"
    _no_split_modules = ["GptOssDecoderLayer"]

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
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

    def get_parallel_plan(self):
        return parallelize.get_ep_plan()

    def get_checkpoint_handler(self, **kwargs):
        checkpoint_keys = kwargs.get("checkpoint_keys", set())
        ep_rank = kwargs.get("ep_rank", 0)
        ep_size = kwargs.get("ep_size", 1)
        is_broadcast = kwargs.get("is_broadcast", False)
        if is_broadcast:
            ep_rank, ep_size = 0, 1
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        return GptOssCheckpointHandler(
            num_experts=self.config.num_experts,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            ep_rank=ep_rank,
            ep_size=ep_size,
            checkpoint_keys=checkpoint_keys or None,
            skip_qkv_merge=getattr(self, "_unfused_for_tp", False),
        )


# ---------------------------------------------------------------------------
# Model (backbone)
# ---------------------------------------------------------------------------


class GptOssModel(GptOssPreTrainedModel):
    def __init__(self, config: GptOssConfig):
        config = _adapt_gpt_oss_config(config)
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GptOssRotaryEmbedding(config=config)
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
        output_attentions = output_attentions if output_attentions is not None else False
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(self.config, "output_router_logits", False)
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

        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        causal_mask = update_causal_mask(
            getattr(self.config, "_attn_implementation", "eager"),
            attention_mask,
            hidden_states,
            cache_position,
            sliding_window=None,  # Per-layer SWA handled in the attention module
            is_training=self.training,
            output_attentions=output_attentions,
        )

        ps = get_parallel_state()
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = get_cp_strategy().prepare_position_embeddings(
            position_embeddings,
            dim=1,
            sp_group=ps.sp_group,
            num_kv_heads=self.config.num_key_value_heads,
        )

        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if decoder_layer is None:
                continue
            if self.gradient_checkpointing and self.training:
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
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states
        return MoeModelOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


# ---------------------------------------------------------------------------
# Causal LM
# ---------------------------------------------------------------------------


class GptOssForCausalLM(GptOssPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _tp_plan = parallelize.MODEL_TP_PLAN

    def __init__(self, config):
        config = _adapt_gpt_oss_config(config)
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def unfuse_for_tp(self):
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
        return {
            "input_fqns": ["model.embed_tokens"],
            "layer_prefix": "model.layers",
            "output_fqns": ["model.norm", "lm_head"],
            "always_keep_fqns": ["model.rotary_emb"],
            "num_layers": self.config.num_hidden_layers,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> MoeCausalLMOutput:
        output_router_logits = getattr(self.config, "output_router_logits", False)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        return MoeCausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            router_logits=outputs.router_logits,
        )


ModelClass = [GptOssForCausalLM]

__all__ = [
    "GptOssForCausalLM",
    "GptOssModel",
    "GptOssPreTrainedModel",
    "GptOssMoEBlock",
]
