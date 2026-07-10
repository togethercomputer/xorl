"""Base attention module shared across all decoder model variants."""

import os
from typing import Callable, List, Optional, Tuple, Unpack

import torch
import torch.nn.functional as F
from torch import nn

from xorl.distributed.sequence_parallel.strategy import get_cp_strategy
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS, AttentionKwargs
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.models.layers.attention.utils import repeat_kv
from xorl.models.layers.normalization import RMS_NORM_FAMILY_NO_RESIDUAL, RMSNorm
from xorl.models.layers.rope import apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """Base multi-head attention shared across all decoder model variants.

    Subclasses override ``_project_qkv()`` / ``_project_output()`` for different
    attention variants (e.g. Multi-head Latent Attention).  Subclasses override
    ``_init_sliding_window()`` for model-specific sliding window logic.

    SP strategy is resolved at forward time from ParallelState via
    ``get_cp_strategy()``.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        qkv_bias = getattr(config, "attention_bias", False)
        self._use_qk_norm = getattr(config, "use_qk_norm", True)
        self.q_dim = config.num_attention_heads * self.head_dim
        self.kv_dim = config.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, self.q_dim + 2 * self.kv_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        if self._use_qk_norm:
            # qk-norms are no-residual sites: serving runs the family-1 kernel.
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, family=RMS_NORM_FAMILY_NO_RESIDUAL)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, family=RMS_NORM_FAMILY_NO_RESIDUAL)
        self.sliding_window = self._init_sliding_window(config)

    # ------------------------------------------------------------------ #
    # Overridable hooks
    # ------------------------------------------------------------------ #

    def _init_sliding_window(self, config):
        """Override in subclasses for model-specific sliding window logic."""
        return getattr(config, "sliding_window", None)

    def unfuse_for_tp(self):
        """Replace fused qkv_proj with separate q_proj, k_proj, v_proj for tensor parallelism."""
        device = self.qkv_proj.weight.device
        dtype = self.qkv_proj.weight.dtype
        has_qkv_bias = self.qkv_proj.bias is not None
        self.q_proj = nn.Linear(self.config.hidden_size, self.q_dim, bias=has_qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.config.hidden_size, self.kv_dim, bias=has_qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.config.hidden_size, self.kv_dim, bias=has_qkv_bias, device=device, dtype=dtype)
        del self.qkv_proj

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard MHA: fused QKV linear -> split -> norm -> RoPE.

        Override for different attention variants (e.g. Multi-head Latent Attention).

        Returns:
            (q, k, v) each with shape [batch, seq, num_heads, head_dim].
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if hasattr(self, "qkv_proj"):
            qkv = self.qkv_proj(hidden_states)
            self._capture_diagnostic_component("qkv", qkv)
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        self._capture_diagnostic_component("q_pre_qk_norm", q)
        self._capture_diagnostic_component("k_pre_qk_norm", k)
        self._capture_diagnostic_component("v", v)
        q = q.view(hidden_shape)
        k = k.view(hidden_shape)
        if self._use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        self._capture_diagnostic_component("q_post_qk_norm", self._flatten_heads_for_diagnostics(q))
        self._capture_diagnostic_component("k_post_qk_norm", self._flatten_heads_for_diagnostics(k))
        v = v.view(hidden_shape)

        cos, sin = position_embeddings
        self._capture_diagnostic_component("rope_cos", cos)
        self._capture_diagnostic_component("rope_sin", sin)
        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            force_native=getattr(self.config, "_rope_native", False),
        )
        self._capture_diagnostic_component("q", self._flatten_heads_for_diagnostics(q))
        self._capture_diagnostic_component("k", self._flatten_heads_for_diagnostics(k))

        # Optionally cast to bfloat16 after RoPE for SGLang numerical alignment
        if getattr(self.config, "_attention_cast_bf16", False):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        return q, k, v

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Reshape [B, S, H, D] -> [B, S, H*D] then O_proj.

        Override for different attention variants (e.g. Multi-head Latent Attention).
        """
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        output = self._project_output_linear(attn_output)
        self._capture_diagnostic_component("o_proj_output", output)
        return output

    def _project_output_linear(self, attn_output: torch.Tensor) -> torch.Tensor:
        split_count = int(os.environ.get("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT", "0") or 0)
        if split_count <= 1:
            return self.o_proj(attn_output)
        sum_fp32 = os.environ.get("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_SUM_FP32", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        keep_fp32 = os.environ.get("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_KEEP_FP32", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        split_layers = os.environ.get("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_LAYERS", "").strip()
        if split_layers and split_layers.lower() not in {"all", "*"}:
            try:
                enabled_layers = {int(item.strip()) for item in split_layers.split(",") if item.strip()}
            except ValueError as exc:
                raise ValueError(
                    f"Invalid XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_LAYERS={split_layers!r}; "
                    "expected comma-separated layer indices"
                ) from exc
            if self.layer_idx not in enabled_layers:
                return self.o_proj(attn_output)

        weight = self.o_proj.weight
        input_dim = attn_output.shape[-1]
        if input_dim % split_count != 0 or weight.shape[1] != input_dim:
            return self.o_proj(attn_output)

        chunk_size = input_dim // split_count
        output = None
        output_dtype = None
        carry_partials = os.environ.get("XORL_DIAGNOSTIC_O_PROJ_TP_SPLIT_CARRY_PARTIALS", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        partials = []
        for idx in range(split_count):
            start = idx * chunk_size
            end = start + chunk_size
            partial = F.linear(
                attn_output[..., start:end],
                weight[:, start:end],
                self.o_proj.bias if idx == 0 else None,
            )
            output_dtype = partial.dtype
            if sum_fp32:
                partial = partial.to(torch.float32)
            if carry_partials:
                partials.append(partial)
            output = partial if output is None else output + partial
        if sum_fp32 and output_dtype is not None and not keep_fp32:
            output = output.to(output_dtype)
        if carry_partials and output is not None:
            output._xorl_o_proj_tp_partials = tuple(partials)
        return output

    def _diagnostic_attention_tp_split_count(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_strategy,
    ) -> int:
        """Return an opt-in head-shard attention split count for K3 diagnostics."""
        split_count = int(os.environ.get("XORL_DIAGNOSTIC_ATTENTION_TP_SPLIT", "0") or 0)
        if split_count <= 1:
            return 0
        if attn_strategy.__class__.__name__ != "NoopStrategy":
            return 0

        split_layers = os.environ.get("XORL_DIAGNOSTIC_ATTENTION_TP_SPLIT_LAYERS", "").strip()
        if split_layers and split_layers.lower() not in {"all", "*"}:
            try:
                enabled_layers = {int(item.strip()) for item in split_layers.split(",") if item.strip()}
            except ValueError as exc:
                raise ValueError(
                    f"Invalid XORL_DIAGNOSTIC_ATTENTION_TP_SPLIT_LAYERS={split_layers!r}; "
                    "expected comma-separated layer indices"
                ) from exc
            if self.layer_idx not in enabled_layers:
                return 0

        if q.shape[-2] % split_count != 0 or k.shape[-2] % split_count != 0 or v.shape[-2] % split_count != 0:
            return 0
        return split_count

    def _diagnostic_attention_eager_candidate_enabled(self) -> bool:
        return self._diagnostic_attention_eager_env_enabled("XORL_DIAGNOSTIC_ATTENTION_EAGER_CANDIDATE")

    def _diagnostic_attention_eager_replace_enabled(self) -> bool:
        return self._diagnostic_attention_eager_env_enabled("XORL_DIAGNOSTIC_ATTENTION_EAGER")

    def _diagnostic_attention_eager_env_enabled(self, env_name: str) -> bool:
        if os.environ.get(env_name, "").strip().lower() not in {"1", "true", "yes", "on"}:
            return False

        split_layers = os.environ.get(f"{env_name}_LAYERS", "").strip()
        if split_layers and split_layers.lower() not in {"all", "*"}:
            try:
                enabled_layers = {int(item.strip()) for item in split_layers.split(",") if item.strip()}
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {env_name}_LAYERS={split_layers!r}; expected comma-separated layer indices"
                ) from exc
            if self.layer_idx not in enabled_layers:
                return False
        return True

    def _compute_diagnostic_attention_eager_candidate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Manual causal eager attention candidate for K3 source localization.

        This deliberately avoids the configured FlashAttention backend while
        keeping the already-materialized q/k/v tensors. It is meant for narrow
        component captures or guarded replacement replays, not normal training.
        """
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        q_heads = query.shape[1]
        kv_heads = key.shape[1]
        if q_heads % kv_heads != 0:
            raise RuntimeError(
                f"Invalid attention head layout: query_heads={q_heads} is not divisible by kv_heads={kv_heads}."
            )
        kv_repeat = q_heads // kv_heads
        key_states = repeat_kv(key, kv_repeat)
        value_states = repeat_kv(value, kv_repeat)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask
            q_len = attn_weights.shape[-2]
            k_len = attn_weights.shape[-1]
            if causal_mask.shape[-1] != k_len:
                causal_mask = causal_mask[..., :k_len]
            if causal_mask.shape[-2] != q_len:
                causal_mask = causal_mask[..., -q_len:, :]
            attn_weights = attn_weights + causal_mask
        elif self.is_causal:
            q_len = attn_weights.shape[-2]
            k_len = attn_weights.shape[-1]
            q_positions = torch.arange(k_len - q_len, k_len, device=attn_weights.device)
            k_positions = torch.arange(k_len, device=attn_weights.device)
            causal = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            if self.sliding_window is not None:
                causal &= k_positions.unsqueeze(0) > (q_positions.unsqueeze(1) - self.sliding_window)
            attn_weights = attn_weights.masked_fill(
                ~causal.view(1, 1, q_len, k_len),
                torch.finfo(attn_weights.dtype).min,
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous()

    def _compute_diagnostic_attention_tp_split(
        self,
        attn_strategy,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        split_count: int,
        **kwargs: Unpack[AttentionKwargs],
    ) -> torch.Tensor:
        """Run attention in TP-like head shards, then concatenate heads."""
        q_chunks = q.chunk(split_count, dim=-2)
        k_chunks = k.chunk(split_count, dim=-2)
        v_chunks = v.chunk(split_count, dim=-2)
        outputs = [
            attn_strategy.compute_attention(self, q_chunk, k_chunk, v_chunk, attention_mask, **kwargs)
            for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks)
        ]
        return torch.cat(outputs, dim=-2)

    @staticmethod
    def _flatten_heads_for_diagnostics(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-2], -1) if tensor.ndim >= 4 else tensor

    def _capture_diagnostic_component(self, name: str, tensor: torch.Tensor) -> None:
        capture = getattr(self, "_diagnostic_capture_component", None)
        if callable(capture):
            capture(name, tensor)

    def _append_past_key_values(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is None:
            return key_states, value_states

        while len(past_key_values) <= self.layer_idx:
            past_key_values.append(None)

        past = past_key_values[self.layer_idx]
        if past is not None:
            past_key_states, past_value_states = past
            key_states = torch.cat((past_key_states.to(key_states.device), key_states), dim=1)
            value_states = torch.cat((past_value_states.to(value_states.device), value_states), dim=1)

        past_key_values[self.layer_idx] = (key_states, value_states)
        return key_states, value_states

    def _append_diagnostic_past_key_values(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past = getattr(self, "_diagnostic_past_key_value", None)
        if past is None and past_key_values is not None and len(past_key_values) > self.layer_idx:
            past = past_key_values[self.layer_idx]

        if past is not None:
            past_key_states, past_value_states = past
            key_states = torch.cat((past_key_states.to(key_states.device), key_states), dim=1)
            value_states = torch.cat((past_value_states.to(value_states.device), value_states), dim=1)

        cached = (key_states.detach(), value_states.detach())
        self._diagnostic_past_key_value = cached

        if past_key_values is not None:
            while len(past_key_values) <= self.layer_idx:
                past_key_values.append(None)
            past_key_values[self.layer_idx] = cached
        return key_states, value_states

    # ------------------------------------------------------------------ #
    # Attention function helpers
    # ------------------------------------------------------------------ #

    def _get_attention_fn(self) -> Callable:
        """Return the registered attention callable (flash, eager, etc.)."""
        return ATTENTION_FUNCTIONS.get(self.config._attn_implementation, eager_attention_forward)

    def _attention_kwargs(self) -> dict:
        """Common kwargs for the attention callable."""
        return dict(
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
        )

    # ------------------------------------------------------------------ #
    # Forward — three-phase pipeline
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_strategy = get_cp_strategy(num_kv_heads=self.config.num_key_value_heads)
        self._capture_diagnostic_component("attention_input", hidden_states)

        # Phase 1: QKV projection + norm + RoPE (+ pre-attention SP communication)
        q, k, v = attn_strategy.project_qkv(self, hidden_states, position_embeddings)
        self._capture_diagnostic_component("q_attn_input", self._flatten_heads_for_diagnostics(q))
        self._capture_diagnostic_component("k_attn_input", self._flatten_heads_for_diagnostics(k))
        self._capture_diagnostic_component("v_attn_input", self._flatten_heads_for_diagnostics(v))
        if kwargs.get("diagnostic_decode_cache", False):
            k, v = self._append_diagnostic_past_key_values(k, v, past_key_values)
        else:
            k, v = self._append_past_key_values(k, v, past_key_values)

        # Phase 2: Attention (ring attention puts P2P communication here)
        split_count = self._diagnostic_attention_tp_split_count(q, k, v, attn_strategy)
        if split_count > 1:
            attn_output = self._compute_diagnostic_attention_tp_split(
                attn_strategy,
                q,
                k,
                v,
                attention_mask,
                split_count,
                **kwargs,
            )
        else:
            attn_output = attn_strategy.compute_attention(self, q, k, v, attention_mask, **kwargs)
        if self._diagnostic_attention_eager_candidate_enabled() or self._diagnostic_attention_eager_replace_enabled():
            eager_attn_output = self._compute_diagnostic_attention_eager_candidate(q, k, v, attention_mask)
            self._capture_diagnostic_component(
                "attn_output_eager_candidate",
                self._flatten_heads_for_diagnostics(eager_attn_output),
            )
            if self._diagnostic_attention_eager_replace_enabled():
                attn_output = eager_attn_output
        self._capture_diagnostic_component("attn_output", self._flatten_heads_for_diagnostics(attn_output))

        # Phase 3: Output projection (+ post-attention SP communication)
        attn_output = attn_strategy.project_output(self, attn_output)

        return attn_output, None
