"""Base attention module shared across all decoder model variants."""

from typing import Callable, Optional, Tuple, Unpack

import torch
from torch import nn

from xorl.models.layers.normalization import RMSNorm
from xorl.models.layers.rope import apply_rotary_pos_emb
from xorl.models.layers.attention.backend import ATTENTION_FUNCTIONS, AttentionKwargs
from xorl.models.layers.attention.backend.eager import eager_attention_forward
from xorl.distributed.sequence_parallel.strategy import get_sp_strategy

class MultiHeadAttention(nn.Module):
    """Base multi-head attention shared across all decoder model variants.

    Subclasses override ``_project_qkv()`` / ``_project_output()`` for different
    attention variants (e.g. Multi-head Latent Attention).  Subclasses override
    ``_init_sliding_window()`` for model-specific sliding window logic.

    SP strategy is resolved at forward time from ParallelState via
    ``get_sp_strategy()``.
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

        self.q_dim = config.num_attention_heads * self.head_dim
        self.kv_dim = config.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_dim + 2 * self.kv_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
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
        self.q_proj = nn.Linear(self.config.hidden_size, self.q_dim, bias=self.config.attention_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.config.hidden_size, self.kv_dim, bias=self.config.attention_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.config.hidden_size, self.kv_dim, bias=self.config.attention_bias, device=device, dtype=dtype)
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
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        q = self.q_norm(q.view(hidden_shape))
        k = self.k_norm(k.view(hidden_shape))
        v = v.view(hidden_shape)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Reshape [B, S, H, D] -> [B, S, H*D] then O_proj.

        Override for different attention variants (e.g. Multi-head Latent Attention).
        """
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        return self.o_proj(attn_output)

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
        **kwargs: Unpack[AttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        

        attn_strategy = get_sp_strategy(num_kv_heads=self.config.num_key_value_heads)

        # Phase 1: QKV projection + norm + RoPE (+ pre-attention SP communication)
        q, k, v = attn_strategy.project_qkv(self, hidden_states, position_embeddings)

        # Phase 2: Attention (ring attention puts P2P communication here)
        attn_output = attn_strategy.compute_attention(self, q, k, v, attention_mask, **kwargs)

        # Phase 3: Output projection (+ post-attention SP communication)
        attn_output = attn_strategy.project_output(self, attn_output)

        return attn_output, None
