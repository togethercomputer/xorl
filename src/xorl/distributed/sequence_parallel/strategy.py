"""Sequence parallel strategy pattern for attention.

Provides pluggable SP strategies that hook into the three-phase attention
pipeline (project_qkv → compute_attention → project_output).  Each strategy
can inject SP communication at the appropriate phase:

- **Noop**: No communication — delegates to module's own methods.
- **Ulysses sync**: All-to-all around attention (project_qkv / project_output).
- **Ulysses async**: Overlapped linear+a2a in project_qkv / project_output.
- **Ring**: Ring attention via all-gather KV + per-step flash attn.
- **Hybrid Ulysses+Ring**: Ulysses a2a (phases 1,3) + ring attention (phase 2).

Adding a new SP method only requires implementing a new CPStrategy subclass.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist

from ...models.layers.rope import apply_rotary_pos_emb
from .async_ulysses import async_ulysses_output_projection, async_ulysses_qkv_projection
from .data import slice_position_embedding
from .ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads


def _scale_cu_seqlens_for_ringattn(kwargs, ringattn_group):
    """Scale full cu_seqlens to local values for ring attention.

    The data pipeline produces cu_seqlens covering the full packed sequence.
    For ring attention, each ring rank only has S/ringattn_size tokens, so cu_seqlens
    boundaries must be scaled down.  Assumes each document is split uniformly
    across ring ranks (i.e. doc lengths are divisible by ringattn_size).

    Returns:
        (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k) — all None
        if cu_seqlens were not provided in kwargs.
    """
    cu_seqlens_q = kwargs.get("cu_seq_lens_q")
    cu_seqlens_k = kwargs.get("cu_seq_lens_k")
    max_seqlen_q = kwargs.get("max_length_q")
    max_seqlen_k = kwargs.get("max_length_k")

    if cu_seqlens_q is None:
        return None, None, None, None

    ringattn_size = dist.get_world_size(ringattn_group)

    # Scale cumulative lengths: each document contributes L/ringattn_size tokens
    cu_seqlens_q = (cu_seqlens_q // ringattn_size).to(torch.int32)
    cu_seqlens_k = (cu_seqlens_k // ringattn_size).to(torch.int32)

    if max_seqlen_q is not None:
        max_seqlen_q = max_seqlen_q // ringattn_size
    if max_seqlen_k is not None:
        max_seqlen_k = max_seqlen_k // ringattn_size

    return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


# ------------------------------------------------------------------ #
# Base class
# ------------------------------------------------------------------ #


class CPStrategy(ABC):
    """Base class for sequence parallel strategies.

    Each strategy implements three phases that map onto
    ``MultiHeadAttention.forward()``:

    1. ``project_qkv``  — QKV projection + pre-attention SP communication
    2. ``compute_attention`` — core attention (ring attention puts P2P here)
    3. ``project_output`` — post-attention SP communication + output projection
    """

    @abstractmethod
    def project_qkv(self, module, hidden_states, position_embeddings):
        """Phase 1: QKV projection + pre-attention SP communication.

        Default strategies delegate to ``module._project_qkv()`` for the
        model-specific projection (MHA, MLA, etc.) then add SP communication.
        """
        ...

    @abstractmethod
    def compute_attention(self, module, q, k, v, attention_mask, **kwargs):
        """Phase 2: Core attention computation.

        Most strategies just call the attention function directly.
        Ring attention would put its P2P KV transfer loop here.
        """
        ...

    @abstractmethod
    def project_output(self, module, attn_output):
        """Phase 3: Post-attention SP communication + output projection.

        Default strategies add SP communication then delegate to
        ``module._project_output()`` for the model-specific output projection.
        """
        ...

    @abstractmethod
    def prepare_position_embeddings(self, position_embeddings, dim, sp_group, num_kv_heads):
        """Prepare position embeddings (slice for sync, keep full for async)."""
        ...


# ------------------------------------------------------------------ #
# NoopStrategy — no sequence parallelism
# ------------------------------------------------------------------ #


class NoopStrategy(CPStrategy):
    """No sequence parallelism — delegates to module's own methods."""

    def project_qkv(self, module, hidden_states, position_embeddings):
        return module._project_qkv(hidden_states, position_embeddings)

    def compute_attention(self, module, q, k, v, attention_mask, **kwargs):
        attn_fn = module._get_attention_fn()
        attn_output, _ = attn_fn(module, q, k, v, attention_mask, **module._attention_kwargs(), **kwargs)
        return attn_output

    def project_output(self, module, attn_output):
        return module._project_output(attn_output)

    def prepare_position_embeddings(self, position_embeddings, **kwargs):
        return position_embeddings


# ------------------------------------------------------------------ #
# Ulysses sync — all-to-all around attention
# ------------------------------------------------------------------ #


class UlyssesSyncStrategy(CPStrategy):
    """Ulysses SP (sync): fused KV all-to-all with GQA head expansion.

    Used when ulysses_size > num_kv_heads.  Communication happens in
    project_qkv (pre-attention a2a) and project_output (post-attention a2a).
    """

    def __init__(self, group, ulysses_size: int):
        self.group = group
        self.ulysses_size = ulysses_size

    def project_qkv(self, module, hidden_states, position_embeddings):
        # Lazy-imported to break an import cycle with xorl.models.layers.attention
        from ...models.layers.attention.utils import repeat_kv  # noqa: PLC0415

        # Model-specific QKV projection (MHA, MLA, etc.)
        q, k, v = module._project_qkv(hidden_states, position_embeddings)

        # GQA expand if ulysses_size > num_kv_heads
        kv_head_num = k.shape[2]
        if self.ulysses_size > kv_head_num:
            assert self.ulysses_size % kv_head_num == 0, (
                f"ulysses_size ({self.ulysses_size}) must be divisible by num_key_value_heads ({kv_head_num})"
            )
            n_repeat = self.ulysses_size // kv_head_num
            # repeat_kv expects [batch, num_heads, seq, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            k = repeat_kv(k, n_repeat)
            v = repeat_kv(v, n_repeat)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # Pre-attention a2a: scatter seq, gather heads
        if q.ndim == 4 and q.size(0) == 1:
            q = q.squeeze(0)
            k = k.squeeze(0)
            v = v.squeeze(0)

            # Q all-to-all (separate — Q has different head count from K/V in GQA)
            q = gather_seq_scatter_heads(q, seq_dim=0, head_dim=1, group=self.group)
            # Fused KV all-to-all: interleave K/V heads into one tensor, single a2a
            kv = torch.stack([k, v], dim=2)  # [S, H_kv, 2, D]
            kv = kv.reshape(k.size(0), 2 * k.size(1), k.size(2))  # [S, 2*H_kv, D]
            kv = gather_seq_scatter_heads(kv, seq_dim=0, head_dim=1, group=self.group)
            kv = kv.reshape(kv.size(0), kv.size(1) // 2, 2, kv.size(2))  # [S_full, H_kv/SP, 2, D]
            k = kv[:, :, 0, :].contiguous()
            v = kv[:, :, 1, :].contiguous()

            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        else:
            q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, group=self.group)
            # Fused KV a2a for 4D tensors
            kv = torch.stack([k, v], dim=3)  # [B, S, H_kv, 2, D]
            kv = kv.reshape(k.size(0), k.size(1), 2 * k.size(2), k.size(3))  # [B, S, 2*H_kv, D]
            kv = gather_seq_scatter_heads(kv, seq_dim=1, head_dim=2, group=self.group)
            kv = kv.reshape(kv.size(0), kv.size(1), kv.size(2) // 2, 2, kv.size(3))  # [B, S_full, H_kv/SP, 2, D]
            k = kv[:, :, :, 0, :].contiguous()
            v = kv[:, :, :, 1, :].contiguous()

        return q, k, v

    def compute_attention(self, module, q, k, v, attention_mask, **kwargs):
        attn_fn = module._get_attention_fn()
        attn_output, _ = attn_fn(module, q, k, v, attention_mask, **module._attention_kwargs(), **kwargs)
        return attn_output

    def project_output(self, module, attn_output):
        # Post-attention a2a: gather heads, scatter seq
        if attn_output.ndim == 4 and attn_output.size(0) == 1:
            attn_output = attn_output.squeeze(0)
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1, group=self.group)
            attn_output = attn_output.unsqueeze(0)
        else:
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2, group=self.group)

        return module._project_output(attn_output)

    def prepare_position_embeddings(self, position_embeddings, dim, sp_group, **kwargs):
        return slice_position_embedding(position_embeddings, dim=dim, sp_group=sp_group)


# ------------------------------------------------------------------ #
# Ulysses async — overlapped linear + a2a communication
# ------------------------------------------------------------------ #


class UlyssesAsyncStrategy(CPStrategy):
    """Ulysses SP (async): overlaps QKV linear projections with a2a.

    Used when ulysses_size <= num_kv_heads.  The linear projection and
    a2a communication are overlapped for higher throughput.  This requires
    direct access to the module's weights (cannot delegate to _project_qkv).
    """

    def __init__(self, group, ulysses_size: int):
        self.group = group
        self.ulysses_size = ulysses_size

    def project_qkv(self, module, hidden_states, position_embeddings):
        # QLoRA fallback: weight is None (packed in quantized buffers).
        # Fall back to sync-style: module._project_qkv() + synchronous a2a.
        if not hasattr(module, "qkv_proj") or module.qkv_proj.weight is None:
            q, k, v = module._project_qkv(hidden_states, position_embeddings)
            if q.ndim == 4 and q.size(0) == 1:
                q = q.squeeze(0)
                k = k.squeeze(0)
                v = v.squeeze(0)
                q = gather_seq_scatter_heads(q, seq_dim=0, head_dim=1, group=self.group)
                kv = torch.stack([k, v], dim=2).reshape(k.size(0), 2 * k.size(1), k.size(2))
                kv = gather_seq_scatter_heads(kv, seq_dim=0, head_dim=1, group=self.group)
                kv = kv.reshape(kv.size(0), kv.size(1) // 2, 2, kv.size(2))
                k = kv[:, :, 0, :].contiguous()
                v = kv[:, :, 1, :].contiguous()
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            else:
                q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, group=self.group)
                kv = torch.stack([k, v], dim=3).reshape(k.size(0), k.size(1), 2 * k.size(2), k.size(3))
                kv = gather_seq_scatter_heads(kv, seq_dim=1, head_dim=2, group=self.group)
                kv = kv.reshape(kv.size(0), kv.size(1), kv.size(2) // 2, 2, kv.size(3))
                k = kv[:, :, :, 0, :].contiguous()
                v = kv[:, :, :, 1, :].contiguous()
            return q, k, v

        # Squeeze to 2D [S, H] — async autograd Function requires 2D (manual matmul in backward)
        hs_2d = hidden_states.squeeze(0)

        # Split fused qkv_proj weight into Q/K/V views (no copy)
        q_weight = module.qkv_proj.weight[: module.q_dim]
        k_weight = module.qkv_proj.weight[module.q_dim : module.q_dim + module.kv_dim]
        v_weight = module.qkv_proj.weight[module.q_dim + module.kv_dim :]
        q_bias = k_bias = v_bias = None
        if module.qkv_proj.bias is not None:
            q_bias = module.qkv_proj.bias[: module.q_dim]
            k_bias = module.qkv_proj.bias[module.q_dim : module.q_dim + module.kv_dim]
            v_bias = module.qkv_proj.bias[module.q_dim + module.kv_dim :]

        # Async QKV projection + a2a (overlaps linear compute with NCCL comm)
        q, k, v = async_ulysses_qkv_projection(
            hidden_states=hs_2d,
            seq_dimension=0,
            head_dimension=1,
            q_weight=q_weight,
            q_bias=q_bias,
            k_weight=k_weight,
            k_bias=k_bias,
            v_weight=v_weight,
            v_bias=v_bias,
            norm_type=None,
            norm_q_weight=None,
            norm_q_bias=None,
            norm_k_weight=None,
            norm_k_bias=None,
            normalized_shape=None,
            eps=None,
            unpadded_dim_size=None,
            head_dim=module.head_dim,
            group=self.group,
        )
        # q: [S_full, Hq/SP, D], k: [S_full, Hkv/SP, D], v: [S_full, Hkv/SP, D]

        # Apply RMSNorm and RoPE externally (after a2a, on full-length tensors)
        if getattr(module, "_use_qk_norm", False):
            q = module.q_norm(q)
            k = module.k_norm(k)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        cos, sin = position_embeddings  # full-length S_full (NOT sliced)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def compute_attention(self, module, q, k, v, attention_mask, **kwargs):
        attn_fn = module._get_attention_fn()
        attn_output, _ = attn_fn(module, q, k, v, attention_mask, **module._attention_kwargs(), **kwargs)
        return attn_output

    def project_output(self, module, attn_output):
        # QLoRA fallback: weight is None (packed in quantized buffers).
        # Fall back to sync-style: a2a + module.o_proj() forward.
        if module.o_proj.weight is None:
            if attn_output.ndim == 4 and attn_output.size(0) == 1:
                attn_output = attn_output.squeeze(0)
                attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1, group=self.group)
                attn_output = attn_output.unsqueeze(0)
            else:
                attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2, group=self.group)
            return module._project_output(attn_output)

        # Async output projection: a2a + o_proj (backward overlaps a2a with weight grad)
        attn_output = attn_output.squeeze(0)  # [S_full, Hq/SP, D]
        attn_output = attn_output.reshape(attn_output.size(0), -1)  # [S_full, Hq/SP * D]
        attn_output = async_ulysses_output_projection(
            hidden_states=attn_output,
            seq_dimension=0,
            head_dimension=1,
            proj_weight=module.o_proj.weight,
            proj_bias=module.o_proj.bias,
            unpadded_dim_size=None,
            group=self.group,
        )
        attn_output = attn_output.unsqueeze(0)  # [1, S_local, hidden_size]
        return attn_output

    def prepare_position_embeddings(self, position_embeddings, **kwargs):
        return position_embeddings  # Async: full-length (RoPE applied after a2a)


# ------------------------------------------------------------------ #
# Ring attention
# ------------------------------------------------------------------ #


class RingAttentionStrategy(CPStrategy):
    """Ring attention without Ulysses.

    Each ring rank holds a shard of the sequence.  KV is all-gathered across
    the ring group, and flash attention is computed per-step with online LSE
    merging.  No communication in phases 1 or 3.
    """

    def __init__(self, ringattn_group):
        self.ringattn_group = ringattn_group

    def project_qkv(self, module, hidden_states, position_embeddings):
        return module._project_qkv(hidden_states, position_embeddings)

    def compute_attention(self, module, q, k, v, attention_mask, **kwargs):
        from .ring_attention import ring_flash_attention_forward  # noqa: PLC0415

        attn_kwargs = module._attention_kwargs()
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = _scale_cu_seqlens_for_ringattn(
            kwargs, self.ringattn_group
        )

        return ring_flash_attention_forward(
            q,
            k,
            v,
            ringattn_group=self.ringattn_group,
            softmax_scale=attn_kwargs.get("scaling"),
            dropout_p=attn_kwargs.get("dropout", 0.0),
            causal=getattr(module, "is_causal", True),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    def project_output(self, module, attn_output):
        return module._project_output(attn_output)

    def prepare_position_embeddings(self, position_embeddings, dim, sp_group, **kwargs):
        return slice_position_embedding(position_embeddings, dim=dim, sp_group=sp_group)


# ------------------------------------------------------------------ #
# Hybrid Ulysses + Ring
# ------------------------------------------------------------------ #


class HybridUlyssesRingStrategy(CPStrategy):
    """Hybrid Ulysses + Ring attention.

    Data flow with ``cp_size = ulysses_size * ringattn_size``:

    1. Input arrives sharded by ``cp_size``.
    2. **project_qkv** (Ulysses a2a): gather seq within Ulysses group,
       scatter heads.  After this each rank has ``S/ringattn_size`` tokens with
       ``H/ulysses_size`` heads.
    3. **compute_attention** (Ring): ring flash attention across ring group
       rotates KV across ``ringattn_size`` ranks to attend to all ``S`` tokens.
    4. **project_output** (Ulysses a2a reverse): gather heads, scatter seq.
    """

    def __init__(self, ulysses_group, ringattn_group, ulysses_size: int):
        self.ulysses_group = ulysses_group
        self.ringattn_group = ringattn_group
        self.ulysses_size = ulysses_size
        self._ulysses = UlyssesSyncStrategy(group=ulysses_group, ulysses_size=ulysses_size)

    def project_qkv(self, module, hidden_states, position_embeddings):
        return self._ulysses.project_qkv(module, hidden_states, position_embeddings)

    def compute_attention(self, module, q, k, v, attention_mask, **kwargs):
        from .ring_attention import ring_flash_attention_forward  # noqa: PLC0415

        attn_kwargs = module._attention_kwargs()
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = _scale_cu_seqlens_for_ringattn(
            kwargs, self.ringattn_group
        )

        return ring_flash_attention_forward(
            q,
            k,
            v,
            ringattn_group=self.ringattn_group,
            softmax_scale=attn_kwargs.get("scaling"),
            dropout_p=attn_kwargs.get("dropout", 0.0),
            causal=getattr(module, "is_causal", True),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    def project_output(self, module, attn_output):
        return self._ulysses.project_output(module, attn_output)

    def prepare_position_embeddings(self, position_embeddings, dim, sp_group, **kwargs):
        return slice_position_embedding(position_embeddings, dim=dim, sp_group=sp_group)


# ------------------------------------------------------------------ #
# Strategy resolver
# ------------------------------------------------------------------ #

_NOOP = NoopStrategy()


def get_cp_strategy(num_kv_heads: Optional[int] = None) -> CPStrategy:
    """Resolve the SP strategy from the current ParallelState.

    Returns a singleton NoopStrategy when SP is disabled, or the
    appropriate strategy based on the model's ``num_kv_heads`` and the
    configured parallelism dimensions.

    Priority:
    1. Hybrid Ulysses+Ring (both ulysses_size > 1 and ringattn_size > 1)
    2. Ulysses only (ulysses_size > 1)
    3. Ring only (ringattn_size > 1)

    Args:
        num_kv_heads: Number of key-value heads in the model.  Required when
            Ulysses SP is enabled to choose between sync and async variants.
    """
    from ...distributed.parallel_state import get_parallel_state  # noqa: PLC0415

    ps = get_parallel_state()
    if not ps.cp_enabled:
        return _NOOP

    if ps.ulysses_enabled and ps.ringattn_enabled:
        # Hybrid Ulysses + Ring
        return HybridUlyssesRingStrategy(
            ulysses_group=ps.ulysses_group,
            ringattn_group=ps.ringattn_group,
            ulysses_size=ps.ulysses_size,
        )

    if ps.ulysses_enabled:
        if num_kv_heads is not None and ps.ulysses_size <= num_kv_heads:
            return UlyssesAsyncStrategy(group=ps.ulysses_group, ulysses_size=ps.ulysses_size)
        else:
            return UlyssesSyncStrategy(group=ps.ulysses_group, ulysses_size=ps.ulysses_size)

    if ps.ringattn_enabled:
        return RingAttentionStrategy(ringattn_group=ps.ringattn_group)

    return _NOOP
