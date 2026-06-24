"""DSA Indexer for DeepSeek-V4 ``compress_ratio == 4`` layers.

Adapted from miles ``miles_plugins/models/deepseek_v4/ops/v4_indexer.py``:

* ``MegatronModule`` → ``nn.Module``.
* Megatron's ``TELinear`` with ``parallel_mode="duplicated"`` → plain
  ``nn.Linear`` (replicated by default in FSDP2; not listed in any TP plan).
* Process-group plumbing accepts plain ``tp_group`` + ``cp_group`` arguments
  (or ``None`` for the no-parallelism case) instead of a Megatron
  ``ProcessGroupCollection``.
* ``parallel_state.get_context_parallel_world_size`` → ``cp_group.size()``.
* Megatron-SP ``gather_from_sequence_parallel_region`` only fires when
  ``tp_group is not None and tp_group.size() > 1``; xorl uses Ulysses-style
  sequence parallelism (head-parallel, not seqlen-parallel), so a real
  TP-with-SP indexer path is left as a follow-up — guarded by an explicit
  assertion below to fail loudly if ``tp_group.size() > 1``.
* The miles ``indexer_replay_manager`` (RL rollout-replay infra) is out of
  scope for the V0 LoRA-only port and intentionally omitted; the indexer
  always computes top-k on the fly here.
* Config fields renamed:
  ``dsa_indexer_n_heads/_head_dim/_topk`` → ``index_n_heads/index_head_dim/index_topk``;
  ``qk_pos_emb_head_dim`` → ``qk_rope_head_dim``;
  ``dsv4_compress_rope_theta`` → ``compress_rope_theta``;
  ``rotary_base`` → ``rope_theta``.
"""

import einops
import torch
import torch.nn as nn

from .cp_utils import all_gather_cp, get_freqs_cis_for_cp
from .qat import fp8_simulate_qat
from .rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from .utils import dsv4_kv_qat_enabled, rotate_activation


# ``tilelang_indexer_fwd`` is imported lazily inside ``forward`` because it
# requires tilelang. Hosts without tilelang can still import this module to
# instantiate ``V4Indexer``.


class V4Indexer(nn.Module):
    """Top-k indexer over compressed KV positions.

    Inputs are in SBHD layout (``[seqlen, batch, ...]``); the kernel runs
    batched and returns ``topk_indices : [batch, seqlen, index_topk] int64``.
    """

    def __init__(
        self,
        config,
        tp_group: torch.distributed.ProcessGroup | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ):
        super().__init__()

        from .compressor import DeepSeekV4Compressor  # noqa: PLC0415

        self.config = config
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = 4

        self.tp_group = tp_group
        self.cp_group = cp_group
        self.tp_size = tp_group.size() if tp_group is not None else 1
        self.cp_size = cp_group.size() if cp_group is not None else 1
        # Megatron-style sequence-parallel gathers are not yet wired here;
        # xorl uses Ulysses (head-parallel) SP. For LoRA training with TP=1
        # this is a no-op. Fail loudly if someone tries TP > 1.
        assert self.tp_size == 1, (
            "V4Indexer with TP > 1 is not implemented yet — needs an xorl-style "
            "Ulysses-aware SP gather in place of Megatron's "
            "gather_from_sequence_parallel_region."
        )

        # Replicated weights — not listed in any TP_PLAN; treated as duplicated
        # by FSDP2 across the TP mesh in the same way as the miles
        # ``parallel_mode="duplicated"`` TELinears.
        self.linear_wq_b = nn.Linear(
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
        )
        self.linear_weights_proj = nn.Linear(
            self.hidden_size,
            self.index_n_heads,
            bias=False,
        )

        self.compressor = DeepSeekV4Compressor(
            config=config,
            head_dim=self.index_head_dim,
            compress_ratio=self.compress_ratio,
            rotate=True,
            cp_group=cp_group,
        )

        # Cache the KV-QAT config at construction (matches DeepSeekV4Attention /
        # DeepSeekV4Compressor) so forward does not re-read model config per call.
        self._kv_qat_enabled = dsv4_kv_qat_enabled(config)

        rope_base = config.compress_rope_theta if self.compress_ratio else config.rope_theta
        freqs_cis = wrapped_precompute_freqs_cis(config, rope_head_dim=self.rope_head_dim, base=rope_base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:  hidden states ``[seqlen, batch, hidden_size]``.
            qr: low-rank query ``[seqlen, batch, q_lora_rank]``.

        Returns:
            ``topk_indices : [batch, seqlen, index_topk]`` (int64).
        """
        from .kernel.tilelang_indexer_fwd import _make_causal_cu_seqlens, batched_indexer_fwd  # noqa: PLC0415

        seqlen, bsz, _ = x.size()

        q = self.linear_wq_b(qr)
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)

        rd = self.rope_head_dim
        freqs_cis = get_freqs_cis_for_cp(self.freqs_cis, seqlen, self.cp_size, self.cp_group, stride=1)
        q = q.clone()
        q = einops.rearrange(q, "s b ... -> b s ...")
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = einops.rearrange(q, "b s ... -> s b ...")

        q = rotate_activation(q)
        if self._kv_qat_enabled:
            q = fp8_simulate_qat(q, block_size=128)

        k = self.compressor(x)

        weights = self.linear_weights_proj(x)
        softmax_scale = self.index_head_dim**-0.5
        weights = weights * (self.index_n_heads**-0.5) * softmax_scale

        if self.cp_size > 1 and self.cp_group is not None:
            k = all_gather_cp(k, dim=0, cp_group=self.cp_group)

        seqlen_global = seqlen * self.cp_size
        seqlen_kv = k.shape[0]
        cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_global, seqlen_kv, self.compress_ratio, q.device)
        if self.cp_size > 1 and self.cp_group is not None:
            cp_rank = self.cp_group.rank()
            cu_ks = cu_ks[cp_rank * seqlen : (cp_rank + 1) * seqlen]
            cu_ke = cu_ke[cp_rank * seqlen : (cp_rank + 1) * seqlen]
        index_scores = batched_indexer_fwd(q, k, weights.float(), cu_ks, cu_ke)

        topk_k = min(self.index_topk, index_scores.size(-1))
        topk_indices = index_scores.topk(topk_k, dim=-1)[1]
        return topk_indices
