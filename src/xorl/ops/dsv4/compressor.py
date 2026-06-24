"""DeepSeek-V4 KV compressor.

Adapted from miles ``miles_plugins/models/deepseek_v4/ops/compressor.py``:

* Drops the Megatron ``TransformerConfig`` import; reads
  ``hidden_size`` / ``qk_rope_head_dim`` / ``rms_norm_eps`` /
  ``compress_rope_theta`` from xorl's ``DeepseekV4Config``.
* The compressor body is fp32 throughout (``wkv``, ``wgate``, ``ape``, ``norm``)
  for numerical stability of the variance accumulation; this is intentional and
  aligns with the train/infer FP32-accumulator contract.
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from .cp_utils import all_gather_cp, get_freqs_cis_for_cp
from .qat import fp8_simulate_qat
from .rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from .utils import dsv4_kv_qat_enabled, rotate_activation


class RMSNorm(nn.Module):
    """FP32 RMSNorm. The compressor pipeline runs in FP32 end-to-end, so this
    is intentionally a pure-PyTorch FP32 norm rather than xorl's BF16-cast norm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _overlap_transform(tensor: torch.Tensor, *, compress_ratio: int, head_dim: int, value=0) -> torch.Tensor:
    """Overlap-transform for ``compress_ratio == 4``.

    For each token group of size ``ratio``, split into (first_half, second_half)
    halves along ``head_dim`` and re-arrange them across a doubled ratio axis
    (``2 * ratio``), shifting the first half by one group so that adjacent groups
    overlap by ``ratio`` positions.
    """
    b, s, _, _ = tensor.size()
    new_tensor = tensor.new_full((b, s, 2 * compress_ratio, head_dim), value)
    new_tensor[:, :, compress_ratio:] = tensor[:, :, :, head_dim:]
    new_tensor[:, 1:, :compress_ratio] = tensor[:, :-1, :, :head_dim]
    return new_tensor


class DeepSeekV4Compressor(nn.Module):
    """Per-layer KV compressor.

    ``compress_ratio == 4``: uses overlap_transform so adjacent groups overlap by
    ``ratio`` positions (DSA path).
    ``compress_ratio == 128``: static-pool, no overlap.

    Args:
        config: a ``DeepseekV4Config`` (only ``hidden_size``, ``qk_rope_head_dim``,
            ``rms_norm_eps``, ``compress_rope_theta`` are read).
        head_dim: latent dim of the compressor (128 for the indexer compressor,
            512 for the attention compressor).
        compress_ratio: 4 or 128.
        rotate: when True, applies the Hadamard rotation + optional FP8 QAT to
            the compressor output (DSA-indexer path).
        cp_group: optional context-parallel process group; when ``None`` /
            size 1, all CP gathers/slices become no-ops.
    """

    def __init__(
        self,
        config,
        head_dim: int,
        compress_ratio: int,
        rotate: bool,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ):
        super().__init__()

        dim = config.hidden_size
        rope_head_dim = config.qk_rope_head_dim
        norm_eps = config.rms_norm_eps

        # Internal consistency only. Literal Flash dims (head_dim ∈ {128, 512},
        # rope_head_dim == 64, norm_eps == 1e-6) are encoded in
        # DeepseekV4Config defaults — re-asserting them here would prevent
        # unit tests from running at compact dims.
        assert compress_ratio in {4, 128}
        assert rope_head_dim % 2 == 0, "rope_head_dim must be even for view_as_complex"
        assert head_dim > rope_head_dim

        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = head_dim - rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.cp_rank = cp_group.rank() if cp_group is not None else 0

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, norm_eps)

        for p in [self.ape, self.wkv.weight, self.wgate.weight]:
            p._keep_fp32 = True

        # Cache KV-QAT config at construction so forward_raw does not re-read
        # model config on every call.
        self._kv_qat_enabled = dsv4_kv_qat_enabled(config)

        base = config.compress_rope_theta
        freqs_cis = wrapped_precompute_freqs_cis(config, rope_head_dim=rope_head_dim, base=base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def overlap_transform_raw(self, tensor: torch.Tensor, value=0):
        return _overlap_transform(tensor, compress_ratio=self.compress_ratio, head_dim=self.head_dim, value=value)

    def overlap_transform_with_cp(self, tensor: torch.Tensor, value=0) -> torch.Tensor:
        if self.cp_size == 1:
            return self.overlap_transform_raw(tensor, value)

        tensor = all_gather_cp(tensor, dim=1, cp_group=self.cp_group)
        tensor = self.overlap_transform_raw(tensor, value)

        G_local = tensor.shape[1] // self.cp_size
        start = self.cp_rank * G_local
        return tensor[:, start : start + G_local, :, :]

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        # ``ape`` / ``wkv`` / ``wgate`` are nominally fp32 (the compressor
        # pipeline runs in fp32 end-to-end for variance-accumulation
        # stability) — but ``self.wkv(x_fp32)`` below promotes regardless,
        # so the bf16-cast smoke path is functionally OK.
        bsz, seqlen_local, _ = x.size()
        ratio, overlap, _ = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype

        assert (seqlen_local >= ratio) and (seqlen_local % ratio == 0), f"{seqlen_local=} {ratio=}"
        if self.cp_size > 1 and overlap:
            assert seqlen_local % (ratio * 2) == 0, f"{seqlen_local=} {ratio=} {self.cp_size=} overlap={overlap}"

        x_fp32 = x.float()
        # ``wkv`` / ``wgate`` / ``ape`` are nominally fp32 in the model
        # spec; promote the weights here so the compute path is fp32
        # end-to-end regardless of how the model was cast (FSDP-friendly
        # bf16 vs the original training-time fp32 layout).
        kv = F.linear(x_fp32, self.wkv.weight.float())
        score = F.linear(x_fp32, self.wgate.weight.float())
        ape_fp32 = self.ape.float()

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + ape_fp32

        if overlap:
            kv = self.overlap_transform_with_cp(kv, 0)
            score = self.overlap_transform_with_cp(score, float("-inf"))

        score_softmax = score.softmax(dim=2)
        kv = (kv * score_softmax).sum(dim=2)

        kv = self.norm(kv.to(dtype))

        freqs_cis = get_freqs_cis_for_cp(self.freqs_cis, seqlen_local, self.cp_size, self.cp_group, stride=ratio)

        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)
            if self._kv_qat_enabled:
                kv = fp8_simulate_qat(kv, 128)
        elif self._kv_qat_enabled:
            kv = kv.clone()
            kv[..., : self.nope_head_dim] = fp8_simulate_qat(kv[..., : self.nope_head_dim], 64)

        return kv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress KV.

        Args:
            x: ``[seqlen, batch, dim]`` (SBHD).
        Returns:
            ``[seqlen // compress_ratio, batch, head_dim]`` (SBHD).
        """
        x_bshd = einops.rearrange(x, "s b d -> b s d")
        k_bshd = self.forward_raw(x_bshd)
        k = einops.rearrange(k_bshd, "b sc d -> sc b d")
        return k
