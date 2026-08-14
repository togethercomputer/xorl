"""DeepSeek-V4 attention layer (xorl native).

Adapted from miles ``miles_plugins/models/deepseek_v4/deepseek_v4.py``.

Per-layer attention has three flavours, switched by
``config.compress_ratios[layer_id]``:

- ``0`` — pure window attention (window=128). No compressor, no indexer.
- ``128`` — window + static-pool compressed KV (compressor only).
- ``4`` — window + 4-token-pool compressed KV + DSA learned indexer that
  selects ``index_topk`` compressed groups per query (the C4 path).

Layout note: this module uses the xorl/HF convention of BSHD
(``[batch, seqlen, hidden]``) at its public boundary. The compressor /
indexer ops we ported from miles use SBHD internally, so the boundary
``rearrange("b s d -> s b d")`` happens here.

The MLA-like projection chain (``wq_a``→``q_norm``→``wq_b`` / shared
``wkv``→``kv_norm``) and the grouped output projection
(``wo_a``→``wo_b`` factored through ``o_groups`` × ``o_lora_rank``) match
miles verbatim so the HF state-dict loader (Phase 5) can be a direct
name-renaming map.
"""

import os

import einops
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from xorl.distributed.canonical_moe import LogicalRowOwnership
from xorl.distributed.parallel_state import get_parallel_state
from xorl.lora.expert_adapter_contract import DSV4_CLAMPED_SWIGLU_LORA_PROGRAM
from xorl.lora.modules.linear import LoraLinear
from xorl.models.base import XorlPreTrainedModel
from xorl.models.layers import ACT2FN
from xorl.models.layers.moe.moe_block import MoEBlock
from xorl.models.layers.moe.router import TopKRouter
from xorl.models.layers.moe.routing_replay import get_replay_stage
from xorl.models.layers.normalization import RMSNorm
from xorl.models.module_utils import DEFAULT_GRADIENT_CHECKPOINTING_METHOD
from xorl.models.outputs import MoeCausalLMOutput, MoeModelOutput
from xorl.ops.dsv4.attention_core import dense_attn_torch, sparse_attn_tilelang, sparse_attn_torch
from xorl.ops.dsv4.compressor import DeepSeekV4Compressor
from xorl.ops.dsv4.cp_utils import (
    Dsv4ExactCPLayout,
    all_gather_cp,
    build_dsv4_exact_cp_layout,
    gather_dsv4_exact_cp_rows,
    get_compress_topk_idxs_cp,
    get_freqs_cis_for_cp,
    get_q_positions_for_cp,
    get_window_topk_idxs_cp,
)
from xorl.ops.dsv4.hyper_connection import DeepSeekV4HyperConnectionUtil
from xorl.ops.dsv4.qat import fp8_simulate_qat
from xorl.ops.dsv4.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from xorl.ops.dsv4.utils import dsv4_kv_qat_enabled
from xorl.ops.dsv4.v4_indexer import V4Indexer


# Defaults: when ``XORL_DSV4_SPARSE_ATTN_IMPL`` is unset, autodetect —
# ``tilelang`` if CUDA is available and the package imports cleanly,
# otherwise the pure-torch ``sparse`` reference. Set the env explicitly
# to override (``tilelang`` / ``sparse`` / ``dense``).
#
# **Env caching contract:** the value of ``XORL_DSV4_SPARSE_ATTN_IMPL`` is read
# once at ``DeepSeekV4Attention`` construction (see ``self._attn_impl``) and not
# re-read in ``forward``. Tests that monkeypatch this env var must set it BEFORE
# constructing the module — patching after construction silently uses the cached
# value.
_ATTN_IMPL_ENV = "XORL_DSV4_SPARSE_ATTN_IMPL"
_ATTN_IMPL_CHOICES = {"tilelang", "sparse", "dense"}


def _move_preserved_param(
    param: nn.Parameter,
    device: torch.device | None,
    non_blocking: bool,
) -> nn.Parameter:
    if device is None:
        return param

    moved = nn.Parameter(
        param.detach().to(device=device, dtype=param.dtype, non_blocking=non_blocking),
        requires_grad=param.requires_grad,
    )
    moved.__dict__.update(param.__dict__)
    if param.grad is not None:
        moved.grad = param.grad.to(device=device, dtype=param.grad.dtype, non_blocking=non_blocking)
    return moved


def _move_preserved_buffer(
    buffer: torch.Tensor,
    device: torch.device | None,
    non_blocking: bool,
) -> torch.Tensor:
    if device is None:
        return buffer
    return buffer.to(device=device, non_blocking=non_blocking)


def _dsv4_model_to(model: nn.Module, *args, preserve_keep_fp32: bool = True, **kwargs) -> nn.Module:
    device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
    if dtype is None or not dtype.is_floating_point:
        return nn.Module.to(model, *args, **kwargs)

    # ``nn.Module.to(dtype)`` casts complex buffers to real tensors and strips
    # the imaginary RoPE phase. It also downcasts DSv4 parameters that are
    # intentionally marked fp32-only. Temporarily hide both from the traversal,
    # cast the rest, then restore them with device moves but without dtype casts.
    preserved_params: list[tuple[nn.Module, str, nn.Parameter]] = []
    preserved_buffers: list[tuple[nn.Module, str, torch.Tensor]] = []

    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if preserve_keep_fp32 and param is not None and getattr(param, "_keep_fp32", False):
                preserved_params.append((module, name, param))
                module._parameters[name] = None
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and torch.is_complex(buffer):
                preserved_buffers.append((module, name, buffer))
                module._buffers[name] = None

    try:
        nn.Module.to(model, *args, **kwargs)
    finally:
        for module, name, param in preserved_params:
            module._parameters[name] = _move_preserved_param(param, device, non_blocking)
        for module, name, buffer in preserved_buffers:
            module._buffers[name] = _move_preserved_buffer(buffer, device, non_blocking)

    return model


def cast_dsv4_model_dtype(
    model: nn.Module,
    torch_dtype: torch.dtype,
    *,
    preserve_keep_fp32: bool = True,
) -> nn.Module:
    """Cast DSv4 params/buffers while preserving complex RoPE caches and optional fp32 carve-outs."""
    return _dsv4_model_to(model, torch_dtype, preserve_keep_fp32=preserve_keep_fp32)


def _resolve_dsv4_torch_dtype(value, config, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    if value is None or value == "auto":
        value = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None) or default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        value = value.removeprefix("torch.")
        dtype = getattr(torch, value, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    raise TypeError(f"Unsupported DeepseekV4 dtype: {value!r}")


def _resolve_dsv4_weight_path(pretrained_model_name_or_path, subfolder: str | None):
    if subfolder and os.path.isdir(pretrained_model_name_or_path):
        return os.path.join(os.fspath(pretrained_model_name_or_path), subfolder)
    return pretrained_model_name_or_path


def _select_attn_impl():
    # When unset, autodetect: prefer ``tilelang`` if both CUDA is available
    # and the kernel module imports cleanly. Otherwise fall back to the
    # pure-torch ``sparse`` reference path. Anyone running on H100 without
    # an explicit env now gets the fast path by default; CPU/no-tilelang
    # environments still get the safe fallback.
    impl = os.environ.get(_ATTN_IMPL_ENV)
    if impl is None:
        if torch.cuda.is_available():
            try:
                import tilelang  # noqa: F401, PLC0415

                impl = "tilelang"
            except Exception:
                impl = "sparse"
        else:
            impl = "sparse"
    if impl not in _ATTN_IMPL_CHOICES:
        raise ValueError(f"{_ATTN_IMPL_ENV} must be one of {_ATTN_IMPL_CHOICES}, got {impl!r}")
    return impl


class _ExactBatchInvariantRmsNorm(torch.autograd.Function):
    """Serving-value RMSNorm: BI Triton kernel forward, surrogate VJP backward.

    The deterministic serving contract patches standalone RMSNorms to the
    batch-invariant kernel; its bytes differ from sgl_kernel/native rmsnorm by
    one BF16 ulp at rounding boundaries. The kernel carries no autograd, so
    backward differentiates the native recompute (the norm weight is frozen
    base state and receives no gradient).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (  # noqa: PLC0415
            rms_norm_batch_invariant,
        )

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = rms_norm_batch_invariant(flat, weight, eps=eps).reshape_as(x)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        with torch.enable_grad():
            x_live = x.detach().requires_grad_(True)
            hidden = x_live.float()
            variance = hidden.pow(2).mean(-1, keepdim=True)
            surrogate = (hidden * torch.rsqrt(variance + ctx.eps)).to(x.dtype) * weight
            (grad_x,) = torch.autograd.grad(surrogate, (x_live,), grad_output, create_graph=False)
        return grad_x, None, None


class DeepSeekV4Attention(nn.Module):
    """V4 sparse-MLA attention with per-layer ``compress_ratio``.

    Args:
        config: ``DeepseekV4Config`` instance.
        layer_id: 0-based layer index. Reads
            ``config.compress_ratios[layer_id]`` to pick the variant.
        tp_group: optional TP process group. Only ``tp_size == 1`` is
            supported in this V0 (xorl SP gather is a follow-up).
        cp_group: optional CP process group. ``None`` / size 1 = no CP.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        tp_group: torch.distributed.ProcessGroup | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ):
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        # Cache dispatch/config values at construction so the forward path does
        # not pay env/config lookups for every layer × every step.
        self._attn_impl = _select_attn_impl()
        self._kv_qat_enabled = dsv4_kv_qat_enabled(config)
        self._exact_attention = bool(getattr(config, "_dsv4_flash_exact_mode", False))

        self.tp_group = tp_group
        self.cp_group = cp_group
        self.tp_size = tp_group.size() if tp_group is not None else 1
        self.cp_size = cp_group.size() if cp_group is not None else 1
        assert self.tp_size == 1, (
            "DeepSeekV4Attention with TP > 1 is not implemented yet — needs an "
            "xorl-style Ulysses-aware SP gather/scatter to replace miles's "
            "Megatron gather_from_sequence_parallel_region calls."
        )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // self.tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // self.tp_size
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_id] if config.compress_ratios is not None else 0
        self.eps = config.rms_norm_eps

        # Internal consistency. The literal Flash dims (1024 / 512 / 64 / 128)
        # live in DeepseekV4Config defaults — we don't re-assert them here so
        # the layer remains usable at small dims for unit tests.
        assert self.nope_head_dim == self.head_dim - self.rope_head_dim
        assert self.n_heads % self.n_groups == 0, (
            f"num_attention_heads ({self.n_heads}) must be divisible by o_groups ({self.n_groups})"
        )
        assert self.compress_ratio in (0, 4, 128)

        # Per-head fp32 scalar entered into the softmax denominator.
        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.attn_sink._keep_fp32 = True

        # Q double-LoRA projections.
        # wq_a: replicated (parallel_mode="duplicated" in miles); not in TP_PLAN.
        # wq_b: colwise (sharded across tp on the head axis); listed in TP_PLAN.
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # Shared KV down-projection (replicated). Last 64 dims = RoPE.
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)

        # Output projection factored through groups × o_lora_rank.
        # wo_a operates per-group (see einsum trick in forward).
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        self.softmax_scale = self.head_dim**-0.5

        # Compressor + (optionally) indexer for non-window layers.
        if self.compress_ratio:
            self.compressor = DeepSeekV4Compressor(
                config=config,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                rotate=False,
                cp_group=self.cp_group,
            )
            self.indexer = (
                V4Indexer(config=config, tp_group=self.tp_group, cp_group=self.cp_group)
                if self.compress_ratio == 4
                else None
            )
        else:
            self.compressor = None
            self.indexer = None

        # RoPE freqs for the attention KV stream. Compressed layers use the
        # compress theta; window layers use the base theta. The ordinary Miles
        # path disables YaRN smoothing for window-only layers, while the exact
        # lane follows the pinned SGLang model and retains the checkpoint YaRN
        # schedule for C0 as well.
        rope_base = config.compress_rope_theta if self.compress_ratio else config.rope_theta
        yarn_disabled = not self.compress_ratio and not self._exact_attention
        freqs_cis = wrapped_precompute_freqs_cis(
            config, rope_head_dim=self.rope_head_dim, base=rope_base, yarn_disabled=yarn_disabled
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        # The table above is one lru_cache-shared tensor object registered on
        # every layer with the same (base, yarn) args. Loaders that snapshot
        # named_buffers() (which deduplicates by object) and restore after
        # to_empty() would silently leave every layer but the first holder
        # with zeroed RoPE, so rebuild_shared_freqs_cis() re-registers the
        # tables post-materialization from these stashed args.
        self._freqs_cis_rebuild_args = (rope_base, yarn_disabled, self.rope_head_dim)

    def to(self, *args, **kwargs):
        return _dsv4_model_to(self, *args, preserve_keep_fp32=False, **kwargs)

    def _capture_diagnostic_component(self, name: str, value: torch.Tensor) -> None:
        capture = self.__dict__.get("_diagnostic_capture_component")
        if callable(capture):
            capture(name, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        exact_cp_layout: Dsv4ExactCPLayout | None = None,
        decode_carry_offset: int | None = None,
    ) -> torch.Tensor:
        """Run attention.

        Args:
            hidden_states: ``[batch, seqlen, hidden_size]``.

        Returns:
            ``[batch, seqlen, hidden_size]``.
        """
        x = hidden_states  # BSHD throughout

        def validate_lora_metadata(where: str) -> None:
            if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
                from .native_payload import _validate_all_single_adapter_batch_infos  # noqa: PLC0415

                _validate_all_single_adapter_batch_infos(
                    x.device.index,
                    where=f"layer {self.layer_id} attention {where}",
                )

        validate_lora_metadata("entry")
        self._capture_diagnostic_component("attention_input", x)
        bsz, seqlen_local, _ = x.size()
        # Decode-cache carry replays M=1 segments over persistent serving
        # cache state. The offset is an explicit per-call argument so pipeline
        # interleaving and checkpoint recompute cannot overwrite it.
        carry_offset = decode_carry_offset if self._exact_attention else None
        carry_state = None
        if carry_offset is not None:
            if self.cp_size != 1 or bsz != 1:
                raise RuntimeError("DSV4 exact decode-cache carry admits one request without context parallelism")
            carry_state = self.__dict__.get("_dsv4_decode_state")
            if carry_state is None:
                from xorl.ops.dsv4.exact_attention import Dsv4DecodeCarryState  # noqa: PLC0415

                carry_state = Dsv4DecodeCarryState()
                self._dsv4_decode_state = carry_state
            # Absolute positions index the full RoPE table directly.
            freqs_cis = self.freqs_cis
        else:
            freqs_cis = get_freqs_cis_for_cp(self.freqs_cis, seqlen_local, self.cp_size, self.cp_group)
        seqlen_global = seqlen_local * self.cp_size
        if exact_cp_layout is not None:
            if not self._exact_attention or carry_offset is not None:
                raise RuntimeError("DSV4 exact CP layout is only valid for an exact prefill call")
            if exact_cp_layout.compute_rows != seqlen_local:
                raise RuntimeError(
                    "DSV4 exact CP layout does not match the local attention shape: "
                    f"layout={exact_cp_layout.compute_rows}, attention={seqlen_local}"
                )
            rope_positions = exact_cp_layout.local_request_positions
            if rope_positions.numel() and int(rope_positions.max().item()) >= self.freqs_cis.size(0):
                raise ValueError(
                    "DSV4 RoPE cache is too short for the CP request layout: "
                    f"max position={int(rope_positions.max().item())}, "
                    f"cache length={self.freqs_cis.size(0)}"
                )
            freqs_cis = self.freqs_cis.index_select(
                0,
                rope_positions.to(device=self.freqs_cis.device),
            ).to(device=x.device)
            q_positions = rope_positions
            seqlen_global = int(exact_cp_layout.global_logical_rows.numel())
        else:
            q_positions = get_q_positions_for_cp(
                seqlen_local, cp_size=self.cp_size, cp_group=self.cp_group, device=x.device
            )
        # SGLang's DSV4 CP path shards queries for execution, but explicitly
        # all-gathers and reranges BF16 KV plus compressor scores back into
        # logical token order before the cache/compressor kernels.  The trainer
        # owns contiguous CP shards, so rank-order concatenation is already that
        # logical order.  Keep Q local (and carry its absolute positions) while
        # replaying the literal serving attention against the gathered source.
        exact_query_positions = (
            q_positions if self._exact_attention and self.cp_size > 1 and exact_cp_layout is None else None
        )
        exact_kv_freqs_cis = freqs_cis
        if exact_query_positions is not None:
            exact_kv_freqs_cis = get_freqs_cis_for_cp(
                self.freqs_cis,
                seqlen_global,
                1,
                None,
            )
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # ---------------- Q ----------------
        q_lora_pre_norm = self.wq_a(x)
        self._capture_diagnostic_component("q_pre_qk_norm", q_lora_pre_norm)
        if self._exact_attention:
            # The serving deterministic contract runs standalone RMSNorms
            # through the batch-invariant Triton kernel; sgl_kernel/native
            # rmsnorm differs by one BF16 ulp at rounding boundaries. Forward is the serving
            # kernel bytes; backward is the trainer-owned surrogate VJP.
            q_lora = _ExactBatchInvariantRmsNorm.apply(q_lora_pre_norm, self.q_norm.weight, self.eps)
        else:
            q_lora = self.q_norm(q_lora_pre_norm)  # [B, S, q_lora_rank]
        self._capture_diagnostic_component("q_post_qk_norm", q_lora)
        qr = q_lora  # saved for the indexer below
        q = self.wq_b(q_lora)  # [B, S, n_heads * head_dim]
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))  # [B, S, H, D]
        if self._exact_attention:
            from xorl.ops.dsv4.exact_attention import exact_q_norm_rope  # noqa: PLC0415

            q = exact_q_norm_rope(q, freqs_cis, self.eps, position_offset=carry_offset or 0)
        else:
            q_dtype = q.dtype
            q = q.float()
            q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)).to(q_dtype)
            # ``apply_rotary_emb`` (xorl.ops.dsv4.rope) writes into ``q[..., -rd:]``
            # in place, so the slice's storage must be exclusively ours. Without
            # ``q.clone()``, the in-place rotary would mutate the upstream
            # ``self.wq_b(...)`` activation that's still held by autograd, which
            # both poisons the backward graph and tangles up FSDP's all-gather
            # buffer. Same reasoning for ``kv_vanilla.clone()`` below. This
            # ~doubles peak transient memory on the Q/KV path.
            q = q.clone()
            apply_rotary_emb(q[..., -rd:], freqs_cis)
        validate_lora_metadata("Q projection and RoPE")
        self._capture_diagnostic_component("q", q)

        # ---------------- KV (single shared stream) ----------------
        kv_pre_norm = self.wkv(x)
        validate_lora_metadata("KV projection")
        self._capture_diagnostic_component("k_pre_qk_norm", kv_pre_norm)
        if self._exact_attention:
            if exact_cp_layout is not None:
                global_kv = None
                global_x = None
                if self.cp_size > 1:
                    # Gather raw WKV rows in logical request order.  The exact
                    # attention program then invokes serving's fused FP32
                    # norm/RoPE/FP8 store once over each complete request,
                    # matching the CP1 cache-byte boundary.
                    global_kv = gather_dsv4_exact_cp_rows(
                        kv_pre_norm,
                        dim=1,
                        layout=exact_cp_layout,
                        cp_group=self.cp_group,
                    )
                    if ratio:
                        global_x = gather_dsv4_exact_cp_rows(
                            x,
                            dim=1,
                            layout=exact_cp_layout,
                            cp_group=self.cp_group,
                        )

                o = torch.zeros_like(q)
                for local_rows, global_rows in zip(
                    exact_cp_layout.local_request_row_indices,
                    exact_cp_layout.global_request_row_indices,
                ):
                    if local_rows.numel() == 0:
                        continue
                    request_q = q.index_select(1, local_rows)
                    request_positions = exact_cp_layout.local_request_positions.index_select(0, local_rows)
                    request_length = int(global_rows.numel())
                    request_freqs = get_freqs_cis_for_cp(self.freqs_cis, request_length, 1, None)
                    if self.cp_size > 1:
                        request_kv = global_kv.index_select(1, global_rows)
                        request_x = global_x.index_select(1, global_rows) if ratio else None
                    else:
                        request_kv = kv_pre_norm.index_select(1, local_rows)
                        request_x = x.index_select(1, local_rows) if ratio else None

                    if ratio == 0:
                        from xorl.ops.dsv4.exact_attention import exact_c0_attention  # noqa: PLC0415

                        request_o = exact_c0_attention(
                            request_q,
                            request_kv,
                            self.kv_norm.weight,
                            self.attn_sink,
                            request_freqs,
                            self.eps,
                            self.softmax_scale,
                            query_positions=request_positions,
                            kv_preprocessed=False,
                        )
                    else:
                        from xorl.ops.dsv4.exact_attention import exact_compressed_attention  # noqa: PLC0415

                        request_o = exact_compressed_attention(
                            request_q,
                            request_kv,
                            request_x,
                            self.kv_norm.weight,
                            self.attn_sink,
                            request_freqs,
                            self.compressor.wkv.weight,
                            self.compressor.wgate.weight,
                            self.compressor.ape,
                            self.compressor.norm.weight,
                            self.eps,
                            self.softmax_scale,
                            ratio,
                            query_positions=request_positions,
                            kv_preprocessed=False,
                        )
                    o = o.index_copy(1, local_rows, request_o)
            else:
                exact_kv_pre_norm = kv_pre_norm
                exact_x = x
                if self.cp_size > 1:
                    exact_kv_pre_norm = all_gather_cp(exact_kv_pre_norm, dim=1, cp_group=self.cp_group)
                    if ratio:
                        exact_x = all_gather_cp(exact_x, dim=1, cp_group=self.cp_group)
                if ratio == 0:
                    from xorl.ops.dsv4.exact_attention import exact_c0_attention  # noqa: PLC0415

                    o = exact_c0_attention(
                        q,
                        exact_kv_pre_norm,
                        self.kv_norm.weight,
                        self.attn_sink,
                        exact_kv_freqs_cis,
                        self.eps,
                        self.softmax_scale,
                        carry_state=carry_state,
                        position_offset=carry_offset or 0,
                        query_positions=exact_query_positions,
                        kv_preprocessed=False,
                    )
                else:
                    from xorl.ops.dsv4.exact_attention import exact_compressed_attention  # noqa: PLC0415

                    o = exact_compressed_attention(
                        q,
                        exact_kv_pre_norm,
                        exact_x,
                        self.kv_norm.weight,
                        self.attn_sink,
                        exact_kv_freqs_cis,
                        self.compressor.wkv.weight,
                        self.compressor.wgate.weight,
                        self.compressor.ape,
                        self.compressor.norm.weight,
                        self.eps,
                        self.softmax_scale,
                        ratio,
                        carry_state=carry_state,
                        position_offset=carry_offset or 0,
                        query_positions=exact_query_positions,
                        kv_preprocessed=False,
                    )
            kv_vanilla = None
        else:
            kv_vanilla = self.kv_norm(kv_pre_norm)  # [B, S, D]
            kv_vanilla = kv_vanilla.clone()  # in-place rotary; see ``q.clone()`` note above.
            apply_rotary_emb(kv_vanilla[..., -rd:], freqs_cis)
            if self._kv_qat_enabled:
                kv_vanilla[..., : self.nope_head_dim] = fp8_simulate_qat(kv_vanilla[..., : self.nope_head_dim], 64)
            self._capture_diagnostic_component("k_post_qk_norm", kv_vanilla)

        if not self._exact_attention:
            # ---------------- topk indices: window + (optional) compress ----------------
            topk_idxs = get_window_topk_idxs_cp(q_positions, window_size=win, cp_size=self.cp_size, bsz=bsz)
            kv_compress_offset = seqlen_global
            if ratio:
                if self.indexer is not None:
                    # Indexer expects SBHD; rearrange at the boundary.
                    x_sbd = einops.rearrange(x, "b s d -> s b d")
                    qr_sbd = einops.rearrange(qr, "b s d -> s b d")
                    compress_topk_idxs = self.indexer(x_sbd, qr_sbd)  # [B, S, index_topk]
                    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
                    topk_idx_mask = (compress_topk_idxs >= q_first_invalid_group) | (compress_topk_idxs < 0)
                    compress_topk_idxs = torch.where(topk_idx_mask, -1, compress_topk_idxs + kv_compress_offset)
                else:
                    compress_topk_idxs = get_compress_topk_idxs_cp(
                        q_positions,
                        ratio=ratio,
                        cp_size=self.cp_size,
                        bsz=bsz,
                    )
                topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
            topk_idxs = topk_idxs.int()
            # ---------------- compressed KV ----------------
            kv_compress = None
            if ratio:
                kv_compress = self.compressor.forward_raw(x)  # [B, S//ratio, D]

            # ---------------- All-gather across CP for global KV ----------------
            if self.cp_size > 1:
                kv_vanilla = all_gather_cp(kv_vanilla, dim=1, cp_group=self.cp_group)
                if kv_compress is not None:
                    kv_compress = all_gather_cp(kv_compress, dim=1, cp_group=self.cp_group)

            if kv_compress is not None:
                kv = torch.cat([kv_vanilla, kv_compress], dim=1)
                assert kv_compress_offset == kv_vanilla.size(1)
            else:
                kv = kv_vanilla

            # ---------------- Sparse attention ----------------
            attn_sink = self.attn_sink.float()
            if self._attn_impl == "tilelang":
                o = sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, self.softmax_scale)
            elif self._attn_impl == "sparse":
                o = sparse_attn_torch(q, kv, attn_sink, topk_idxs, self.softmax_scale)
            else:  # "dense"
                o = dense_attn_torch(q, kv, attn_sink, topk_idxs, self.softmax_scale)
        validate_lora_metadata("attention core")
        self._capture_diagnostic_component("attn_output", o)

        # Inverse RoPE on the rope slice of the output.
        if self._exact_attention:
            from xorl.ops.dsv4.exact_attention import exact_inverse_rope  # noqa: PLC0415

            o = exact_inverse_rope(o, freqs_cis, position_offset=carry_offset or 0)
        else:
            apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
        validate_lora_metadata("inverse RoPE")
        self._capture_diagnostic_component("attn_output_gated", o)

        # ---------------- Grouped output projection ----------------
        # o : [B, S, H, D] -> [B, S, n_local_groups, n_heads*D / n_local_groups]
        o = o.view(bsz, seqlen_local, self.n_local_groups, -1)
        native_wo_a_payload = self.wo_a._modules.get("native_base_payload")
        if native_wo_a_payload is not None:
            from .native_payload import dsv4_native_grouped_wo_a  # noqa: PLC0415

            o_base = dsv4_native_grouped_wo_a(
                o.flatten(0, 1),
                self.wo_a,
                groups=self.n_local_groups,
                out_per_group=self.o_lora_rank,
            ).unflatten(0, (bsz, seqlen_local))
        else:
            wo_a_w = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o_base = torch.einsum("bsgd,grd->bsgr", o, wo_a_w)

        # When ``wo_a`` has been replaced with ``LoraLinear`` (attention LoRA
        # targeting "wo_a"), the standard ``LoraLinear.forward`` path is
        # bypassed by the einsum above. Recompute the LoRA delta in the
        # grouped form so the adapter actually contributes:
        #
        #   mid : F.linear(o, lora_A)                  -> [B, S, g, r]
        #   lora_B reshaped to [g, o_lora_rank, r]
        #   delta = einsum("bsgr,gor->bsgo", mid, lora_B_g) * scaling
        #
        # ``lora_A`` is shared across groups (one down-projection of dim
        # n_heads*head_dim/n_groups); ``lora_B`` carries per-group output
        # mixtures via its first dim. This is the natural map of a flat
        # LoraLinear onto a grouped projection — the adapter is rank-r
        # per-group on the up-side and shared on the down-side.
        if isinstance(self.wo_a, LoraLinear) and native_wo_a_payload is None:
            lora_A = self.wo_a.lora_A
            lora_B = self.wo_a.lora_B
            mid = F.linear(o.to(lora_A.dtype), lora_A)
            lora_B_g = lora_B.view(self.n_local_groups, self.o_lora_rank, -1)
            delta = torch.einsum("bsgr,gor->bsgo", mid, lora_B_g) * self.wo_a.scaling
            o_base = o_base + delta.to(o_base.dtype)
        self._capture_diagnostic_component("o_proj_output", o_base)

        return self.wo_b(o_base.flatten(2))


class DeepseekV4MLP(nn.Module):
    """SwiGLU dense MLP. Used for the DSv4 shared expert.

    With ``config.swiglu_limit > 0`` the gate pre-activation is clamped to
    ``[-limit, +limit]`` before ``silu`` (training-stability fix that ships
    with the 0415 ckpt).
    """

    def __init__(self, config, intermediate_size: int | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class DeepseekV4MoE(MoEBlock):
    """V4 sparse MoE block.

    Inherits from xorl's ``MoEBlock`` to reuse the gate / experts / dispatch /
    LoRA-injection / EP plumbing, and overrides only what differs:

    - The router is replaced with ``TopKRouter(scoring_func="sqrtsoftplus",
      topk_method=...)`` — ``noaux_tc`` for non-hash layers, ``None`` for
      hash layers (selection comes from ``tid2eid``).
    - For non-hash layers, an ``e_score_correction_bias`` parameter is
      attached to ``self.gate`` (HF convention puts it on the gate, not on
      the router).
    - For hash layers (``layer_id < num_hash_layers``), a frozen
      ``[vocab_size, num_experts_per_tok]`` lookup is registered as a
      buffer; ``forward(input_ids=...)`` is required.
    - A single shared expert (``DeepseekV4MLP``, intermediate sized at
      ``moe_intermediate_size * n_shared_experts``) is added to the routed
      output before returning.

    Routing-replay support: ``MoEBlock._regather_routing`` dispatches on
    ``self.router.scoring_func`` and recovers ``sqrtsoftplus`` weights from
    cached expert indices via ``sqrt(softplus(logits))``; ``self.route()``
    overrides the parent's flow to integrate replay while still passing
    ``expert_bias`` / ``tid2eid`` / ``input_ids`` on the non-replay path.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        moe_implementation: str | None = None,
    ):
        super().__init__(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            norm_topk_prob=config.norm_topk_prob,
            moe_implementation=moe_implementation or getattr(config, "_moe_implementation", "triton"),
            train_router=getattr(config, "train_router", True),
            record_routing_weights=getattr(config, "record_routing_weights", True),
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0)),
        )
        self.config = config
        self.layer_id = layer_id
        self.is_hash_layer = layer_id < int(config.num_hash_layers)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self._dsv4_exact_native = bool(getattr(config, "_dsv4_flash_exact_mode", False))
        self.experts.ep_dispatch = getattr(config, "_ep_dispatch", "alltoall")
        self.experts.deepep_buffer_size_gb = getattr(config, "_deepep_buffer_size_gb", 2.0)
        self.experts.deepep_num_sms = getattr(config, "_deepep_num_sms", 20)
        self.experts.deepep_async_combine = getattr(config, "_deepep_async_combine", False)
        self.experts.alltoall_combine_hidden_chunk_size = getattr(config, "_alltoall_combine_hidden_chunk_size", 0)
        self.experts.expert_lora_semantics = DSV4_CLAMPED_SWIGLU_LORA_PROGRAM

        # Replace the parent's softmax router with our V4 one.
        self.router = TopKRouter(
            num_experts=self.num_experts,
            top_k=self.top_k,
            scoring_func="sqrtsoftplus",
            topk_method=None if self.is_hash_layer else "noaux_tc",
            # MXFP4 Marlin returns the unscaled routed partial. SGLang applies
            # the 1.5 factor only when it joins that partial with the TP-shared
            # expert, immediately before the ordered TP/EP reduction.
            routed_scaling_factor=(None if self._dsv4_exact_native else getattr(config, "routed_scaling_factor", None)),
            exact_sqrtsoftplus_serving=self._dsv4_exact_native,
        )

        if not self.is_hash_layer:
            # HF puts the bias on ``mlp.gate.e_score_correction_bias`` — match
            # that for state-dict round-tripping. ``requires_grad=False``: the
            # bias only enters top-k argmax (selection), not the routing-weight
            # autograd path, so SGD gradients are always zero. DeepSeek updates
            # this OOB via an aux-loss controller during training. Marking it
            # frozen here keeps optimizer / LoRA-trainable enumeration honest.
            self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
            self.gate.e_score_correction_bias._keep_fp32 = True
        else:
            # Frozen vocab→top_k lookup; populated from the HF checkpoint.
            self.register_buffer(
                "tid2eid",
                torch.zeros(config.vocab_size, self.top_k, dtype=torch.int32),
                persistent=True,
            )

        n_shared = int(getattr(config, "n_shared_experts", 0) or 0)
        self.shared_experts = (
            DeepseekV4MLP(config, intermediate_size=config.moe_intermediate_size * n_shared) if n_shared > 0 else None
        )

    def supports_routing_replay(self) -> bool:
        """Exact serving-kernel routing is deterministic but not replayable."""

        return not self._dsv4_exact_native

    def _capture_diagnostic_component(self, name: str, value: torch.Tensor) -> None:
        capture = self.__dict__.get("_diagnostic_capture_component")
        if callable(capture):
            capture(name, value)

    def route(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None):
        """V4-specific route: passes ``expert_bias`` / ``tid2eid`` / ``input_ids``
        into the router, and integrates routing replay for gradient-checkpoint
        determinism (parallel to :meth:`MoEBlock.route`).

        Returns ``(routing_weights, selected_experts, router_logits)`` matching
        the parent's signature so downstream paths (forward, EP dispatch) stay
        compatible.
        """
        if self._dsv4_exact_native:
            # The pinned sampler's router calls linear_bf16_fp32 ->
            # torch.mm(x, w.t(), out_dtype=fp32), but under its deterministic
            # contract torch.mm is patched to the batch-invariant persistent
            # Triton GEMM: matmul_persistent(x, w.t().contiguous()).to(fp32)
            # — a BF16-output kernel widened to FP32. Call that kernel
            # directly (never patch mm globally in a training graph). A
            # cuBLAS BF16 GEMM differs by one BF16 ulp on rounding-boundary
            # logits, which perturbs the sqrt-softplus weights and every
            # routed contribution downstream.
            from sglang.srt.batch_invariant_ops.batch_invariant_ops import (  # noqa: PLC0415
                matmul_persistent,
            )

            gate_weight = self.gate.weight
            if hidden_states.dtype is not torch.bfloat16 or gate_weight.dtype is not torch.bfloat16:
                raise TypeError("DSV4 exact router requires BF16 hidden states and BF16 gate weights")
            flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
            router_logits = (
                matmul_persistent(flat, gate_weight.t().contiguous())
                .to(torch.float32)
                .reshape(*hidden_states.shape[:-1], gate_weight.shape[0])
            )
        elif getattr(self, "config", None) is not None and getattr(self.config, "_router_fp32", False):
            router_logits = F.linear(hidden_states.float(), self.gate.weight.float())
        else:
            router_logits = self.gate(hidden_states)

        if self._dsv4_exact_native:
            stage = get_replay_stage()
            if stage is not None and self._routing_replay is not None:
                raise RuntimeError(
                    "Exact DSV4 serving-kernel routing does not admit gradient-checkpoint routing replay"
                )
            if self.train_router:
                raise RuntimeError("Exact DSV4 first-lane routing must remain frozen")
            if self.is_hash_layer:
                if input_ids is None:
                    raise ValueError("Exact DSV4 hash-routed layers require input_ids")
                from sglang.kernels.ops.attention.dsv4 import hash_topk  # noqa: PLC0415

                flat_input_ids = input_ids.reshape(-1)
                diagnostic_range = None
                if os.environ.get("XORL_DSV4_HASH_TOPK_DIAGNOSTICS") == "1":
                    diagnostic_range = (
                        int(flat_input_ids.min().item()),
                        int(flat_input_ids.max().item()),
                    )
                    if diagnostic_range[0] < 0 or diagnostic_range[1] >= self.tid2eid.shape[0]:
                        raise RuntimeError(
                            f"Exact DSV4 hash-topk layer {self.layer_id} received token range "
                            f"{diagnostic_range} for table rows {self.tid2eid.shape[0]}"
                        )
                    torch.cuda.synchronize(flat_input_ids.device)
                try:
                    routing_weights, selected_experts = hash_topk(
                        router_logits=router_logits,
                        input_ids=flat_input_ids,
                        tid2eid=self.tid2eid,
                        num_fused_shared_experts=0,
                        routed_scaling_factor=self.routed_scaling_factor,
                        scoring_func="sqrtsoftplus",
                        # Serving chains its gate projection into hash_topk with
                        # CUDA programmatic dependent launch.  The trainer gate is
                        # an ordinary PyTorch F.linear and does not trigger that
                        # dependency. Waiting for an unarmed PDL predecessor is an
                        # illegal-access race on short/tail batches; the non-PDL
                        # specialization executes the identical per-warp math.
                        use_pdl=False,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Exact DSV4 hash-topk failed at layer {self.layer_id}; "
                        f"rows={flat_input_ids.numel()} token_range={diagnostic_range} "
                        f"table_shape={tuple(self.tid2eid.shape)} table_stride={self.tid2eid.stride()}"
                    ) from exc
            else:
                from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate  # noqa: PLC0415

                routing_weights, selected_experts = moe_fused_gate(
                    router_logits,
                    self.gate.e_score_correction_bias,
                    topk=self.top_k,
                    scoring_func="sqrtsoftplus",
                    num_fused_shared_experts=0,
                    renormalize=True,
                    routed_scaling_factor=self.routed_scaling_factor,
                    apply_routed_scaling_factor_on_output=False,
                )
            routing_weights = routing_weights.detach()
            if getattr(self, "_diagnostic_capture_routing", False):
                self._diagnostic_last_routing = {
                    "router_logits": router_logits,
                    "router_routing_weights": routing_weights,
                    "router_selected_experts": selected_experts,
                }
            return routing_weights, selected_experts, router_logits

        # --- Routing replay (mirrors MoEBlock.route) ---
        # On all replay-active stages, expert selection is determined once on
        # the record path (under no_grad) and then reused on replay; routing
        # weights are recomputed via _regather_routing so the autograd graph
        # structure (sqrtsoftplus -> gather -> renorm) is identical between
        # forward and checkpoint recompute.
        stage = get_replay_stage()
        replay = self._routing_replay

        if stage is not None and replay is not None:
            if stage == "record":
                with torch.no_grad():
                    if self.is_hash_layer:
                        assert input_ids is not None, "hash-routed layer requires input_ids"
                        _, selected_experts = self.router(
                            router_logits,
                            hidden_states.dtype,
                            tid2eid=self.tid2eid,
                            input_ids=input_ids.reshape(-1),
                        )
                    else:
                        _, selected_experts = self.router(
                            router_logits,
                            hidden_states.dtype,
                            expert_bias=self.gate.e_score_correction_bias,
                        )
                replay.record(selected_experts)
            elif stage == "replay_forward":
                selected_experts = replay.pop_forward()
            elif stage == "replay_backward":
                selected_experts = replay.pop_backward()
            else:
                # Defensive: any new stage name added to ``get_replay_stage()``
                # without a branch here would leave ``selected_experts``
                # undefined and crash with ``NameError`` in
                # ``_regather_routing``. Fail loudly instead.
                raise ValueError(
                    f"Unrecognized replay stage {stage!r}; expected one of "
                    "{'record', 'replay_forward', 'replay_backward'}."
                )

            selected_experts, routing_weights = self._regather_routing(
                router_logits, selected_experts, hidden_states.dtype
            )

            if self.record_routing_weights:
                if stage == "record":
                    replay.record_weights(routing_weights)
                elif stage == "replay_backward":
                    cached_weights = replay.pop_backward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(
                            torch.float32 if self._dsv4_exact_native else hidden_states.dtype
                        )
                elif stage == "replay_forward":
                    cached_weights = replay.pop_forward_weights()
                    if cached_weights is not None:
                        routing_weights = cached_weights.to(
                            torch.float32 if self._dsv4_exact_native else hidden_states.dtype
                        )
        else:
            if self.is_hash_layer:
                assert input_ids is not None, "hash-routed layer requires input_ids"
                routing_weights, selected_experts = self.router(
                    router_logits,
                    hidden_states.dtype,
                    tid2eid=self.tid2eid,
                    input_ids=input_ids.reshape(-1),
                )
            else:
                routing_weights, selected_experts = self.router(
                    router_logits,
                    hidden_states.dtype,
                    expert_bias=self.gate.e_score_correction_bias,
                )

        if not self.train_router:
            routing_weights = routing_weights.detach()

        if getattr(self, "_diagnostic_capture_routing", False):
            self._diagnostic_last_routing = {
                "router_logits": router_logits,
                "router_routing_weights": routing_weights,
                "router_selected_experts": selected_experts,
            }

        return routing_weights, selected_experts, router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        live_token_count: int | None = None,
    ):
        """Forward pass.

        Args:
            hidden_states: ``[batch, seqlen, hidden_size]``.
            input_ids: ``[batch, seqlen]`` int — required only for hash
                layers; ignored by non-hash layers.

        Returns:
            ``(output [batch, seqlen, hidden_size], router_logits [N, num_experts])``.
        """
        if self._dsv4_exact_native:
            return self._forward_exact_native(hidden_states, input_ids, live_token_count)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights, selected_experts, router_logits = self.route(flat_hidden_states, input_ids=input_ids)

        if self.moe_implementation == "eager":
            final_hidden_states = self._eager_forward(flat_hidden_states, routing_weights, selected_experts)
        else:
            final_hidden_states = self.experts(flat_hidden_states, routing_weights, selected_experts)

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(flat_hidden_states)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def _forward_exact_native(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        live_token_count: int | None,
    ):
        """Mirror serving DP-attention gather, TP8 MoE partial, and rank sum."""

        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        # Unified combine: DSV4 folds BF16-transported partials with the same
        # canonical adjacent-pair FP32 tree as Qwen/GLM; only the variable-row
        # transport remains DSV4-specific (see dsv4_native_combine).
        from xorl.models.layers.moe.dsv4_native_combine import (  # noqa: PLC0415
            compact_rank_padded_rows,
            exchange_variable_and_canonical_fold,
            gather_ids_for_ep_combine,
            gather_tokens_for_ep_combine,
            row_counts_for_ep_combine,
        )

        from .native_payload import (  # noqa: PLC0415
            _validate_all_single_adapter_batch_infos,
            dsv4_join_routed_shared_partial,
            dsv4_native_shared_expert_tp_partial,
        )

        def validate_lora_metadata(where: str) -> None:
            if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
                _validate_all_single_adapter_batch_infos(
                    local_hidden.device.index,
                    where=f"layer {self.layer_id} {where}",
                )

        if input_ids is None and self.is_hash_layer:
            raise ValueError("Exact DSV4 hash-routed layers require input_ids")
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        local_hidden = hidden_states.reshape(-1, hidden_dim)
        local_rows = local_hidden.shape[0]
        live_token_count = local_rows if live_token_count is None else int(live_token_count)
        if not 0 <= live_token_count <= local_rows:
            raise ValueError(f"Exact DSV4 live token count {live_token_count} is outside packed row count {local_rows}")
        parallel_state = get_parallel_state()
        ep_group = parallel_state.ep_group
        ep_size = int(parallel_state.ep_size)
        ep_rank = int(parallel_state.ep_rank)
        ownership = LogicalRowOwnership(
            dp_size=int(parallel_state.dp_size),
            cp_size=int(parallel_state.cp_size),
            dp_rank=int(parallel_state.dp_rank),
            cp_rank=(int(parallel_state.cp_rank) if parallel_state.cp_enabled else 0),
            contributor_count=ep_size,
        )
        if int(parallel_state.tp_size) != 1:
            raise RuntimeError("Exact DSV4 requires stage-local body TP1")
        if ep_size != 8:
            raise RuntimeError("Exact DSV4 MoE combine requires EP8")
        if ep_group is None or ep_rank != ownership.source_ordinal:
            raise RuntimeError(
                "Exact DSV4 EP order must match DP-major/CP-minor logical row ownership: "
                f"ep_rank={ep_rank}, source_ordinal={ownership.source_ordinal}"
            )
        get_group_ranks = getattr(dist, "get_process_group_ranks", None)
        if get_group_ranks is not None and ownership.cp_size > 1:
            cp_group = parallel_state.sp_group
            if cp_group is None:
                raise RuntimeError("Exact DSV4 CP-owned rows require a stage-local CP group")
            ep_ranks = tuple(get_group_ranks(ep_group))
            expected_cp_ranks = tuple(ep_ranks[index] for index in ownership.context_source_ordinals)
            if tuple(get_group_ranks(cp_group)) != expected_cp_ranks:
                raise RuntimeError("Exact DSV4 CP membership does not match the DP-major owner plane")
        row_counts = row_counts_for_ep_combine(live_token_count, local_hidden.device, ep_group)
        if sum(row_counts) == 0:
            return torch.zeros_like(hidden_states), hidden_states.new_zeros(
                (local_rows, self.num_experts), dtype=torch.float32
            )
        padded_rows = max(row_counts)
        gathered_hidden_padded = gather_tokens_for_ep_combine(
            local_hidden[:live_token_count],
            ep_group,
            padded_rows,
        )
        gathered_hidden = compact_rank_padded_rows(
            gathered_hidden_padded,
            padded_rows=padded_rows,
            row_counts=row_counts,
        )
        self._capture_diagnostic_component("moe_native_gathered_input", gathered_hidden)
        if input_ids is None:
            local_ids = torch.zeros(live_token_count, dtype=torch.long, device=hidden_states.device)
        else:
            local_ids = input_ids.reshape(-1)[:live_token_count]
        gathered_ids_padded = gather_ids_for_ep_combine(
            local_ids,
            ep_group,
            padded_rows,
        )
        gathered_ids = compact_rank_padded_rows(
            gathered_ids_padded,
            padded_rows=padded_rows,
            row_counts=row_counts,
        )
        self._capture_diagnostic_component("moe_native_gathered_ids", gathered_ids)
        routing_weights, selected_experts, router_logits = self.route(
            gathered_hidden,
            input_ids=gathered_ids,
        )
        self._capture_diagnostic_component("moe_native_gathered_routing", router_logits)
        self._capture_diagnostic_component("moe_native_local_ids", selected_experts)
        # Serving all-gathers rank-major LoRA metadata and scopes it across the
        # gathered MLP, so every EP rank applies its routed and shared-expert
        # adapter slice before the canonical rank fold.
        adapter_partial_live = True
        # Enter through Module.__call__ so the independently wrapped expert
        # FSDP unit materializes its native payload and LoRA factors before
        # the Marlin runner receives their pointers.
        routed = self.experts(
            gathered_hidden,
            routing_weights,
            selected_experts,
            dsv4_exact_native=True,
            dsv4_exact_lora_live=adapter_partial_live,
        )
        validate_lora_metadata("routed expert return")
        self._capture_diagnostic_component("moe_native_routed", routed)
        if self.shared_experts is None:
            raise RuntimeError("Exact DSV4 requires the official shared expert")
        shared = dsv4_native_shared_expert_tp_partial(
            gathered_hidden,
            self.shared_experts,
            tp_rank=ep_rank,
            tp_size=ep_size,
            diagnostic_capture=self._capture_diagnostic_component,
            lora_live=adapter_partial_live,
        )
        validate_lora_metadata("shared expert return")
        local_partial = dsv4_join_routed_shared_partial(
            routed,
            shared,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        validate_lora_metadata("routed/shared join")
        self._capture_diagnostic_component("moe_native_local_partial", local_partial)
        combined = exchange_variable_and_canonical_fold(
            local_partial,
            ep_group,
            row_counts,
            ownership.source_ordinal,
        )
        validate_lora_metadata("EP exchange/combine")
        self._capture_diagnostic_component("moe_native_combined", combined)
        if combined.shape[0] != live_token_count:
            raise RuntimeError(
                f"Exact DSV4 combine returned {combined.shape[0]} rows for {live_token_count} live local rows"
            )
        padded_combined = torch.cat((combined, local_hidden[live_token_count:] * 0.0), dim=0)
        own_start = sum(row_counts[: ownership.source_ordinal])
        local_router_logits = router_logits[own_start : own_start + live_token_count]
        local_router_logits = torch.cat(
            (
                local_router_logits,
                router_logits.new_zeros((local_rows - live_token_count, router_logits.shape[-1])),
            ),
            dim=0,
        )
        return (
            padded_combined.reshape(batch_size, sequence_length, hidden_dim),
            local_router_logits,
        )


class DeepseekV4DecoderLayer(nn.Module):
    """A single V4 transformer layer wrapped with the HyperConnection mixer.

    Carries the per-layer fp32 HC parameters (``hc_attn_{fn,base,scale}`` and
    ``hc_ffn_{fn,base,scale}``) at the layer level — matching the HF state-dict
    layout (``model.layers.{N}.hc_attn_fn`` etc.) so the loader doesn't need
    a special-case rename.

    The layer operates in 4-D BSHD: ``[batch, seqlen, hc_mult, hidden]``.
    ``layer_pre`` collapses the ``hc_mult`` dim before each sublayer, the
    sublayer (attention or MoE) runs in 3-D, then ``layer_post`` re-mixes
    into 4-D.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        tp_group: torch.distributed.ProcessGroup | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
        moe_implementation: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_mult = int(config.hc_mult)

        self.self_attn = DeepSeekV4Attention(config, layer_id=layer_id, tp_group=tp_group, cp_group=cp_group)
        self.mlp = DeepseekV4MoE(config, layer_id=layer_id, moe_implementation=moe_implementation)

        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Hyper-connection per-layer fp32 params.
        hc_dim = self.hc_mult * self.hidden_size
        mix_size = (2 + self.hc_mult) * self.hc_mult

        for prefix in ("hc_attn", "hc_ffn"):
            # Register the params first, then set the ``_keep_fp32`` marker on
            # the registered Parameters. The init_empty_weights helper
            # snapshots ``param.__dict__`` at register time and forwards it
            # as kwargs to ``Parameter.__new__`` — extra attrs set before
            # ``setattr`` would land in those kwargs and crash.
            setattr(self, f"{prefix}_fn", nn.Parameter(torch.empty(mix_size, hc_dim, dtype=torch.float32)))
            setattr(self, f"{prefix}_base", nn.Parameter(torch.empty(mix_size, dtype=torch.float32)))
            setattr(self, f"{prefix}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32)))
            for suffix in ("fn", "base", "scale"):
                getattr(self, f"{prefix}_{suffix}")._keep_fp32 = True

        self.hc_util = DeepSeekV4HyperConnectionUtil(config)
        self._exact_mhc = bool(getattr(config, "_dsv4_flash_exact_mode", False))

    def _capture_diagnostic_component(self, name: str, value: torch.Tensor) -> None:
        capture = self.__dict__.get("_diagnostic_capture_component")
        if callable(capture):
            capture(name, value)

    def forward(
        self,
        hidden_states_4d: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        output_router_logits: bool = False,
        live_token_count: int | None = None,
        exact_cp_layout: Dsv4ExactCPLayout | None = None,
        decode_carry_offset: int | None = None,
    ) -> torch.Tensor:
        """Run the layer.

        Args:
            hidden_states_4d: ``[batch, seqlen, hc_mult, hidden_size]``.
            input_ids: ``[batch, seqlen]`` int — required only when this
                layer's MoE is hash-routed.

        Returns:
            ``[batch, seqlen, hc_mult, hidden_size]``.
        """
        # ---- Attention sublayer ----
        if self._exact_mhc:
            h3d, post, comb = self.hc_util.layer_pre_norm_exact(
                hidden_states_4d,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm.weight,
            )
        else:
            h3d, post, comb = self.hc_util.layer_pre(
                hidden_states_4d, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
            h3d = self.input_layernorm(h3d)
        self._capture_diagnostic_component("input_norm", h3d)
        attn_out = self.self_attn(
            h3d,
            exact_cp_layout=exact_cp_layout,
            decode_carry_offset=decode_carry_offset,
        )
        post_fn = self.hc_util.layer_post_exact if self._exact_mhc else self.hc_util.layer_post
        hidden_states_4d = post_fn(attn_out, hidden_states_4d, post, comb)

        # ---- MoE / FFN sublayer ----
        if self._exact_mhc:
            h3d, post, comb = self.hc_util.layer_pre_norm_exact(
                hidden_states_4d,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.post_attention_layernorm.weight,
            )
        else:
            h3d, post, comb = self.hc_util.layer_pre(
                hidden_states_4d, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
            )
            h3d = self.post_attention_layernorm(h3d)
        self._capture_diagnostic_component("post_attention_norm", h3d)
        ffn_out, router_logits = self.mlp(
            h3d,
            input_ids=input_ids,
            live_token_count=live_token_count,
        )
        if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
            from .native_payload import _validate_all_single_adapter_batch_infos  # noqa: PLC0415

            _validate_all_single_adapter_batch_infos(
                h3d.device.index,
                where=f"layer {self.layer_id} MoE module return",
            )
        hidden_states_4d = post_fn(ffn_out, hidden_states_4d, post, comb)
        if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
            _validate_all_single_adapter_batch_infos(
                h3d.device.index,
                where=f"layer {self.layer_id} FFN hyperconnection post",
            )

        if output_router_logits:
            return hidden_states_4d, router_logits
        return hidden_states_4d


class DeepseekV4PreTrainedModel(XorlPreTrainedModel):
    """Base for DSv4 model classes.

    ``_init_weights`` provides reasonable defaults for *every* parameter type
    DSv4 introduces — including the fp32 HC params, the fp32 attn_sink, and
    the fp32 compressor weights. Without these, ``MoEExperts.gate_up_proj``
    and the various ``torch.empty`` HC params land NaN-inited.
    """

    config_class = None  # set by importer to avoid circular import
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]

    def rebuild_shared_freqs_cis(self) -> int:
        """Re-register the lru_cache-shared RoPE tables after to_empty loads.

        named_buffers() deduplicates shared tensor objects, so buffer restore
        recovers each table under only its first FQN and every other layer's
        RoPE runs on zeroed storage (layers other than the first restored FQN
        would otherwise emit zero RoPE slices). Rebuild one
        table per distinct (base, yarn, head_dim) argument tuple per device
        and share it across the layers that use it.
        """

        shared: dict[tuple, torch.Tensor] = {}
        rebuilt = 0
        for module in self.modules():
            args = getattr(module, "_freqs_cis_rebuild_args", None)
            if args is None or "freqs_cis" not in getattr(module, "_buffers", {}):
                continue
            rope_base, yarn_disabled, rope_head_dim = args
            device = module.freqs_cis.device
            key = (rope_base, yarn_disabled, rope_head_dim, device)
            table = shared.get(key)
            if table is None:
                table = wrapped_precompute_freqs_cis(
                    self.config,
                    rope_head_dim=rope_head_dim,
                    base=rope_base,
                    yarn_disabled=yarn_disabled,
                ).to(device)
                shared[key] = table
            module.register_buffer("freqs_cis", table, persistent=False)
            rebuilt += 1
        return rebuilt

    def get_ignore_modules_in_mixed_precision(self):
        """Keep native dense payload bytes outside the decoder BF16 policy."""

        if getattr(self.config, "_dsv4_flash_exact_mode", False):
            from .native_payload import Dsv4NativeBlockFp8Payload  # noqa: PLC0415

            # Routed payloads are already owned by the separately wrapped EP
            # expert unit. Only dense child payload holders belong here.
            return (Dsv4NativeBlockFp8Payload,)
        return None

    def to(self, *args, **kwargs):
        return _dsv4_model_to(self, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, state_dict=None, **kwargs):
        """HF AutoModel-compatible loader for DeepSeek-V4 snapshots.

        ``transformers.AutoModelForCausalLM`` requires registered model
        classes to expose ``from_pretrained``. xorl models do not inherit
        ``transformers.PreTrainedModel``, so DSv4 implements the subset needed
        to instantiate from a HF ``config.json`` and load the HF safetensors
        layout through the existing xorl checkpoint handler.
        """
        if model_args:
            raise TypeError(f"{cls.__name__}.from_pretrained does not accept positional model args: {model_args!r}")

        output_loading_info = bool(kwargs.pop("output_loading_info", False))
        torch_dtype_arg = kwargs.pop("torch_dtype", None)
        dtype_arg = kwargs.pop("dtype", None)
        if torch_dtype_arg is None:
            torch_dtype_arg = dtype_arg
        target_dtype_arg = kwargs.pop("target_dtype", None)
        attn_implementation = kwargs.pop("attn_implementation", None)
        moe_implementation = kwargs.pop("moe_implementation", None)
        init_device = kwargs.pop("init_device", "cpu")
        strict = bool(kwargs.pop("strict", False))
        dequantize_fp8 = bool(kwargs.pop("dequantize_fp8", True))
        kwargs.pop("progress", None)
        subfolder = kwargs.pop("subfolder", None)

        if kwargs.pop("ignore_mismatched_sizes", False):
            raise NotImplementedError("DeepseekV4ForCausalLM.from_pretrained does not support ignore_mismatched_sizes")

        device_map = kwargs.pop("device_map", None)
        if device_map in ("cpu", {"": "cpu"}):
            init_device = "cpu"
        elif device_map not in (None,):
            raise NotImplementedError("DeepseekV4ForCausalLM.from_pretrained does not support device_map")

        # Accepted by HF's PreTrainedModel loader but not meaningful for this
        # xorl-native path.
        for unused in (
            "_from_auto",
            "_commit_hash",
            "adapter_kwargs",
            "cache_dir",
            "force_download",
            "local_files_only",
            "low_cpu_mem_usage",
            "proxies",
            "revision",
            "token",
            "trust_remote_code",
            "use_safetensors",
            "weights_only",
        ):
            kwargs.pop(unused, None)

        from .configuration_deepseek_v4 import DeepseekV4Config  # noqa: PLC0415

        if config is None:
            config = DeepseekV4Config.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=subfolder,
            )
        elif not isinstance(config, DeepseekV4Config):
            config = DeepseekV4Config.from_hf_config(config)

        if moe_implementation is not None:
            config._moe_implementation = moe_implementation

        torch_dtype = _resolve_dsv4_torch_dtype(torch_dtype_arg, config)
        target_dtype = _resolve_dsv4_torch_dtype(target_dtype_arg, config, default=torch_dtype)
        load_summary = None

        if state_dict is not None:
            model = cls._from_config(config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
            from .checkpoint_handler import load_hf_state_dict_into_model  # noqa: PLC0415

            load_summary = load_hf_state_dict_into_model(
                model,
                state_dict,
                strict=strict,
                dequantize_fp8=dequantize_fp8,
                target_dtype=target_dtype,
            )
        else:
            if init_device not in {"cpu", "cuda", "npu"}:
                raise ValueError(
                    f"init_device must be one of 'cpu', 'cuda', or 'npu' when loading weights; got {init_device!r}"
                )
            from xorl.models.module_utils import all_ranks_load_weights, init_empty_weights  # noqa: PLC0415

            with init_empty_weights():
                model = cls._from_config(config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
            model._dsv4_dequantize_fp8 = dequantize_fp8
            model._dsv4_target_dtype = target_dtype
            all_ranks_load_weights(
                model,
                _resolve_dsv4_weight_path(pretrained_model_name_or_path, subfolder),
                init_device=init_device,
            )

        model.eval()
        if output_loading_info:
            return model, {"dsv4_load_summary": load_summary}
        return model

    @classmethod
    def _from_config(cls, config, **kwargs):
        """Create DSv4 with the active xorl parallel groups wired in."""
        torch_dtype = kwargs.pop("torch_dtype", None)
        attn_implementation = kwargs.pop("attn_implementation", None)

        if attn_implementation:
            config._attn_implementation = attn_implementation

        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()
        tp_group = parallel_state.tp_group if parallel_state.tp_enabled else None
        cp_group = parallel_state.sp_group if parallel_state.cp_enabled else None
        model = cls(
            config,
            tp_group=tp_group,
            cp_group=cp_group,
            moe_implementation=getattr(config, "_moe_implementation", None),
        )

        if torch_dtype is not None:
            cast_dsv4_model_dtype(model, torch_dtype)

        return model

    def get_checkpoint_handler(self, **kwargs):
        from .checkpoint_handler import DeepseekV4CheckpointHandler  # noqa: PLC0415

        if kwargs.get("is_broadcast", False):
            ep_rank, ep_size = 0, 1
        else:
            ep_rank = kwargs.get("ep_rank", 0)
            ep_size = kwargs.get("ep_size", 1)

        return DeepseekV4CheckpointHandler(
            self.config,
            checkpoint_keys=kwargs.get("checkpoint_keys", set()),
            ep_rank=ep_rank,
            ep_size=ep_size,
            dequantize_fp8=getattr(self, "_dsv4_dequantize_fp8", True),
            target_dtype=getattr(self, "_dsv4_target_dtype", torch.bfloat16),
        )

    def _init_weights(self, module):
        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        elif hasattr(module, "gate_up_proj") and isinstance(module.gate_up_proj, nn.Parameter):
            # ``MoEExperts``: stacked expert weights.
            module.gate_up_proj.data.normal_(0.0, std)
            if hasattr(module, "down_proj") and isinstance(module.down_proj, nn.Parameter):
                module.down_proj.data.normal_(0.0, std)
        # Catch direct parameters that other branches didn't cover:
        # hc_attn_*, hc_ffn_*, model-level hc_head_*, attn_sink, compressor's
        # ``ape``, e_score_correction_bias. Init unconditionally — these are
        # ``torch.empty``-allocated and can hold arbitrary garbage including
        # NaN/Inf.
        #
        # Note on ``e_score_correction_bias``: it is registered as an *extra*
        # attribute on a ``nn.Linear`` (``self.gate``) rather than via the
        # standard ``bias`` slot. The ``isinstance(module, nn.Linear)`` branch
        # above runs and zero-inits ``module.bias`` (which is ``None`` because
        # the gate is constructed with ``bias=False``); ``e_score_correction_bias``
        # is a separate Parameter that surfaces via
        # ``named_parameters(recurse=False)`` here, where this catch-all
        # zero-inits it.
        for name, p in module.named_parameters(recurse=False):
            if name == "attn_sink" or name.startswith("hc_") or name in ("ape", "e_score_correction_bias"):
                p.data.zero_()


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    """Transformer with HyperConnection wrapping.

    Layout:

    1. ``embed_tokens(input_ids)`` -> 3-D BSHD.
    2. ``hc_util.block_expand`` -> 4-D BSHD ``[B, S, hc_mult, H]``.
    3. ``N`` ``DeepseekV4DecoderLayer`` runs in 4-D.
    4. ``hc_util.block_head`` (using model-level ``hc_head_*`` params)
       collapses to 3-D.
    5. Final ``norm`` -> 3-D output.
    """

    def __init__(
        self,
        config,
        tp_group: torch.distributed.ProcessGroup | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
        moe_implementation: str | None = None,
    ):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.hc_mult = int(config.hc_mult)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=config.pad_token_id)

        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    config,
                    layer_id=i,
                    tp_group=tp_group,
                    cp_group=cp_group,
                    moe_implementation=moe_implementation,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Model-level HC head params (named to match HF: ``model.hc_head_*``).
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        for p in (self.hc_head_fn, self.hc_head_base, self.hc_head_scale):
            p._keep_fp32 = True

        # Stateless HC math.
        self.hc_util = DeepSeekV4HyperConnectionUtil(config)
        self.gradient_checkpointing = False
        self.post_init()

    def _capture_diagnostic_component(self, name: str, value: torch.Tensor) -> None:
        capture = self.__dict__.get("_diagnostic_capture_component")
        if callable(capture):
            capture(name, value)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> MoeModelOutput:
        """Args:
            input_ids: ``[batch, seqlen]`` int.

        Returns:
            ``[batch, seqlen, hidden_size]``.
        """
        pp_stage_is_first = kwargs.pop("_pp_stage_is_first", None)
        pp_stage_is_last = kwargs.pop("_pp_stage_is_last", None)
        if (pp_stage_is_first is None) != (pp_stage_is_last is None):
            raise ValueError("DeepSeek-V4 pipeline stage flags must be supplied together")
        is_pipeline_stage = pp_stage_is_first is not None
        incoming_hyperconnection_state = bool(is_pipeline_stage and not pp_stage_is_first)
        decode_cache_carry = False
        carry_position_offset = 0
        if getattr(self.config, "_dsv4_flash_exact_mode", False):
            decode_cache_carry = bool(kwargs.pop("decode_cache_carry", False))
            if decode_cache_carry:
                # The decode-cache scorer passes absolute position_ids
                # (torch.arange(start, end)); the segment offset selects the
                # carried serving state to append to.
                if position_ids is None:
                    raise ValueError("DSV4 decode-cache carry requires absolute position_ids")
                carry_position_offset = int(position_ids.reshape(-1)[0].item())
        del attention_mask
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("DeepseekV4Model.forward requires input_ids or inputs_embeds")
            hidden_states = self.embed_tokens(input_ids)  # [B, S, H]
        else:
            hidden_states = inputs_embeds
            if input_ids is None and int(getattr(self.config, "num_hash_layers", 0)) > 0:
                raise ValueError("input_ids are required when DeepSeek-V4 hash-routed layers are enabled")
        if incoming_hyperconnection_state:
            if hidden_states.ndim != 4 or hidden_states.shape[2] != self.hc_mult:
                raise ValueError(
                    "A non-first DeepSeek-V4 pipeline stage requires the live 4-D hyperconnection state; "
                    f"got shape {tuple(hidden_states.shape)}"
                )
        elif hidden_states.ndim != 3:
            raise ValueError(f"DeepSeek-V4 input stage requires a 3-D hidden state; got {tuple(hidden_states.shape)}")

        live_token_count = None
        exact_cp_layout = None
        packed_sequence_length = int(hidden_states.shape[1])
        if getattr(self.config, "_dsv4_flash_exact_mode", False):
            parallel_state = get_parallel_state()
            cp_logical_rows = kwargs.pop("_cp_logical_row_indices", None)
            cp_request_ids = kwargs.pop("_cp_request_ids", None)
            cp_request_positions = kwargs.pop("_cp_request_positions", None)
            cp_live_mask = kwargs.pop("_cp_live_mask", None)
            sample_lengths = kwargs.pop("_r3_sample_lengths", None)
            num_samples = kwargs.pop("num_samples", None)
            if isinstance(sample_lengths, torch.Tensor):
                sample_lengths = sample_lengths.detach().cpu().reshape(-1).tolist()
            if not decode_cache_carry:
                if hidden_states.shape[0] != 1:
                    raise RuntimeError(f"Exact DSV4 row layout requires batch=1; got batch={hidden_states.shape[0]}")
                side_channels = (cp_logical_rows, cp_request_ids, cp_request_positions, cp_live_mask)
                if any(value is None for value in side_channels):
                    if int(parallel_state.cp_size) > 1:
                        raise ValueError("Exact DSV4 CP requires collator row/request/live side channels")
                    cp_logical_rows = torch.arange(packed_sequence_length, device=hidden_states.device).view(1, -1)
                    cp_request_ids = torch.full_like(cp_logical_rows, -1)
                    cp_request_positions = torch.zeros_like(cp_logical_rows)
                    cp_live_mask = torch.zeros_like(cp_logical_rows, dtype=torch.bool)
                    if num_samples is not None and int(num_samples) == 0:
                        fallback_lengths = []
                    elif sample_lengths is not None:
                        fallback_lengths = [int(length) for length in sample_lengths]
                    else:
                        fallback_lengths = [packed_sequence_length]
                    cursor = 0
                    for request_id, length in enumerate(fallback_lengths):
                        if length < 0 or cursor + length > packed_sequence_length:
                            raise ValueError(
                                f"Exact DSV4 request lengths {fallback_lengths} do not fit {packed_sequence_length} rows"
                            )
                        cp_request_ids[:, cursor : cursor + length] = request_id
                        cp_request_positions[:, cursor : cursor + length] = torch.arange(
                            length,
                            device=hidden_states.device,
                        )
                        cp_live_mask[:, cursor : cursor + length] = True
                        cursor += length
                cp_logical_rows = cp_logical_rows.to(device=hidden_states.device)
                cp_request_ids = cp_request_ids.to(device=hidden_states.device)
                cp_request_positions = cp_request_positions.to(device=hidden_states.device)
                cp_live_mask = cp_live_mask.to(device=hidden_states.device)
                packed_sequence_length = int(cp_logical_rows.numel())
                local_live_count = int(cp_live_mask.count_nonzero().item())
                compute_rows_tensor = torch.tensor([local_live_count], dtype=torch.int64, device=hidden_states.device)
                if dist.is_available() and dist.is_initialized():
                    # FSDP collectives require equal activation shapes; EP's
                    # canonical fold and CP's padded gather require the same.
                    # Equalize across every applicable owner plane rather than
                    # assuming one topology subsumes the others.
                    fsdp_group = (
                        parallel_state.fsdp_group if bool(getattr(parallel_state, "fsdp_enabled", False)) else None
                    )
                    ep_group = parallel_state.ep_group if bool(getattr(parallel_state, "ep_enabled", False)) else None
                    sp_group = parallel_state.sp_group if int(parallel_state.cp_size) > 1 else None
                    compute_groups = (fsdp_group, ep_group, sp_group)
                    reduced_groups = set()
                    for compute_group in compute_groups:
                        if compute_group is None or id(compute_group) in reduced_groups:
                            continue
                        dist.all_reduce(compute_rows_tensor, op=dist.ReduceOp.MAX, group=compute_group)
                        reduced_groups.add(id(compute_group))
                compute_rows = max(1, int(compute_rows_tensor.item()))
                exact_cp_layout = build_dsv4_exact_cp_layout(
                    cp_logical_rows,
                    cp_request_ids,
                    cp_request_positions,
                    cp_live_mask,
                    compute_rows=compute_rows,
                    cp_group=(parallel_state.sp_group if int(parallel_state.cp_size) > 1 else None),
                )
                live_token_count = exact_cp_layout.local_live_count
                if incoming_hyperconnection_state:
                    if hidden_states.shape[1] != packed_sequence_length:
                        raise ValueError(
                            "DeepSeek-V4 pipeline wire rows do not match storage-order CP metadata: "
                            f"wire={hidden_states.shape[1]}, metadata={packed_sequence_length}"
                        )
                    # Physical PP buffers and generic varlen metadata describe
                    # storage rows.  The live HyperConnection state occupies a
                    # compact prefix of that wire; strip its transport padding
                    # before executing this stage's decoder layers.  Do not
                    # index by local_storage_indices here: the preceding stage
                    # already put live values in compact order.
                    hidden_states = hidden_states[:, :compute_rows]
                else:
                    compact_h = hidden_states.index_select(1, exact_cp_layout.local_storage_indices)
                    hidden_states = F.pad(compact_h, (0, 0, 0, compute_rows - live_token_count))
                if input_ids is not None:
                    if input_ids.shape[1] != packed_sequence_length:
                        raise ValueError(
                            "DeepSeek-V4 original input IDs do not match storage-order CP metadata: "
                            f"ids={input_ids.shape[1]}, metadata={packed_sequence_length}"
                        )
                    compact_ids = input_ids.index_select(1, exact_cp_layout.local_storage_indices)
                    input_ids = F.pad(compact_ids, (0, compute_rows - live_token_count), value=0)
            else:
                live_token_count = packed_sequence_length
        del position_ids
        del kwargs

        output_router_logits = (
            self.config.output_router_logits if output_router_logits is None else output_router_logits
        )
        all_router_logits = [] if output_router_logits else None
        h4d = (
            hidden_states if incoming_hyperconnection_state else self.hc_util.block_expand(hidden_states)
        )  # [B, S, hc_mult, H]

        use_outer_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and self._gradient_checkpointing_method == DEFAULT_GRADIENT_CHECKPOINTING_METHOD
        )
        for layer in self.layers:
            if layer is None:
                continue
            if use_outer_checkpoint:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    h4d,
                    input_ids=input_ids,
                    output_router_logits=output_router_logits,
                    live_token_count=live_token_count,
                    exact_cp_layout=exact_cp_layout,
                    decode_carry_offset=(carry_position_offset if decode_cache_carry else None),
                )
            else:
                layer_outputs = layer(
                    h4d,
                    input_ids=input_ids,
                    output_router_logits=output_router_logits,
                    live_token_count=live_token_count,
                    exact_cp_layout=exact_cp_layout,
                    decode_carry_offset=(carry_position_offset if decode_cache_carry else None),
                )

            if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
                from .native_payload import _validate_all_single_adapter_batch_infos  # noqa: PLC0415

                _validate_all_single_adapter_batch_infos(
                    h4d.device.index,
                    where=f"decoder layer {layer.layer_id} module return",
                )

            if output_router_logits:
                h4d, router_logits = layer_outputs
                if exact_cp_layout is not None:
                    live_router_logits = router_logits[: exact_cp_layout.local_live_count]
                    storage_router_logits = router_logits.new_zeros((packed_sequence_length, *router_logits.shape[1:]))
                    router_logits = storage_router_logits.index_copy(
                        0,
                        exact_cp_layout.local_storage_indices,
                        live_router_logits,
                    )
                all_router_logits.append(router_logits)
            else:
                h4d = layer_outputs

        if is_pipeline_stage and not pp_stage_is_last:
            # Preserve the live multi-stream residual state across the PP wire.
            # Collapsing/re-expanding here would be a different model program.
            # Keep live values in compact-prefix order, but restore the fixed
            # storage-row capacity expected by PP buffers and cu_seq_lens.
            if exact_cp_layout is not None:
                transport_padding = packed_sequence_length - h4d.shape[1]
                if transport_padding < 0:
                    raise RuntimeError(
                        "DeepSeek-V4 compact compute rows exceed the pipeline storage capacity: "
                        f"compute={h4d.shape[1]}, storage={packed_sequence_length}; exact DSV4 PP must "
                        "align storage rows across owner planes before building its schedule"
                    )
                h4d = F.pad(h4d, (0, 0, 0, 0, 0, transport_padding))
            return MoeModelOutput(
                last_hidden_state=h4d,
                router_logits=tuple(all_router_logits) if all_router_logits is not None else None,
            )

        h3d = self.hc_util.block_head(h4d, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        self._capture_diagnostic_component("hc_head_output", h3d)
        h3d = self.norm(h3d)
        self._capture_diagnostic_component("final_norm", h3d)
        if exact_cp_layout is not None:
            live_h3d = h3d[:, : exact_cp_layout.local_live_count]
            storage_h3d = h3d.new_zeros((h3d.shape[0], packed_sequence_length, h3d.shape[-1]))
            h3d = storage_h3d.index_copy(1, exact_cp_layout.local_storage_indices, live_h3d)
        return MoeModelOutput(
            last_hidden_state=h3d,
            router_logits=tuple(all_router_logits) if all_router_logits is not None else None,
        )


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel):
    """Causal-LM head wrapping :class:`DeepseekV4Model`."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    # Keep the generic alias while the exact DSV4 PP path uses the explicit
    # original-ID contract below.
    _pp_requires_input_ids_on_all_stages = True
    _pp_carries_hyperconnection_state = True
    _pp_requires_original_input_ids = True

    def __init__(
        self,
        config,
        tp_group: torch.distributed.ProcessGroup | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
        moe_implementation: str | None = None,
    ):
        super().__init__(config)
        self.model = DeepseekV4Model(
            config, tp_group=tp_group, cp_group=cp_group, moe_implementation=moe_implementation
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
        if getattr(config, "_dsv4_flash_exact_mode", False):
            from .native_payload import attach_dsv4_native_payloads  # noqa: PLC0415

            attach_dsv4_native_payloads(self, config)

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
            "num_layers": self.config.num_hidden_layers,
            "pipeline_boundary_state": {
                "rank": 4,
                "dtype": "bfloat16",
                "shape_suffix": (int(self.config.hc_mult), int(self.config.hidden_size)),
                "state": "completed_hyperconnection_residual",
            },
        }

    def _configure_pp_stage(self, *, stage_idx: int, num_stages: int) -> None:
        """Give model-level HyperConnection head parameters one PP owner."""

        if stage_idx == num_stages - 1:
            return
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            self.model.register_parameter(name, None)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> MoeCausalLMOutput:
        """Return hidden states for xorl's external CE/loss path."""
        output_router_logits = kwargs.pop("output_router_logits", self.config.output_router_logits)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        return MoeCausalLMOutput(last_hidden_state=outputs.last_hidden_state, router_logits=outputs.router_logits)

    def get_parallel_plan(self):
        """EP plan for the routed-expert weights. Consumed by
        ``xorl.distributed.torch_parallelize.build_parallelize_model`` to
        slice each expert tensor along ``Shard(0)`` across the EP mesh.
        """
        from . import parallelize  # noqa: PLC0415

        return parallelize.get_ep_plan()


ModelClass = DeepseekV4ForCausalLM

__all__ = [
    "DeepSeekV4Attention",
    "DeepseekV4MLP",
    "DeepseekV4MoE",
    "DeepseekV4DecoderLayer",
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
    "cast_dsv4_model_dtype",
]
