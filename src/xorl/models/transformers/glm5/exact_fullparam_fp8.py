"""Full-parameter block-FP8 master-weight components for exact GLM-5.2 training.

The byte-parity mechanism is *publish the bytes you consumed*: an FP32 master
is quantized to serving block-FP8 bytes exactly once per optimizer step by
:meth:`refresh_quantized_cache`; every trainer forward in that step consumes
the cached bytes through the unchanged exact SGLang kernel; weight sync
publishes the very same tensors.  Trainer forward bytes therefore equal the
sampler's served bytes by construction, and the quantization program's
arithmetic never crosses engines.

The forward fails closed if the master has mutated since the last refresh: a
stale cache must never score tokens.  Backward is the trainer-owned
straight-through treatment: activation gradients flow through the dequantized
cached bytes in the declared BF16 program (the same base-branch treatment the
frozen lane's surrogate uses), and the master's weight gradient is the plain
FP32 GEMM ``grad_outputᵀ @ input`` (∂Q/∂W treated as identity).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xorl.ops.exact.block_fp8_native import (
    _sglang_native_block_fp8_linear_value,
    pack_fp8_as_float32,
    unpack_float32_as_fp8,
)
from xorl.ops.exact.fused_silu_and_mul import one_round_swiglu


logger = logging.getLogger(__name__)

GLM52_EXACT_TP1_FULLPARAM_FP8_CONTRACT_VERSION = "glm52_exact_tp1_fullparam_block_fp8_v1"
GLM52_EXACT_FULLPARAM_ROUTER_CONTRACT_VERSION = "glm52_exact_fullparam_router_bf16_view_v1"

_FP8_DTYPE = torch.float8_e4m3fn
_BLOCK = 128


def _master_identity_tuple(master: Tensor) -> tuple[int, int]:
    """(data_ptr, version) of the storage the optimizer actually mutates.

    Under FSDP2 the master is a DTensor whose local shard is the mutated
    storage; recording the wrapper's identity would miss ``.data`` swaps of
    the shard. In both cases any in-place update bumps the recorded version
    and any re-wrap changes the recorded pointer.
    """

    try:
        from torch.distributed.tensor import DTensor  # noqa: PLC0415

        if isinstance(master, DTensor):
            local = master._local_tensor
            return (local.data_ptr(), local._version)
    except ImportError:  # pragma: no cover - DTensor ships with torch.distributed
        pass
    return (master.data_ptr(), master._version)


def _master_full_tensor(master: Tensor) -> Tensor:
    """The complete FP32 master bytes, gathering FSDP2 shards when present.

    ``full_tensor()`` is an all-gather: every rank of the mesh must call the
    refresh together (the step boundary is already a collective point).  The
    gather is byte-exact — no arithmetic touches the master bytes.
    """

    try:
        from torch.distributed.tensor import DTensor  # noqa: PLC0415

        if isinstance(master, DTensor):
            return master.full_tensor().detach()
    except ImportError:  # pragma: no cover
        pass
    return master


def _master_mutation_source(module: nn.Module, name: str) -> Tensor:
    """Resolve the storage whose mutation defines staleness for master ``name``.

    FSDP2 swaps a wrapped parameter between the sharded DTensor (between
    forwards — the storage the optimizer mutates) and a freshly allocated
    unsharded plain working copy (inside the unshard window).  Recording or
    checking identity on whichever object happens to be visible is unsound:
    the two states alternate legitimately without any master mutation (gate:
    the FSDP2 test's forward-vs-publication asymmetry).  Identity must
    therefore track the SHARDED storage through both swap states: whenever
    the visible attribute is the sharded DTensor it is stashed on the module,
    and a later plain-tensor sighting (the unshard window) resolves to the
    stash.  A module that was never wrapped resolves to the visible plain
    master unchanged, and a re-wrap overwrites the stash at the next
    sharded-state sighting, so the pre-wrap identity fails closed.
    """

    master = getattr(module, name)
    stash = module.__dict__.setdefault("_sharded_master_stash", {})
    try:
        from torch.distributed.tensor import DTensor  # noqa: PLC0415

        if isinstance(master, DTensor):
            stash[name] = master
            return master
    except ImportError:  # pragma: no cover
        pass
    stashed = stash.get(name)
    return stashed if stashed is not None else master


def quantize_master_to_serving_bytes(master: Tensor) -> tuple[Tensor, Tensor]:
    """Run the block-FP8 quantization program Q on one FP32 master.

    Returns ``(weight_fp8, weight_scale_inv)`` in the serving layout:
    float8_e4m3fn ``[out, in]`` plus FP32 ``[ceil(out/128), ceil(in/128)]``
    multiplicative-dequant scales.  This single program point feeds both the
    trainer forward cache and weight-sync publication; its arithmetic is
    deliberately not part of the cross-engine contract (the sampler receives
    these bytes verbatim), but it must be well defined: non-finite masters
    RAISE instead of minting non-finite scales.
    """

    if master.dtype is not torch.float32:
        raise TypeError(f"Full-param FP8 quantization requires an FP32 master, got {master.dtype}")
    if master.dim() != 2:
        raise ValueError(f"Full-param FP8 quantization requires a 2D master, got shape {tuple(master.shape)}")
    if master.device.type != "cuda":
        raise RuntimeError("Full-param FP8 quantization requires CUDA (pinned Triton quantizer)")
    if not bool(torch.isfinite(master).all()):
        raise ValueError("Full-param FP8 master contains non-finite values; refusing to quantize")

    from xorl.ops.quantize import block_fp8_quantize_gkn  # noqa: PLC0415

    weight_fp8, scale_inv = block_fp8_quantize_gkn(master.detach().contiguous(), _BLOCK)
    if weight_fp8.dtype is not _FP8_DTYPE or scale_inv.dtype is not torch.float32:
        raise RuntimeError(
            f"Quantization program produced {weight_fp8.dtype}/{scale_inv.dtype}, expected {_FP8_DTYPE}/{torch.float32}"
        )
    expected_scale_shape = (math.ceil(master.shape[0] / _BLOCK), math.ceil(master.shape[1] / _BLOCK))
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise RuntimeError(
            f"Quantization program produced scale shape {tuple(scale_inv.shape)}, expected {expected_scale_shape}"
        )
    return weight_fp8, scale_inv


def quantize_expert_masters_to_serving_bytes(masters: Tensor) -> tuple[Tensor, Tensor]:
    """Run Q per expert on a stacked ``[E, out, in]`` FP32 master bank.

    Expert-boundary-preserving by construction: each expert's 2D
    master is quantized independently, so no 128×128 tile and no scale entry
    ever spans two experts, and the publication unit is the expert.  The
    output layout matches the serving bank: float8 ``[E, out, in]`` plus FP32
    ``[E, out/128, in/128]`` scales.

    Both dimensions must be 128-aligned.  This is the geometric precondition
    for fuse-after-quantize equivalence (quantizing the fused ``[gate; up]``
    rows equals concatenating per-projection quantizations), which GLM-5.2's
    routed geometry satisfies; anything else RAISES rather than silently
    minting straddling tiles.
    """

    if masters.dim() != 3:
        raise ValueError(f"Expert quantization requires a stacked [E, out, in] master, got {tuple(masters.shape)}")
    num_experts, out_features, in_features = masters.shape
    if num_experts <= 0:
        raise ValueError("Expert quantization requires at least one expert")
    if out_features % _BLOCK or in_features % _BLOCK:
        raise ValueError(
            "GLM-5.2 full-param expert quantization requires 128-aligned expert projections, "
            f"got ({out_features}, {in_features}); straddling tiles would break "
            "fuse-after-quantize equivalence"
        )

    weight_fp8 = torch.empty_like(masters, dtype=_FP8_DTYPE)
    scale_inv = torch.empty(
        (num_experts, out_features // _BLOCK, in_features // _BLOCK),
        dtype=torch.float32,
        device=masters.device,
    )
    for expert_index in range(num_experts):
        expert_weight, expert_scale = quantize_master_to_serving_bytes(masters[expert_index].contiguous())
        weight_fp8[expert_index].copy_(expert_weight)
        scale_inv[expert_index].copy_(expert_scale)
    return weight_fp8, scale_inv


class _Glm52FullParamBlockFP8LinearFunction(torch.autograd.Function):
    """Exact cached-byte forward with straight-through master backward."""

    @staticmethod
    def forward(ctx, input: Tensor, weight_master: Tensor, module) -> Tensor:
        output = module._exact_forward_value(input)
        if output.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 full-param FP8 forward produced {output.dtype}, expected torch.bfloat16")
        if tuple(output.shape) != (*input.shape[:-1], module.out_features):
            raise RuntimeError(
                "GLM-5.2 full-param FP8 output shape mismatch: "
                f"got {tuple(output.shape)}, expected {(*input.shape[:-1], module.out_features)}"
            )
        ctx.module = module
        # Saving the master alongside the input gives autograd its normal
        # version-counter protection: mutating the master between this exact
        # forward and the straight-through backward raises.
        ctx.save_for_backward(input.detach(), weight_master)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, _weight_master = ctx.saved_tensors
        grad_input, grad_master = ctx.module._straight_through_vjp(
            input,
            grad_output,
            needs_input_grad=ctx.needs_input_grad[:2],
        )
        return grad_input, grad_master, None


class Glm52ExactTP1BlockFP8FullParamLinear(nn.Module):
    """Trainable block-FP8 TP1 linear whose forward consumes published bytes.

    State:

    - ``weight_master``: FP32 ``[out, in]`` trainable master;
    - ``quantized_weight_f32``: packed fp8-as-float32 byte cache (the lane's
      byte-safe transport convention);
    - ``weight_scale_inv``: FP32 ``[ceil(out/128), ceil(in/128)]`` cache.

    The caches are refreshed from the master exactly once per optimizer step
    via :meth:`refresh_quantized_cache` and are what the forward consumes and
    :meth:`publishable_weight_bytes` exposes.  Any master mutation without a
    refresh makes the module RAISE on the next forward or publication.
    """

    contract_version = GLM52_EXACT_TP1_FULLPARAM_FP8_CONTRACT_VERSION
    _glm52_exact_fullparam_component = True
    glm52_fullparam_payload_kind = "block_fp8_linear"
    # The FP32 master and the packed byte cache must never be cast by an FSDP
    # mixed-precision policy.
    fsdp_requires_full_precision = True
    _engagement_logged = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        block_size: tuple[int, int] = (_BLOCK, _BLOCK),
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("GLM-5.2 full-param block-FP8 projections are bias-free")
        if tuple(block_size) != (_BLOCK, _BLOCK):
            raise ValueError(f"Only the official GLM block shape (128, 128) is supported, got {block_size}")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("GLM-5.2 full-param block-FP8 dimensions must be positive")
        if in_features % 4:
            raise ValueError("GLM-5.2 full-param block-FP8 in_features must be divisible by four for byte packing")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block_size = (_BLOCK, _BLOCK)
        self.weight_master = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.register_buffer(
            "quantized_weight_f32",
            torch.empty((out_features, in_features // 4), dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.empty(
                (math.ceil(out_features / _BLOCK), math.ceil(in_features / _BLOCK)),
                dtype=torch.float32,
                device=device,
            ),
        )
        # (data_ptr, _version) of the master at the last cache refresh.  None
        # means the cache has never been seeded and the module must not score.
        self._master_identity: tuple[int, int] | None = None
        self._refresh_ordinal = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(cls, module: nn.Linear) -> "Glm52ExactTP1BlockFP8FullParamLinear":
        """Adopt a BF16/FP32 ``nn.Linear``: master = widened weight, cache = Q(master)."""

        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
        if module.bias is not None:
            raise ValueError("GLM-5.2 full-param block-FP8 projections are bias-free")
        replacement = cls(
            module.in_features,
            module.out_features,
            device=module.weight.device,
        )
        with torch.no_grad():
            replacement.weight_master.copy_(module.weight.detach().to(torch.float32))
        replacement.refresh_quantized_cache()
        return replacement

    def load_prequantized(self, weight: Tensor, weight_scale_inv: Tensor) -> None:
        """Seed the cache with checkpoint bytes and the master with their dequantization.

        This preserves step-0 identity: before the first optimizer step the
        trainer scores and publishes the original checkpoint bytes — never
        Q(dequant(checkpoint)), which is not a byte round-trip.
        """

        if weight.dtype is not _FP8_DTYPE:
            raise TypeError(f"Prequantized weight must be {_FP8_DTYPE}, got {weight.dtype}")
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"FP8 weight shape {tuple(weight.shape)} does not match ({self.out_features}, {self.in_features})"
            )
        if weight_scale_inv.dtype is not torch.float32:
            raise TypeError(f"weight_scale_inv must remain FP32, got {weight_scale_inv.dtype}")
        if tuple(weight_scale_inv.shape) != tuple(self.weight_scale_inv.shape):
            raise ValueError(
                f"FP8 scale shape {tuple(weight_scale_inv.shape)} does not match {tuple(self.weight_scale_inv.shape)}"
            )
        if not bool(torch.all(torch.isfinite(weight_scale_inv))):
            raise ValueError("weight_scale_inv contains non-finite values")
        if self.weight_master.device.type != "cuda":
            raise RuntimeError("GLM-5.2 full-param prequantized seeding requires CUDA (pinned dequant kernel)")

        from xorl.ops.quantize import block_fp8_dequantize_gkn  # noqa: PLC0415

        with torch.no_grad():
            device = self.weight_master.device
            weight = weight.to(device).contiguous()
            weight_scale_inv = weight_scale_inv.to(device).contiguous()
            self.quantized_weight_f32.copy_(pack_fp8_as_float32(weight))
            self.weight_scale_inv.copy_(weight_scale_inv)
            dequantized = block_fp8_dequantize_gkn(weight, weight_scale_inv, _BLOCK).to(torch.float32)
            self.weight_master.copy_(dequantized)
        self._record_master_identity()

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def _record_master_identity(self) -> None:
        self._master_identity = _master_identity_tuple(_master_mutation_source(self, "weight_master"))
        self._refresh_ordinal += 1

    def _master_is_stale(self) -> bool:
        if self._master_identity is None:
            return True
        return self._master_identity != _master_identity_tuple(_master_mutation_source(self, "weight_master"))

    def _assert_cache_fresh(self, site: str) -> None:
        if self._master_identity is None:
            raise RuntimeError(
                f"GLM-5.2 full-param block-FP8 {site} before the quantized cache was seeded; "
                "call load_prequantized() or refresh_quantized_cache() first"
            )
        if self._master_is_stale():
            raise RuntimeError(
                f"GLM-5.2 full-param block-FP8 {site} with a stale quantized cache: the FP32 master "
                "mutated after the last refresh_quantized_cache(). Refresh at the optimizer-step "
                "boundary before scoring or publishing; a stale cache must never score tokens."
            )

    def _validate_refresh_source(self) -> None:
        if self.weight_master.device.type != "cuda":
            raise RuntimeError("GLM-5.2 full-param cache refresh requires CUDA (pinned Triton quantizer)")
        if self.weight_master.dtype is not torch.float32:
            raise TypeError("GLM-5.2 full-param cache refresh requires an FP32 master")

    def _gather_refresh_master(self) -> Tensor:
        return _master_full_tensor(_master_mutation_source(self, "weight_master")).contiguous()

    def _stage_quantized_cache(self, full_master: Tensor) -> tuple[Tensor, Tensor]:
        if full_master.dtype is not torch.float32 or tuple(full_master.shape) != (
            self.out_features,
            self.in_features,
        ):
            raise RuntimeError(
                "GLM-5.2 full-param gathered linear master has the wrong dtype or shape: "
                f"{full_master.dtype} {tuple(full_master.shape)}"
            )
        return quantize_master_to_serving_bytes(full_master)

    @torch.no_grad()
    def _commit_quantized_cache(self, staged: tuple[Tensor, Tensor]) -> None:
        weight_fp8, scale_inv = staged
        self.quantized_weight_f32.copy_(pack_fp8_as_float32(weight_fp8))
        self.weight_scale_inv.copy_(scale_inv)
        self._record_master_identity()

    @torch.no_grad()
    def refresh_quantized_cache(self) -> None:
        """Quantize the current master into the consumed-and-published byte cache."""

        self._validate_refresh_source()
        self._commit_quantized_cache(self._stage_quantized_cache(self._gather_refresh_master()))

    def publishable_weight_bytes(self) -> tuple[Tensor, Tensor]:
        """Return the exact serving byte pair the forward consumes.

        The returned tensors are views of the module's caches, not copies:
        weight synchronization publishing them publishes the very bytes the
        trainer scored with.  RAISES if the cache is stale or unseeded.
        """

        self._assert_cache_fresh("publication")
        return self._cached_fp8_weight(), self.weight_scale_inv

    def _cached_fp8_weight(self) -> Tensor:
        return unpack_float32_as_fp8(
            self.quantized_weight_f32,
            (self.out_features, self.in_features),
        )

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _validate_engaged_contract(self, input: Tensor) -> None:
        if self.weight_master.dtype is not torch.float32:
            raise TypeError("GLM-5.2 full-param block-FP8 master must remain FP32")
        if not self.weight_master.requires_grad:
            raise RuntimeError(
                "GLM-5.2 full-param block-FP8 master must be trainable; use the frozen native lane otherwise"
            )
        if input.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 full-param block-FP8 requires BF16 activations, got {input.dtype}")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"GLM-5.2 full-param block-FP8 input width {input.shape[-1]} does not match {self.in_features}"
            )
        if not input.is_contiguous():
            raise ValueError("GLM-5.2 full-param block-FP8 requires contiguous sampler-layout activations")
        if input.numel() == 0:
            raise ValueError("GLM-5.2 full-param block-FP8 does not admit an empty TP1 component batch")
        self._assert_cache_fresh("forward")

    def _log_engagement_once(self) -> None:
        cls = type(self)
        if not cls._engagement_logged:
            cls._engagement_logged = True
            logger.info(
                "GLM-5.2 full-param block-FP8 linear engaged: contract=%s in=%d out=%d block=%s "
                "(cached-byte SGLang W8A8 forward, straight-through FP32 master backward)",
                self.contract_version,
                self.in_features,
                self.out_features,
                self.block_size,
            )

    def _exact_forward_value(self, input: Tensor) -> Tensor:
        """Run the unchanged exact SGLang W8A8 dispatch on the cached bytes."""

        if input.device.type != "cuda":
            raise RuntimeError("GLM-5.2 full-param block-FP8 forward requires CUDA and pinned SGLang kernels")
        if self.quantized_weight_f32.device != input.device or self.weight_scale_inv.device != input.device:
            raise RuntimeError("GLM-5.2 full-param block-FP8 caches and activations must share one CUDA device")
        if self.weight_master.device != input.device:
            raise RuntimeError("GLM-5.2 full-param block-FP8 master and activations must share one CUDA device")
        rows = input.numel() // self.in_features
        input_2d = input.view(rows, self.in_features)
        output = _sglang_native_block_fp8_linear_value(
            input_2d,
            self._cached_fp8_weight().contiguous(),
            self.weight_scale_inv.contiguous(),
            block_size=self.block_size,
        )
        return output.view(*input.shape[:-1], self.out_features)

    def _dequantized_cached_weight(self) -> Tensor:
        from xorl.ops.quantize import block_fp8_dequantize_gkn  # noqa: PLC0415

        return block_fp8_dequantize_gkn(
            self._cached_fp8_weight().contiguous(),
            self.weight_scale_inv.contiguous(),
            _BLOCK,
        ).to(torch.float32)

    def _straight_through_vjp(
        self,
        input: Tensor,
        grad_output: Tensor,
        *,
        needs_input_grad: tuple[bool, bool],
    ) -> tuple[Tensor | None, Tensor | None]:
        """Trainer-owned backward; forward bytes remain the only contract.

        Activation gradient flows through the dequantized cached bytes in the
        declared BF16 linear program — the same treatment the frozen lane's
        surrogate applies to its base branch.  The master gradient is the
        straight-through FP32 GEMM ``grad_outputᵀ @ input``.
        """

        need_input, need_master = needs_input_grad
        if not (need_input or need_master):
            return None, None

        grad_2d = grad_output.reshape(-1, self.out_features)
        input_2d = input.reshape(-1, self.in_features)

        grad_input = None
        if need_input:
            with torch.enable_grad(), torch.autocast(device_type=input.device.type, enabled=False):
                base_input = input.detach().requires_grad_(True)
                base_weight = self._dequantized_cached_weight().to(base_input.dtype)
                base_output = F.linear(base_input, base_weight)
                (grad_input,) = torch.autograd.grad(
                    base_output,
                    base_input,
                    grad_outputs=grad_output.to(base_output.dtype),
                )

        grad_master = None
        if need_master:
            grad_master = grad_2d.float().t().matmul(input_2d.float())

        return grad_input, grad_master

    def forward(self, input: Tensor) -> Tensor:
        self._validate_engaged_contract(input)
        self._log_engagement_once()
        return _Glm52FullParamBlockFP8LinearFunction.apply(input, self.weight_master, self)


GLM52_FULLPARAM_DENSE_MLP_CONTRACT_VERSION = "glm52_fullparam_dense_mlp_v2"


class Glm52FullParamDenseMLP(nn.Module):
    """Full-param dense MLP: fused gate/up full-param linear, SwiGLU, down.

    Mirrors serving exact mode's dense composition: fused ``[gate; up]`` base,
    one-round FP32 SwiGLU, then down.  The fuse boundary is byte-safe because
    GLM's intermediate size is 128-aligned, so checkpoint bytes and scales
    concatenate across the gate/up boundary without any tile straddling
    (fuse-after-quantize equivalence).
    """

    contract_version = GLM52_FULLPARAM_DENSE_MLP_CONTRACT_VERSION
    _glm52_exact_fullparam_component = True
    glm52_fullparam_payload_kind = "block_fp8_dense_mlp"
    fsdp_requires_full_precision = True

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if intermediate_size % _BLOCK:
            raise ValueError(
                "GLM-5.2 full-param dense MLP requires a 128-aligned intermediate size for the "
                f"fused gate/up boundary, got {intermediate_size}"
            )
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_up_proj = Glm52ExactTP1BlockFP8FullParamLinear(
            hidden_size,
            2 * intermediate_size,
            device=device,
        )
        self.down_proj = Glm52ExactTP1BlockFP8FullParamLinear(
            intermediate_size,
            hidden_size,
            device=device,
        )

    def load_prequantized_split(
        self,
        gate_weight: Tensor,
        gate_scale_inv: Tensor,
        up_weight: Tensor,
        up_scale_inv: Tensor,
        down_weight: Tensor,
        down_scale_inv: Tensor,
    ) -> None:
        """Seed from the checkpoint's unfused gate/up/down byte pairs."""

        for name, tensor in (("gate", gate_weight), ("up", up_weight)):
            if tuple(tensor.shape) != (self.intermediate_size, self.hidden_size):
                raise ValueError(
                    f"GLM-5.2 dense {name}_proj weight must be "
                    f"({self.intermediate_size}, {self.hidden_size}), got {tuple(tensor.shape)}"
                )
        fused_weight = torch.cat((gate_weight, up_weight), dim=0)
        fused_scale = torch.cat((gate_scale_inv, up_scale_inv), dim=0)
        self.gate_up_proj.load_prequantized(fused_weight, fused_scale)
        self.down_proj.load_prequantized(down_weight, down_scale_inv)

    def _validate_refresh_source(self) -> None:
        self.gate_up_proj._validate_refresh_source()
        self.down_proj._validate_refresh_source()

    def _gather_refresh_masters(self) -> tuple[Tensor, Tensor]:
        return self.gate_up_proj._gather_refresh_master(), self.down_proj._gather_refresh_master()

    def _stage_quantized_caches(
        self, full_masters: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        gate_up_master, down_master = full_masters
        return (
            self.gate_up_proj._stage_quantized_cache(gate_up_master),
            self.down_proj._stage_quantized_cache(down_master),
        )

    @torch.no_grad()
    def _commit_quantized_caches(self, staged: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]) -> None:
        self.gate_up_proj._commit_quantized_cache(staged[0])
        self.down_proj._commit_quantized_cache(staged[1])

    def refresh_quantized_cache(self) -> None:
        self._validate_refresh_source()
        self._commit_quantized_caches(self._stage_quantized_caches(self._gather_refresh_masters()))

    def publishable_split_projections(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Row-sliced views of the fused cache in checkpoint (unfused) form.

        The slices are views: publishing them publishes the consumed bytes.
        Row slices never straddle 128-tiles because intermediate_size is
        128-aligned (enforced at construction).
        """

        fused_weight, fused_scale = self.gate_up_proj.publishable_weight_bytes()
        scale_rows = self.intermediate_size // _BLOCK
        return (
            fused_weight.narrow(0, 0, self.intermediate_size),
            fused_scale.narrow(0, 0, scale_rows),
            fused_weight.narrow(0, self.intermediate_size, self.intermediate_size),
            fused_scale.narrow(0, scale_rows, scale_rows),
        )

    def publishable_checkpoint_projections(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """The checkpoint-form byte tuple the serving LOADER actually consumes.

        Field order matches the ``block_fp8_dense_mlp`` payload schema:
        ``(gate, gate_scale_inv, up, up_scale_inv, down, down_scale_inv)``.
        The gate/up tensors are zero-copy row views of the fused cache (the
        loader fuses gate rows first, then up rows — the inverse of these
        views); down is the down cache verbatim.  Staleness raises through
        the owning linears' publication gates.
        """

        gate_weight, gate_scale, up_weight, up_scale = self.publishable_split_projections()
        down_weight, down_scale = self.down_proj.publishable_weight_bytes()
        return gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale

    def forward(self, input: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(input)
        # SGLang exact mode universally selects SiluAndMul.forward_exact:
        # SiLU and multiply in FP32 with one final rounding.  The historical
        # trainer op rounded between SiLU and multiply, so it is not the same
        # numerical program.
        activated = one_round_swiglu(gate_up)
        return self.down_proj(activated)


class _Glm52FullParamRouterGemmFunction(torch.autograd.Function):
    """Pinned batch-invariant router forward on the cached BF16 view.

    The forward value program is the existing ``bi_router_gemm`` on
    BF16 weight bytes; only the byte source changes (cached view of an FP32
    master).  The backward mirrors the contracted router backward's ordinary
    matmuls but keeps the master gradient in FP32 (the master is FP32; no
    quantization sits between master view and kernel, so this is ordinary
    autograd through one explicit rounding, not a straight-through
    estimator).
    """

    @staticmethod
    def forward(ctx, hidden: Tensor, weight_master: Tensor, module) -> Tensor:
        effective = module._effective_weight
        output = module._router_value(hidden)
        if output.shape != (hidden.shape[0], module.num_experts):
            raise RuntimeError(
                "GLM-5.2 full-param router output shape mismatch: "
                f"got {tuple(output.shape)}, expected {(hidden.shape[0], module.num_experts)}"
            )
        ctx.save_for_backward(hidden, effective, weight_master)
        return output

    @staticmethod
    def backward(ctx, grad_logits: Tensor):
        hidden, effective, _weight_master = ctx.saved_tensors
        grad_logits = grad_logits.to(torch.float32)
        grad_hidden = grad_master = None
        if ctx.needs_input_grad[0]:
            grad_hidden = (grad_logits @ effective.float()).to(hidden.dtype)
        if ctx.needs_input_grad[1]:
            grad_master = grad_logits.t().matmul(hidden.float())
        return grad_hidden, grad_master, None


class Glm52ExactFullParamRouterWeight(nn.Module):
    """FP32 router master with the single explicit BF16 view both engines consume.

    The router GEMM contract is unchanged (``bi_router_gemm`` on BF16 bytes);
    unfreezing only changes which bytes fill the weight.  The BF16 view is
    materialized once per optimizer step by :meth:`refresh_effective_view`,
    consumed by the trainer forward, and exposed verbatim to weight sync by
    :meth:`publishable_weight_bytes`.  ``e_score_correction_bias`` stays
    outside this module and remains frozen.
    """

    contract_version = GLM52_EXACT_FULLPARAM_ROUTER_CONTRACT_VERSION
    _glm52_exact_fullparam_component = True
    glm52_fullparam_payload_kind = "bf16_router"
    fsdp_requires_full_precision = True
    _engagement_logged = False

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0 or hidden_size <= 0:
            raise ValueError("GLM-5.2 full-param router dimensions must be positive")
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.weight_master = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.register_buffer(
            "_effective_weight",
            torch.empty((num_experts, hidden_size), dtype=torch.bfloat16, device=device),
        )
        self._master_identity: tuple[int, int] | None = None
        self._refresh_ordinal = 0

    def load_from_bf16(self, weight: Tensor) -> None:
        """Seed from checkpoint BF16 bytes; the widened master reproduces them exactly."""

        if weight.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 router checkpoint weight must be BF16, got {weight.dtype}")
        if tuple(weight.shape) != (self.num_experts, self.hidden_size):
            raise ValueError(
                f"Router weight shape {tuple(weight.shape)} does not match ({self.num_experts}, {self.hidden_size})"
            )
        with torch.no_grad():
            device = self.weight_master.device
            self._effective_weight.copy_(weight.to(device))
            # BF16 -> FP32 widening is exact, so refresh_effective_view()
            # would regenerate these very bytes; there is no router analog of
            # the FP8 step-0 round-trip hazard.
            self.weight_master.copy_(weight.to(device=device, dtype=torch.float32))
        self._record_master_identity()

    def _record_master_identity(self) -> None:
        self._master_identity = _master_identity_tuple(_master_mutation_source(self, "weight_master"))
        self._refresh_ordinal += 1

    def _master_is_stale(self) -> bool:
        if self._master_identity is None:
            return True
        return self._master_identity != _master_identity_tuple(_master_mutation_source(self, "weight_master"))

    def _assert_view_fresh(self, site: str) -> None:
        if self._master_identity is None:
            raise RuntimeError(
                f"GLM-5.2 full-param router {site} before the BF16 view was seeded; "
                "call load_from_bf16() or refresh_effective_view() first"
            )
        if self._master_is_stale():
            raise RuntimeError(
                f"GLM-5.2 full-param router {site} with a stale BF16 view: the FP32 master mutated "
                "after the last refresh_effective_view(). Refresh at the optimizer-step boundary."
            )

    def _validate_refresh_source(self) -> None:
        if self.weight_master.dtype is not torch.float32:
            raise TypeError("GLM-5.2 full-param router master must remain FP32")

    def _gather_refresh_master(self) -> Tensor:
        return _master_full_tensor(_master_mutation_source(self, "weight_master"))

    def _stage_effective_view(self, full_master: Tensor) -> Tensor:
        if full_master.dtype is not torch.float32 or tuple(full_master.shape) != (
            self.num_experts,
            self.hidden_size,
        ):
            raise RuntimeError(
                "GLM-5.2 full-param gathered router master has the wrong dtype or shape: "
                f"{full_master.dtype} {tuple(full_master.shape)}"
            )
        if not bool(torch.isfinite(full_master).all()):
            raise ValueError("GLM-5.2 full-param router master contains non-finite values")
        return full_master.to(torch.bfloat16)

    @torch.no_grad()
    def _commit_effective_view(self, effective: Tensor) -> None:
        self._effective_weight.copy_(effective)
        self._record_master_identity()

    @torch.no_grad()
    def refresh_effective_view(self) -> None:
        self._validate_refresh_source()
        self._commit_effective_view(self._stage_effective_view(self._gather_refresh_master()))

    def publishable_weight_bytes(self) -> Tensor:
        """Return the BF16 byte view weight sync must publish (not a copy)."""

        self._assert_view_fresh("publication")
        return self._effective_weight

    def _log_engagement_once(self) -> None:
        cls = type(self)
        if not cls._engagement_logged:
            cls._engagement_logged = True
            logger.info(
                "GLM-5.2 full-param router engaged: contract=%s experts=%d hidden=%d "
                "(pinned bi_router_gemm on the published BF16 view, FP32 master autograd)",
                self.contract_version,
                self.num_experts,
                self.hidden_size,
            )

    def _router_value(self, hidden_states: Tensor) -> Tensor:
        """Run the lane's pinned batch-invariant router GEMM on the cached view."""

        if hidden_states.device.type != "cuda":
            raise RuntimeError("GLM-5.2 full-param router forward requires CUDA and the pinned BI router kernel")
        if self._effective_weight.device != hidden_states.device:
            raise RuntimeError("GLM-5.2 full-param router view and activations must share one CUDA device")

        from xorl.ops.sglang.batch_invariant_ops import bi_router_gemm  # noqa: PLC0415

        return bi_router_gemm(hidden_states, self._effective_weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 full-param router requires BF16 hidden states, got {hidden_states.dtype}")
        if hidden_states.dim() != 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "GLM-5.2 full-param router requires flattened [tokens, hidden] BF16 input: "
                f"got {tuple(hidden_states.shape)}, expected [tokens, {self.hidden_size}]"
            )
        if hidden_states.numel() == 0:
            raise ValueError("GLM-5.2 full-param router does not admit an empty token batch")
        if not hidden_states.is_contiguous():
            raise ValueError("GLM-5.2 full-param router requires contiguous activations")
        if self.weight_master.dtype is not torch.float32:
            raise TypeError("GLM-5.2 full-param router master must remain FP32")
        if not self.weight_master.requires_grad:
            raise RuntimeError("GLM-5.2 full-param router master must be trainable; keep the frozen router otherwise")
        self._assert_view_fresh("forward")
        self._log_engagement_once()
        return _Glm52FullParamRouterGemmFunction.apply(hidden_states, self.weight_master, self)


GLM52_FULLPARAM_ROUTING_SURROGATE_CONTRACT_VERSION = "glm52_fullparam_routing_weight_surrogate_v1"

# The serving renormalization epsilon (sglang topk _RENORMALIZE_SUM_EPSILON,
# shared by the canonical biased_grouped_topk and xorl's generic route).
_ROUTING_RENORMALIZE_EPSILON = 1e-20

_routing_surrogate_engagement_logged = False


class _Glm52FullParamRoutingWeightsSurrogate(torch.autograd.Function):
    """Trainer-owned gradient attachment for SERVING routing-weight values.

    The GLM-5.2 serving top-k programs are gradient-opaque (the canonical
    ``biased_grouped_topk`` may resolve to an in-place CUDA kernel with no
    ``grad_fn``), and the generic trainer route detaches routing weights for
    every mode that does not train routers. Full-param mode trains routers,
    and their only gradient path is the combine multiply:
    ``d loss/d routing`` flows back from the expert banks and must continue
    into the router logits.

    Forward: returns the serving-program VALUES verbatim (byte-identical —
    selection and weights stay exactly what the K3 stream scored).
    Backward: the analytic vjp of the differentiable regather program at the
    same logits — ``sigmoid(logits).gather(ids)``, renormalized with the
    serving epsilon when the branch renormalizes, times the branch's scale
    multiplier — evaluated in FP32.  Top-k SELECTION gradients are zero
    almost everywhere and are correctly omitted (the ids are constants).
    """

    @staticmethod
    def forward(
        ctx,
        router_logits: Tensor,
        serving_weights: Tensor,
        selected_experts: Tensor,
        renormalize: bool,
        scale: float,
    ) -> Tensor:
        if router_logits.dim() != 2 or serving_weights.dim() != 2:
            raise ValueError(
                "GLM-5.2 routing surrogate requires [tokens, experts] logits and [tokens, topk] weights; "
                f"got {tuple(router_logits.shape)} and {tuple(serving_weights.shape)}"
            )
        if tuple(selected_experts.shape) != tuple(serving_weights.shape):
            raise ValueError(
                "GLM-5.2 routing surrogate requires ids and weights of one shape; "
                f"got {tuple(selected_experts.shape)} and {tuple(serving_weights.shape)}"
            )
        ctx.save_for_backward(router_logits.detach(), selected_experts)
        ctx.renormalize = bool(renormalize)
        ctx.scale = float(scale)
        # The VALUES are the serving program's, untouched; only the autograd
        # edge to the logits is new.
        return serving_weights.detach()

    @staticmethod
    def backward(ctx, grad_weights: Tensor):
        router_logits, selected_experts = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast(device_type=router_logits.device.type, enabled=False):
            logits_ref = router_logits.float().detach().requires_grad_(True)
            gathered = torch.sigmoid(logits_ref).gather(1, selected_experts.long())
            if ctx.renormalize:
                weights = gathered / (
                    gathered.sum(dim=-1, keepdim=True, dtype=torch.float32) + _ROUTING_RENORMALIZE_EPSILON
                )
            else:
                weights = gathered
            if ctx.scale != 1.0:
                weights = weights * ctx.scale
            (grad_logits,) = torch.autograd.grad(
                weights,
                logits_ref,
                grad_outputs=grad_weights.float(),
            )
        return grad_logits.to(router_logits.dtype), None, None, None, None


def glm52_fullparam_routing_weights_with_grad(
    router_logits: Tensor,
    serving_weights: Tensor,
    selected_experts: Tensor,
    *,
    renormalize: bool,
    scale: float,
) -> Tensor:
    """Attach the trainer-owned routing-weight gradient to serving values."""

    global _routing_surrogate_engagement_logged
    if not _routing_surrogate_engagement_logged:
        _routing_surrogate_engagement_logged = True
        logger.info(
            "GLM-5.2 full-param routing-weight surrogate engaged: contract=%s "
            "(forward = serving values verbatim; backward = analytic regather vjp "
            "sigmoid-gather%s x%.6g into the router logits)",
            GLM52_FULLPARAM_ROUTING_SURROGATE_CONTRACT_VERSION,
            "-renormalize" if renormalize else "",
            scale,
        )
    return _Glm52FullParamRoutingWeightsSurrogate.apply(
        router_logits,
        serving_weights,
        selected_experts,
        renormalize,
        scale,
    )


__all__ = [
    "GLM52_EXACT_FULLPARAM_ROUTER_CONTRACT_VERSION",
    "GLM52_EXACT_TP1_FULLPARAM_FP8_CONTRACT_VERSION",
    "GLM52_FULLPARAM_DENSE_MLP_CONTRACT_VERSION",
    "GLM52_FULLPARAM_ROUTING_SURROGATE_CONTRACT_VERSION",
    "Glm52ExactFullParamRouterWeight",
    "Glm52ExactTP1BlockFP8FullParamLinear",
    "Glm52FullParamDenseMLP",
    "glm52_fullparam_routing_weights_with_grad",
    "quantize_expert_masters_to_serving_bytes",
    "quantize_master_to_serving_bytes",
]
