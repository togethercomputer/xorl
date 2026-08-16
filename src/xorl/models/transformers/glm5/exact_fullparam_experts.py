"""Full-parameter block-FP8 routed-expert bank for the exact GLM-5.2 lane.

The bank extends :class:`Glm52NativeBlockFP8Experts` with FP32 expert masters
in the serving GKN orientation, a per-step quantized byte cache (the parent's
packed tensors, re-registered as BUFFERS here — see ``__init__``), and a
straight-through backward.  The value path is the parent's literal SGLang
fused block-FP8 expert sequence on the cached bytes; only the byte source
changes.

Quantization is expert-boundary-preserving: each expert's 2D GKN matrix is
quantized independently (`quantize_expert_masters_to_serving_bytes`), so no
tile or scale spans two experts and the publication unit is the expert.
Because Q's 128x128 tiles are square and elementwise beyond the tile absmax,
quantizing in the GKN orientation equals transposing a row-major quantization
byte-for-byte; the cache is produced directly in the serving layout.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    _master_full_tensor,
    _master_identity_tuple,
    _master_mutation_source,
    quantize_expert_masters_to_serving_bytes,
)
from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
from xorl.ops.block_fp8_native import pack_fp8_as_float32


logger = logging.getLogger(__name__)

GLM52_FULLPARAM_ROUTED_EXPERTS_CONTRACT_VERSION = "glm52_fullparam_block_fp8_routed_experts_v1"


def _row_major_copy(tensor: Tensor) -> Tensor:
    """A standard-stride copy (``.contiguous()`` is a no-op on size-1 dims and
    can leave byte-view-hostile strides behind a transpose)."""

    output = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    output.copy_(tensor)
    return output


class Glm52ExpertCheckpointPublication:
    """One globally-indexed expert of a full-param bank, in checkpoint form.

    Payload-producer adapter for checkpoint-form publication: exposes
    exactly the split per-expert tensors the serving loader consumes —
    ``gate_proj``/``up_proj``/``down_proj`` weights and scales in the official
    row-major checkpoint orientation.  All byte movement is the inverse of the
    production ``NativeBlockFP8ExpertPairBuffer`` fuse (a pure permutation, no
    arithmetic).  Staleness is delegated to the owning bank: every publication
    RAISES if any expert master mutated after the last cache refresh.
    """

    glm52_fullparam_payload_kind = "block_fp8_expert"

    def __init__(self, bank: "Glm52FullParamBlockFP8RoutedExperts", local_slot: int, global_expert_id: int):
        self._bank = bank
        self.local_slot = int(local_slot)
        self.global_expert_id = int(global_expert_id)

    @property
    def contract_version(self) -> str:
        return self._bank.contract_version

    def publishable_checkpoint_tensors(self) -> tuple[Tensor, ...]:
        return self._bank.publishable_expert_checkpoint_tensors(self.local_slot)


class _Glm52FullParamRoutedExpertsFunction(torch.autograd.Function):
    """Cached-byte expert value forward with straight-through master backward."""

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        routing: Tensor,
        local_ids: Tensor,
        gate_up_master: Tensor,
        down_master: Tensor,
        routed_scaling_factor: float,
        module,
    ) -> Tensor:
        output = module._sampler_value(
            hidden,
            routing,
            local_ids,
            routed_scaling_factor=float(routed_scaling_factor),
        )
        expected_shape = (hidden.shape[0], module.hidden_size)
        if output.dtype is not torch.bfloat16 or tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "GLM-5.2 full-param routed expert output contract mismatch: "
                f"got {output.dtype} {tuple(output.shape)}, expected torch.bfloat16 {expected_shape}"
            )
        ctx.module = module
        ctx.routed_scaling_factor = float(routed_scaling_factor)
        # Saving the masters gives autograd its version-counter protection
        # against mutation between the exact forward and the surrogate.
        ctx.save_for_backward(hidden.detach(), routing.detach(), local_ids, gate_up_master, down_master)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        hidden, routing, local_ids, _gate_up_master, _down_master = ctx.saved_tensors
        gradients = ctx.module._straight_through_vjp(
            hidden,
            routing,
            local_ids,
            grad_output=grad_output,
            routed_scaling_factor=ctx.routed_scaling_factor,
            needs_input_grad=(
                ctx.needs_input_grad[0],
                ctx.needs_input_grad[1],
                ctx.needs_input_grad[3],
                ctx.needs_input_grad[4],
            ),
        )
        grad_hidden, grad_routing, grad_gate_up, grad_down = gradients
        return grad_hidden, grad_routing, None, grad_gate_up, grad_down, None, None


class Glm52FullParamBlockFP8RoutedExperts(Glm52NativeBlockFP8Experts):
    """One rank-local expert bank with FP32 masters and a published byte cache.

    State (``num_experts`` is the rank-local bank size, as in the parent):

    - ``gate_up_weight_master``: FP32 ``[E, hidden, 2*intermediate]`` (GKN);
    - ``down_weight_master``: FP32 ``[E, intermediate, hidden]`` (GKN);
    - parent packed tensors + scales, re-registered as BUFFERS: the per-step
      quantized byte cache, outside every sharding/optimizer plan and
      refreshed only by :meth:`refresh_quantized_cache`.
    """

    contract_version = GLM52_FULLPARAM_ROUTED_EXPERTS_CONTRACT_VERSION
    _glm52_exact_fullparam_component = True
    glm52_fullparam_payload_kind = "block_fp8_expert_bank"
    fsdp_requires_full_precision = True
    _engagement_logged = False
    _CACHE_BUFFER_NAMES = (
        "gate_up_packed_weight_f32",
        "gate_up_weight_scale_inv",
        "down_packed_weight_f32",
        "down_weight_scale_inv",
    )

    def _apply(self, fn, recurse: bool = True):
        """Byte-cache cast protection, reimplemented for BUFFER caches.

        The parent's ``_apply`` protects the caches as parameters; here they
        are buffers (see ``__init__``), so hold them out of ``_buffers``, run
        the plain ``nn.Module._apply`` (skipping the parent's parameter-based
        protection), and restore them with device movement but never a dtype
        cast — casting packed FP8-as-FP32 bytes would corrupt them.
        """

        probe = fn(torch.empty(0, dtype=torch.float32, device=self.gate_up_packed_weight_f32.device))
        protected = {name: self._buffers.pop(name) for name in self._CACHE_BUFFER_NAMES}
        try:
            result = super(Glm52NativeBlockFP8Experts, self)._apply(fn, recurse=recurse)
            replacements = {
                name: (
                    torch.empty_like(buffer, dtype=torch.float32, device=probe.device)
                    if buffer.is_meta
                    else buffer.to(device=probe.device, dtype=torch.float32)
                )
                for name, buffer in protected.items()
            }
        except Exception:
            self._buffers.update(protected)
            raise
        self._buffers.update(replacements)
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Keep the parent's fail-closed cast refusal on the cache tensors,
        # which its parameter walk no longer sees now that they are buffers.
        for name in self._CACHE_BUFFER_NAMES:
            key = f"{prefix}{name}"
            tensor = state_dict.get(key)
            buffer = getattr(self, name)
            if tensor is not None and (tensor.dtype is not torch.float32 or tuple(tensor.shape) != tuple(buffer.shape)):
                raise TypeError(
                    f"GLM native FP8 state {key} must be FP32 {tuple(buffer.shape)}, "
                    f"got {tensor.dtype} {tuple(tensor.shape)}; refusing a load_state_dict cast"
                )
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        hidden_act: str = "silu",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            num_experts,
            hidden_size,
            intermediate_size,
            hidden_act=hidden_act,
            device=device,
        )
        # The byte caches are BUFFERS in the trainer bank (the serving parent
        # registers frozen parameters): FSDP2's ``fully_shard`` converts every
        # parameter to a DTensor, and the step-boundary refresh writes plain
        # quantizer output into the caches — a DTensor cache would reject the
        # write. Buffers keep the byte caches out of the sharding plan
        # entirely, matching the dense composite's
        # ``quantized_weight_f32``/``weight_scale_inv`` buffers.
        for name in self._CACHE_BUFFER_NAMES:
            cache = self._parameters.pop(name)
            self.register_buffer(name, cache.detach())
        self.gate_up_weight_master = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=True,
        )
        self.down_weight_master = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                hidden_size,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=True,
        )
        self._master_identity: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._refresh_ordinal = 0
        # Global expert ownership (EP): assigned once by the admission from
        # the model's EP placement; None means checkpoint-form publication is
        # not admitted (fail closed, never guess an offset).
        self._global_expert_start: int | None = None
        self._global_num_experts: int | None = None

    # ------------------------------------------------------------------
    # Global expert indexing (EP ownership)
    # ------------------------------------------------------------------

    def assign_global_expert_range(self, expert_start: int, num_global_experts: int) -> None:
        """Record which contiguous global expert block this rank-local bank owns.

        Mirrors the production ``NativeBlockFP8ExpertPairBuffer`` ownership
        rule (``expert_start = ep_rank * local_experts``); checkpoint-form
        publication is refused until this is assigned.
        """

        expert_start = int(expert_start)
        num_global_experts = int(num_global_experts)
        if num_global_experts <= 0 or num_global_experts % self.num_experts:
            raise ValueError(
                f"Global expert count {num_global_experts} must be a positive multiple of the "
                f"local bank size {self.num_experts}"
            )
        if expert_start < 0 or expert_start % self.num_experts or expert_start + self.num_experts > num_global_experts:
            raise ValueError(
                f"Global expert start {expert_start} is not a valid EP-local block of size "
                f"{self.num_experts} within {num_global_experts} experts"
            )
        if self._global_expert_start is not None and (
            self._global_expert_start != expert_start or self._global_num_experts != num_global_experts
        ):
            raise RuntimeError(
                "GLM-5.2 full-param expert bank global range reassignment: "
                f"({self._global_expert_start}, {self._global_num_experts}) -> "
                f"({expert_start}, {num_global_experts}); EP ownership is fixed for a bank's lifetime"
            )
        self._global_expert_start = expert_start
        self._global_num_experts = num_global_experts

    @property
    def global_expert_ids(self) -> tuple[int, ...]:
        if self._global_expert_start is None:
            raise RuntimeError(
                "GLM-5.2 full-param expert bank has no assigned global expert range; "
                "call assign_global_expert_range() at admission before checkpoint-form publication"
            )
        return tuple(range(self._global_expert_start, self._global_expert_start + self.num_experts))

    @property
    def num_global_experts(self) -> int:
        """Total global expert count this bank's EP ownership was declared for."""

        if self._global_num_experts is None:
            raise RuntimeError(
                "GLM-5.2 full-param expert bank has no assigned global expert range; "
                "call assign_global_expert_range() at admission before checkpoint-form publication"
            )
        return self._global_num_experts

    def checkpoint_publications(self) -> list[tuple[int, Glm52ExpertCheckpointPublication]]:
        """(global_expert_id, publication adapter) for every local expert."""

        return [
            (global_id, Glm52ExpertCheckpointPublication(self, local_slot, global_id))
            for local_slot, global_id in enumerate(self.global_expert_ids)
        ]

    def publishable_expert_checkpoint_tensors(self, local_slot: int) -> tuple[Tensor, ...]:
        """One expert's cached bytes in the split checkpoint form the loader consumes.

        Field order matches the ``block_fp8_expert`` payload schema:
        ``(gate, gate_scale_inv, up, up_scale_inv, down, down_scale_inv)``,
        row-major ``[intermediate, hidden]`` gate/up and ``[hidden,
        intermediate]`` down — exactly the inverse of the production pair
        buffer's ``cat((gate, up), dim=0).T`` fuse.  The transposes are byte
        permutations of the cache (materialized contiguously; no arithmetic
        touches the bytes).  RAISES on a stale cache.
        """

        local_slot = int(local_slot)
        if not 0 <= local_slot < self.num_experts:
            raise ValueError(f"Expert slot {local_slot} outside the local bank [0, {self.num_experts})")
        self._assert_cache_fresh("publication")
        intermediate = self.intermediate_size
        scale_blocks = intermediate // 128
        gate_up = self.gate_up_proj[local_slot]  # [hidden, 2*intermediate] (GKN)
        gate_up_scale = self.gate_up_weight_scale_inv[local_slot]  # [hidden/128, 2*intermediate/128]
        down = self.down_proj[local_slot]  # [intermediate, hidden] (GKN)
        down_scale = self.down_weight_scale_inv[local_slot]  # [intermediate/128, hidden/128]
        return (
            _row_major_copy(gate_up[:, :intermediate].t()),
            _row_major_copy(gate_up_scale[:, :scale_blocks].t()),
            _row_major_copy(gate_up[:, intermediate:].t()),
            _row_major_copy(gate_up_scale[:, scale_blocks:].t()),
            _row_major_copy(down.t()),
            _row_major_copy(down_scale.t()),
        )

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def _current_master_identity(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            _master_identity_tuple(_master_mutation_source(self, "gate_up_weight_master")),
            _master_identity_tuple(_master_mutation_source(self, "down_weight_master")),
        )

    def _record_master_identity(self) -> None:
        self._master_identity = self._current_master_identity()
        self._refresh_ordinal += 1

    def _assert_cache_fresh(self, site: str) -> None:
        if self._master_identity is None:
            raise RuntimeError(
                f"GLM-5.2 full-param routed experts {site} before the quantized cache was seeded; "
                "call load_prequantized() or refresh_quantized_cache() first"
            )
        if self._master_identity != self._current_master_identity():
            raise RuntimeError(
                f"GLM-5.2 full-param routed experts {site} with a stale quantized cache: an expert "
                "master mutated after the last refresh_quantized_cache(). Refresh at the "
                "optimizer-step boundary; a stale cache must never score tokens."
            )

    def _validate_refresh_source(self) -> None:
        if self.gate_up_weight_master.device.type != "cuda":
            raise RuntimeError("GLM-5.2 full-param expert cache refresh requires CUDA (pinned Triton quantizer)")
        if self.gate_up_weight_master.dtype is not torch.float32 or self.down_weight_master.dtype is not torch.float32:
            raise TypeError("GLM-5.2 full-param expert cache refresh requires FP32 masters")

    def _gather_refresh_masters(self) -> tuple[Tensor, Tensor]:
        return (
            _master_full_tensor(_master_mutation_source(self, "gate_up_weight_master")).contiguous(),
            _master_full_tensor(_master_mutation_source(self, "down_weight_master")).contiguous(),
        )

    def _stage_quantized_caches(self, full_masters: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        gate_up_master, down_master = full_masters
        expected_gate_up = (self.num_experts, self.hidden_size, 2 * self.intermediate_size)
        expected_down = (self.num_experts, self.intermediate_size, self.hidden_size)
        if gate_up_master.dtype is not torch.float32 or tuple(gate_up_master.shape) != expected_gate_up:
            raise RuntimeError(
                "GLM-5.2 full-param gathered gate/up expert master has the wrong dtype or shape: "
                f"{gate_up_master.dtype} {tuple(gate_up_master.shape)}"
            )
        if down_master.dtype is not torch.float32 or tuple(down_master.shape) != expected_down:
            raise RuntimeError(
                "GLM-5.2 full-param gathered down expert master has the wrong dtype or shape: "
                f"{down_master.dtype} {tuple(down_master.shape)}"
            )
        gate_up_fp8, gate_up_scale = quantize_expert_masters_to_serving_bytes(gate_up_master)
        down_fp8, down_scale = quantize_expert_masters_to_serving_bytes(down_master)
        return gate_up_fp8, gate_up_scale, down_fp8, down_scale

    @torch.no_grad()
    def _commit_quantized_caches(self, staged: tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
        gate_up_fp8, gate_up_scale, down_fp8, down_scale = staged
        self.gate_up_packed_weight_f32.copy_(
            pack_fp8_as_float32(gate_up_fp8).reshape(self.gate_up_packed_weight_f32.shape)
        )
        self.gate_up_weight_scale_inv.copy_(gate_up_scale)
        self.down_packed_weight_f32.copy_(pack_fp8_as_float32(down_fp8).reshape(self.down_packed_weight_f32.shape))
        self.down_weight_scale_inv.copy_(down_scale)
        self._record_master_identity()

    @torch.no_grad()
    def refresh_quantized_cache(self) -> None:
        """Quantize both master banks into the consumed-and-published caches."""

        self._validate_refresh_source()
        self._commit_quantized_caches(self._stage_quantized_caches(self._gather_refresh_masters()))

    def load_prequantized(self, gate_up, gate_up_scale_inv, down, down_scale_inv) -> None:
        """Seed caches with checkpoint bytes and masters with their dequantization.

        Step-0 identity: before the first optimizer step the bank
        scores and publishes the original checkpoint bytes — never
        Q(dequant(checkpoint)).
        """

        if self.gate_up_weight_master.device.type != "cuda":
            raise RuntimeError("GLM-5.2 full-param expert prequantized seeding requires CUDA (pinned dequant kernel)")
        super().load_prequantized(gate_up, gate_up_scale_inv, down, down_scale_inv)
        try:
            from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang block-FP8 dequantization is required for master seeding") from exc
        with torch.no_grad():
            self.gate_up_weight_master.copy_(
                block_quant_dequant(
                    self.gate_up_proj,
                    self.gate_up_weight_scale_inv,
                    [128, 128],
                    torch.bfloat16,
                ).to(torch.float32)
            )
            self.down_weight_master.copy_(
                block_quant_dequant(
                    self.down_proj,
                    self.down_weight_scale_inv,
                    [128, 128],
                    torch.bfloat16,
                ).to(torch.float32)
            )
        self._record_master_identity()

    def publishable_expert_bytes(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the exact serving byte views the forward consumes.

        Field order matches the serving ``load_prequantized`` contract:
        ``(gate_up_fp8, gate_up_scale_inv, down_fp8, down_scale_inv)``.
        """

        self._assert_cache_fresh("publication")
        return (
            self.gate_up_proj,
            self.gate_up_weight_scale_inv,
            self.down_proj,
            self.down_weight_scale_inv,
        )

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _validate_engaged_contract(self, hidden: Tensor, routing: Tensor, local_ids: Tensor) -> None:
        for name in ("gate_up_weight_master", "down_weight_master"):
            master = getattr(self, name)
            if master.dtype is not torch.float32:
                raise TypeError(f"GLM-5.2 full-param expert {name} must remain FP32")
            if not master.requires_grad:
                raise RuntimeError(
                    f"GLM-5.2 full-param expert {name} must be trainable; use the frozen native bank otherwise"
                )
        for name in (
            "gate_up_packed_weight_f32",
            "gate_up_weight_scale_inv",
            "down_packed_weight_f32",
            "down_weight_scale_inv",
        ):
            if getattr(self, name).requires_grad:
                raise RuntimeError(f"GLM-5.2 full-param expert byte cache {name} must remain frozen")
        # Stored-rows admission: the local-id bound below checks against the
        # DECLARED EP-local size, so
        # the declaration is only a safe bound if the storage actually holds
        # that many rows.  A bank whose masters or caches store fewer rows
        # (e.g. an EP slice applied twice during construction) must refuse
        # here rather than fault in the fused kernel.
        stored_rows = {
            name: int(getattr(self, name).shape[0])
            for name in (
                "gate_up_weight_master",
                "down_weight_master",
                "gate_up_packed_weight_f32",
                "gate_up_weight_scale_inv",
                "down_packed_weight_f32",
                "down_weight_scale_inv",
            )
        }
        mismatched = {name: count for name, count in stored_rows.items() if count != int(self.num_experts)}
        if mismatched:
            raise RuntimeError(
                f"GLM-5.2 full-param expert bank declares {int(self.num_experts)} EP-local experts but "
                f"stores {mismatched}; a bank storing a different row count than it declares is a corrupt "
                "construction (e.g. an EP slice applied more or less than exactly once)"
            )
        if hidden.ndim != 2 or hidden.dtype is not torch.bfloat16 or hidden.shape[1] != self.hidden_size:
            raise TypeError(
                "GLM-5.2 full-param routed hidden states must be BF16 [rows, hidden_size]; "
                f"got {hidden.dtype} {tuple(hidden.shape)}"
            )
        if not hidden.is_contiguous() or hidden.shape[0] == 0:
            raise ValueError("GLM-5.2 full-param routed hidden states must be non-empty and contiguous")
        if routing.ndim != 2 or routing.dtype is not torch.float32 or routing.shape[0] != hidden.shape[0]:
            raise TypeError(
                f"GLM-5.2 full-param routing weights must be FP32 [rows, topk], got "
                f"{routing.dtype} {tuple(routing.shape)}"
            )
        if tuple(local_ids.shape) != tuple(routing.shape) or local_ids.dtype is not torch.int32:
            raise TypeError(
                "GLM-5.2 full-param local expert IDs must be int32 with the routing shape; "
                f"got {local_ids.dtype} {tuple(local_ids.shape)}"
            )
        if not routing.is_contiguous() or not local_ids.is_contiguous():
            raise ValueError("GLM-5.2 full-param routing weights and local IDs must use contiguous route-major layouts")
        if not bool(torch.all(torch.isfinite(routing))):
            raise ValueError("GLM-5.2 full-param routing weights contain non-finite values")
        if local_ids.numel() and (bool(torch.any(local_ids < -1)) or bool(torch.any(local_ids >= self.num_experts))):
            raise ValueError(
                f"GLM-5.2 full-param local IDs admit only the -1 sentinel or slots [0, {self.num_experts - 1}]"
            )
        self._assert_cache_fresh("forward")

    def _log_engagement_once(self) -> None:
        cls = type(self)
        if not cls._engagement_logged:
            cls._engagement_logged = True
            logger.info(
                "GLM-5.2 full-param routed experts engaged: contract=%s experts=%d hidden=%d "
                "intermediate=%d (cached-byte SGLang fused block-FP8 value path, "
                "straight-through FP32 master backward)",
                self.contract_version,
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
            )

    def _sampler_value(
        self,
        hidden: Tensor,
        routing: Tensor,
        local_ids: Tensor,
        *,
        routed_scaling_factor: float,
    ) -> Tensor:
        """Run the parent's literal fused block-FP8 expert sequence on the cache.

        This mirrors ``Glm52NativeBlockFP8Experts.sglang_ep_native_routed_partial``
        argument for argument; only the frozen-parameter admission is replaced
        by the full-param cache admission (performed in the engaged contract).
        """

        if hidden.device.type != "cuda":
            raise RuntimeError("GLM-5.2 full-param expert forward requires CUDA and pinned public SGLang")
        if (
            self.gate_up_packed_weight_f32.device != hidden.device
            or self.down_packed_weight_f32.device != hidden.device
        ):
            raise RuntimeError("GLM-5.2 full-param expert caches and activations must share one CUDA device")

        from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415
        from xorl.ops.moe.sglang_fused_moe_strided import fused_experts_impl_strided  # noqa: PLC0415

        MoEExperts._ensure_sglang_server_args()
        return fused_experts_impl_strided(
            hidden.contiguous(),
            self.gate_up_proj.transpose(1, 2),
            self.down_proj.transpose(1, 2),
            routing,
            local_ids,
            activation="silu",
            is_gated=True,
            use_fp8_w8a8=True,
            w1_scale=self.gate_up_weight_scale_inv.transpose(1, 2),
            w2_scale=self.down_weight_scale_inv.transpose(1, 2),
            block_shape=[128, 128],
            filter_expert=True,
            routed_scaling_factor=routed_scaling_factor,
        )

    # ``_dequantized_cached_experts`` and ``_surrogate_program`` are inherited
    # from :class:`Glm52NativeBlockFP8Experts` (moved to the parent so the
    # frozen bank's activation-only backward shares the identical checked
    # program; the trainable subclass behavior is byte-unchanged — its stored
    # bank size equals its declared ``num_experts``).

    def _straight_through_vjp(
        self,
        hidden: Tensor,
        routing: Tensor,
        local_ids: Tensor,
        *,
        grad_output: Tensor,
        routed_scaling_factor: float,
        needs_input_grad: tuple[bool, bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Trainer-owned straight-through backward to the FP32 GKN masters.

        The surrogate differentiates the dequantized cached bytes; ∂Q/∂W is
        treated as identity, and the [E, N, K] leaf gradients map onto the
        [E, K, N] masters by transposition (a byte permutation, no rounding).
        """

        need_hidden, need_routing, need_gate_up, need_down = needs_input_grad
        if not any(needs_input_grad):
            return None, None, None, None

        gate_up_deq, down_deq = self._dequantized_cached_experts()
        with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden_ref = hidden.float().detach().requires_grad_(need_hidden)
            routing_ref = routing.float().detach().requires_grad_(need_routing)
            gate_up_ref = gate_up_deq.float().detach().requires_grad_(need_gate_up)
            down_ref = down_deq.float().detach().requires_grad_(need_down)
            output = self._surrogate_program(
                hidden_ref,
                routing_ref,
                local_ids,
                gate_up_ref,
                down_ref,
                routed_scaling_factor=routed_scaling_factor,
            )
            requested = [
                reference
                for reference, needed in zip(
                    (hidden_ref, routing_ref, gate_up_ref, down_ref),
                    needs_input_grad,
                    strict=True,
                )
                if needed
            ]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_outputs=grad_output.float(),
                allow_unused=False,
            )

        iterator = iter(computed)
        grad_hidden = next(iterator) if need_hidden else None
        grad_routing = next(iterator) if need_routing else None
        grad_gate_up = next(iterator).transpose(1, 2).contiguous() if need_gate_up else None
        grad_down = next(iterator).transpose(1, 2).contiguous() if need_down else None
        return grad_hidden, grad_routing, grad_gate_up, grad_down

    def forward(
        self,
        hidden_states: Tensor,
        routing_weights: Tensor,
        selected_experts: Tensor | None = None,
        *,
        sglang_ep_native_local_ids: Tensor | None = None,
        routed_scaling_factor: float = 1.0,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            raise TypeError(f"Unexpected GLM-5.2 full-param routed expert arguments: {sorted(kwargs)}")
        if selected_experts is not None:
            raise RuntimeError("Global ids must not bypass GLM canonical native-EP mapping")
        if sglang_ep_native_local_ids is None:
            raise RuntimeError("GLM-5.2 full-param routed experts require canonical rank-local IDs")
        routed_scaling_factor = float(routed_scaling_factor)
        if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
            raise ValueError("GLM-5.2 routed scaling factor must be finite and positive")
        self._validate_engaged_contract(hidden_states, routing_weights, sglang_ep_native_local_ids)
        self._log_engagement_once()
        return _Glm52FullParamRoutedExpertsFunction.apply(
            hidden_states,
            routing_weights,
            sglang_ep_native_local_ids,
            self.gate_up_weight_master,
            self.down_weight_master,
            routed_scaling_factor,
            self,
        )


__all__ = [
    "GLM52_FULLPARAM_ROUTED_EXPERTS_CONTRACT_VERSION",
    "Glm52ExpertCheckpointPublication",
    "Glm52FullParamBlockFP8RoutedExperts",
]
