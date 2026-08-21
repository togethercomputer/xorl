"""Frozen native block-FP8 state and exact SGLang dense dispatch.

The packed parameter uses the same public storage convention as XoRL QLoRA:
four float8 bytes are viewed as one float32 element.  This makes the payload a
normal reshardable parameter while retaining every checkpoint byte.  It must
be wrapped without an FSDP mixed-precision policy; casting the packed float32
values numerically would corrupt the embedded bytes.
"""

from __future__ import annotations

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)

NATIVE_BLOCK_FP8_CONTRACT_VERSION = "xorl_native_block_fp8_sglang_v1"
NATIVE_BLOCK_FP8_FROZEN_DGRAD_CONTRACT_VERSION = "xorl_native_block_fp8_frozen_dgrad_v1"
_FP8_DTYPE = torch.float8_e4m3fn


def _sglang_native_block_fp8_linear_value(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Run the shared exact-value SGLang W8A8 dispatch.

    Gradient policy belongs to the caller: :class:`NativeBlockFP8Linear`
    remains scoring-only, while the exact QLoRA wrapper supplies its validated
    surrogate through a custom autograd boundary.
    """

    if input.device.type != "cuda":
        raise RuntimeError("Native block-FP8 forward requires CUDA and the pinned SGLang Triton kernel")
    if input.dtype is not torch.bfloat16:
        raise TypeError(f"Native block-FP8 forward requires BF16 activations, got {input.dtype}")
    if weight.dtype is not _FP8_DTYPE:
        raise TypeError(f"Native block-FP8 weight must remain float8_e4m3fn, got {weight.dtype}")
    if weight_scale_inv.dtype is not torch.float32:
        raise TypeError(f"Native block-FP8 scales must remain FP32, got {weight_scale_inv.dtype}")
    if input.device != weight.device or input.device != weight_scale_inv.device:
        raise RuntimeError("Native block-FP8 weight, scales, and input must be on the same CUDA device")

    try:
        from sglang.srt.layers.quantization.fp8_utils import (  # noqa: PLC0415
            triton_w8a8_block_fp8_linear,
        )
    except Exception as exc:
        raise RuntimeError("Pinned public SGLang block-FP8 kernel is required for native FP8") from exc

    return triton_w8a8_block_fp8_linear(
        input,
        weight,
        list(block_size),
        weight_scale_inv,
    )


def pack_fp8_as_float32(weight: torch.Tensor) -> torch.Tensor:
    """Return a contiguous float32 view containing the exact float8 bytes."""

    if weight.dtype is not _FP8_DTYPE:
        raise TypeError(f"Expected float8_e4m3fn weight, got {weight.dtype}")
    if weight.numel() % 4:
        raise ValueError(f"FP8 weight has {weight.numel()} elements; byte packing requires a multiple of four")
    return weight.contiguous().view(torch.uint8).view(torch.float32)


def unpack_float32_as_fp8(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Recover a float8 view without numerical conversion."""

    if packed.dtype is not torch.float32:
        raise TypeError(
            f"Packed native-FP8 state has dtype {packed.dtype}, expected float32; "
            "an FSDP mixed-precision cast may have corrupted it"
        )
    expected = 1
    for dim in shape:
        expected *= dim
    if packed.numel() * 4 != expected:
        raise ValueError(
            f"Packed native-FP8 state has {packed.numel() * 4} bytes, expected {expected} for shape {shape}"
        )
    return packed.contiguous().view(torch.uint8).view(_FP8_DTYPE).reshape(shape)


def validate_native_fp8_state_metadata(
    module: nn.Module,
    metadata: dict[str, tuple[torch.dtype, tuple[int, ...]]],
    *,
    prefix: str = "",
) -> None:
    """Fail before DCP load if serialized dtype/shape metadata can cast bytes.

    DCP callers must build ``metadata`` from the checkpoint reader before
    invoking ``set_model_state_dict``.  State-dict hooks below cover ordinary
    ``load_state_dict``; this preflight covers loaders that copy shards without
    calling module hooks.
    """

    expected = {
        f"{prefix}{name}": (parameter.dtype, tuple(parameter.shape))
        for name, parameter in module.named_parameters()
        if "packed_weight_f32" in name or name.endswith("weight_scale_inv")
    }
    missing = sorted(set(expected) - set(metadata))
    mismatched = {
        name: (metadata[name], contract)
        for name, contract in expected.items()
        if name in metadata and metadata[name] != contract
    }
    if missing or mismatched:
        raise ValueError(f"Native FP8 DCP metadata mismatch: missing={missing[:8]} mismatched={mismatched}")


def validate_native_fp8_dcp_checkpoint(
    checkpoint_path: str,
    expected_state: dict[str, torch.Tensor],
    *,
    state_prefix: str = "model.",
) -> None:
    """Read DCP metadata and reject any castable native-FP8 payload pre-load."""

    from torch.distributed.checkpoint import FileSystemReader  # noqa: PLC0415

    state_metadata = FileSystemReader(checkpoint_path).read_metadata().state_dict_metadata
    tensor_metadata = {}
    for name, metadata in state_metadata.items():
        properties = getattr(metadata, "properties", None)
        size = getattr(metadata, "size", None)
        if properties is not None and size is not None:
            tensor_metadata[name] = (properties.dtype, tuple(size))
    expected_metadata = {
        f"{state_prefix}{name}": (tensor.dtype, tuple(tensor.shape))
        for name, tensor in expected_state.items()
        if "packed_weight_f32" in name or name.endswith("weight_scale_inv")
    }
    missing = sorted(set(expected_metadata) - set(tensor_metadata))
    mismatched = {
        name: (tensor_metadata[name], contract)
        for name, contract in expected_metadata.items()
        if name in tensor_metadata and tensor_metadata[name] != contract
    }
    if missing or mismatched:
        raise ValueError(f"Native FP8 DCP metadata mismatch: missing={missing[:8]} mismatched={mismatched}")


def _validate_state_dict_contract(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    expected: dict[str, tuple[torch.dtype, tuple[int, ...]]],
) -> None:
    for name, (dtype, shape) in expected.items():
        key = f"{prefix}{name}"
        tensor = state_dict.get(key)
        if tensor is None:
            continue
        if tensor.dtype is not dtype or tuple(tensor.shape) != shape:
            raise TypeError(
                f"Native FP8 state {key} must be {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}; "
                "refusing a load_state_dict cast"
            )


class _NativeBlockFP8FrozenDgradFunction(torch.autograd.Function):
    """Exact-value forward with the frozen-trunk activation backward.

    Forward: the UNCHANGED SGLang W8A8 dispatch on the module's frozen bytes
    (byte-identical to the scoring-only path — the value program is not
    touched).  Backward: dgrad only, ``grad_output @ dequant(cache)`` in the
    declared BF16 linear program — the same base-branch treatment the
    trainable full-param composites and the exact QLoRA surrogate apply to
    quantized bytes.  There is deliberately NO wgrad and NO master/cache
    mutation: the trunk is frozen, so the only gradient this boundary may
    produce is the activation gradient that lets upstream trainable
    parameters learn.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, module, out_start: int, out_end: int, in_start: int, in_end: int):
        weight = module.fp8_weight()[out_start:out_end, in_start:in_end].contiguous()
        scale = module.weight_scale_inv[
            out_start // 128 : (out_end + 127) // 128,
            in_start // 128 : (in_end + 127) // 128,
        ].contiguous()
        output = _sglang_native_block_fp8_linear_value(
            input,
            weight,
            scale,
            block_size=module.block_size,
        )
        ctx.module = module
        ctx.ranges = (out_start, out_end, in_start, in_end)
        # dgrad needs only the frozen weight slice (recovered in backward from
        # the frozen module); the input is deliberately NOT saved — there is
        # no wgrad at a frozen boundary.
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        out_start, out_end, in_start, in_end = ctx.ranges
        grad_input = ctx.module._frozen_activation_dgrad(
            grad_output,
            output_range=(out_start, out_end),
            input_range=(in_start, in_end),
        )
        return grad_input, None, None, None, None, None


class NativeBlockFP8Linear(nn.Module):
    """Frozen W8A8 block-FP8 linear whose base weight is never dequantized.

    SGLang is imported only on an engaged CUDA forward.  Construction, config
    inspection, meta initialization, DCP planning, and ordinary non-FP8 model
    imports therefore do not depend on SGLang.
    """

    fsdp_requires_full_precision = True
    contract_version = NATIVE_BLOCK_FP8_CONTRACT_VERSION
    frozen_dgrad_contract_version = NATIVE_BLOCK_FP8_FROZEN_DGRAD_CONTRACT_VERSION
    # Fail-closed default: the scoring-only lanes never admit an activation
    # backward.  A training admission that OWNS the trainable-set semantics
    # (the GLM-5.2 full-param admission) must opt in per module via
    # :meth:`enable_frozen_activation_dgrad`.
    _frozen_dgrad_admitted = False
    _frozen_dgrad_engagement_logged = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: tuple[int, int] = (128, 128),
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Native block-FP8 dimensions must be positive")
        if in_features % 4:
            raise ValueError("Native block-FP8 in_features must be divisible by four for byte packing")
        if tuple(block_size) != (128, 128):
            raise ValueError(f"Only the official GLM block shape (128, 128) is supported, got {block_size}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block_size = tuple(block_size)
        self.packed_weight_f32 = nn.Parameter(
            torch.empty((out_features, in_features // 4), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                ((out_features + 127) // 128, (in_features + 127) // 128),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )

    @classmethod
    def from_linear(cls, module: nn.Linear) -> "NativeBlockFP8Linear":
        if module.bias is not None:
            raise ValueError("Native block-FP8 linear does not support bias")
        return cls(
            module.in_features,
            module.out_features,
            device=module.weight.device,
        )

    def _apply(self, fn, recurse: bool = True):
        # Preserve packed bytes and scale values across model.to(dtype=...).
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.packed_weight_f32.device))
        protected = {name: self._parameters.pop(name) for name in ("packed_weight_f32", "weight_scale_inv")}
        try:
            result = super()._apply(fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                if parameter.is_meta:
                    value = torch.empty_like(parameter, dtype=torch.float32, device=probe.device)
                else:
                    value = parameter.to(device=probe.device, dtype=torch.float32)
                replacements[name] = nn.Parameter(value, requires_grad=False)
        except Exception:
            self._parameters.update(protected)
            raise
        self._parameters.update(replacements)
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
        _validate_state_dict_contract(
            state_dict,
            prefix,
            {
                "packed_weight_f32": (torch.float32, tuple(self.packed_weight_f32.shape)),
                "weight_scale_inv": (torch.float32, tuple(self.weight_scale_inv.shape)),
            },
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

    def load_prequantized(self, weight: torch.Tensor, weight_scale_inv: torch.Tensor) -> None:
        """Copy one official weight/scale pair with strict byte and shape checks."""

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
        packed = pack_fp8_as_float32(weight)
        with torch.no_grad():
            self.packed_weight_f32.copy_(packed.to(self.packed_weight_f32.device))
            self.weight_scale_inv.copy_(weight_scale_inv.to(self.weight_scale_inv.device))

    def fp8_weight(self) -> torch.Tensor:
        return unpack_float32_as_fp8(
            self.packed_weight_f32,
            (self.out_features, self.in_features),
        )

    def enable_frozen_activation_dgrad(self) -> None:
        """Admit the validated frozen-trunk activation backward.

        Idempotent per module.  The forward VALUE program is byte-unchanged;
        only the refusal on grad-requiring inputs is replaced by the checked
        BF16 dequant-program dgrad (no wgrad, no master/cache mutation —
        frozen means frozen).  Only a trainable-set admission that must
        backpropagate THROUGH this frozen module may call this; scoring-only
        lanes keep the fail-closed refusal.
        """

        self._frozen_dgrad_admitted = True
        cls = NativeBlockFP8Linear
        if not cls._frozen_dgrad_engagement_logged:
            cls._frozen_dgrad_engagement_logged = True
            logger.info(
                "Native block-FP8 frozen-trunk activation dgrad engaged: contract=%s "
                "(forward bytes unchanged; dgrad = grad_output @ dequant(cache) in the "
                "declared BF16 program; wgrad none; frozen bytes immutable)",
                self.frozen_dgrad_contract_version,
            )

    def _frozen_activation_dgrad(
        self,
        grad_output: torch.Tensor,
        *,
        output_range: tuple[int, int],
        input_range: tuple[int, int],
    ) -> torch.Tensor:
        """dgrad through the dequantized frozen bytes in the BF16 program.

        Dequantization is the same program point the trainable composites'
        surrogate uses (``block_fp8_dequantize_gkn`` -> FP32 -> one explicit
        BF16 rounding); the GEMM is the declared BF16 linear backward
        ``grad_output @ W``.  Gated bitwise against the reference
        dequant-matmul autograd (tests/ops/test_block_fp8_frozen_dgrad.py).
        """

        from xorl.ops.quantize import block_fp8_dequantize_gkn  # noqa: PLC0415

        out_start, out_end = output_range
        in_start, in_end = input_range
        weight = self.fp8_weight()[out_start:out_end, in_start:in_end].contiguous()
        scale = self.weight_scale_inv[
            out_start // 128 : (out_end + 127) // 128,
            in_start // 128 : (in_end + 127) // 128,
        ].contiguous()
        dequantized = block_fp8_dequantize_gkn(weight, scale, 128).to(torch.float32).to(torch.bfloat16)
        grad_2d = grad_output.reshape(-1, out_end - out_start).to(torch.bfloat16)
        return grad_2d.matmul(dequantized).reshape(*grad_output.shape[:-1], in_end - in_start)

    @staticmethod
    def _validate_partition(value: tuple[int, int] | None, size: int, name: str) -> tuple[int, int]:
        if value is None:
            return 0, size
        start, end = value
        if start < 0 or end > size or start >= end:
            raise ValueError(f"Invalid native block-FP8 {name} range {value} for size {size}")
        if start % 128 or (end != size and end % 128):
            raise ValueError(f"Native block-FP8 {name} range {value} must follow 128-element block boundaries")
        return start, end

    def forward_partition(
        self,
        input: torch.Tensor,
        *,
        output_range: tuple[int, int] | None = None,
        input_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Run an aligned output-row/input-column partition through SGLang.

        This supports callers that preserve a model-specific reduction policy
        by assigning disjoint, block-aligned projection slices to ranks.
        """

        if self.packed_weight_f32.requires_grad or self.weight_scale_inv.requires_grad:
            raise RuntimeError("Native block-FP8 base weights and scales must remain frozen")
        grad_engaged = torch.is_grad_enabled() and input.requires_grad
        if grad_engaged and not self._frozen_dgrad_admitted:
            raise RuntimeError(
                "Native block-FP8 phase-one forward is scoring-only; activation backward requires a validated kernel"
            )
        out_start, out_end = self._validate_partition(output_range, self.out_features, "output")
        in_start, in_end = self._validate_partition(input_range, self.in_features, "input")
        if input.shape[-1] != in_end - in_start:
            raise ValueError(
                f"Native block-FP8 input width {input.shape[-1]} does not match selected range {in_start}:{in_end}"
            )
        if grad_engaged:
            # Same slicing + same kernel inside the autograd boundary; the
            # value bytes are identical to the scoring-only path below.
            return _NativeBlockFP8FrozenDgradFunction.apply(input, self, out_start, out_end, in_start, in_end)
        weight = self.fp8_weight()[out_start:out_end, in_start:in_end].contiguous()
        scale = self.weight_scale_inv[
            out_start // 128 : (out_end + 127) // 128,
            in_start // 128 : (in_end + 127) // 128,
        ].contiguous()
        output = _sglang_native_block_fp8_linear_value(
            input,
            weight,
            scale,
            block_size=self.block_size,
        )
        return output

    def forward(
        self,
        input: torch.Tensor | None = None,
        *,
        return_dequantized_weight: bool = False,
        output_range: tuple[int, int] | None = None,
        input_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if return_dequantized_weight:
            if input is not None or output_range is not None or input_range is not None:
                raise ValueError("Native block-FP8 weight materialization does not accept activation or range inputs")
            if self.packed_weight_f32.device.type != "cuda":
                raise RuntimeError("Native block-FP8 weight materialization requires CUDA")
            from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant  # noqa: PLC0415

            return block_quant_dequant(
                self.fp8_weight(),
                self.weight_scale_inv,
                list(self.block_size),
                torch.bfloat16,
            )
        if input is None:
            raise ValueError("Native block-FP8 linear forward requires an activation input")
        return self.forward_partition(input, output_range=output_range, input_range=input_range)


__all__ = [
    "NATIVE_BLOCK_FP8_CONTRACT_VERSION",
    "NATIVE_BLOCK_FP8_FROZEN_DGRAD_CONTRACT_VERSION",
    "NativeBlockFP8Linear",
    "pack_fp8_as_float32",
    "unpack_float32_as_fp8",
    "validate_native_fp8_dcp_checkpoint",
    "validate_native_fp8_state_metadata",
]
