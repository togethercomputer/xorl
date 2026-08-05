"""Frozen native block-FP8 state and exact SGLang dense dispatch.

The packed parameter uses the same public storage convention as XoRL QLoRA:
four float8 bytes are viewed as one float32 element.  This makes the payload a
normal reshardable parameter while retaining every checkpoint byte.  It must
be wrapped without an FSDP mixed-precision policy; casting the packed float32
values numerically would corrupt the embedded bytes.
"""

from __future__ import annotations

import torch
from torch import nn


NATIVE_BLOCK_FP8_CONTRACT_VERSION = "xorl_native_block_fp8_sglang_v1"
_FP8_DTYPE = torch.float8_e4m3fn


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


class NativeBlockFP8Linear(nn.Module):
    """Frozen W8A8 block-FP8 linear whose base weight is never dequantized.

    SGLang is imported only on an engaged CUDA forward.  Construction, config
    inspection, meta initialization, DCP planning, and ordinary non-FP8 model
    imports therefore do not depend on SGLang.
    """

    fsdp_requires_full_precision = True
    contract_version = NATIVE_BLOCK_FP8_CONTRACT_VERSION

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
        if torch.is_grad_enabled() and input.requires_grad:
            raise RuntimeError(
                "Native block-FP8 phase-one forward is scoring-only; activation backward requires a validated kernel"
            )
        out_start, out_end = self._validate_partition(output_range, self.out_features, "output")
        in_start, in_end = self._validate_partition(input_range, self.in_features, "input")
        if input.shape[-1] != in_end - in_start:
            raise ValueError(
                f"Native block-FP8 input width {input.shape[-1]} does not match selected range {in_start}:{in_end}"
            )
        if input.device.type != "cuda":
            raise RuntimeError("Native block-FP8 forward requires CUDA and the pinned SGLang Triton kernel")
        if input.dtype is not torch.bfloat16:
            raise TypeError(f"Native block-FP8 forward requires BF16 activations, got {input.dtype}")
        if self.weight_scale_inv.dtype is not torch.float32:
            raise TypeError(f"Native block-FP8 scales must remain FP32, got {self.weight_scale_inv.dtype}")

        weight = self.fp8_weight()[out_start:out_end, in_start:in_end].contiguous()
        scale = self.weight_scale_inv[
            out_start // 128 : (out_end + 127) // 128,
            in_start // 128 : (in_end + 127) // 128,
        ].contiguous()
        if weight.device != input.device or self.weight_scale_inv.device != input.device:
            raise RuntimeError("Native block-FP8 weight, scales, and input must be on the same CUDA device")

        try:
            from sglang.srt.layers.quantization.fp8_utils import (  # noqa: PLC0415
                triton_w8a8_block_fp8_linear,
            )
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang block-FP8 kernel is required for native FP8") from exc

        output = triton_w8a8_block_fp8_linear(
            input,
            weight,
            list(self.block_size),
            scale,
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
    "NativeBlockFP8Linear",
    "pack_fp8_as_float32",
    "unpack_float32_as_fp8",
    "validate_native_fp8_dcp_checkpoint",
    "validate_native_fp8_state_metadata",
]
