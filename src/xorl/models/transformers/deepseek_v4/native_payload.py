"""Immutable native DeepSeek-V4-Flash checkpoint payload ownership.

The ordinary DSV4 loader materializes BF16 weights.  Exact replay additionally
needs the original E4M3/E8M0 dense pairs and the packed E2M1/E8M0 routed
experts.  These holders make those bytes first-class model state so FSDP/DCP
cannot silently cast them while the serving-value forward is brought up one
physical family at a time.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch
from torch import nn


DSV4_NATIVE_PAYLOAD_CONTRACT_VERSION = "dsv4_flash_native_payload_v1"
_DIAGNOSTIC_SINGLE_ADAPTER_BATCH_INFOS = {}


def pack_bytes_as_float32(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack exact tensor bytes into an FSDP-safe float32 parameter payload."""

    raw = tensor.contiguous().view(torch.uint8).flatten()
    logical_bytes = raw.numel()
    padded_bytes = math.ceil(logical_bytes / 4) * 4
    if padded_bytes != logical_bytes:
        raw = torch.nn.functional.pad(raw, (0, padded_bytes - logical_bytes))
    return raw.view(torch.float32), logical_bytes


def unpack_float32_as_bytes(
    packed: torch.Tensor,
    *,
    logical_bytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Recover a typed tensor view without numerical conversion."""

    if packed.dtype is not torch.float32:
        raise TypeError(f"Packed DSV4 payload must remain float32, got {packed.dtype}")
    expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if logical_bytes != expected_bytes:
        raise ValueError(
            f"DSV4 payload metadata has {logical_bytes} bytes but {dtype} {shape} requires {expected_bytes}"
        )
    raw = packed.contiguous().view(torch.uint8)[:logical_bytes]
    return raw.view(dtype).reshape(shape)


def pack_expert_rows_as_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Pack one-byte expert rows into FSDP-safe FP32 without mixing experts."""

    if tensor.ndim < 1 or tensor.element_size() != 1:
        raise ValueError("DSV4 expert payload packing requires a non-scalar one-byte tensor")
    raw = tensor.contiguous().view(torch.uint8)
    padded_row_bytes = math.ceil(raw.shape[-1] / 4) * 4
    if padded_row_bytes != raw.shape[-1]:
        raw = torch.nn.functional.pad(raw, (0, padded_row_bytes - raw.shape[-1]))
    return raw.view(torch.float32)


def unpack_expert_rows_from_float32(
    packed: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Recover one-byte expert rows after EP/FSDP transport."""

    if packed.dtype is not torch.float32:
        raise TypeError(f"Packed DSV4 expert payload must remain float32, got {packed.dtype}")
    if torch.empty((), dtype=dtype).element_size() != 1:
        raise ValueError(f"DSV4 expert payload dtype must occupy one byte, got {dtype}")
    if tuple(packed.shape[:-1]) != shape[:-1]:
        raise ValueError(f"DSV4 expert payload prefix shape mismatch: {tuple(packed.shape[:-1])} != {shape[:-1]}")
    raw = packed.contiguous().view(torch.uint8)[..., : shape[-1]]
    return raw.contiguous().view(dtype).reshape(shape)


class Dsv4NativeBlockFp8Payload(nn.Module):
    """One frozen E4M3 weight plus its native E8M0 128x128 scale grid."""

    contract_version = DSV4_NATIVE_PAYLOAD_CONTRACT_VERSION
    fsdp_requires_full_precision = True

    def __init__(self, out_features: int, in_features: int, *, device=None) -> None:
        super().__init__()
        if out_features <= 0 or in_features <= 0:
            raise ValueError("DSV4 native FP8 dimensions must be positive")
        weight_bytes = out_features * in_features
        scale_shape = (math.ceil(out_features / 128), math.ceil(in_features / 128))
        scale_bytes = math.prod(scale_shape)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.scale_shape = scale_shape
        self.weight_logical_bytes = weight_bytes
        self.scale_logical_bytes = scale_bytes
        self.packed_weight_f32 = nn.Parameter(
            torch.empty(math.ceil(weight_bytes / 4), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.packed_scale_f32 = nn.Parameter(
            torch.empty(math.ceil(scale_bytes / 4), dtype=torch.float32, device=device),
            requires_grad=False,
        )

    def _apply(self, fn, recurse: bool = True):
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.packed_weight_f32.device))
        protected = {name: self._parameters.pop(name) for name in ("packed_weight_f32", "packed_scale_f32")}
        try:
            result = super()._apply(fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                value = (
                    torch.empty_like(parameter, device=probe.device)
                    if parameter.is_meta
                    else parameter.to(device=probe.device, dtype=torch.float32)
                )
                replacements[name] = nn.Parameter(value, requires_grad=False)
        except Exception:
            self._parameters.update(protected)
            raise
        self._parameters.update(replacements)
        return result

    def fp8_weight(self) -> torch.Tensor:
        return unpack_float32_as_bytes(
            self.packed_weight_f32,
            logical_bytes=self.weight_logical_bytes,
            dtype=torch.float8_e4m3fn,
            shape=(self.out_features, self.in_features),
        )

    def e8m0_scale(self) -> torch.Tensor:
        return unpack_float32_as_bytes(
            self.packed_scale_f32,
            logical_bytes=self.scale_logical_bytes,
            dtype=torch.float8_e8m0fnu,
            shape=self.scale_shape,
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize owned bytes while this payload's FSDP unit is gathered.

        Exact-mode dense payload holders are wrapped as independent FSDP units
        so their packed FP32 parameters bypass the decoder's BF16 policy.  A
        direct call to ``fp8_weight()`` does not run the FSDP pre-forward hook:
        it can therefore hand a Triton kernel a DTensor whose logical shape is
        global but whose local storage is only one shard.  Calling the module
        gathers the bytes first.  Clone the typed views before the post-forward
        reshard invalidates the all-gather buffer.
        """

        return self.fp8_weight().clone(), self.e8m0_scale().clone()

    def load_prequantized(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        if weight.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"DSV4 dense payload weight must be E4M3, got {weight.dtype}")
        if scale.dtype is not torch.float8_e8m0fnu:
            raise TypeError(f"DSV4 dense payload scale must be E8M0, got {scale.dtype}")
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"DSV4 dense payload weight shape mismatch: {tuple(weight.shape)}")
        if tuple(scale.shape) != self.scale_shape:
            raise ValueError(f"DSV4 dense payload scale shape mismatch: {tuple(scale.shape)}")
        packed_weight, _ = pack_bytes_as_float32(weight)
        packed_scale, _ = pack_bytes_as_float32(scale)
        with torch.no_grad():
            self.packed_weight_f32.copy_(packed_weight.to(self.packed_weight_f32.device))
            self.packed_scale_f32.copy_(packed_scale.to(self.packed_scale_f32.device))


def _payload_values_for_backward(payload: Dsv4NativeBlockFp8Payload) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize payload bytes inside a backward pass.

    Module __call__ gathers the FSDP-sharded packed parameters via the
    pre-forward hook, but firing that hook during autograd (surrogate VJPs)
    trips FSDP root state. Explicitly unshard/reshard instead and read the
    typed views directly.
    """

    unshard = getattr(payload, "unshard", None)
    reshard = getattr(payload, "reshard", None)
    if callable(unshard):
        unshard()
    try:
        return payload.fp8_weight().clone(), payload.e8m0_scale().clone()
    finally:
        if callable(reshard):
            reshard()


def _dequantize_native_block_fp8_for_backward(payload: Dsv4NativeBlockFp8Payload) -> torch.Tensor:
    weight, scales = _payload_values_for_backward(payload)
    weight = weight.float()
    scales = scales.float()
    expanded = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (weight * expanded[: payload.out_features, : payload.in_features]).to(torch.bfloat16)


def _dequantize_native_block_fp8(payload: Dsv4NativeBlockFp8Payload) -> torch.Tensor:
    # Call forward() directly: the payload holds only whole (non-sharded)
    # packed buffers, and Module.__call__ would fire FSDP pre-forward hooks —
    # fatal when this runs inside a backward (surrogate VJPs).
    weight, scales = payload()
    weight = weight.float()
    scales = scales.float()
    expanded = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (weight * expanded[: payload.out_features, : payload.in_features]).to(torch.bfloat16)


def _native_block_fp8_value(input: torch.Tensor, payload: Dsv4NativeBlockFp8Payload) -> torch.Tensor:
    from xorl.ops.exact.block_fp8_native import _sglang_native_block_fp8_linear_value  # noqa: PLC0415

    rows = input.numel() // payload.in_features
    weight, scales = payload()
    output = _sglang_native_block_fp8_linear_value(
        input.reshape(rows, payload.in_features).contiguous(),
        weight.contiguous(),
        scales.float().contiguous(),
    )
    return output.reshape(*input.shape[:-1], payload.out_features)


class _Dsv4NativeBlockFp8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, module) -> torch.Tensor:
        ctx.module = module
        return _native_block_fp8_value(input, module.native_base_payload)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None
        payload = ctx.module.native_base_payload
        weight = _dequantize_native_block_fp8_for_backward(payload)
        rows = grad_output.numel() // payload.out_features
        grad_input = grad_output.reshape(rows, payload.out_features).float() @ weight.float()
        return grad_input.to(grad_output.dtype).reshape(*grad_output.shape[:-1], payload.in_features), None


class Dsv4NativeBlockFp8Linear(nn.Linear):
    """An ``nn.Linear``-compatible leaf whose only base image is native FP8."""

    fsdp_requires_full_precision = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            raise RuntimeError("Exact DSV4 native FP8 linears are bias-free")
        return _Dsv4NativeBlockFp8Function.apply(input, self)


@lru_cache(maxsize=64)
def _single_adapter_batch_info(device_index: int, rows: int):
    from sglang.srt.lora.utils import LoRABatchInfo  # noqa: PLC0415

    device = torch.device("cuda", device_index)
    batch_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.ones(1, dtype=torch.int32, device=device),
        scalings=torch.ones(1, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )
    if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
        # Keep every diagnostic entry visible even when the LRU evicts it.
        # Attention operates on rank-local rows while routed experts operate
        # on EP-gathered rows, so checking only the current shape can miss an
        # overwrite of the smaller metadata allocation.
        _DIAGNOSTIC_SINGLE_ADAPTER_BATCH_INFOS[(device_index, rows)] = batch_info
    return batch_info


def _validate_single_adapter_batch_info(batch_info, rows: int, *, where: str) -> None:
    """Fail at the first corrupted cached LoRA metadata boundary.

    The exact DSV4 lane reuses the same one-segment device metadata at every
    adapted projection. A stray write from an earlier kernel otherwise becomes
    a misleading illegal access inside the next LoRA GEMM. This check runs only
    with the synchronized base-Marlin diagnostic.
    """

    observed = {
        "seg_indptr": batch_info.seg_indptr.detach().cpu().tolist(),
        "weight_indices": batch_info.weight_indices.detach().cpu().tolist(),
        "lora_ranks": batch_info.lora_ranks.detach().cpu().tolist(),
        "seg_lens": batch_info.seg_lens.detach().cpu().tolist(),
    }
    expected = {
        "seg_indptr": [0, rows],
        "weight_indices": [0],
        "lora_ranks": [1],
        "seg_lens": [rows],
    }
    if observed != expected:
        raise RuntimeError(
            f"Exact DSV4 cached LoRA metadata was corrupted before {where}: expected={expected} observed={observed}"
        )


def _validate_all_single_adapter_batch_infos(device_index: int, *, where: str) -> None:
    """Validate every diagnostic LoRA metadata allocation on one device."""

    for (cached_device_index, rows), batch_info in tuple(_DIAGNOSTIC_SINGLE_ADAPTER_BATCH_INFOS.items()):
        if cached_device_index == device_index:
            _validate_single_adapter_batch_info(
                batch_info,
                rows,
                where=f"{where} (cached_rows={rows})",
            )


class _Dsv4NativeBlockFp8LoraFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_A, lora_B, module):
        if (module.active_r, module.active_lora_alpha, module._active_scaling()) != (
            1,
            1,
            1.0,
        ):
            raise RuntimeError("Exact DSV4 native FP8 LoRA requires rank=1, alpha=1, scaling=1")
        effective_A = lora_A.to(torch.bfloat16).contiguous()
        effective_B = lora_B.to(torch.bfloat16).contiguous()
        base = _native_block_fp8_value(input, module.native_base_payload)
        try:
            from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
            from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned SGLang LoRA kernels are required for exact DSV4") from exc
        rows = input.numel() // module.in_features
        input_2d = input.reshape(rows, module.in_features).contiguous()
        batch_info = _single_adapter_batch_info(input.device.index, rows)
        low_rank = sgemm_lora_a_fwd(input_2d, effective_A.unsqueeze(0), batch_info)
        output = sgemm_lora_b_fwd(
            low_rank,
            effective_B.unsqueeze(0),
            batch_info,
            base_output=base.reshape(rows, module.out_features),
        )
        ctx.module = module
        ctx.save_for_backward(input.detach(), effective_A, effective_B, lora_A, lora_B)
        return output.reshape(*input.shape[:-1], module.out_features)

    @staticmethod
    def backward(ctx, grad_output):
        input, effective_A, effective_B, _master_A, _master_B = ctx.saved_tensors
        module = ctx.module
        x = input.reshape(-1, module.in_features).float()
        dy = grad_output.reshape(-1, module.out_features).float()
        a = effective_A.float()
        b = effective_B.float()
        low_rank = x @ a.transpose(0, 1)
        grad_B = dy.transpose(0, 1) @ low_rank
        grad_low_rank = dy @ b
        grad_A = grad_low_rank.transpose(0, 1) @ x
        grad_input = grad_low_rank @ a
        if ctx.needs_input_grad[0]:
            base_weight = _dequantize_native_block_fp8_for_backward(module.native_base_payload)
            grad_input = grad_input + dy @ base_weight.float()
            grad_input = grad_input.to(input.dtype).reshape_as(input)
        else:
            grad_input = None
        return grad_input, grad_A, grad_B, None


def dsv4_native_block_fp8_lora(input: torch.Tensor, module: nn.Module) -> torch.Tensor:
    return _Dsv4NativeBlockFp8LoraFunction.apply(
        input,
        module.lora_A,
        module.lora_B,
        module,
    )


def _native_grouped_wo_a_value(
    input: torch.Tensor,
    payload: Dsv4NativeBlockFp8Payload,
    *,
    groups: int,
    out_per_group: int,
) -> torch.Tensor:
    """Run the frozen sampler's literal dequantized grouped ``wo_a`` program."""

    if input.device.type != "cuda" or input.dtype is not torch.bfloat16:
        raise TypeError("Exact DSV4 grouped wo_a requires CUDA BF16 activations")
    if input.ndim != 3 or input.shape[1] != groups:
        raise ValueError(f"Grouped wo_a requires [tokens, {groups}, width], got {tuple(input.shape)}")
    _, _, width = input.shape
    if payload.in_features != width or payload.out_features != groups * out_per_group:
        raise ValueError("Grouped wo_a payload geometry does not match the logical projection")

    weight, scales = payload()
    # The qualified sampler sets SGLANG_OPT_FP8_WO_A_GEMM=0.  Its loader
    # dequantizes each checkpoint 128x128 block in FP32, casts the resulting
    # matrix to BF16, then executes this exact grouped einsum.  Preserve the
    # checkpoint-native FP8/E8M0 bytes in the payload and materialize only this
    # serving-value compute view while the FSDP payload is gathered.
    scale_rows, scale_cols = payload.scale_shape
    weight_bf16 = (
        (weight.float().view(scale_rows, 128, scale_cols, 128) * scales.float().view(scale_rows, 1, scale_cols, 1))
        .reshape(payload.out_features, payload.in_features)
        .to(torch.bfloat16)
    )
    return torch.einsum(
        "tgd,grd->tgr",
        input,
        weight_bf16.view(groups, out_per_group, width),
    )


class _Dsv4NativeGroupedWoALoraFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_A, lora_B, module, groups, out_per_group):
        if (module.active_r, module.active_lora_alpha, module._active_scaling()) != (
            1,
            1,
            1.0,
        ):
            raise RuntimeError("Exact DSV4 grouped wo_a requires rank=1 alpha=1")
        tokens = input.shape[0]
        base = _native_grouped_wo_a_value(
            input,
            module.native_base_payload,
            groups=groups,
            out_per_group=out_per_group,
        )
        effective_A = lora_A.to(torch.bfloat16).contiguous()
        effective_B = lora_B.to(torch.bfloat16).view(groups, out_per_group, 1)
        try:
            from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd  # noqa: PLC0415
            from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned SGLang LoRA kernels are required for grouped wo_a") from exc
        batch_info = _single_adapter_batch_info(input.device.index, tokens)
        if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
            _validate_single_adapter_batch_info(
                batch_info,
                tokens,
                where=getattr(module.native_base_payload, "logical_name", "grouped_wo_a"),
            )
        outputs = []
        for group in range(groups):
            low_rank = sgemm_lora_a_fwd(
                input[:, group, :].contiguous(),
                effective_A.unsqueeze(0),
                batch_info,
            )
            # SGLang's serving kernel performs ``BF16(base + BF16(delta))``.
            # Keep the same arithmetic while avoiding a strided in-place base
            # view: the low-level LoRA-B kernel's output ABI is token-major.
            delta = sgemm_lora_b_fwd(
                low_rank,
                effective_B[group].unsqueeze(0),
                batch_info,
            )
            outputs.append(base[:, group, :] + delta)
        ctx.module = module
        ctx.groups = groups
        ctx.out_per_group = out_per_group
        ctx.save_for_backward(input.detach(), effective_A, effective_B)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        input, effective_A, effective_B = ctx.saved_tensors
        module = ctx.module
        groups = ctx.groups
        out_per_group = ctx.out_per_group
        x = input.float()
        dy = grad_output.float()
        a = effective_A.float()
        b = effective_B.float()
        low_rank = x @ a.transpose(0, 1)
        grad_B_grouped = torch.einsum("tgo,tgr->gor", dy, low_rank)
        grad_low_rank = torch.einsum("tgo,gor->tgr", dy, b)
        grad_A = torch.einsum("tgr,tgd->rd", grad_low_rank, x)
        grad_input = grad_low_rank @ a
        if ctx.needs_input_grad[0]:
            base_weight = _dequantize_native_block_fp8_for_backward(module.native_base_payload).view(
                groups, out_per_group, module.in_features
            )
            grad_input = grad_input + torch.einsum("tgo,god->tgd", dy, base_weight.float())
            grad_input = grad_input.to(input.dtype)
        else:
            grad_input = None
        return (
            grad_input,
            grad_A,
            grad_B_grouped.reshape(groups * out_per_group, 1),
            None,
            None,
            None,
        )


def dsv4_native_grouped_wo_a(
    input: torch.Tensor,
    module: nn.Module,
    *,
    groups: int,
    out_per_group: int,
) -> torch.Tensor:
    """Serving-value grouped base plus active rank-one LoRA."""

    if hasattr(module, "lora_A"):
        return _Dsv4NativeGroupedWoALoraFunction.apply(
            input,
            module.lora_A,
            module.lora_B,
            module,
            groups,
            out_per_group,
        )
    return _native_grouped_wo_a_value(
        input,
        module.native_base_payload,
        groups=groups,
        out_per_group=out_per_group,
    )


class Dsv4NativeMxfp4ExpertPayload(nn.Module):
    """One EP-shardable SGLang-layout routed-expert MXFP4 payload bank."""

    contract_version = DSV4_NATIVE_PAYLOAD_CONTRACT_VERSION
    fsdp_requires_full_precision = True

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, *, device=None) -> None:
        super().__init__()
        if hidden_size % 32 or intermediate_size % 32:
            raise ValueError("DSV4 MXFP4 dimensions must be divisible by the 32-value scale group")
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.w13_shape = (num_experts, 2 * intermediate_size, hidden_size // 2)
        self.w2_shape = (num_experts, hidden_size, intermediate_size // 2)
        self.w13_scale_shape = (num_experts, 2 * intermediate_size, hidden_size // 32)
        self.w2_scale_shape = (num_experts, hidden_size, intermediate_size // 32)
        self.packed_w13_weight_f32 = nn.Parameter(
            torch.empty(
                *self.w13_shape[:-1],
                math.ceil(self.w13_shape[-1] / 4),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.packed_w2_weight_f32 = nn.Parameter(
            torch.empty(
                *self.w2_shape[:-1],
                math.ceil(self.w2_shape[-1] / 4),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.packed_w13_scale_f32 = nn.Parameter(
            torch.empty(
                *self.w13_scale_shape[:-1],
                math.ceil(self.w13_scale_shape[-1] / 4),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.packed_w2_scale_f32 = nn.Parameter(
            torch.empty(
                *self.w2_scale_shape[:-1],
                math.ceil(self.w2_scale_shape[-1] / 4),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )

    def _apply(self, fn, recurse: bool = True):
        # All native bytes are carried in FP32 containers so FSDP can shard
        # them. Protect those containers from model-wide numerical casts.
        first = self.packed_w13_weight_f32
        probe = fn(torch.empty(0, dtype=torch.float32, device=first.device))
        names = (
            "packed_w13_weight_f32",
            "packed_w2_weight_f32",
            "packed_w13_scale_f32",
            "packed_w2_scale_f32",
        )
        protected = {name: self._parameters.pop(name) for name in names}
        try:
            result = super()._apply(fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                value = (
                    torch.empty_like(parameter, device=probe.device)
                    if parameter.is_meta
                    else parameter.to(device=probe.device, dtype=torch.float32)
                )
                replacements[name] = nn.Parameter(value, requires_grad=False)
        except Exception:
            self._parameters.update(protected)
            raise
        self._parameters.update(replacements)
        return result

    @property
    def w13_weight(self) -> torch.Tensor:
        return unpack_expert_rows_from_float32(
            self.packed_w13_weight_f32,
            dtype=torch.int8,
            shape=(*self.packed_w13_weight_f32.shape[:-2], *self.w13_shape[-2:]),
        )

    @property
    def w2_weight(self) -> torch.Tensor:
        return unpack_expert_rows_from_float32(
            self.packed_w2_weight_f32,
            dtype=torch.int8,
            shape=(*self.packed_w2_weight_f32.shape[:-2], *self.w2_shape[-2:]),
        )

    @property
    def w13_weight_scale_inv(self) -> torch.Tensor:
        return unpack_expert_rows_from_float32(
            self.packed_w13_scale_f32,
            dtype=torch.float8_e8m0fnu,
            shape=(*self.packed_w13_scale_f32.shape[:-2], *self.w13_scale_shape[-2:]),
        )

    @property
    def w2_weight_scale_inv(self) -> torch.Tensor:
        return unpack_expert_rows_from_float32(
            self.packed_w2_scale_f32,
            dtype=torch.float8_e8m0fnu,
            shape=(*self.packed_w2_scale_f32.shape[:-2], *self.w2_scale_shape[-2:]),
        )


def _prepare_mxfp4_marlin_bank(
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    size_n: int,
    size_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint-native MXFP4 bytes to SGLang's Marlin layout."""

    from sglang.kernels.ops.quantization.gptq_marlin_repack import (  # noqa: PLC0415
        gptq_marlin_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import (  # noqa: PLC0415
        marlin_permute_scales,
    )
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (  # noqa: PLC0415
        mxfp4_marlin_process_scales,
    )

    permutation = torch.empty(0, dtype=torch.int32, device=weight.device)
    numerical_scales = scales.to(torch.bfloat16)
    repacked = []
    processed_scales = []
    for expert_idx in range(weight.shape[0]):
        repacked.append(
            gptq_marlin_repack(
                b_q_weight=weight[expert_idx].view(torch.int32).T.contiguous(),
                perm=permutation,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
        )
        marlin_scales = marlin_permute_scales(
            s=numerical_scales[expert_idx].T.contiguous(),
            size_k=size_k,
            size_n=size_n,
            group_size=32,
        )
        processed_scales.append(
            mxfp4_marlin_process_scales(
                marlin_scales,
                input_dtype=torch.bfloat16,
            )
        )
    return torch.stack(repacked), torch.stack(processed_scales)


def _dsv4_marlin_quant_info(
    payload: Dsv4NativeMxfp4ExpertPayload,
    *,
    expert_map: torch.Tensor,
    global_num_experts: int,
):
    """Return a pointer/version-validated local Marlin payload cache."""

    from sglang.srt.layers.moe.moe_runner.marlin import (  # noqa: PLC0415
        MarlinMoeQuantInfo,
    )

    # Key the cache by the immutable owned parameters, not by the transient
    # full-parameter views produced by FSDP.  Their data pointers can change
    # with input geometry/allocation pressure even though the checkpoint bytes
    # are identical; rebuilding would drop repacked tensors while an async
    # Marlin launch may still be reading them.
    source_parameters = (
        payload.packed_w13_weight_f32,
        payload.packed_w2_weight_f32,
        payload.packed_w13_scale_f32,
        payload.packed_w2_scale_f32,
    )
    key = tuple((parameter._version, tuple(parameter.shape), parameter.dtype) for parameter in source_parameters)
    cache = payload.__dict__.get("_marlin_repacked_cache")
    if cache is None or cache["key"] != key:
        w13, s13 = _prepare_mxfp4_marlin_bank(
            payload.w13_weight,
            payload.w13_weight_scale_inv,
            size_n=2 * payload.intermediate_size,
            size_k=payload.hidden_size,
        )
        w2, s2 = _prepare_mxfp4_marlin_bank(
            payload.w2_weight,
            payload.w2_weight_scale_inv,
            size_n=payload.hidden_size,
            size_k=payload.intermediate_size,
        )
        cache = {
            "key": key,
            "w13": w13,
            "w2": w2,
            "s13": s13,
            "s2": s2,
        }
        payload.__dict__["_marlin_repacked_cache"] = cache
    if cache["w13"].device.type == "cuda":
        stream = torch.cuda.current_stream(cache["w13"].device)
        for name in ("w13", "w2", "s13", "s2"):
            cache[name].record_stream(stream)
    return MarlinMoeQuantInfo(
        w13_qweight=cache["w13"],
        w2_qweight=cache["w2"],
        w13_scales=cache["s13"],
        w2_scales=cache["s2"],
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        is_k_full=True,
        expert_map=expert_map,
        global_num_experts=global_num_experts,
    )


def _dsv4_moe_lora_info(
    *,
    tokens: int,
    hidden_size: int,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
):
    from sglang.srt.lora.lora_moe_runners import LoRAInfo  # noqa: PLC0415

    device = gate_a.device
    local_experts = gate_a.shape[0]
    # SGLang stacks the independently trainable gate/up A ranks along the
    # rank axis and their B outputs along the projection-output axis.
    gate_up_a = torch.cat((gate_a.transpose(-1, -2), up_a.transpose(-1, -2)), dim=1).unsqueeze(0)
    gate_up_b = torch.cat((gate_b.transpose(-1, -2), up_b.transpose(-1, -2)), dim=1).unsqueeze(0)
    return LoRAInfo(
        gate_up_lora_a_weights=gate_up_a.to(torch.bfloat16).contiguous(),
        gate_up_lora_b_weights=gate_up_b.to(torch.bfloat16).contiguous(),
        down_lora_a_weights=down_a.transpose(-1, -2).unsqueeze(0).to(torch.bfloat16).contiguous(),
        down_lora_b_weights=down_b.transpose(-1, -2).unsqueeze(0).to(torch.bfloat16).contiguous(),
        seg_indptr=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        req_to_lora=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.ones(1, dtype=torch.int32, device=device),
        adapter_enabled=torch.ones(1, dtype=torch.int32, device=device),
        token_lora_mapping=torch.zeros(tokens, dtype=torch.int32, device=device),
        max_lora_rank=1,
        num_experts=local_experts,
        has_active_lora=True,
        hidden_size=hidden_size,
    )


def _build_dsv4_moe_runner_config(
    config_cls,
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
) -> object:
    """Freeze the trainer-side serving runner contract for exact DSV4."""
    return config_cls(
        num_experts=256,
        num_local_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        layer_id=0,
        top_k=top_k,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        inplace=False,
        dsv4_exact_mode=True,
    )


def _dsv4_native_mxfp4_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    experts,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: PLC0415
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner  # noqa: PLC0415
    from sglang.srt.layers.moe.token_dispatcher.standard import (  # noqa: PLC0415
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: PLC0415
    from sglang.srt.layers.moe.utils import MoeRunnerBackend  # noqa: PLC0415

    from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

    if hidden_states.device.type != "cuda" or hidden_states.dtype is not torch.bfloat16:
        raise TypeError("Exact DSV4 routed experts require CUDA BF16 activations")
    if routing_weights.dtype is not torch.float32:
        raise TypeError("Exact DSV4 Marlin routing weights must remain FP32")
    parallel_state = get_parallel_state()
    if not parallel_state.ep_enabled or parallel_state.ep_size != 8:
        raise RuntimeError("Exact DSV4 routed experts require the admitted EP8 topology")
    ep_rank = int(parallel_state.ep_rank)
    payload = experts.native_mxfp4_payload
    local_experts = int(payload.w13_weight.shape[0])
    if local_experts * int(parallel_state.ep_size) != 256:
        raise RuntimeError("Exact DSV4 routed payload must contain 32 experts per EP8 rank")
    local_start = ep_rank * local_experts
    local_ids = selected_experts.to(torch.int64) - local_start
    local_ids = torch.where(
        (local_ids >= 0) & (local_ids < local_experts),
        local_ids,
        torch.full_like(local_ids, -1),
    ).to(torch.int32)
    logical_rows = hidden_states.shape[0]
    # No row padding: the trainer must launch exactly the serving runner's M
    # per gathered segment. Serving decode launches M=1; the retired pad to
    # DSV4_MARLIN_MIN_QUALIFIED_ROWS made the trainer's live row ride an
    # M=10 chunk (1 live + 9 masked) whose Marlin bytes diverge from the
    # serving M=1 launch value-dependently on rounding-boundary inputs.
    # Stability for every M <= 10 launch
    # is owned by the shared chunked-Marlin program.
    run_hidden_states = hidden_states.contiguous()
    run_routing_weights = routing_weights.contiguous()
    run_local_ids = local_ids.contiguous()
    expert_map = torch.full((256,), -1, dtype=torch.int32, device=hidden_states.device)
    expert_map[local_start : local_start + local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device=hidden_states.device
    )
    quant_info = _dsv4_marlin_quant_info(
        payload,
        expert_map=expert_map,
        global_num_experts=256,
    )
    config = _build_dsv4_moe_runner_config(
        MoeRunnerConfig,
        local_experts=local_experts,
        hidden_size=payload.hidden_size,
        intermediate_size=payload.intermediate_size,
        top_k=selected_experts.shape[1],
    )
    dispatch = StandardDispatchOutput(
        hidden_states=run_hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=run_routing_weights,
            topk_ids=run_local_ids,
            router_logits=torch.empty(
                run_hidden_states.shape[0],
                256,
                dtype=torch.float32,
                device=hidden_states.device,
            ),
        ),
    )
    if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") == "1":
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (  # noqa: PLC0415
            fused_marlin_moe,
        )

        output = fused_marlin_moe(
            hidden_states=run_hidden_states,
            w1=quant_info.w13_qweight,
            w2=quant_info.w2_qweight,
            w1_scale=quant_info.w13_scales,
            w2_scale=quant_info.w2_scales,
            gating_output=dispatch.topk_output.router_logits,
            topk_weights=run_routing_weights,
            topk_ids=run_local_ids,
            num_bits=4,
            is_k_full=True,
            inplace=False,
            routed_scaling_factor=1.5,
            clamp_limit=10.0,
            expert_map=expert_map,
            global_num_experts=256,
        )
        # This branch is an explicitly opt-in diagnostic control.  Surface an
        # asynchronous Marlin fault at its own call boundary instead of at an
        # unrelated kernel in the following transformer layer.
        try:
            torch.cuda.synchronize(hidden_states.device)
        except RuntimeError as exc:
            layer_idx = getattr(payload, "layer_idx", "unknown")
            raise RuntimeError(f"Diagnostic DSV4 base Marlin failed after layer {layer_idx}") from exc
        _validate_all_single_adapter_batch_infos(
            hidden_states.device.index,
            where=f"base Marlin layer {getattr(payload, 'layer_idx', 'unknown')}",
        )
        return output[:logical_rows], local_ids
    lora_info = _dsv4_moe_lora_info(
        tokens=run_hidden_states.shape[0],
        hidden_size=payload.hidden_size,
        gate_a=gate_a,
        gate_b=gate_b,
        up_a=up_a,
        up_b=up_b,
        down_a=down_a,
        down_b=down_b,
    )
    output = (
        MoeRunner(
            MoeRunnerBackend.MARLIN,
            config,
            lora_enabled=True,
        )
        .run(dispatch, quant_info, lora_info=lora_info)
        .hidden_states
    )
    return output[:logical_rows], local_ids


class _Dsv4NativeMxfp4RoutedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        routing_weights,
        selected_experts,
        gate_a,
        gate_b,
        up_a,
        up_b,
        down_a,
        down_b,
        experts,
        lora_live,
    ):
        factors = (gate_a, gate_b, up_a, up_b, down_a, down_b)
        # The gather-aware exact lane sets ``lora_live`` on every EP rank so
        # each rank contributes its local expert-bank delta.  Keeping the gate
        # here also supports an explicit base-only partial when requested.
        effective = factors if lora_live else tuple(torch.zeros_like(f) for f in factors)
        output, local_ids = _dsv4_native_mxfp4_forward(
            hidden_states,
            routing_weights,
            selected_experts,
            *effective,
            experts,
        )
        ctx.experts = experts
        ctx.lora_live = lora_live
        ctx.save_for_backward(
            hidden_states.detach(),
            routing_weights.detach(),
            local_ids,
            gate_a,
            gate_b,
            up_a,
            up_b,
            down_a,
            down_b,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from xorl.models.transformers.deepseek_v4.checkpoint_handler import (  # noqa: PLC0415
            _dequantize_mxfp4_packed_int8,
        )

        (
            hidden_states,
            routing_weights,
            local_ids,
            gate_a,
            gate_b,
            up_a,
            up_b,
            down_a,
            down_b,
        ) = ctx.saved_tensors
        payload = ctx.experts.native_mxfp4_payload
        lora_live = ctx.lora_live
        need_x, need_r = ctx.needs_input_grad[:2]
        factor_needs = ctx.needs_input_grad[3:9]
        grad_x = torch.zeros_like(hidden_states) if need_x else None
        grad_r = torch.zeros_like(routing_weights) if need_r else None
        masters = (gate_a, gate_b, up_a, up_b, down_a, down_b)
        factor_grads = [torch.zeros_like(master) if needed else None for master, needed in zip(masters, factor_needs)]
        # When serving skips this rank's LoRA partial, the value is constant
        # in the factors: differentiate the zero-factor program for x/r and
        # keep the exact-zero factor gradients initialized above.
        factor_targets = factor_needs if lora_live else (False,) * 6

        with torch.enable_grad():
            for expert_idx in range(payload.w13_weight.shape[0]):
                positions = (local_ids == expert_idx).nonzero(as_tuple=False)
                if positions.numel() == 0:
                    continue
                token_ids, slot_ids = positions[:, 0], positions[:, 1]
                x = hidden_states[token_ids].detach().requires_grad_(need_x)
                weight = routing_weights[token_ids, slot_ids].detach().requires_grad_(need_r)
                effective = [
                    (master[expert_idx].detach() if lora_live else torch.zeros_like(master[expert_idx]))
                    .to(torch.bfloat16)
                    .requires_grad_(needed)
                    for master, needed in zip(masters, factor_targets)
                ]
                w13 = _dequantize_mxfp4_packed_int8(
                    payload.w13_weight[expert_idx],
                    payload.w13_weight_scale_inv[expert_idx],
                )
                w2 = _dequantize_mxfp4_packed_int8(
                    payload.w2_weight[expert_idx],
                    payload.w2_weight_scale_inv[expert_idx],
                )
                intermediate = payload.intermediate_size
                gate = torch.nn.functional.linear(x, w13[:intermediate])
                gate = gate + (x @ effective[0]) @ effective[1]
                up = torch.nn.functional.linear(x, w13[intermediate:])
                up = up + (x @ effective[2]) @ effective[3]
                gate = gate.clamp(max=10.0)
                up = up.clamp(min=-10.0, max=10.0)
                activated = torch.nn.functional.silu(gate) * up
                down = torch.nn.functional.linear(activated, w2)
                down = down + (activated @ effective[4]) @ effective[5]
                weighted = (down.float() * weight.float().unsqueeze(-1)).to(torch.bfloat16)
                targets = []
                target_roles = []
                if need_x:
                    targets.append(x)
                    target_roles.append(("x", None))
                if need_r:
                    targets.append(weight)
                    target_roles.append(("r", None))
                for factor_idx, (factor, needed) in enumerate(zip(effective, factor_targets)):
                    if needed:
                        targets.append(factor)
                        target_roles.append(("factor", factor_idx))
                grads = torch.autograd.grad(
                    weighted,
                    targets,
                    grad_outputs=grad_output[token_ids],
                    allow_unused=False,
                )
                for role, grad in zip(target_roles, grads):
                    if role[0] == "x":
                        grad_x.index_add_(0, token_ids, grad.to(grad_x.dtype))
                    elif role[0] == "r":
                        grad_r[token_ids, slot_ids] = grad.to(grad_r.dtype)
                    else:
                        factor_grads[role[1]][expert_idx].add_(grad.to(factor_grads[role[1]].dtype))
        return (
            grad_x,
            grad_r,
            None,
            *factor_grads,
            None,
            None,
        )


def dsv4_native_mxfp4_routed_partial(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    experts,
    *,
    lora_live: bool = True,
) -> torch.Tensor:
    """Literal local MXFP4-Marlin routed partial with trainable rank-one LoRA.

    Gather-aware serving installs rank-major LoRA metadata on every EP rank,
    so the exact caller keeps ``lora_live`` enabled for every local expert-bank
    partial.  Passing ``False`` remains the explicit base-only program.
    """

    if (experts.active_r, experts.active_lora_alpha, experts._active_scaling()) != (
        1,
        1,
        1.0,
    ):
        raise RuntimeError("Exact DSV4 routed experts require rank=1 alpha=1")
    factors = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        factors.extend(experts._active_lora_views(projection))
    return _Dsv4NativeMxfp4RoutedFunction.apply(
        hidden_states,
        routing_weights,
        selected_experts,
        *factors,
        experts,
        lora_live,
    )


def _native_fp8_slice_value(
    input: torch.Tensor,
    payload: Dsv4NativeBlockFp8Payload,
    *,
    output_start: int = 0,
    output_end: int | None = None,
    input_start: int = 0,
    input_end: int | None = None,
) -> torch.Tensor:
    from xorl.ops.exact.block_fp8_native import (  # noqa: PLC0415
        _sglang_native_block_fp8_linear_value,
    )

    output_end = payload.out_features if output_end is None else output_end
    input_end = payload.in_features if input_end is None else input_end
    if any(value % 128 for value in (output_start, output_end, input_start, input_end)):
        raise ValueError("Exact DSV4 FP8 TP slices must stay 128-aligned")
    full_weight, full_scales = payload()
    weight = full_weight[output_start:output_end, input_start:input_end]
    scales = full_scales[
        output_start // 128 : output_end // 128,
        input_start // 128 : input_end // 128,
    ]
    return _sglang_native_block_fp8_linear_value(
        input.contiguous(),
        weight.contiguous(),
        scales.float().contiguous(),
    )


def _sglang_dense_lora_add(
    input: torch.Tensor,
    base: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    from sglang.kernels.ops.gemm.sgemm_lora_a import (  # noqa: PLC0415
        sgemm_lora_a_fwd,
    )
    from sglang.kernels.ops.gemm.sgemm_lora_b import (  # noqa: PLC0415
        sgemm_lora_b_fwd,
    )

    batch_info = _single_adapter_batch_info(input.device.index, input.shape[0])
    low_rank = sgemm_lora_a_fwd(
        input.contiguous(),
        lora_a.to(torch.bfloat16).unsqueeze(0).contiguous(),
        batch_info,
    )
    return sgemm_lora_b_fwd(
        low_rank,
        lora_b.to(torch.bfloat16).unsqueeze(0).contiguous(),
        batch_info,
        base_output=base,
    )


def _sglang_gate_up_lora_add(
    input: torch.Tensor,
    base: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
) -> torch.Tensor:
    """Match ``MergedColumnParallelLinearWithLoRA.run_gate_up_lora``."""

    from sglang.kernels.ops.gemm.gate_up_lora_b import (  # noqa: PLC0415
        gate_up_lora_b_fwd,
    )
    from sglang.kernels.ops.gemm.sgemm_lora_a import (  # noqa: PLC0415
        sgemm_lora_a_fwd,
    )

    if gate_b.shape != up_b.shape:
        raise ValueError("DSV4 shared gate/up LoRA slices must have equal shapes")
    batch_info = _single_adapter_batch_info(input.device.index, input.shape[0])
    a = torch.cat((gate_a, up_a), dim=0).to(torch.bfloat16).unsqueeze(0)
    b = torch.cat((gate_b, up_b), dim=0).to(torch.bfloat16).unsqueeze(0)
    low_rank = sgemm_lora_a_fwd(
        input.contiguous(),
        a.contiguous(),
        batch_info,
        stack_num=2,
    )
    return gate_up_lora_b_fwd(
        low_rank,
        b.contiguous(),
        batch_info,
        gate_b.shape[0],
        base,
    )


def _dsv4_native_shared_tp_forward(
    input: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    module,
    tp_rank: int,
    tp_size: int,
    diagnostic_capture=None,
) -> torch.Tensor:
    intermediate = module.intermediate_size
    if intermediate % tp_size:
        raise ValueError("DSV4 shared intermediate size must divide the TP width")
    width = intermediate // tp_size
    start, end = tp_rank * width, (tp_rank + 1) * width
    from xorl.ops.exact.block_fp8_native import (  # noqa: PLC0415
        _sglang_native_block_fp8_linear_value,
    )

    gate_payload = module.gate_proj.native_base_payload
    up_payload = module.up_proj.native_base_payload
    gate_weight, gate_scale = gate_payload()
    up_weight, up_scale = up_payload()
    gate_up_weight = torch.cat(
        (
            gate_weight[start:end],
            up_weight[start:end],
        ),
        dim=0,
    ).contiguous()
    gate_up_scale = (
        torch.cat(
            (
                gate_scale[start // 128 : end // 128],
                up_scale[start // 128 : end // 128],
            ),
            dim=0,
        )
        .float()
        .contiguous()
    )
    gate_up_base = _sglang_native_block_fp8_linear_value(
        input.contiguous(),
        gate_up_weight,
        gate_up_scale,
    )
    gate_up = _sglang_gate_up_lora_add(
        input,
        gate_up_base,
        gate_a,
        gate_b[start:end],
        up_a,
        up_b[start:end],
    )
    if gate_up.shape != (input.shape[0], 2 * width):
        raise RuntimeError(
            "DSV4 shared gate/up fused output has the wrong local shape: "
            f"got {tuple(gate_up.shape)}, expected {(input.shape[0], 2 * width)}"
        )
    if callable(diagnostic_capture):
        diagnostic_capture("moe_native_shared_gate_value", gate_up[:, :width])
        diagnostic_capture("moe_native_shared_gate_up", gate_up)
    activated = torch.empty(
        (input.shape[0], width),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    from sglang.kernels.ops.attention.dsv4 import (  # noqa: PLC0415
        silu_and_mul_clamp,
    )

    silu_and_mul_clamp(gate_up, activated, 10.0)
    if callable(diagnostic_capture):
        diagnostic_capture("moe_native_shared_act", activated)
    down_base = _native_fp8_slice_value(
        activated,
        module.down_proj.native_base_payload,
        input_start=start,
        input_end=end,
    )
    output = _sglang_dense_lora_add(
        activated,
        down_base,
        down_a[:, start:end],
        down_b,
    )
    if callable(diagnostic_capture):
        diagnostic_capture("moe_native_shared_down", output)
    return output


class _Dsv4NativeSharedTpFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        gate_a,
        gate_b,
        up_a,
        up_b,
        down_a,
        down_b,
        module,
        tp_rank,
        tp_size,
        diagnostic_capture,
        lora_live,
    ):
        factors = (gate_a, gate_b, up_a, up_b, down_a, down_b)
        # The gather-aware exact lane sets ``lora_live`` on every EP rank so
        # each rank contributes its shared-expert TP adapter slice.  Keeping
        # the gate here also supports an explicit base-only partial.
        effective = factors if lora_live else tuple(torch.zeros_like(f) for f in factors)
        output = _dsv4_native_shared_tp_forward(
            input,
            *effective,
            module,
            tp_rank,
            tp_size,
            diagnostic_capture,
        )
        ctx.module = module
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        ctx.lora_live = lora_live
        ctx.save_for_backward(
            input.detach(),
            gate_a,
            gate_b,
            up_a,
            up_b,
            down_a,
            down_b,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, *masters = ctx.saved_tensors
        module = ctx.module
        lora_live = ctx.lora_live
        width = module.intermediate_size // ctx.tp_size
        start, end = ctx.tp_rank * width, (ctx.tp_rank + 1) * width
        need_x = ctx.needs_input_grad[0]
        factor_needs = ctx.needs_input_grad[1:7]
        # When serving skips this rank's LoRA partial, the value is constant in
        # the factors: differentiate the zero-factor program for grad_x and
        # return exact-zero factor gradients.
        factor_targets = factor_needs if lora_live else (False,) * 6
        with torch.enable_grad():
            x = input.detach().requires_grad_(need_x)
            effective = [
                (master.detach() if lora_live else torch.zeros_like(master)).to(torch.bfloat16).requires_grad_(needed)
                for master, needed in zip(masters, factor_targets)
            ]
            gate_weight = _dequantize_native_block_fp8_for_backward(module.gate_proj.native_base_payload)[start:end]
            up_weight = _dequantize_native_block_fp8_for_backward(module.up_proj.native_base_payload)[start:end]
            down_weight = _dequantize_native_block_fp8_for_backward(module.down_proj.native_base_payload)[:, start:end]
            gate = torch.nn.functional.linear(x, gate_weight)
            gate = gate + (x @ effective[0].T) @ effective[1][start:end].T
            up = torch.nn.functional.linear(x, up_weight)
            up = up + (x @ effective[2].T) @ effective[3][start:end].T
            activated = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0)
            down = torch.nn.functional.linear(activated, down_weight)
            down = down + (activated @ effective[4][:, start:end].T) @ effective[5].T
            targets = []
            target_indices = []
            if need_x:
                targets.append(x)
                target_indices.append(0)
            for idx, (factor, needed) in enumerate(zip(effective, factor_targets), start=1):
                if needed:
                    targets.append(factor)
                    target_indices.append(idx)
            computed = torch.autograd.grad(
                down,
                targets,
                grad_outputs=grad_output,
                allow_unused=False,
            )
        grads = [None] * 7
        for index, grad in zip(target_indices, computed):
            target = input if index == 0 else masters[index - 1]
            grads[index] = grad.to(target.dtype)
        if not lora_live:
            for idx, needed in enumerate(factor_needs, start=1):
                if needed:
                    grads[idx] = torch.zeros_like(masters[idx - 1])
        return (*grads, None, None, None, None, None)


def dsv4_native_shared_expert_tp_partial(
    input: torch.Tensor,
    module,
    *,
    tp_rank: int,
    tp_size: int,
    diagnostic_capture=None,
    lora_live: bool = True,
) -> torch.Tensor:
    """Serving-value shared-expert TP slice, before the ordered rank sum.

    Gather-aware serving installs rank-major LoRA metadata on every EP rank,
    so the exact caller keeps ``lora_live`` enabled for every shared-expert TP
    slice.  Passing ``False`` remains the explicit base-only program.
    """

    projections = (module.gate_proj, module.up_proj, module.down_proj)
    for projection in projections:
        if (
            projection.active_r,
            projection.active_lora_alpha,
            projection._active_scaling(),
        ) != (1, 1, 1.0):
            raise RuntimeError("Exact DSV4 shared experts require rank=1 alpha=1")
    return _Dsv4NativeSharedTpFunction.apply(
        input,
        module.gate_proj.lora_A,
        module.gate_proj.lora_B,
        module.up_proj.lora_A,
        module.up_proj.lora_B,
        module.down_proj.lora_A,
        module.down_proj.lora_B,
        module,
        tp_rank,
        tp_size,
        diagnostic_capture,
        lora_live,
    )


class _Dsv4RoutedSharedJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed: torch.Tensor, shared: torch.Tensor, scale: float):
        from xorl.distributed.canonical_moe import canonical_moe_leaf_fp32_v1  # noqa: PLC0415

        ctx.scale = scale
        ctx.routed_dtype = routed.dtype
        ctx.shared_dtype = shared.dtype
        return canonical_moe_leaf_fp32_v1(shared, routed, routed_scale=scale)

    @staticmethod
    def backward(ctx, grad_output):
        grad_routed = (grad_output.float() * ctx.scale).to(ctx.routed_dtype)
        grad_shared = grad_output.to(ctx.shared_dtype)
        return grad_routed, grad_shared, None


def dsv4_join_routed_shared_partial(
    routed: torch.Tensor,
    shared: torch.Tensor,
    *,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Match the exact serving leaf: FP32 FMA, then one BF16/FP16 store."""

    return _Dsv4RoutedSharedJoin.apply(routed, shared, routed_scaling_factor)


def attach_dsv4_native_payloads(model: nn.Module, config) -> None:
    """Attach complete immutable payload holders before checkpoint loading."""

    dense_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # These official-checkpoint families are native BF16 and have no
            # scale sidecars: output head, router gates, compressor wkv/wgate,
            # and the indexer's score projection.  They must retain their real
            # Parameters; attaching an empty FP8 holder and later stripping the
            # BF16 weight would corrupt model ownership before first forward.
            if (
                name == "lm_head"
                or name.endswith(".mlp.gate")
                or ".compressor." in name
                or name.endswith(".indexer.linear_weights_proj")
            ):
                continue
            if hasattr(module, "native_base_payload"):
                raise ValueError("DSV4 native dense payload was attached twice")
            payload = Dsv4NativeBlockFp8Payload(
                module.out_features,
                module.in_features,
                device=module.weight.device,
            )
            payload.logical_name = name
            module.add_module("native_base_payload", payload)
            module.__class__ = Dsv4NativeBlockFp8Linear
            dense_count += 1

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != int(config.num_hidden_layers):
        raise ValueError("DSV4 native payload attachment requires the complete decoder stack")
    for layer_idx, layer in enumerate(layers):
        experts = layer.mlp.experts
        if hasattr(experts, "native_mxfp4_payload"):
            raise ValueError("DSV4 native routed payload was attached twice")
        payload = Dsv4NativeMxfp4ExpertPayload(
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            device=experts.gate_up_proj.device,
        )
        payload.layer_idx = layer_idx
        experts.add_module("native_mxfp4_payload", payload)
        # The independently wrapped expert FSDP unit owns these packed byte
        # carriers. Its MP policy must neither cast the payload nor coerce the
        # exact FP32 routing weights at the module boundary.
        experts.fsdp_requires_full_precision = True
    model._dsv4_native_dense_payload_count = dense_count
    model._dsv4_native_routed_payload_count = len(layers)


def strip_dsv4_dequantized_base_parameters(model: nn.Module) -> tuple[int, int]:
    """Remove placeholder BF16 bases before FSDP/checkpoint materialization."""

    dense_count = 0
    for module in model.modules():
        if "native_base_payload" in module._modules:
            if "weight" not in module._parameters:
                raise RuntimeError(f"DSV4 native linear {type(module).__name__} has no registered weight to strip")
            module.register_parameter("weight", None)
            dense_count += 1

    routed_count = 0
    layers = getattr(getattr(model, "model", None), "layers", ())
    for layer in layers:
        experts = layer.mlp.experts
        if "native_mxfp4_payload" not in experts._modules:
            raise RuntimeError("DSV4 exact routed experts lost native MXFP4 ownership")
        experts.register_parameter("gate_up_proj", None)
        experts.register_parameter("down_proj", None)
        routed_count += 1
    model._dsv4_stripped_dense_base_count = dense_count
    model._dsv4_stripped_routed_base_count = routed_count
    return dense_count, routed_count


__all__ = [
    "DSV4_NATIVE_PAYLOAD_CONTRACT_VERSION",
    "Dsv4NativeBlockFp8Payload",
    "Dsv4NativeBlockFp8Linear",
    "Dsv4NativeMxfp4ExpertPayload",
    "attach_dsv4_native_payloads",
    "dsv4_native_block_fp8_lora",
    "dsv4_native_grouped_wo_a",
    "dsv4_native_mxfp4_routed_partial",
    "dsv4_native_shared_expert_tp_partial",
    "dsv4_join_routed_shared_partial",
    "pack_bytes_as_float32",
    "pack_expert_rows_as_float32",
    "unpack_float32_as_bytes",
    "unpack_expert_rows_from_float32",
    "strip_dsv4_dequantized_base_parameters",
]
