"""Exact GLM-5.2 absorbed-MLA ``kv_b_proj`` with active LoRA.

SGLang never invokes ``kv_b_proj`` as an ordinary linear in its absorbed MLA
path.  It dequantizes the frozen block-FP8 weight, creates deliberately laid
out K/V views, performs one base BMM before attention and another after
attention, then applies two factored LoRA correction kernels to each result.
This module owns that complete physical program behind explicit ``q`` and
``v`` forward branches.  The split lets both real attention sites enter the
same module boundary, so FSDP hooks cover every packed-parameter access and
autograd accumulates both branches into the shared logical factors.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor, nn

from xorl.models.transformers.glm5.exact_lora_contract import glm52_exact_lora_scaling
from xorl.ops.block_fp8_native import NativeBlockFP8Linear


GLM52_EXACT_TP1_ABSORBED_KV_B_QLORA_CONTRACT_VERSION = "glm52_exact_tp1_absorbed_kv_b_qlora_v2"

_GLM52_NUM_HEADS = 64
_GLM52_QK_NOPE_HEAD_DIM = 192
_GLM52_QK_ROPE_HEAD_DIM = 64
_GLM52_V_HEAD_DIM = 256
_GLM52_KV_LORA_RANK = 512
_GLM52_FIXED_GRAPH_LORA_SLOTS = 8


@lru_cache(maxsize=64)
def _single_adapter_absorbed_batch_info(device_index: int, rows: int, rank: int, scaling: float):
    """Cache the trainer's immutable one-active-adapter kernel metadata."""

    from sglang.srt.lora.utils import LoRABatchInfo  # noqa: PLC0415

    device = torch.device("cuda", device_index)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.full((1,), rank, dtype=torch.int32, device=device),
        scalings=torch.full((1,), scaling, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )


class _Glm52ExactTP1AbsorbedKvBQFunction(torch.autograd.Function):
    """Own the pre-attention base BMM and literal q-side correction."""

    @staticmethod
    def forward(
        ctx,
        q_nope: Tensor,
        lora_A: Tensor,
        lora_B: Tensor,
        batch_info: Any,
        module: Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
    ) -> Tensor:
        effective_A = lora_A.to(torch.bfloat16).contiguous()
        effective_B = lora_B.to(torch.bfloat16).contiguous()
        output = module._exact_q_value(q_nope, effective_A, effective_B, batch_info)
        ctx.set_materialize_grads(False)
        ctx.module = module
        # The masters are version-counter sentinels; the surrogate consumes
        # only the exact forward's BF16 factor bytes widened to FP32.
        ctx.save_for_backward(q_nope.detach(), effective_A, effective_B, lora_A, lora_B)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor | None):
        q_nope, effective_A, effective_B, _lora_A_master, _lora_B_master = ctx.saved_tensors
        if grad_output is None:
            return None, None, None, None, None
        grad_q_nope, grad_A, grad_B = ctx.module._surrogate_q_vjp(
            q_nope,
            effective_A,
            effective_B,
            grad_output,
            needs_input_grad=ctx.needs_input_grad[:3],
        )
        return grad_q_nope, grad_A, grad_B, None, None


class _Glm52ExactTP1AbsorbedKvBVFunction(torch.autograd.Function):
    """Own the post-attention base BMM and literal v-side correction."""

    @staticmethod
    def forward(
        ctx,
        attn_latent: Tensor,
        lora_A: Tensor,
        lora_B: Tensor,
        batch_info: Any,
        module: Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
    ) -> Tensor:
        effective_A = lora_A.to(torch.bfloat16).contiguous()
        effective_B = lora_B.to(torch.bfloat16).contiguous()
        output = module._exact_v_value(attn_latent, effective_A, effective_B, batch_info)
        ctx.set_materialize_grads(False)
        ctx.module = module
        ctx.save_for_backward(attn_latent.detach(), effective_A, effective_B, lora_A, lora_B)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor | None):
        attn_latent, effective_A, effective_B, _lora_A_master, _lora_B_master = ctx.saved_tensors
        if grad_output is None:
            return None, None, None, None, None
        grad_attn_latent, grad_A, grad_B = ctx.module._surrogate_v_vjp(
            attn_latent,
            effective_A,
            effective_B,
            grad_output,
            needs_input_grad=ctx.needs_input_grad[:3],
        )
        return grad_attn_latent, grad_A, grad_B, None, None


class Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(NativeBlockFP8Linear):
    """Official-shape TP1 absorbed ``kv_b_proj`` with two explicit uses."""

    _glm52_exact_active_lora_component = True
    adapter_gradient_producer_family = "module_managed"
    contract_version = GLM52_EXACT_TP1_ABSORBED_KV_B_QLORA_CONTRACT_VERSION
    logical_factor_names = ("lora_A", "lora_B")
    max_lora_rank: int
    fixed_graph_lora_slots = _GLM52_FIXED_GRAPH_LORA_SLOTS

    def __init__(
        self,
        *,
        num_heads: int = _GLM52_NUM_HEADS,
        qk_nope_head_dim: int = _GLM52_QK_NOPE_HEAD_DIM,
        v_head_dim: int = _GLM52_V_HEAD_DIM,
        kv_lora_rank: int = _GLM52_KV_LORA_RANK,
        r: int = 1,
        lora_alpha: int = 1,
        bias: bool = False,
        device: torch.device | str | None = None,
        tp_size: int = 1,
    ) -> None:
        geometry = (num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank)
        official = (
            _GLM52_NUM_HEADS,
            _GLM52_QK_NOPE_HEAD_DIM,
            _GLM52_V_HEAD_DIM,
            _GLM52_KV_LORA_RANK,
        )
        if geometry != official:
            raise ValueError(
                "GLM-5.2 exact absorbed kv_b supports only official "
                f"(heads, qk_nope, v_head, kv_rank)={official}, got {geometry}"
            )
        scaling = glm52_exact_lora_scaling(r, lora_alpha)
        if bias:
            raise ValueError("GLM-5.2 exact absorbed kv_b is bias-free")
        if tp_size != 1:
            raise ValueError(f"GLM-5.2 exact absorbed kv_b admits only effective TP1, got TP{tp_size}")

        out_features = num_heads * (qk_nope_head_dim + v_head_dim)
        super().__init__(kv_lora_rank, out_features, device=device)
        self.num_heads = int(num_heads)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.kv_lora_rank = int(kv_lora_rank)
        self.r = self.active_r = int(r)
        self.lora_alpha = self.active_lora_alpha = int(lora_alpha)
        self.scaling = scaling
        self.max_lora_rank = r
        self.tp_size = 1
        self.lora_A = nn.Parameter(torch.empty((r, kv_lora_rank), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.empty((out_features, r), dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self._validated_batch_info_signature: tuple[Any, ...] | None = None

    def _apply(self, fn, recurse: bool = True):
        # Packed FP8 bytes/scales and optimizer masters are all encoded as
        # FP32 state. A model-wide BF16 conversion may move but never cast or
        # replace live non-meta Parameters.
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.lora_A.device))
        names = ("packed_weight_f32", "weight_scale_inv", "lora_A", "lora_B")
        protected = {name: self._parameters.pop(name) for name in names}
        try:
            result = nn.Module._apply(self, fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                if parameter.is_meta:
                    value = torch.empty_like(parameter, dtype=torch.float32, device=probe.device)
                    replacements[name] = nn.Parameter(value, requires_grad=parameter.requires_grad)
                else:
                    with torch.no_grad():
                        parameter.data = parameter.data.to(device=probe.device, dtype=torch.float32)
                        if parameter.grad is not None:
                            parameter.grad.data = parameter.grad.data.to(device=probe.device, dtype=torch.float32)
                    replacements[name] = parameter
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
        for name, shape in (
            ("lora_A", (self.r, self.kv_lora_rank)),
            ("lora_B", (self.out_features, self.r)),
        ):
            tensor = state_dict.get(f"{prefix}{name}")
            if tensor is not None and (tensor.dtype is not torch.float32 or tuple(tensor.shape) != shape):
                raise TypeError(
                    f"GLM-5.2 exact absorbed kv_b state {prefix}{name} must be FP32 {shape}, "
                    f"got {tensor.dtype} {tuple(tensor.shape)}"
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

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        self.scaling = glm52_exact_lora_scaling(lora_rank, lora_alpha)
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha

    def _validate_state_contract(self, device: torch.device) -> None:
        if (
            self.r,
            self.active_r,
            self.lora_alpha,
            self.active_lora_alpha,
            self.scaling,
            self.tp_size,
            self.max_lora_rank,
            self.fixed_graph_lora_slots,
        ) != (
            self.r,
            self.r,
            self.lora_alpha,
            self.lora_alpha,
            glm52_exact_lora_scaling(self.r, self.lora_alpha),
            1,
            self.r,
            _GLM52_FIXED_GRAPH_LORA_SLOTS,
        ):
            raise RuntimeError("GLM-5.2 exact absorbed kv_b rank/alpha/scaling/TP/slot contract was mutated")
        expected_factors = {
            "lora_A": (self.lora_A, (self.r, self.kv_lora_rank)),
            "lora_B": (self.lora_B, (self.out_features, self.r)),
        }
        for name, (factor, shape) in expected_factors.items():
            if factor.dtype is not torch.float32 or tuple(factor.shape) != shape:
                raise TypeError(f"{name} must remain FP32 with shape {shape}, got {factor.dtype} {tuple(factor.shape)}")
        if self.packed_weight_f32.requires_grad or self.weight_scale_inv.requires_grad:
            raise RuntimeError("GLM-5.2 absorbed kv_b block-FP8 base weights and scales must remain frozen")
        for state in (self.lora_A, self.lora_B, self.packed_weight_f32, self.weight_scale_inv):
            if state.device != device:
                raise RuntimeError("GLM-5.2 exact absorbed kv_b state and activation must share one CUDA device")

    def _validate_batch_info(self, batch_info: Any, rows: int, device: torch.device) -> int:
        try:
            from sglang.srt.lora.utils import LoRABatchInfo  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang LoRA metadata is required") from exc
        if not isinstance(batch_info, LoRABatchInfo):
            raise TypeError(f"Expected SGLang LoRABatchInfo, got {type(batch_info).__name__}")

        num_segments = batch_info.num_segments or batch_info.bs
        tensors = {
            "seg_lens": batch_info.seg_lens,
            "seg_indptr": batch_info.seg_indptr,
            "weight_indices": batch_info.weight_indices,
            "lora_ranks": batch_info.lora_ranks,
            "scalings": batch_info.scalings,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor) or tensor.device != device:
                raise RuntimeError(f"Absorbed kv_b metadata {name} must be a tensor on {device}")
            if tensor.ndim != 1 or not tensor.is_contiguous():
                raise ValueError(f"Absorbed kv_b metadata {name} must be a contiguous one-dimensional tensor")
        signature = (
            id(batch_info),
            rows,
            device.index,
            batch_info.use_cuda_graph,
            batch_info.bs,
            batch_info.num_segments,
            batch_info.max_len,
            batch_info.expected_tokens,
            id(batch_info.permutation),
            None if batch_info.permutation is None else batch_info.permutation._version,
            *(item for tensor in tensors.values() for item in (id(tensor), tensor._version)),
        )
        # The first warm call validates device contents. Unchanged metadata is
        # then graph-capture safe: tensor version counters catch every ordinary
        # in-place routing update without another GPU-to-CPU synchronization.
        if signature == self._validated_batch_info_signature:
            return batch_info.lora_ranks.numel()
        if any(tensors[name].dtype is not torch.int32 for name in ("seg_lens", "seg_indptr", "weight_indices")):
            raise TypeError("Absorbed kv_b segment lengths, pointers, and weight indices must be int32")
        if batch_info.lora_ranks.dtype is not torch.int32:
            raise TypeError("Absorbed kv_b LoRA ranks must be int32")
        if batch_info.scalings.dtype is not torch.float32:
            raise TypeError("Absorbed kv_b LoRA scalings must be FP32")

        slot_count = batch_info.lora_ranks.numel()
        if batch_info.scalings.numel() != slot_count:
            raise ValueError("Absorbed kv_b rank and scaling slot counts differ")
        if batch_info.use_cuda_graph:
            if (batch_info.bs, num_segments, slot_count) != (
                _GLM52_FIXED_GRAPH_LORA_SLOTS,
                _GLM52_FIXED_GRAPH_LORA_SLOTS,
                _GLM52_FIXED_GRAPH_LORA_SLOTS,
            ):
                raise ValueError("Exact absorbed kv_b fixed-graph metadata requires eight adapter segment slots")
            if batch_info.permutation is None:
                raise ValueError("Exact absorbed kv_b fixed-graph metadata requires the real merged-row permutation")
            expected_sizes = {
                "seg_lens": _GLM52_FIXED_GRAPH_LORA_SLOTS,
                "seg_indptr": _GLM52_FIXED_GRAPH_LORA_SLOTS + 1,
                "weight_indices": _GLM52_FIXED_GRAPH_LORA_SLOTS,
                "lora_ranks": _GLM52_FIXED_GRAPH_LORA_SLOTS,
                "scalings": _GLM52_FIXED_GRAPH_LORA_SLOTS,
            }
        else:
            if (batch_info.bs, num_segments, slot_count) != (1, 1, 1) or batch_info.permutation is not None:
                raise ValueError("Exact absorbed kv_b trainer metadata requires one eager active-adapter segment")
            expected_sizes = {
                "seg_lens": 1,
                "seg_indptr": 2,
                "weight_indices": 1,
                "lora_ranks": 1,
                "scalings": 1,
            }
        actual_sizes = {name: tensor.numel() for name, tensor in tensors.items()}
        if actual_sizes != expected_sizes:
            raise ValueError(f"Exact absorbed kv_b metadata sizes must be {expected_sizes}, got {actual_sizes}")
        indptr = batch_info.seg_indptr[: num_segments + 1].detach().to(device="cpu", dtype=torch.int64)
        lengths = indptr[1:] - indptr[:-1]
        declared_lengths = batch_info.seg_lens[:num_segments].detach().to(device="cpu", dtype=torch.int64)
        indices = batch_info.weight_indices[:num_segments].detach().to(device="cpu", dtype=torch.int64)
        if int(indptr[0]) != 0 or bool(torch.any(lengths < 0)) or int(indptr[-1]) != rows:
            raise ValueError("Absorbed kv_b segments must monotonically cover every activation row exactly once")
        if not torch.equal(lengths, declared_lengths):
            raise ValueError("Absorbed kv_b segment lengths and pointers disagree")
        if bool(torch.any(indices[lengths > 0] != 0)):
            raise ValueError("Every live exact absorbed kv_b segment must select adapter slot zero")

        ranks = batch_info.lora_ranks.detach().to(device="cpu", dtype=torch.int64)
        scalings = batch_info.scalings.detach().to(device="cpu", dtype=torch.float32)
        expected_ranks = torch.zeros(slot_count, dtype=torch.int64)
        expected_scalings = torch.zeros(slot_count, dtype=torch.float32)
        expected_ranks[0] = self.r
        expected_scalings[0] = self.scaling
        if not torch.equal(ranks, expected_ranks) or not torch.equal(scalings, expected_scalings):
            raise ValueError(
                "Exact absorbed kv_b requires the configured rank/scaling in slot0 "
                "and every padded slot rank0/scale0"
            )
        if batch_info.use_cuda_graph:
            expected_indices = torch.arange(_GLM52_FIXED_GRAPH_LORA_SLOTS, dtype=torch.int64)
            expected_lengths = torch.zeros(_GLM52_FIXED_GRAPH_LORA_SLOTS, dtype=torch.int64)
            expected_lengths[0] = rows
            if not torch.equal(indices, expected_indices) or not torch.equal(lengths, expected_lengths):
                raise ValueError(
                    "Exact absorbed kv_b fixed graph requires one live slot0 segment and seven empty slots"
                )
        if batch_info.max_len is None or int(batch_info.max_len) != rows:
            raise ValueError(f"Absorbed kv_b metadata max_len={batch_info.max_len} does not match rows={rows}")
        if batch_info.expected_tokens is not None and int(batch_info.expected_tokens) != rows:
            raise ValueError(f"Absorbed kv_b metadata expects {batch_info.expected_tokens} rows, received {rows}")
        if batch_info.permutation is not None:
            if batch_info.permutation.device != device or batch_info.permutation.dtype is not torch.int32:
                raise TypeError("Absorbed kv_b permutation must be int32 on the activation device")
            if batch_info.permutation.ndim != 1 or not batch_info.permutation.is_contiguous():
                raise ValueError("Absorbed kv_b permutation must be a contiguous one-dimensional tensor")
            permutation = batch_info.permutation[:rows].detach().to(device="cpu", dtype=torch.int64)
            if not torch.equal(permutation, torch.arange(rows, dtype=torch.int64)):
                raise ValueError("Exact absorbed kv_b fixed-graph permutation must be the real slot0 identity order")
        self._validated_batch_info_signature = signature
        return slot_count

    def _prepare_branch(
        self, input: Tensor, branch: str, batch_info: Any | None
    ) -> tuple[Tensor, Any, tuple[int, ...]]:
        if branch == "q":
            input_width = self.qk_nope_head_dim
        elif branch == "v":
            input_width = self.kv_lora_rank
        else:
            raise ValueError(f"GLM-5.2 exact absorbed kv_b branch must be 'q' or 'v', got {branch!r}")
        if input.ndim != 4 or tuple(input.shape[-2:]) != (self.num_heads, input_width):
            raise ValueError(
                f"GLM-5.2 exact absorbed kv_b {branch} input must have [batch, sequence, heads, dim] shape "
                f"ending in ({self.num_heads}, {input_width}), got {tuple(input.shape)}"
            )
        if input.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 exact absorbed kv_b {branch} input must be BF16, got {input.dtype}")
        if input.device.type != "cuda":
            raise RuntimeError("GLM-5.2 exact absorbed kv_b branches require CUDA")
        leading_shape = tuple(input.shape[:2])
        rows = math.prod(leading_shape)
        if rows <= 0:
            raise ValueError("GLM-5.2 exact absorbed kv_b requires at least one CP-local row")
        if branch == "q":
            fused_head_width = self.qk_nope_head_dim + _GLM52_QK_ROPE_HEAD_DIM
            expected_stride = (
                leading_shape[1] * self.num_heads * fused_head_width,
                self.num_heads * fused_head_width,
                fused_head_width,
                1,
            )
            if input.stride() != expected_stride:
                raise ValueError(
                    "GLM-5.2 exact absorbed kv_b q input must be the official 192-wide slice of a "
                    f"256-wide q head, with stride {expected_stride}; got {input.stride()}"
                )
        elif not input.is_contiguous():
            raise ValueError("GLM-5.2 exact absorbed kv_b v input must use the contiguous [B,S,H,512] layout")
        self._validate_state_contract(input.device)
        if batch_info is None:
            batch_info = _single_adapter_absorbed_batch_info(
                input.device.index, rows, self.r, self.scaling
            )
        self._validate_batch_info(batch_info, rows, input.device)
        try:
            flat_input = input.view(rows, self.num_heads, input_width)
        except RuntimeError as exc:
            raise ValueError(
                f"GLM-5.2 exact absorbed kv_b {branch} input must flatten batch/sequence as a view"
            ) from exc
        return flat_input, batch_info, leading_shape

    def _materialize_absorbed_weights(self) -> tuple[Tensor, Tensor]:
        """Reproduce S4's block-FP8 post-load dequantization and K/V layout."""

        try:
            from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang block-FP8 dequantization is required") from exc
        weight = block_quant_dequant(
            self.fp8_weight(),
            self.weight_scale_inv,
            list(self.block_size),
            torch.bfloat16,
        )
        if weight.dtype is not torch.bfloat16 or tuple(weight.shape) != (self.out_features, self.in_features):
            raise RuntimeError(
                f"SGLang kv_b dequantization produced {weight.dtype} {tuple(weight.shape)}, expected "
                f"BF16 {(self.out_features, self.in_features)}"
            )
        w_kc, w_vc = weight.unflatten(
            0,
            (self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        # These apparently redundant transforms are literal S4 post-load
        # layout construction and determine the operand strides seen by bmm.
        w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        w_vc = w_vc.contiguous().transpose(1, 2)
        return w_kc, w_vc

    def _physical_factor_buffers(
        self,
        effective_A: Tensor,
        effective_B: Tensor,
        batch_info: Any,
    ) -> tuple[Tensor, Tensor]:
        if effective_A.dtype is not torch.bfloat16 or effective_B.dtype is not torch.bfloat16:
            raise TypeError("GLM-5.2 exact absorbed kv_b effective factors must be BF16")
        slot_count = batch_info.lora_ranks.numel()
        # Adapter-slot padding is real; the live rank dimension is not padded.
        A_buffer = effective_A.new_zeros((slot_count, self.r, self.kv_lora_rank))
        B_buffer = effective_B.new_zeros((slot_count, self.out_features, self.r))
        A_buffer[0].copy_(effective_A)
        B_buffer[0].copy_(effective_B)
        return A_buffer, B_buffer

    def _exact_q_value(
        self,
        q_nope: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        batch_info: Any,
    ) -> Tensor:
        try:
            from sglang.kernels.ops.gemm.kv_b_lora_absorbed import (  # noqa: PLC0415
                step_a_q_fwd,
                step_b_q_fwd,
            )
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang absorbed kv_b q kernels are required") from exc
        w_kc, _ = self._materialize_absorbed_weights()
        q_output = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
        A_buffer, B_buffer = self._physical_factor_buffers(effective_A, effective_B, batch_info)
        q_lora_a = step_a_q_fwd(
            q_nope,
            B_buffer,
            batch_info,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        step_b_q_fwd(q_lora_a, A_buffer, batch_info, q_output)
        return q_output

    def _exact_v_value(
        self,
        attn_latent: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        batch_info: Any,
    ) -> Tensor:
        try:
            from sglang.kernels.ops.gemm.kv_b_lora_absorbed import (  # noqa: PLC0415
                step_a_v_fwd,
                step_b_v_fwd,
            )
        except Exception as exc:
            raise RuntimeError("Pinned public SGLang absorbed kv_b v kernels are required") from exc
        _, w_vc = self._materialize_absorbed_weights()
        flat_output = torch.empty(
            (attn_latent.shape[0], self.num_heads * self.v_head_dim),
            dtype=attn_latent.dtype,
            device=attn_latent.device,
        )
        v_output = flat_output.view(-1, self.num_heads, self.v_head_dim)
        torch.bmm(
            attn_latent.transpose(0, 1),
            w_vc,
            out=v_output.transpose(0, 1),
        )
        A_buffer, B_buffer = self._physical_factor_buffers(effective_A, effective_B, batch_info)
        v_lora_a = step_a_v_fwd(attn_latent, A_buffer, batch_info)
        step_b_v_fwd(
            v_lora_a,
            B_buffer,
            batch_info,
            v_output,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )
        return v_output

    def _surrogate_q_vjp(
        self,
        q_nope: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        grad_output: Tensor,
        *,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Differentiate the q base in BF16 and its LoRA branch in FP32."""

        need_input, need_A, need_B = needs_input_grad
        if not any(needs_input_grad):
            return None, None, None
        with torch.enable_grad(), torch.autocast(device_type=q_nope.device.type, enabled=False):
            base_grad_input = None
            if need_input:
                base_input = q_nope.detach().requires_grad_(True)
                w_kc, _ = self._materialize_absorbed_weights()
                base_output = torch.bmm(base_input.transpose(0, 1), w_kc).transpose(0, 1)
                (base_grad_input,) = torch.autograd.grad(base_output, base_input, grad_outputs=grad_output)

            reference_q = q_nope.float().detach().requires_grad_(need_input)
            reference_A = effective_A.float().detach().requires_grad_(need_A)
            reference_B = effective_B.float().detach().requires_grad_(need_B)
            B_k = reference_B.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.r,
            )[:, : self.qk_nope_head_dim]
            q_low_rank = torch.einsum("shd,hdr->shr", reference_q, B_k)
            q_correction = self.scaling * torch.einsum("shr,rc->shc", q_low_rank, reference_A)
            requested = []
            labels = []
            for label, required, value in (
                ("input", need_input, reference_q),
                ("A", need_A, reference_A),
                ("B", need_B, reference_B),
            ):
                if required:
                    labels.append(label)
                    requested.append(value)
            gradients = torch.autograd.grad(
                q_correction,
                requested,
                grad_outputs=grad_output.float(),
                allow_unused=False,
            )
        by_label = dict(zip(labels, gradients, strict=True))
        grad_input = None
        if need_input:
            grad_input = base_grad_input.float() + by_label["input"].float()
        return grad_input, by_label.get("A"), by_label.get("B")

    def _surrogate_v_vjp(
        self,
        attn_latent: Tensor,
        effective_A: Tensor,
        effective_B: Tensor,
        grad_output: Tensor,
        *,
        needs_input_grad: tuple[bool, bool, bool],
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Differentiate the v base in BF16 and its LoRA branch in FP32."""

        need_input, need_A, need_B = needs_input_grad
        if not any(needs_input_grad):
            return None, None, None
        with torch.enable_grad(), torch.autocast(device_type=attn_latent.device.type, enabled=False):
            base_grad_input = None
            if need_input:
                base_input = attn_latent.detach().requires_grad_(True)
                _, w_vc = self._materialize_absorbed_weights()
                base_output = torch.bmm(base_input.transpose(0, 1), w_vc).transpose(0, 1)
                (base_grad_input,) = torch.autograd.grad(base_output, base_input, grad_outputs=grad_output)

            reference_attn = attn_latent.float().detach().requires_grad_(need_input)
            reference_A = effective_A.float().detach().requires_grad_(need_A)
            reference_B = effective_B.float().detach().requires_grad_(need_B)
            B_v = reference_B.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.r,
            )[:, self.qk_nope_head_dim :]
            v_low_rank = torch.einsum("shc,rc->shr", reference_attn, reference_A)
            v_correction = self.scaling * torch.einsum("shr,hdr->shd", v_low_rank, B_v)
            requested = []
            labels = []
            for label, required, value in (
                ("input", need_input, reference_attn),
                ("A", need_A, reference_A),
                ("B", need_B, reference_B),
            ):
                if required:
                    labels.append(label)
                    requested.append(value)
            gradients = torch.autograd.grad(
                v_correction,
                requested,
                grad_outputs=grad_output.float(),
                allow_unused=False,
            )
        by_label = dict(zip(labels, gradients, strict=True))
        grad_input = None
        if need_input:
            grad_input = base_grad_input.float() + by_label["input"].float()
        return grad_input, by_label.get("A"), by_label.get("B")

    def forward_partition(self, *args, **kwargs) -> Tensor:
        raise RuntimeError(
            "GLM-5.2 exact absorbed kv_b cannot run as a direct projection; "
            "invoke the module with branch='q' or branch='v'"
        )

    def forward(
        self,
        input: Tensor | None = None,
        *,
        branch: str | None = None,
        batch_info: Any | None = None,
        return_dequantized_weight: bool = False,
    ) -> Tensor:
        """Run one real absorbed use through the module's FSDP hook boundary."""

        if return_dequantized_weight:
            if input is not None or branch is not None or batch_info is not None:
                raise ValueError("Frozen kv_b materialization does not accept an activation, branch, or metadata")
            return super().forward(return_dequantized_weight=True)
        if branch is None:
            raise RuntimeError(
                "GLM-5.2 exact absorbed kv_b cannot run as a direct projection; "
                "invoke the module with branch='q' or branch='v'"
            )
        if input is None:
            raise ValueError(f"GLM-5.2 exact absorbed kv_b branch={branch!r} requires an activation")
        flat_input, resolved_batch_info, leading_shape = self._prepare_branch(input, branch, batch_info)
        if branch == "q":
            flat_output = _Glm52ExactTP1AbsorbedKvBQFunction.apply(
                flat_input,
                self.lora_A,
                self.lora_B,
                resolved_batch_info,
                self,
            )
            output_width = self.kv_lora_rank
        else:
            flat_output = _Glm52ExactTP1AbsorbedKvBVFunction.apply(
                flat_input,
                self.lora_A,
                self.lora_B,
                resolved_batch_info,
                self,
            )
            output_width = self.v_head_dim
        return flat_output.view(*leading_shape, self.num_heads, output_width)


__all__ = [
    "GLM52_EXACT_TP1_ABSORBED_KV_B_QLORA_CONTRACT_VERSION",
    "Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA",
]
