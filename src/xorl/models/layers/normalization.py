import importlib
import os
import warnings
from typing import Callable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from xorl.ops.batch_invariant_ops import (
    RMS_NORM_FAMILIES,
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    RMSNormFamily,
    bi_fused_add_rms_norm,
    bi_rms_norm,
    fused_rms_norm_backward,
    is_batch_invariant_mode_enabled,
    is_batch_invariant_op_enabled,
    is_trunk_linear_contract_enabled,
    set_batch_invariant_mode,
)


RMSNormMode = Literal["eager", "native", "compile", "sglang", "sglang_fused", "sglang_jit", "sglang_kernel"]
_RMSNORM_MODE: RMSNormMode = "native"
_COMPILED_NATIVE_RMS_NORM: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None
_COMPILED_EAGER_RMS_NORM: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None
_COMPILED_ZERO_CENTERED_RMS_NORM: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None
_SGLANG_JIT_NORM = None
_SGLANG_KERNEL_NORM = None


def set_rmsnorm_mode(mode: RMSNormMode) -> None:
    global _RMSNORM_MODE
    if mode not in {"eager", "native", "compile", "sglang", "sglang_fused", "sglang_jit", "sglang_kernel"}:
        raise ValueError(f"Unsupported rmsnorm_mode: {mode}")
    _RMSNORM_MODE = mode


def get_rmsnorm_mode() -> RMSNormMode:
    return _RMSNORM_MODE


def eager_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """Eager implementation of RMSNorm."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


def native_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    return F.rms_norm(hidden_states, (weight.shape[0],), weight, eps=variance_epsilon)


def sglang_residual_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (hidden_states * weight.to(torch.float32)).to(input_dtype)


class _FusedSglangRMSNorm(torch.autograd.Function):
    """Differentiable wrapper for :func:`sglang_rms_norm_batch_invariant`.

    Forward runs the fused batch-invariant Triton kernel (bit-exact to
    ``sglang_residual_rms_norm``); backward uses the closed-form RMSNorm
    gradient.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):
        out = bi_rms_norm(hidden_states, weight, variance_epsilon, family=RMS_NORM_FAMILY_RESIDUAL_TREE)
        ctx.save_for_backward(hidden_states, weight)
        ctx.variance_epsilon = variance_epsilon
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight = ctx.saved_tensors
        grad_normed, grad_weight = fused_rms_norm_backward(hidden_states, weight, ctx.variance_epsilon, grad_output)
        return grad_normed.to(hidden_states.dtype), grad_weight.to(weight.dtype), None


class _BatchInvariantRMSNorm(torch.autograd.Function):
    """Differentiable wrapper for the ``serving_no_residual`` (family-1) kernel — the
    kernel behind the aten::rms_norm interpose and serving's no-residual RMSNorm dispatch.

    This is NOT :class:`_FusedSglangRMSNorm`: the family-1 kernel and the fused sglang
    residual-tree kernel disagree at 1 ulp on rare bf16 boundary values, and the qk-norm
    contract is family-1. Backward reuses the closed-form RMSNorm gradient (gradients do
    not enter the forward K3)."""

    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):
        out = bi_rms_norm(hidden_states, weight, variance_epsilon, family=RMS_NORM_FAMILY_NO_RESIDUAL)
        ctx.save_for_backward(hidden_states, weight)
        ctx.variance_epsilon = variance_epsilon
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight = ctx.saved_tensors
        grad_normed, grad_weight = fused_rms_norm_backward(hidden_states, weight, ctx.variance_epsilon, grad_output)
        return grad_normed.to(hidden_states.dtype), grad_weight.to(weight.dtype), None


class _FusedSglangResidualRMSNorm(torch.autograd.Function):
    """Differentiable wrapper for :func:`fused_add_rms_norm_batch_invariant`.

    Forward fuses ``residual_out = hidden_states + residual`` and the RMSNorm in
    Triton (bit-exact to the eager add + ``sglang_residual_rms_norm``). Backward
    sums the norm gradient and the downstream residual gradient (``residual_out``
    feeds both the normalization and the next layer's residual stream).
    """

    @staticmethod
    def forward(ctx, hidden_states, residual, weight, variance_epsilon):
        out, residual_out = bi_fused_add_rms_norm(
            hidden_states, residual, weight, variance_epsilon, family=RMS_NORM_FAMILY_RESIDUAL_TREE
        )
        ctx.save_for_backward(residual_out, weight)
        ctx.variance_epsilon = variance_epsilon
        ctx.input_dtype = hidden_states.dtype
        ctx.residual_dtype = residual.dtype
        return out, residual_out

    @staticmethod
    def backward(ctx, grad_output, grad_residual_out):
        residual_out, weight = ctx.saved_tensors
        grad_total, grad_weight = fused_rms_norm_backward(
            residual_out,
            weight,
            ctx.variance_epsilon,
            grad_output,
            grad_residual_out=grad_residual_out,
        )
        return (
            grad_total.to(ctx.input_dtype),
            grad_total.to(ctx.residual_dtype),
            grad_weight.to(weight.dtype),
            None,
        )


def fast_sglang_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Fast, differentiable, bit-exact replacement for the no-residual eager
    ``sglang_residual_rms_norm``. Falls back to the eager path off CUDA."""
    if not hidden_states.is_cuda:
        return sglang_residual_rms_norm(hidden_states, weight, variance_epsilon)
    return _FusedSglangRMSNorm.apply(hidden_states, weight, variance_epsilon)


def fast_batch_invariant_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Differentiable, bit-exact match of serving's no-residual (family-1)
    batch-invariant RMSNorm (the aten::rms_norm interpose kernel). Used for the
    qk-norm term of the trunk contract lane. Falls back to the native path off CUDA."""
    if not hidden_states.is_cuda:
        return native_rms_norm(hidden_states, weight, variance_epsilon)
    return _BatchInvariantRMSNorm.apply(hidden_states, weight, variance_epsilon)


def fast_sglang_residual_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast, differentiable, bit-exact replacement for the eager
    ``residual_out = hidden_states + residual`` followed by
    ``sglang_residual_rms_norm(residual_out)``. Returns ``(out, residual_out)``.
    Falls back to the eager path off CUDA."""
    if not hidden_states.is_cuda:
        residual_out = hidden_states + residual
        return (
            sglang_residual_rms_norm(residual_out, weight, variance_epsilon),
            residual_out,
        )
    return _FusedSglangResidualRMSNorm.apply(hidden_states, residual, weight, variance_epsilon)


def _get_sglang_jit_norm():
    global _SGLANG_JIT_NORM
    if _SGLANG_JIT_NORM is None:
        try:
            sglang_jit_norm = importlib.import_module("sglang.jit_kernel.norm")
        except Exception as exc:
            raise RuntimeError(
                "rmsnorm_mode='sglang_jit' requires sglang.jit_kernel.norm on PYTHONPATH "
                "(for example via SGLANG_REPO=/path/to/sglang/python)."
            ) from exc
        _SGLANG_JIT_NORM = sglang_jit_norm
    return _SGLANG_JIT_NORM


def _get_sglang_kernel_norm():
    global _SGLANG_KERNEL_NORM
    if _SGLANG_KERNEL_NORM is None:
        try:
            _SGLANG_KERNEL_NORM = importlib.import_module("sgl_kernel")
        except Exception as exc:
            raise RuntimeError(
                "rmsnorm_mode='sglang_kernel' requires sgl_kernel on PYTHONPATH "
                "(for example via an SGLang-capable xorl diagnostic venv)."
            ) from exc
    return _SGLANG_KERNEL_NORM


def _as_2d_token_hidden(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def sglang_jit_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Use SGLang's JIT RMSNorm kernels for forward-only parity diagnostics.

    The CUDA residual path mirrors SGLang's in-place fused_add_rmsnorm: the
    returned first tensor is normalized output and the second is the residual
    sum. CPU falls back to the native implementation so unit tests can exercise
    the mode without requiring the JIT CUDA extension.
    """
    if not hidden_states.is_cuda:
        if residual is None:
            return native_rms_norm(hidden_states, weight, variance_epsilon)
        residual_out = hidden_states + residual
        return native_rms_norm(residual_out, weight, variance_epsilon), residual_out

    sglang_jit_norm = _get_sglang_jit_norm()
    out = hidden_states.contiguous().clone()
    out_2d = _as_2d_token_hidden(out)
    kernel_weight = weight.to(out.dtype).contiguous()
    if residual is None:
        sglang_jit_norm.rmsnorm(out_2d, kernel_weight, out=out_2d, eps=variance_epsilon)
        return out

    residual_out = residual.contiguous().clone()
    residual_out_2d = _as_2d_token_hidden(residual_out)
    sglang_jit_norm.fused_add_rmsnorm(out_2d, residual_out_2d, kernel_weight, variance_epsilon)
    return out, residual_out


def sglang_kernel_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Use SGLang's production sgl_kernel RMSNorm ops for forward-only diagnostics."""
    if not hidden_states.is_cuda:
        if residual is None:
            return native_rms_norm(hidden_states, weight, variance_epsilon)
        residual_out = hidden_states + residual
        return native_rms_norm(residual_out, weight, variance_epsilon), residual_out

    sgl_kernel = _get_sglang_kernel_norm()
    out = hidden_states.contiguous().clone()
    out_2d = _as_2d_token_hidden(out)
    kernel_weight = weight.to(out.dtype).contiguous()
    if residual is None:
        normed = sgl_kernel.rmsnorm(out_2d, kernel_weight, variance_epsilon)
        if normed is not None:
            return normed.reshape_as(out)
        return out

    residual_out = residual.contiguous().clone()
    residual_out_2d = _as_2d_token_hidden(residual_out)
    sgl_kernel.fused_add_rmsnorm(out_2d, residual_out_2d, kernel_weight, variance_epsilon)
    return out, residual_out


def compiled_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    global _COMPILED_NATIVE_RMS_NORM
    if _COMPILED_NATIVE_RMS_NORM is None:
        _COMPILED_NATIVE_RMS_NORM = torch.compile(native_rms_norm)
    return _COMPILED_NATIVE_RMS_NORM(hidden_states, weight, variance_epsilon)


def compiled_eager_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """Compiled fp32-upcast eager RMSNorm. Use when ``F.rms_norm``'s bf16 path is
    too imprecise (e.g., GLM-4 MoE 92+ norm layers compounding bf16 error)."""
    global _COMPILED_EAGER_RMS_NORM
    if _COMPILED_EAGER_RMS_NORM is None:
        _COMPILED_EAGER_RMS_NORM = torch.compile(eager_rms_norm)
    return _COMPILED_EAGER_RMS_NORM(hidden_states, weight, variance_epsilon)


def eager_zero_centered_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    output = hidden_states.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + variance_epsilon)
    output = output * (1.0 + weight.float())
    return output.type_as(hidden_states)


def native_zero_centered_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    output = F.rms_norm(
        hidden_states.float(),
        (weight.shape[0],),
        1.0 + weight.float(),
        eps=variance_epsilon,
    )
    return output.type_as(hidden_states)


def native_zero_centered_rms_norm_without_batch_invariant(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    if hidden_states.is_cuda:
        if is_batch_invariant_mode_enabled() and is_batch_invariant_op_enabled("rms_norm"):
            with set_batch_invariant_mode(False):
                return native_zero_centered_rms_norm(hidden_states, weight, variance_epsilon)
    return native_zero_centered_rms_norm(hidden_states, weight, variance_epsilon)


def compiled_zero_centered_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    global _COMPILED_ZERO_CENTERED_RMS_NORM
    if _COMPILED_ZERO_CENTERED_RMS_NORM is None:
        _COMPILED_ZERO_CENTERED_RMS_NORM = torch.compile(native_zero_centered_rms_norm)
    return _COMPILED_ZERO_CENTERED_RMS_NORM(hidden_states, weight, variance_epsilon)


def fast_zero_centered_batch_invariant_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Zero-centered (Gemma-style) analogue of :func:`fast_batch_invariant_rms_norm`.

    Family-1 term of the Qwen3.5 norm-seed contract: the fp32-upcast no-residual
    norm (qk-norm / layer-0 input norm) runs the same batch-invariant kernel the
    aten::rms_norm interpose serves, with ``1 + weight`` folded in fp32 exactly as
    :func:`native_zero_centered_rms_norm` does — so the trunk-contract lane
    (``XORL_BI_TRUNK_LINEAR=1``) is bit-identical to the interpose lane without a
    global interpose. Real gradients via the closed-form RMSNorm backward.
    Falls back to the native path off CUDA."""
    if not hidden_states.is_cuda:
        return native_zero_centered_rms_norm(hidden_states, weight, variance_epsilon)
    output = _BatchInvariantRMSNorm.apply(hidden_states.float(), 1.0 + weight.float(), variance_epsilon)
    return output.type_as(hidden_states)


_WARNED_UNDECLARED_FAMILY: set[tuple[str, bool]] = set()


def _check_undeclared_family(mode: RMSNormMode, residual_present: bool) -> None:
    """Loud tripwire for the norm-seed hazard: an RMSNorm call in a serving-parity
    mode under batch-invariant mode without a declared kernel family reaches a
    family implicitly (the silent-flip class). Warns once per
    (mode, call-shape); raises when XORL_RMSNORM_REQUIRE_FAMILY is set."""
    implicit = "serving_residual_tree (fused)" if residual_present else "serving_no_residual (aten interpose)"
    message = (
        f"RMSNorm call without a declared kernel family in mode={mode!r} under "
        f"batch-invariant mode: dispatching implicitly to the {implicit} family. "
        "Family mismatches against serving are silent 1-ulp K3 seeds; declare "
        "family= at the call site or module construction."
    )
    if os.environ.get("XORL_RMSNORM_REQUIRE_FAMILY", "").lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(message)
    key = (mode, residual_present)
    if key not in _WARNED_UNDECLARED_FAMILY:
        _WARNED_UNDECLARED_FAMILY.add(key)
        warnings.warn(message, stacklevel=3)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    ``family`` declares which batch-invariant kernel family the site's serving
    counterpart executes (see ``xorl.ops.batch_invariant_ops``):
    ``"serving_no_residual"`` for qk-norms and layer-0 input layernorms,
    ``"serving_residual_tree"`` for residual-carrying norms (input layernorm at
    layer>0, post-attention layernorm, final norm). The declaration replaces the
    per-call ``force_sglang_residual=<expr>`` guards: it is inert outside the
    ``sglang``/``sglang_fused`` modes and reproduces their dispatch bit-for-bit
    inside them, while making a family flip a loud error instead of a silent
    1-ulp K3 seed.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        mode: Optional[RMSNormMode] = None,
        family: Optional[RMSNormFamily] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.mode: RMSNormMode = mode or get_rmsnorm_mode()
        if family is not None and family not in RMS_NORM_FAMILIES:
            raise ValueError(f"Unknown RMSNorm family {family!r}; expected one of {RMS_NORM_FAMILIES}")
        self.family: Optional[RMSNormFamily] = family

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        force_sglang_residual: bool = False,
        force_sglang_residual_kernel: bool = False,
        family: Optional[RMSNormFamily] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if family is not None and family not in RMS_NORM_FAMILIES:
            raise ValueError(f"Unknown RMSNorm family {family!r}; expected one of {RMS_NORM_FAMILIES}")
        family = family or self.family

        if family == RMS_NORM_FAMILY_NO_RESIDUAL:
            # Structural guard: serving never runs the no-residual family on a
            # residual stream; any residual-tree request here is a family flip.
            if residual is not None:
                raise ValueError(
                    "RMSNorm site declared family='serving_no_residual' but was called "
                    "with a residual stream; residual site-classes are 'serving_residual_tree'"
                )
            if force_sglang_residual or force_sglang_residual_kernel:
                raise ValueError(
                    "RMSNorm site declared family='serving_no_residual' but was forced "
                    "onto the sglang residual tree; this is a family flip"
                )
        elif family == RMS_NORM_FAMILY_RESIDUAL_TREE:
            # Single-tensor residual-tree sites (pre-summed input norm at layer>0,
            # final norm) dispatch exactly like the legacy
            # ``force_sglang_residual=(mode in sglang modes)`` call-site guards.
            if residual is None and not force_sglang_residual_kernel:
                force_sglang_residual = force_sglang_residual or self.mode in ("sglang", "sglang_fused")
        elif (
            not force_sglang_residual
            and not force_sglang_residual_kernel
            and self.mode in ("sglang", "sglang_fused")
            and (is_batch_invariant_mode_enabled() or is_trunk_linear_contract_enabled())
        ):
            _check_undeclared_family(self.mode, residual is not None)

        diag_add_cast_bf16 = os.environ.get("XORL_DIAGNOSTIC_RESIDUAL_ADD_CAST_BF16", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # The fused sglang path performs the residual add inside its kernel
        # (bit-exact to torch's fp32-accumulated bf16 add), so skip the eager
        # add when it applies. The diagnostic bf16 recast is not modelled by the
        # kernel, so fall back to the eager add if it is requested.
        use_fused_residual = (
            self.mode == "sglang_fused"
            and residual is not None
            and not force_sglang_residual
            and not force_sglang_residual_kernel
            and not diag_add_cast_bf16
        )

        residual_out: Optional[torch.Tensor] = None
        norm_input = hidden_states
        if residual is not None and not use_fused_residual:
            residual_out = hidden_states + residual
            if diag_add_cast_bf16:
                residual_out = residual_out.to(torch.bfloat16)
            norm_input = residual_out

        if force_sglang_residual_kernel:
            if residual is not None:
                out, residual_out = sglang_kernel_rms_norm(
                    hidden_states,
                    self.weight,
                    self.variance_epsilon,
                    residual=residual,
                )
            else:
                out = sglang_kernel_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif force_sglang_residual:
            if self.mode == "sglang_fused":
                out = fast_sglang_rms_norm(norm_input, self.weight, self.variance_epsilon)
            else:
                out = sglang_residual_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "eager":
            out = eager_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "native":
            out = native_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "compile":
            out = compiled_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "sglang":
            if residual_out is not None or force_sglang_residual:
                out = sglang_residual_rms_norm(norm_input, self.weight, self.variance_epsilon)
            else:
                out = native_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "sglang_fused":
            if use_fused_residual:
                out, residual_out = fast_sglang_residual_rms_norm(
                    hidden_states, residual, self.weight, self.variance_epsilon
                )
            elif residual is not None:
                # Diagnostic bf16 recast path: residual already added above.
                out = fast_sglang_rms_norm(norm_input, self.weight, self.variance_epsilon)
            elif is_trunk_linear_contract_enabled():
                # Trunk contract lane: no-residual norms (qk-norm) must bit-match
                # serving's family-1 batch-invariant kernel, with real gradients.
                out = fast_batch_invariant_rms_norm(norm_input, self.weight, self.variance_epsilon)
            else:
                out = native_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "sglang_jit":
            if residual is not None:
                out, residual_out = sglang_jit_rms_norm(
                    hidden_states,
                    self.weight,
                    self.variance_epsilon,
                    residual=residual,
                )
            else:
                out = sglang_jit_rms_norm(norm_input, self.weight, self.variance_epsilon)
        elif self.mode == "sglang_kernel":
            if residual is not None:
                out, residual_out = sglang_kernel_rms_norm(
                    hidden_states,
                    self.weight,
                    self.variance_epsilon,
                    residual=residual,
                )
            else:
                out = sglang_kernel_rms_norm(norm_input, self.weight, self.variance_epsilon)
        else:
            raise ValueError(f"Unsupported rmsnorm_mode: {self.mode}")

        if residual_out is not None and prenorm:
            return out, residual_out
        return out

    def extra_repr(self):
        family = f", family={self.family}" if self.family is not None else ""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, mode={self.mode}{family}"
