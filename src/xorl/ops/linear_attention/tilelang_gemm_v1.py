"""In-repo shim that re-adds tilelang's `gemm_v1` (fast `tl::gemm_ss/_rs/_sr` template
GEMM via the `tl_gemm` builtin) on top of a *stock* upstream tilelang, without forking it.

tilelang 0.1.9 removed the explicit `gemm_v1` op and unified on `T.gemm` (a Python "tileop"
lowering that inlines the WGMMA sequence ~4-5x slower for warp-specialized kernels like
FlashQLA GDN). The fast C++ template (`tl::gemm_ss`, `src/tl_templates/cuda/gemm_sm90.h`) and
its `tl.tl_gemm` codegen builtin still ship in tilelang >=0.1.10 — they're just no longer
reachable from the frontend. This shim re-exposes them:

  * `gemm_v1(...)` — frontend that tags a GEMM node with the ``use_tl_gemm_template`` annotation.
  * `GemmV1Template` — lowering impl that builds the `tl::gemm_ss<...>` string (ported from
    tilelang 0.1.8 `src/op/gemm.cc::GemmNode::Lower`) and emits it via the `tl_gemm` builtin;
    layout inference is delegated to the stock per-instruction impl so layouts match.
  * `patch()` — installs `T.gemm_v1` and dispatches annotated nodes to `GemmV1Template`
    (called once, before any FlashQLA kernel is traced). Mirrors the vendored quack
    `cute_dsl_ptxas.patch()` injection pattern.

Requires a tilelang with the `tl_gemm` builtin (>=0.1.10) and, for the TMA-load path that
FlashQLA's kernels request via ``T.copy(..., prefer_instruction="tma")``, PR #2303.
"""

from __future__ import annotations

import tilelang.language as T
from tilelang import _ffi_api
from tilelang._typing import BarrierType, BufferLikeType
from tilelang.language.gemm_op import _gemm_impl
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.tileop.gemm.registry import resolve_gemm_impl
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment
from tilelang.utils.target import target_is_cdna, target_is_cuda, target_is_hopper
from tvm import tirx
from tvm.target import Target


GEMM_INST_WGMMA = "cuda.wgmma"
_ANNOTATION = "use_tl_gemm_template"


def _gemm_inst_key(gemm_node, thread_nums, target: Target) -> str:
    return str(_ffi_api.GemmGetGemmInstructionKey(gemm_node, int(thread_nums), target))


def _make_access_ptr_from_region(buf, region, rw_mask: int):
    """Python port of C++ MakeAccessPtrFromRegion(region, rw_mask, require_2d=True)."""
    shape = buf.shape
    ndim = len(shape)
    mins = [r.min for r in region.region]
    if ndim == 1:
        offset = mins[0]
        extent = region.region[0].extent
    else:
        strides = [None] * ndim
        cur = tirx.const(1, shape[0].dtype)
        for i in range(ndim - 1, -1, -1):
            strides[i] = cur
            cur = cur * shape[i]
        offset = tirx.const(0, shape[0].dtype)
        for i in range(ndim - 2):
            offset = offset + mins[i] * strides[i]
        extent = region.region[ndim - 2].extent * region.region[ndim - 1].extent
    return buf.access_ptr(rw_mask, offset=offset, extent=extent)


def _const_bool(prim_expr) -> bool:
    val = getattr(prim_expr, "value", prim_expr)
    return bool(val)


class GemmV1Template(GemmBase):
    """Emit the fast `tl::gemm_ss/_rs/_sr` template call via the `tl.tl_gemm` builtin."""

    def infer_layout(self, target: Target, thread_nums: int):
        # Reuse the stock per-instruction layout inference so the shared/fragment layouts the
        # template consumes match what the rest of the pipeline expects.
        gemm_inst = _gemm_inst_key(self.gemm_node, thread_nums, target)
        impl = resolve_gemm_impl(gemm_inst, target)
        return impl(self.gemm_node).infer_layout(target, thread_nums)

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        thread_nums = thread_bounds.extent
        gemm_inst = _gemm_inst_key(self.gemm_node, thread_nums, target)
        warp_m, warp_n = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, gemm_inst)

        if is_fragment(self.A):
            assert not self.trans_A, "gemm_rs requires the A operand to be non-transposed."
            op_name = "tl::gemm_rs"
        elif is_fragment(self.B):
            op_name = "tl::gemm_sr"
        else:
            op_name = "tl::gemm_ss"
        assert is_fragment(self.C), "gemm_v1 requires the C/accumulator operand to be a fragment."

        parts = [
            str(int(self.M)),
            str(int(self.N)),
            str(int(self.K)),
            str(int(warp_m)),
            str(int(warp_n)),
            str(int(bool(self.trans_A))),
            str(int(bool(self.trans_B))),
            str(int(_const_bool(self.clear_accum))),
        ]
        if target_is_cuda(target):
            parts += [
                str(int(self.stride_A)),
                str(int(self.stride_B)),
                str(int(self.offset_A)),
                str(int(self.offset_B)),
            ]
        if target_is_cdna(target):
            parts += [str(int(self.k_pack))]
        elif target_is_hopper(target):
            parts += ["true" if gemm_inst == GEMM_INST_WGMMA else "false"]
        if target_is_hopper(target) and int(self.wg_wait) != 0:
            parts += [str(int(self.wg_wait))]

        instance = f"{op_name}<{', '.join(parts)}>"

        a_ptr = _make_access_ptr_from_region(self.A, self.ARegion, 1)
        b_ptr = _make_access_ptr_from_region(self.B, self.BRegion, 1)
        c_ptr = _make_access_ptr_from_region(self.C, self.CRegion, 3)

        @T.prim_func
        def _gemm_template() -> None:
            T.evaluate(
                tirx.call_intrin(
                    "handle",
                    tirx.op.Op.get("tl.tl_gemm"),
                    tirx.StringImm(instance),
                    a_ptr,
                    b_ptr,
                    c_ptr,
                )
            )

        return _Simplify(_gemm_template, inline_let=True)


def gemm_v1(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tirx.PrimExpr:
    """GEMM v1: fast C++ template path (`tl::gemm_ss/_rs/_sr`) via the `tl.tl_gemm` builtin."""
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
        annotations={_ANNOTATION: 1},
    )


_patched = False


def _node_uses_template(gemm_node) -> bool:
    ann = getattr(gemm_node, "annotations", None)
    return ann is not None and ann.get(_ANNOTATION) is not None


def patch() -> None:
    """Install `T.gemm_v1` and route nodes annotated with ``use_tl_gemm_template`` to the
    fast template lowering. Idempotent; safe to call before tracing FlashQLA kernels."""
    global _patched
    if _patched:
        return
    from tilelang.tileop.gemm import Gemm  # noqa: PLC0415

    T.gemm_v1 = gemm_v1  # so `T.gemm_v1(...)` resolves in traced kernels

    _orig_lower = Gemm.lower
    _orig_infer = Gemm.infer_layout

    def _lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr):
        if _node_uses_template(self):
            return GemmV1Template(self).lower(layout_map, target, thread_bounds, thread_var, mbar_phase_expr)
        return _orig_lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr)

    def _infer(self, target, thread_nums):
        if _node_uses_template(self):
            return GemmV1Template(self).infer_layout(target, thread_nums)
        return _orig_infer(self, target, thread_nums)

    Gemm.lower = _lower
    Gemm.infer_layout = _infer
    _patched = True
