"""Backend selection for the Gated Delta Rule (GDN) chunk kernel.

Two implementations are available:

* ``fla``      — the vendored Flash-Linear-Attention Triton kernels
                 (:mod:`xorl.ops.linear_attention.ops.gated_delta_rule`). Default.
                 Supports Ulysses context parallelism via ``cp_context``.
* ``flashqla`` — the vendored Qwen FlashQLA fused TileLang kernels
                 (:mod:`xorl.ops._vendored.flashqla`). Hopper (SM90) only,
                 requires ``tilelang``. Faster fwd/bwd. The single-GPU kernel is
                 CP-unaware; under Ulysses CP it is driven by xorl's native CP
                 orchestration via :mod:`xorl.ops.linear_attention.flashqla_cp`
                 (see :func:`flashqla_chunk_gated_delta_rule_cp`).

Select the backend with the ``XORL_GDN_BACKEND`` environment variable
(``fla`` | ``flashqla``). ``flashqla`` requires head dim 128; for other head dims
the caller falls back to ``fla`` (see :func:`warn_cp_fallback_once`).

``XORL_GDN_FLASHQLA_AUTOCP`` overrides FlashQLA's ``auto_cp`` intra-card CP
heuristic outside an exact Qwen3.5 model program.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable


GDN_BACKEND_ENV = "XORL_GDN_BACKEND"
FLASHQLA_AUTOCP_ENV = "XORL_GDN_FLASHQLA_AUTOCP"
_VALID_BACKENDS = ("fla", "flashqla")

_flashqla_chunk: Callable[..., Any] | None = None
_flashqla_chunk_cp: Callable[..., Any] | None = None
_warned_cp_fallback = False


def get_gdn_backend() -> str:
    """Return the configured GDN chunk backend (``"fla"`` or ``"flashqla"``)."""
    from xorl.ops.linear_attention.modules.bi_contract import _is_gdn_contract_enabled  # noqa: PLC0415

    if _is_gdn_contract_enabled():
        return "fla"
    backend = os.environ.get(GDN_BACKEND_ENV, "fla").strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{GDN_BACKEND_ENV}={backend!r} is invalid; expected one of {_VALID_BACKENDS}.",
        )
    return backend


def resolve_flashqla_auto_cp(auto_cp: bool | None) -> bool:
    """Resolve FlashQLA's ``auto_cp`` mode outside the exact Qwen program.

    auto_cp's intra-card CP heuristic (Be*H<=40 gate + warmup h0-drop approximation)
    breaks batch-invariance and changes math, so it is pinned OFF whenever the GDN
    exact Qwen model program is active; otherwise it defaults to ON.
    """
    from xorl.ops.linear_attention.modules.bi_contract import _is_gdn_contract_enabled  # noqa: PLC0415

    if _is_gdn_contract_enabled():
        return False
    if auto_cp is not None:
        return auto_cp
    env = os.environ.get(FLASHQLA_AUTOCP_ENV)
    if env is not None:
        return env.strip().lower() in {"1", "true", "yes", "on"}
    return True


def warn_cp_fallback_once() -> None:
    """Warn (once) that a FlashQLA request fell back to the FLA Triton GDN kernel.

    Under the exact GDN contract a silent backend swap is a byte hazard, not
    a performance note: the contract RAISES instead. (Today the contract pins
    the backend to ``fla`` before any FlashQLA request can be made, so this
    is defense-in-depth against a future reordering of backend resolution.)
    """
    from xorl.ops.linear_attention.modules.bi_contract import _is_gdn_contract_enabled  # noqa: PLC0415

    if _is_gdn_contract_enabled():
        raise RuntimeError(
            "Exact Qwen3.5 GDN: a FlashQLA->FLA backend fallback was requested while the GDN "
            "contract is active; silently swapping the kernel program is not admitted"
        )
    global _warned_cp_fallback
    if not _warned_cp_fallback:
        warnings.warn(
            f"{GDN_BACKEND_ENV}=flashqla was requested, but FlashQLA requires 128-dim heads; "
            "the FLA Triton GDN kernel is used for these layers instead.",
            stacklevel=2,
        )
        _warned_cp_fallback = True


def flashqla_chunk_gated_delta_rule(**kwargs: Any) -> Any:
    """Lazily import and call the vendored FlashQLA chunk kernel.

    The import is deferred because FlashQLA imports ``tilelang`` and validates the
    GPU compute capability (SM90) at import time; keeping it lazy means the default
    ``fla`` path never requires ``tilelang`` to be installed.
    """
    global _flashqla_chunk
    if _flashqla_chunk is None:
        try:
            # Re-add tilelang's fast `gemm_v1` (tl::gemm_ss template) on top of the stock
            # upstream tilelang; FlashQLA's kernels call `T.gemm_v1`. Must run before tracing.
            from xorl.ops.linear_attention import tilelang_gemm_v1  # noqa: PLC0415

            tilelang_gemm_v1.patch()
            from xorl.ops._vendored.flashqla import chunk_gated_delta_rule as _chunk  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Failed to import the FlashQLA GDN backend ({GDN_BACKEND_ENV}=flashqla). "
                "It requires tilelang>=0.1.10 (with the tl_gemm builtin + PR #2303 "
                "prefer_instruction) and a Hopper (SM90) GPU. "
                f"Original error: {exc}",
            ) from exc
        _flashqla_chunk = _chunk
    return _flashqla_chunk(**kwargs)


def flashqla_chunk_gated_delta_rule_cp(**kwargs: Any) -> Any:
    """Lazily import and call the FlashQLA chunk kernel under xorl's native CP.

    Like :func:`flashqla_chunk_gated_delta_rule`, the import is deferred (FlashQLA pulls
    in ``tilelang`` and validates SM90 at import time). This path drives the FlashQLA
    interior with xorl's Ulysses/sequence-parallel orchestration; see
    :mod:`xorl.ops.linear_attention.flashqla_cp`.
    """
    global _flashqla_chunk_cp
    if _flashqla_chunk_cp is None:
        try:
            from xorl.ops.linear_attention import tilelang_gemm_v1  # noqa: PLC0415

            tilelang_gemm_v1.patch()
            from xorl.ops.linear_attention.flashqla_cp import (  # noqa: PLC0415
                flashqla_chunk_gated_delta_rule_cp as _chunk_cp,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Failed to import the FlashQLA CP GDN backend ({GDN_BACKEND_ENV}=flashqla with CP). "
                "It requires tilelang>=0.1.10 (with the tl_gemm builtin + PR #2303 "
                "prefer_instruction) and a Hopper (SM90) GPU. "
                f"Original error: {exc}",
            ) from exc
        _flashqla_chunk_cp = _chunk_cp
    return _flashqla_chunk_cp(**kwargs)
