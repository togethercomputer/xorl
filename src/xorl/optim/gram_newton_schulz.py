"""
Gram Newton-Schulz orthogonalization for Muon.

This is an in-tree adaptation of Dao-AILab/gram-newton-schulz that keeps the
algorithm local to Xorl so it can be used inside the existing FSDP/DTensor
Muon optimizer without depending on the external package's single-GPU Muon
wrapper.
"""

import importlib
import os
from functools import lru_cache
from itertools import combinations
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..utils import logging


logger = logging.get_logger(__name__)

SYMMETRIC_KERNEL_TILE_SIZE = 256
MOST_NEGATIVE_GRAM_EIGENVALUE = -4e-4


def _torch_sym_baddbmm(A: Tensor, B: Tensor, C: Tensor, alpha: float = 1.0, beta: float = 1.0) -> Tensor:
    if A.ndim == 2:
        return torch.addmm(C, A, B, beta=beta, alpha=alpha)
    return torch.baddbmm(C, A, B, beta=beta, alpha=alpha)


def _torch_mm_add(A: Tensor, B: Tensor, C: Tensor, beta: float = 1.0) -> Tensor:
    if A.ndim == 2:
        return torch.addmm(C, A, B, beta=beta)
    return torch.baddbmm(C, A, B, beta=beta)


_TORCH_BACKEND = SimpleNamespace(
    sym_mm=lambda A, B: A @ B,
    sym_baddbmm=_torch_sym_baddbmm,
    mm=lambda A, B: A @ B,
    mm_add=_torch_mm_add,
)


def _quack_supports_symmetric_gemm_dtype(dtype: torch.dtype, capability: tuple[int, int]) -> bool:
    if dtype == torch.float32:
        return capability[0] >= 10
    return dtype in (torch.float16, torch.bfloat16)


def _ensure_cutlass_arch_for_current_device() -> None:
    if os.environ.get("CUTE_DSL_ARCH") or not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()
    if major == 11 and minor == 0:
        major, minor = 10, 1

    suffix = "a" if major >= 9 else ""
    os.environ["CUTE_DSL_ARCH"] = f"sm_{major}{minor}{suffix}"


def _import_quack_gemm_interface():
    try:
        return importlib.import_module("quack.gemm_interface")
    except Exception as exc:
        raise ImportError("Muon Gram Newton-Schulz requires the upstream `quack-kernels` package") from exc


@lru_cache(maxsize=1)
def _make_quack_backend():
    _ensure_cutlass_arch_for_current_device()
    gemm_interface = _import_quack_gemm_interface()
    gemm = gemm_interface.gemm
    gemm_add = gemm_interface.gemm_add
    gemm_symmetric = gemm_interface.gemm_symmetric

    return SimpleNamespace(
        sym_mm=lambda A, B: gemm_symmetric(A, B),
        sym_baddbmm=lambda A, B, C, alpha=1.0, beta=1.0: gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta),
        mm=lambda A, B: gemm(A, B),
        mm_add=lambda A, B, C, beta=1.0: gemm_add(A, B, C=C, beta=beta),
    )


def expand_ns_coefficients(
    ns_coefficients: Sequence[float] | Sequence[Sequence[float]],
    ns_steps: int,
) -> List[Tuple[float, float, float]]:
    """
    Expand Muon's coefficient config into one coefficient triple per iteration.

    PyTorch Muon uses a single `(a, b, c)` tuple repeated `ns_steps` times.
    Gram Newton-Schulz needs the per-iteration coefficients explicitly.
    """
    if ns_steps <= 0:
        raise ValueError(f"ns_steps must be positive, got {ns_steps}")

    if len(ns_coefficients) == 3 and isinstance(ns_coefficients[0], (float, int)):
        triple = tuple(float(v) for v in ns_coefficients)
        return [triple] * ns_steps

    expanded = [tuple(float(v) for v in coeff) for coeff in ns_coefficients]
    if len(expanded) != ns_steps:
        raise ValueError(
            f"Expected {ns_steps} coefficient triples, got {len(expanded)}. "
            "Pass a single (a, b, c) tuple or one triple per step."
        )
    for coeff in expanded:
        if len(coeff) != 3:
            raise ValueError(f"Each Newton-Schulz coefficient set must have 3 values, got {len(coeff)}")
    return expanded


def simulate_perturbed_gram_newton_schulz(
    x_eigenvalues: np.ndarray,
    coefficients: Sequence[Tuple[float, float, float]],
    perturbation: float,
    reset_iterations: Optional[Iterable[int]] = None,
) -> dict[str, np.ndarray]:
    """Mirror the restart autotune heuristic used in Dao-AILab/gram-newton-schulz."""
    if perturbation >= 0:
        raise ValueError(f"perturbation must be negative, got {perturbation}")

    q_values: dict[str, np.ndarray] = {}
    x_eigenvalues = np.asarray(x_eigenvalues, dtype=np.float64).copy()
    reset_iterations = set(reset_iterations or ())
    q = np.ones_like(x_eigenvalues)

    with np.errstate(over="ignore", invalid="ignore"):
        for iteration, (a, b, c) in enumerate(coefficients):
            if iteration == 0 or iteration in reset_iterations:
                if iteration != 0:
                    x_eigenvalues *= q
                r = x_eigenvalues**2 + perturbation
                q = np.ones_like(x_eigenvalues)

            z = a + r * (b + r * c)
            q *= z
            r *= z**2
            q_values[f"Q_{iteration}"] = q.astype(np.float64)

    return q_values


def stability_metric(q_values: dict[str, np.ndarray]) -> float:
    def condition(values: np.ndarray) -> float:
        abs_values = np.abs(values)
        return float(abs_values.max() / abs_values.min())

    return max(condition(values) for values in q_values.values())


def find_best_restarts(
    coefficients: Sequence[Tuple[float, float, float]],
    *,
    num_restarts: int = 1,
    most_negative_gram_eigenvalue: float = MOST_NEGATIVE_GRAM_EIGENVALUE,
) -> List[int]:
    """
    Find restart positions with the same heuristic as Dao-AILab's autotuner.

    Restart positions are zero-based iteration indices at which the algorithm
    restarts before executing that iteration. For example, `[2]` means restart
    after finishing the second iteration.
    """
    if num_restarts < 0:
        raise ValueError(f"num_restarts must be non-negative, got {num_restarts}")

    possible_positions = list(range(1, len(coefficients)))
    x_eigenvalues = np.logspace(0, -10, 10000)

    if num_restarts == 0:
        return []
    if num_restarts > len(possible_positions):
        raise ValueError(f"Cannot have {num_restarts} restarts with only {len(coefficients)} Newton-Schulz iterations")

    best_restarts: Optional[List[int]] = None
    best_max_q = float("inf")

    for restart_combo in combinations(possible_positions, num_restarts):
        restart_positions = list(restart_combo)
        q_values = simulate_perturbed_gram_newton_schulz(
            x_eigenvalues,
            coefficients,
            most_negative_gram_eigenvalue,
            reset_iterations=restart_positions,
        )
        max_q = stability_metric(q_values)
        if max_q < best_max_q:
            best_max_q = max_q
            best_restarts = restart_positions

    if best_restarts is None or not np.isfinite(best_max_q) or best_max_q >= 1e8:
        raise ValueError(
            "Failed to find numerically stable Gram Newton-Schulz restarts. "
            "Increase the restart count or provide explicit restart positions."
        )

    return best_restarts


class GramNewtonSchulzOrthogonalizer:
    """Orthogonalize a 2D matrix with Gram Newton-Schulz and optional Quack kernels."""

    def __init__(
        self,
        *,
        ns_coefficients: Sequence[Tuple[float, float, float]],
        ns_epsilon: float = 1e-7,
        ns_use_quack_kernels: bool = True,
        gram_newton_schulz_restart_iterations: Optional[Sequence[int]] = None,
    ) -> None:
        if ns_epsilon <= 0:
            raise ValueError(f"ns_epsilon must be positive, got {ns_epsilon}")

        self.ns_coefficients = tuple(tuple(float(v) for v in coeff) for coeff in ns_coefficients)
        if not self.ns_coefficients:
            raise ValueError("ns_coefficients must contain at least one iteration")
        for coeff in self.ns_coefficients:
            if len(coeff) != 3:
                raise ValueError(f"Each Newton-Schulz coefficient set must have 3 values, got {len(coeff)}")

        self.ns_epsilon = ns_epsilon
        self.ns_use_quack_kernels = ns_use_quack_kernels
        self.reset_iterations = tuple(sorted(set(gram_newton_schulz_restart_iterations or ())))
        self._logged_quack_fallback = False
        self._logged_quack_dtype_fallback = False

    def orthogonalize(self, X: Tensor) -> Tensor:
        """Orthogonalize a 2D matrix and return a tensor with the original shape and dtype."""
        original_shape = X.shape
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.reshape(-1, *X.shape[-2:])

        original_dtype = X.dtype
        X = X.to(torch.float32)

        should_transpose = X.size(-2) > X.size(-1)
        if should_transpose:
            X = X.mT

        X /= X.norm(dim=(-2, -1), keepdim=True).clamp_min(self.ns_epsilon)

        if X.is_cuda:
            # Respect the caller-selected Muon compute dtype on CUDA so dtype
            # sweeps can actually compare bf16 vs fp32 Newton-Schulz behavior.
            compute_dtype = (
                original_dtype if original_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
            )
        else:
            compute_dtype = torch.float32
        X = X.to(compute_dtype)

        if X.size(-2) == X.size(-1):
            X = self._standard_newton_schulz(X)
        else:
            X = self._gram_newton_schulz(X)

        if should_transpose:
            X = X.mT

        return X.to(original_dtype).reshape(original_shape)

    def _select_backend(self, X: Tensor):
        if not self.ns_use_quack_kernels or not X.is_cuda or min(X.size(-2), X.size(-1)) <= SYMMETRIC_KERNEL_TILE_SIZE:
            return _TORCH_BACKEND

        capability = torch.cuda.get_device_capability(X.device)
        if capability[0] < 9:
            return _TORCH_BACKEND
        if not _quack_supports_symmetric_gemm_dtype(X.dtype, capability):
            if not self._logged_quack_dtype_fallback:
                logger.warning_rank0(
                    "Gram Newton-Schulz could not use Quack symmetric GEMM for "
                    f"dtype={X.dtype} on sm_{capability[0]}{capability[1]}; falling back to torch matmuls."
                )
                self._logged_quack_dtype_fallback = True
            return _TORCH_BACKEND

        try:
            return _make_quack_backend()
        except Exception as exc:  # pragma: no cover - exercised only in incomplete CUDA environments
            if not self._logged_quack_fallback:
                logger.warning_rank0(
                    f"Gram Newton-Schulz could not load Quack kernels ({exc!r}); falling back to torch matmuls."
                )
                self._logged_quack_fallback = True
            return _TORCH_BACKEND

    def _gram_newton_schulz(self, X: Tensor) -> Tensor:
        ops = self._select_backend(X)
        R = ops.sym_mm(X, X.mT)

        batch_size = R.size(0)
        identity = (
            torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        )
        Q = None

        for iteration, (a, b, c) in enumerate(self.ns_coefficients):
            if iteration in self.reset_iterations and iteration != 0:
                X = ops.mm(Q, X).contiguous()
                R = ops.sym_mm(X, X.mT)
                Q = None

            Z = ops.sym_baddbmm(R, R, C=R, alpha=c, beta=b)
            if iteration == 0 or iteration in self.reset_iterations:
                Q = Z + a * identity
            else:
                Q = ops.sym_baddbmm(Q, Z, C=Q, beta=a)

            if iteration < len(self.ns_coefficients) - 1 and iteration + 1 not in self.reset_iterations:
                RZ = ops.sym_baddbmm(R, Z, C=R, beta=a)
                R = ops.sym_baddbmm(Z, RZ, C=RZ, beta=a)

        return ops.mm(Q, X)

    def _standard_newton_schulz(self, X: Tensor) -> Tensor:
        ops = self._select_backend(X)
        for a, b, c in self.ns_coefficients:
            gram_matrix = ops.sym_mm(X, X.mT)
            gram_update = ops.sym_baddbmm(gram_matrix, gram_matrix, C=gram_matrix, alpha=c, beta=b)
            X = ops.mm_add(gram_update, X, C=X, beta=a)
        return X
