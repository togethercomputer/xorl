"""Utility ops for DeepSeek-V4.

``rotate_activation`` applies a scaled Walsh-Hadamard transform to the last
dim of its input. The C4-layer compressor and the DSA indexer use this to
redistribute activation energy before FP8 QAT.

Implementation:

- If ``fast_hadamard_transform`` is available, dispatches to its CUDA
  kernel (the production path).
- Otherwise falls back to a pure-torch FWHT (``_fwht_torch``) that runs on
  any device. The math matches bit-for-bit modulo floating-point
  reduction order, so swapping backends does not change forward outputs
  beyond rounding noise.
"""

import torch


try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform
except ImportError:
    _fast_hadamard_transform = None


def _fwht_torch(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure-torch Fast Walsh-Hadamard Transform on the last dim.

    Computes ``(H_D @ x) * scale`` where ``H_D`` is the unnormalized
    Hadamard matrix recursively defined by ``H_2 = [[1, 1], [1, -1]]``.
    Last dim must be a power of two.
    """
    D = x.size(-1)
    if D & (D - 1) != 0 or D == 0:
        raise ValueError(f"FWHT requires last-dim power-of-2; got {D}")

    h = x.float()
    s = 1
    while s < D:
        # Group pairs at stride s along the last dim.
        h = h.reshape(*h.shape[:-1], D // (2 * s), 2, s)
        a, b = h[..., 0, :], h[..., 1, :]
        h = torch.stack([a + b, a - b], dim=-2)
        h = h.flatten(start_dim=-3)
        s *= 2

    return (h * scale).to(x.dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Scaled Hadamard transform used to redistribute activation energy before
    QAT. Consumed by both the attention compressor and the DSA indexer.

    Falls back to a pure-torch FWHT when the ``fast_hadamard_transform``
    CUDA kernel isn't installed (e.g. CPU / dev images).
    """
    assert x.dtype == torch.bfloat16
    scale = x.size(-1) ** -0.5
    if _fast_hadamard_transform is not None:
        return _fast_hadamard_transform(x, scale=scale)
    return _fwht_torch(x, scale)


def dsv4_kv_qat_enabled(config) -> bool:
    """Return whether the DSv4 KV/indexer FP8-QAT path should run.

    Miles gates this on Megatron's ``config.fp8 is not None``. Xorl's HF-style
    config uses ``quantization_config`` for the same intent, while still
    accepting a direct ``fp8`` attribute when present.
    """
    if getattr(config, "fp8", None) is not None:
        return True

    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        return False

    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()

    if not isinstance(quantization_config, dict):
        return False

    return any("fp8" in str(key).lower() or "fp8" in str(value).lower() for key, value in quantization_config.items())
