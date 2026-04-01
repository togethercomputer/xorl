from __future__ import annotations

# Adapted from flash-linear-attention/fla/utils.py.
# Portions of this file are adapted from flash-linear-attention, Copyright (c) 2023-2025 Songlin Yang, licensed under the MIT License.
import contextlib
import functools
import inspect
import os
from collections.abc import Callable
from typing import Any

import torch
import triton


FLA_CACHE_RESULTS = os.getenv("FLA_CACHE_RESULTS", "1") == "1"
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


def tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    last_args: tuple[Any, ...] | None = None
    last_kwargs: dict[str, Any] | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result
        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                same_args = all(a is b for a, b in zip(args, last_args, strict=False))
                same_kwargs = all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items())
                if same_args and same_kwargs:
                    return last_result
        last_result = fn(*args, **kwargs)
        last_args = args
        last_kwargs = kwargs
        return last_result

    return wrapper


def _device_context(device: torch.device | None) -> contextlib.AbstractContextManager[None]:
    if device is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.cuda.device(device.index or 0)


def input_guard(
    fn: Callable[..., Any] | None = None,
    *,
    no_guard_contiguous: bool | list[str] = False,
) -> Callable[..., Any]:
    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(inner)
        param_names = list(signature.parameters.keys())

        @functools.wraps(inner)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            skip_params = set(no_guard_contiguous) if isinstance(no_guard_contiguous, list) else set()
            processed_args: list[Any] = []
            device: torch.device | None = None

            for index, arg in enumerate(args):
                name = param_names[index] if index < len(param_names) else f"arg_{index}"
                if isinstance(arg, torch.Tensor):
                    device = device or arg.device
                    if no_guard_contiguous is True or name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            processed_kwargs: dict[str, Any] = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    device = device or value.device
                    if no_guard_contiguous is True or key in skip_params:
                        processed_kwargs[key] = value
                    else:
                        processed_kwargs[key] = value.contiguous()
                else:
                    processed_kwargs[key] = value

            with _device_context(device):
                return inner(*processed_args, **processed_kwargs)

        return wrapped

    if fn is not None:
        return decorator(fn)
    return decorator


def _cuda_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


IS_GATHER_SUPPORTED = hasattr(triton.language, "gather")
IS_AMD = False
IS_NVIDIA_HOPPER = torch.cuda.is_available() and _cuda_capability()[0] >= 9
IS_NVIDIA_BLACKWELL = torch.cuda.is_available() and _cuda_capability()[0] == 10
IS_TMA_SUPPORTED = False
USE_CUDA_GRAPH = False


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    del arch, tensor_idx
    return False


device_type = "cuda" if torch.cuda.is_available() else "cpu"
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device_type)
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device_type)
