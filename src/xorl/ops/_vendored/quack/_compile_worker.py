# Copyright (c) 2025, Tri Dao.
# Persistent subprocess worker for parallel autotuning pre-compilation.
# Receives length-prefixed MessagePack tasks on stdin, creates FakeTensors
# matching the parent's tensor metadata, and compiles with COMPILE_ONLY=True.
# Stays alive to process multiple configs (amortizes import overhead).

import importlib
import re
import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from . import cache_utils
from ._worker_protocol import recv_message, send_message


cache_utils.COMPILE_ONLY = True

_dtype_map = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}

_ALLOWED_MODULE_PREFIXES = ("quack.", "xorl.ops.quack.")
_QUALNAME_PART_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_compile_function(fn_module: str, fn_qualname: str):
    """Resolve an autotune target only from the Quack kernel namespace."""
    if not isinstance(fn_module, str) or not fn_module.startswith(_ALLOWED_MODULE_PREFIXES):
        raise ValueError("Compile worker target must be a Quack module")
    parts = fn_qualname.split(".") if isinstance(fn_qualname, str) else []
    if not parts or any(not _QUALNAME_PART_RE.fullmatch(part) or part.startswith("__") for part in parts):
        raise ValueError("Compile worker target must use a safe qualified name")

    obj = importlib.import_module(fn_module)
    for part in parts:
        obj = getattr(obj, part)
    obj = getattr(obj, "fn", obj)
    if not callable(obj):
        raise TypeError("Compile worker target must be callable")
    return obj


def _make_fake_tensor(meta):
    shape = meta["shape"]
    stride = meta["stride"]
    dtype = _dtype_map[meta["dtype"]]
    return torch.empty_strided(shape, stride, dtype=dtype, device="cuda")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m xorl.ops.quack._compile_worker MODULE QUALNAME")
    fn = _resolve_compile_function(sys.argv[1], sys.argv[2])

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # Signal ready
    send_message(stdout, "READY")

    while True:
        payload = recv_message(stdin)
        if payload is None:
            break

        tensor_meta = payload["tensor_meta"]
        kwargs = payload["kwargs"]
        config_kwargs = payload["config_kwargs"]

        with FakeTensorMode():
            fake_args = []
            for meta in tensor_meta:
                if isinstance(meta, dict) and "shape" in meta:
                    fake_args.append(_make_fake_tensor(meta))
                else:
                    fake_args.append(meta)
            try:
                fn(*fake_args, **kwargs, **config_kwargs)
                send_message(stdout, "OK")
            except Exception as e:
                send_message(stdout, f"ERR:{e}")


if __name__ == "__main__":
    main()
