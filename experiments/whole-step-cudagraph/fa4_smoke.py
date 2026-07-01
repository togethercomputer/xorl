"""Smoke test: does FA4 (flash_attn.cute) JIT-compile and run under torch nightly 2.14 / cu130?"""

import torch


print("torch", torch.__version__)
try:
    from flash_attn.cute import flash_attn_func

    print("import flash_attn.cute OK")
except Exception as e:
    print("IMPORT FAILED:", type(e).__name__, e)
    raise

dev = "cuda"
B, S, Hn, D = 2, 1024, 16, 128
q = torch.randn(B, S, Hn, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, S, Hn, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, S, Hn, D, device=dev, dtype=torch.bfloat16)
try:
    out = flash_attn_func(q, k, v, causal=True)
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()
    print(f"FA4 FORWARD OK  out.shape={tuple(out.shape)} dtype={out.dtype} mean={out.float().mean().item():.4f}")
except Exception as e:
    import traceback

    print("FA4 FORWARD FAILED:", type(e).__name__)
    traceback.print_exc()
    raise
