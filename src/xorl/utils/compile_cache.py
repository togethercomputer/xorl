import os


def configure_rank_local_compile_caches() -> None:
    """Put compile/autotune caches on rank-local paths under a user-scoped root."""
    cache_user = os.environ.get("USER", "unknown")

    triton_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    triton_base = os.environ.get("TRITON_CACHE_DIR", f"/tmp/triton_cache_{cache_user}")
    triton_cache_dir = os.path.join(triton_base, f"cache_rank{triton_rank}")
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    os.makedirs(triton_cache_dir, exist_ok=True)

    inductor_rank = os.environ.get("LOCAL_RANK", "0")
    inductor_base = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{cache_user}")
    inductor_cache_dir = os.path.join(inductor_base, f"rank{inductor_rank}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    os.makedirs(inductor_cache_dir, exist_ok=True)

    # Quack keeps its own autotune cache separate from Triton's cache.
    quack_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    quack_base = os.environ.get("QUACK_CACHE_DIR", f"/tmp/quack_cache_{cache_user}")
    quack_cache_dir = os.path.join(quack_base, f"cache_rank{quack_rank}")
    os.environ["QUACK_CACHE_DIR"] = quack_cache_dir
    os.makedirs(quack_cache_dir, exist_ok=True)
