# Lazy-load alltoall to break circular import:
#   moe/__init__ → alltoall → ops.group_gemm → ops/__init__ → ops.moe → moe/__init__
def __getattr__(name):
    if name in (
        "preprocess",
        "token_pre_all2all",
        "tokens_post_all2all",
        "AllToAllDispatchContext",
        "alltoall_pre_dispatch",
        "alltoall_post_combine",
    ):
        from .alltoall import (  # noqa: PLC0415
            AllToAllDispatchContext,
            alltoall_post_combine,
            alltoall_pre_dispatch,
            preprocess,
            token_pre_all2all,
            tokens_post_all2all,
        )

        globals().update(
            {
                "preprocess": preprocess,
                "token_pre_all2all": token_pre_all2all,
                "tokens_post_all2all": tokens_post_all2all,
                "AllToAllDispatchContext": AllToAllDispatchContext,
                "alltoall_pre_dispatch": alltoall_pre_dispatch,
                "alltoall_post_combine": alltoall_post_combine,
            }
        )
        return globals()[name]
    if name in (
        "DeepEPBuffer",
        "DEEPEP_AVAILABLE",
        "token_pre_dispatch",
        "tokens_post_combine",
        "get_default_buffer",
        "destroy_default_buffer",
    ):
        from .deepep import (  # noqa: PLC0415
            DEEPEP_AVAILABLE,
            DeepEPBuffer,
            destroy_default_buffer,
            get_default_buffer,
            token_pre_dispatch,
            tokens_post_combine,
        )

        globals().update(
            {
                "DeepEPBuffer": DeepEPBuffer,
                "DEEPEP_AVAILABLE": DEEPEP_AVAILABLE,
                "token_pre_dispatch": token_pre_dispatch,
                "tokens_post_combine": tokens_post_combine,
                "get_default_buffer": get_default_buffer,
                "destroy_default_buffer": destroy_default_buffer,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "preprocess",
    "token_pre_all2all",
    "tokens_post_all2all",
    # Unified alltoall dispatch/combine
    "AllToAllDispatchContext",
    "alltoall_pre_dispatch",
    "alltoall_post_combine",
    # DeepEP dispatch/combine
    "DeepEPBuffer",
    "DEEPEP_AVAILABLE",
    "token_pre_dispatch",
    "tokens_post_combine",
    "get_default_buffer",
    "destroy_default_buffer",
]
