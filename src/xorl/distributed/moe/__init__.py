# Legacy R3 routing replay (server path) — kept for backward compat
from .routing_replay import (
    RoutingReplay as R3RoutingReplay,
    get_current_routing_replay,
    set_current_routing_replay,
)

# Lazy-load moe_layer to break circular import:
#   moe/__init__ → moe_layer → ops.group_gemm → ops/__init__ → ops.moe_experts → moe/__init__
def __getattr__(name):
    if name in (
        "EPGroupGemm",
        "EPGroupGemmWithLoRA",
        "preprocess",
        "token_pre_all2all",
        "tokens_post_all2all",
        "AllToAllDispatchContext",
        "alltoall_pre_dispatch",
        "alltoall_post_combine",
    ):
        from .moe_layer import (
            EPGroupGemm,
            EPGroupGemmWithLoRA,
            preprocess,
            token_pre_all2all,
            tokens_post_all2all,
            AllToAllDispatchContext,
            alltoall_pre_dispatch,
            alltoall_post_combine,
        )

        globals().update(
            {
                "EPGroupGemm": EPGroupGemm,
                "EPGroupGemmWithLoRA": EPGroupGemmWithLoRA,
                "preprocess": preprocess,
                "token_pre_all2all": token_pre_all2all,
                "tokens_post_all2all": tokens_post_all2all,
                "AllToAllDispatchContext": AllToAllDispatchContext,
                "alltoall_pre_dispatch": alltoall_pre_dispatch,
                "alltoall_post_combine": alltoall_post_combine,
            }
        )
        return globals()[name]
    if name in ("DeepEPBuffer", "DEEPEP_AVAILABLE", "token_pre_dispatch", "tokens_post_combine", "get_default_buffer", "destroy_default_buffer"):
        from .deepep import (
            DeepEPBuffer,
            DEEPEP_AVAILABLE,
            token_pre_dispatch,
            tokens_post_combine,
            get_default_buffer,
            destroy_default_buffer,
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
    "EPGroupGemm",
    "EPGroupGemmWithLoRA",
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
    # Legacy R3 routing replay (server path)
    "R3RoutingReplay",
    "get_current_routing_replay",
    "set_current_routing_replay",
]
