# Keep routing_replay eager (no cycles)
from .routing_replay import (
    RoutingReplay,
    get_current_routing_replay,
    set_current_routing_replay,
)

# Lazy-load moe_layer to break circular import:
#   moe/__init__ → moe_layer → ops.group_gemm → ops/__init__ → ops.fused_moe → moe/__init__
def __getattr__(name):
    if name in (
        "EPGroupGemm",
        "EPGroupGemmWithLoRA",
        "preprocess",
        "token_pre_all2all",
        "tokens_post_all2all",
    ):
        from .moe_layer import (
            EPGroupGemm,
            EPGroupGemmWithLoRA,
            preprocess,
            token_pre_all2all,
            tokens_post_all2all,
        )

        globals().update(
            {
                "EPGroupGemm": EPGroupGemm,
                "EPGroupGemmWithLoRA": EPGroupGemmWithLoRA,
                "preprocess": preprocess,
                "token_pre_all2all": token_pre_all2all,
                "tokens_post_all2all": tokens_post_all2all,
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
    # Routing replay for R3
    "RoutingReplay",
    "get_current_routing_replay",
    "set_current_routing_replay",
]
