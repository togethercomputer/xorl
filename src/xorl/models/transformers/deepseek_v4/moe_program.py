"""Named DeepSeek-V4 production MoE numerical program."""

DSV4_DEEPEP_NATIVE_EXACT_V1 = "deepep_native_exact_v1"


def resolve_dsv4_moe_numerical_program(*, exact: bool, ep_dispatch: str, deepep_native_exact: bool) -> str | None:
    if not exact:
        return None
    if not deepep_native_exact or ep_dispatch != "deepep":
        raise ValueError(
            "Exact DeepSeek-V4 server training requires deepep_native_exact=true "
            "with ep_dispatch='deepep'; the retired post-expert diagnostic is not a "
            "production fallback"
        )
    return DSV4_DEEPEP_NATIVE_EXACT_V1


__all__ = [
    "DSV4_DEEPEP_NATIVE_EXACT_V1",
    "resolve_dsv4_moe_numerical_program",
]
