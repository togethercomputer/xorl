"""Full-weight FP8 compute training helpers."""

_CONFIG_EXPORTS = {
    "UnsupportedFP8ConfigError",
    "enrich_sync_quantization_with_fp8_bf16_islands",
    "extract_nemo_fp8_cfg",
    "is_blackwell_device",
    "merge_fp8_bf16_layer_island_excludes",
    "normalize_fp8_training_config",
    "resolve_fp8_bf16_layer_islands",
    "validate_external_fp8_runtime_config",
    "validate_fp8_blackwell_training_policy",
}
_GROUPED_EXPORTS = {
    "fp8_block_loop_group_gemm_same_mn",
    "fp8_block_loop_group_gemm_same_nk",
    "fp8_deep_gemm_group_gemm_same_nk",
    "fp8_group_gemm_same_mn",
    "fp8_group_gemm_same_nk",
    "fp8_quack_group_gemm_same_mn",
    "fp8_quack_group_gemm_same_nk",
    "fp8_scalar_quack_group_gemm_same_nk",
    "fp8_triton_grouped_group_gemm_same_mn",
    "fp8_triton_grouped_group_gemm_same_nk",
    "validate_fp8_grouped_backend",
}
_LINEAR_EXPORTS = {"FP8CorrectionMode", "FP8Linear"}
_PROFILER_EXPORTS = {"clear_linear_error_profile", "get_linear_error_profile", "write_linear_error_profile"}
_UTILS_EXPORTS = {"DEFAULT_FP8_GROUPED_BACKEND", "inject_fp8_training_into_model", "summarize_fp8_training_model"}


def __getattr__(name: str):
    if name in _CONFIG_EXPORTS:
        from . import config_compat  # noqa: PLC0415

        return getattr(config_compat, name)
    if name in _GROUPED_EXPORTS:
        from . import grouped  # noqa: PLC0415

        return getattr(grouped, name)
    if name in _LINEAR_EXPORTS:
        from . import linear  # noqa: PLC0415

        return getattr(linear, name)
    if name in _PROFILER_EXPORTS:
        from . import profiler  # noqa: PLC0415

        return getattr(profiler, name)
    if name in _UTILS_EXPORTS:
        from . import utils  # noqa: PLC0415

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_FP8_GROUPED_BACKEND",
    "FP8CorrectionMode",
    "FP8Linear",
    "UnsupportedFP8ConfigError",
    "clear_linear_error_profile",
    "enrich_sync_quantization_with_fp8_bf16_islands",
    "extract_nemo_fp8_cfg",
    "fp8_block_loop_group_gemm_same_mn",
    "fp8_block_loop_group_gemm_same_nk",
    "fp8_deep_gemm_group_gemm_same_nk",
    "fp8_group_gemm_same_mn",
    "fp8_group_gemm_same_nk",
    "fp8_quack_group_gemm_same_mn",
    "fp8_quack_group_gemm_same_nk",
    "fp8_scalar_quack_group_gemm_same_nk",
    "fp8_triton_grouped_group_gemm_same_mn",
    "fp8_triton_grouped_group_gemm_same_nk",
    "get_linear_error_profile",
    "inject_fp8_training_into_model",
    "is_blackwell_device",
    "merge_fp8_bf16_layer_island_excludes",
    "normalize_fp8_training_config",
    "resolve_fp8_bf16_layer_islands",
    "summarize_fp8_training_model",
    "validate_external_fp8_runtime_config",
    "validate_fp8_blackwell_training_policy",
    "validate_fp8_grouped_backend",
    "write_linear_error_profile",
]
