from .async_ulysses import (
    async_ulysses_output_projection,
    async_ulysses_qkv_projection,
    divide_qkv_linear_bias,
    divide_qkv_linear_weight,
)
from .comm import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_ringattn_group,
    get_ringattn_rank,
    get_ringattn_world_size,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    get_unified_sequence_parallel_group,
    get_unified_sequence_parallel_rank,
    get_unified_sequence_parallel_world_size,
    init_sequence_parallel,
    set_data_parallel_group,
    set_ringattn_group,
    set_ulysses_sequence_parallel_group,
    set_unified_sequence_parallel_group,
)
from .data import (
    gather_outputs,
    sequence_parallel_preprocess,
    slice_input_tensor,
    slice_input_tensor_scale_grad,
    slice_position_embedding,
)
from .strategy import (
    CPStrategy,
    HybridUlyssesRingStrategy,
    NoopStrategy,
    RingAttentionStrategy,
    UlyssesAsyncStrategy,
    UlyssesSyncStrategy,
    get_cp_strategy,
)
from .ulysses import (
    all_to_all_images,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)
from .utils import pad_tensor, unpad_tensor, vlm_images_a2a_meta


def __getattr__(name):
    if name == "ring_flash_attention_forward":
        from .ring_attention import ring_flash_attention_forward  # noqa: PLC0415

        return ring_flash_attention_forward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "init_sequence_parallel",
    "set_data_parallel_group",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "set_ulysses_sequence_parallel_group",
    "get_ulysses_sequence_parallel_world_size",
    "get_ulysses_sequence_parallel_rank",
    "get_ulysses_sequence_parallel_group",
    "set_ringattn_group",
    "get_ringattn_group",
    "get_ringattn_rank",
    "get_ringattn_world_size",
    "set_unified_sequence_parallel_group",
    "get_unified_sequence_parallel_group",
    "get_unified_sequence_parallel_rank",
    "get_unified_sequence_parallel_world_size",
    "slice_input_tensor",
    "slice_input_tensor_scale_grad",
    "slice_position_embedding",
    "sequence_parallel_preprocess",
    "gather_heads_scatter_seq",
    "gather_seq_scatter_heads",
    "all_to_all_images",
    "gather_outputs",
    "vlm_images_a2a_meta",
    "pad_tensor",
    "unpad_tensor",
    "async_ulysses_qkv_projection",
    "async_ulysses_output_projection",
    "divide_qkv_linear_weight",
    "divide_qkv_linear_bias",
    "CPStrategy",
    "NoopStrategy",
    "UlyssesSyncStrategy",
    "UlyssesAsyncStrategy",
    "RingAttentionStrategy",
    "HybridUlyssesRingStrategy",
    "ring_flash_attention_forward",
    "get_cp_strategy",
]
