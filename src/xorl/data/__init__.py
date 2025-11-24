# from ..utils.import_utils import (
#     is_xorl_patch_available,
# )
# from .data_collator import (
#     CollatePipeline,
#     DataCollatorWithPacking,
#     DataCollatorWithPadding,
#     PackingConcatCollator,
#     MakeMicroBatchCollator,
#     TextSequenceShardCollator,
#     UnpackDataCollator,
# )
from .data_loader import DataLoaderBuilder
# from .dataset import (
#     build_energon_dataset,
#     build_interleave_dataset,
#     build_iterative_dataset,
#     build_mapping_dataset,
# )
# from .dummy_dataset import build_dummy_dataset
# from .multimodal.data_collator import (
#     OmniDataCollatorWithPacking,
#     OmniDataCollatorWithPadding,
#     OmniSequenceShardCollator,
# )
# from .multimodal.multimodal_chat_template import build_multimodal_chat_template


# if is_xorl_patch_available():
#     # for internal use only
#     from xorl_patch.data.streaming import (
#         build_byted_dataset,
#         build_multisource_dataset,
#         build_streaming_dataloader,
#         build_vanilla_streaming_dataloader,
#     )
# else:

#     def build_byted_dataset(*args, **kwargs):
#         raise NotImplementedError("build_byted_dataset is not available, please install xorl_patch")

#     def build_multisource_dataset(*args, **kwargs):
#         raise NotImplementedError("build_multisource_dataset is not available, please install xorl_patch")

#     def build_streaming_dataloader(*args, **kwargs):
#         raise NotImplementedError("build_streaming_dataloader is not available, please install xorl_patch")

#     def build_vanilla_streaming_dataloader(*args, **kwargs):
#         raise NotImplementedError("build_vanilla_streaming_dataloader is not available, please install xorl_patch")


__all__ = [
    "DataLoaderBuilder",
]
