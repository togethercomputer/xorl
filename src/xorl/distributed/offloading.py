import enum
from collections import OrderedDict
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.autograd.graph import saved_tensors_hooks


class OffloadPolicy(enum.Enum):
    OFFLOAD = 0
    KEEP_ON_GPU = 1
    IGNORE = 2


def _is_weight(tensor: torch.Tensor) -> bool:
    return type(tensor.grad_fn).__name__ == "TBackward0"


class custom_save_on_cpu(saved_tensors_hooks):
    def __init__(self, gpu_limit_in_gb: float = 0, pin_memory: bool = False, min_offload_size: int = 1024) -> None:
        self.cur_gpu_ram_in_mb = 0.0

        def pack_to_cpu(tensor: torch.Tensor) -> Tuple[OffloadPolicy, torch.device, torch.Tensor]:
            tensor_num_bytes = tensor.element_size() * tensor.nelement()
            if _is_weight(tensor) or tensor_num_bytes <= min_offload_size:
                return (OffloadPolicy.IGNORE, tensor.device, tensor)

            if self.cur_gpu_ram_in_mb < gpu_limit_in_gb * 1024:
                self.cur_gpu_ram_in_mb += tensor_num_bytes / 1024 / 1024
                return (OffloadPolicy.KEEP_ON_GPU, tensor.device, tensor)

            if not pin_memory:
                return (OffloadPolicy.OFFLOAD, tensor.device, tensor.cpu())

            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (OffloadPolicy.OFFLOAD, tensor.device, packed)

        def unpack_from_cpu(packed: Tuple[OffloadPolicy, torch.device, torch.Tensor]) -> torch.Tensor:
            offload_policy, device, tensor = packed

            if offload_policy == OffloadPolicy.IGNORE:
                return tensor
            elif offload_policy == OffloadPolicy.KEEP_ON_GPU:
                tensor_num_bytes = tensor.element_size() * tensor.nelement()
                self.cur_gpu_ram_in_mb -= tensor_num_bytes / 1024 / 1024
                return tensor
            else:
                return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


class ActivationOffloader(saved_tensors_hooks):
    """Stream-overlapped activation offloading with backward prefetching."""

    def __init__(
        self,
        gpu_limit_in_gb: float = 0.0,
        min_offload_size: int = 1024,
        overlap_budget_gb: float = 2.0,
        prefetch_count: int = 2,
    ) -> None:
        self._gpu_limit_bytes = int(gpu_limit_in_gb * (1 << 30))
        self._min_offload_size = min_offload_size
        self._stash_budget_bytes = int(overlap_budget_gb * (1 << 30)) // 2
        self._prefetch_budget_bytes = int(overlap_budget_gb * (1 << 30)) // 2
        self._prefetch_count = prefetch_count

        self._offload_stream: Optional[torch.cuda.Stream] = None
        self._device: Optional[torch.device] = None

        self._bytes_offloaded = 0
        self._bytes_kept_on_gpu = 0

        self._next_id = 0
        self._gpu_used_bytes = 0
        self._tracker: Dict[int, Tuple[torch.Tensor, bool]] = {}
        self._offloaded_ids: List[int] = []
        self._fwd_stash: "OrderedDict[int, Tuple[torch.Tensor, torch.cuda.Event]]" = OrderedDict()
        self._fwd_stash_bytes = 0
        self._prefetch_cache: Dict[int, Tuple[torch.Tensor, torch.cuda.Event]] = {}
        self._prefetch_cache_bytes = 0
        self._bwd_schedule: Optional[List[int]] = None
        self._prefetch_cursor = 0

        def pack(tensor: torch.Tensor) -> int:
            return self._pack_impl(tensor)

        def unpack(tensor_id: int) -> torch.Tensor:
            return self._unpack_impl(tensor_id)

        super().__init__(pack, unpack)

    def __enter__(self):
        self._reset()
        return super().__enter__()

    def consume_stats(self) -> Dict[str, int]:
        stats = {
            "bytes_offloaded": self._bytes_offloaded,
            "bytes_kept_on_gpu": self._bytes_kept_on_gpu,
        }
        self._bytes_offloaded = 0
        self._bytes_kept_on_gpu = 0
        return stats

    def _ensure_stream(self, device: torch.device) -> None:
        if self._offload_stream is None or self._device != device:
            self._device = device
            self._offload_stream = torch.cuda.Stream(device=device)

    def _reset(self) -> None:
        if self._offload_stream is not None and self._prefetch_cache:
            torch.cuda.current_stream(self._device).wait_stream(self._offload_stream)
        self._next_id = 0
        self._gpu_used_bytes = 0
        self._tracker.clear()
        self._offloaded_ids.clear()
        self._fwd_stash.clear()
        self._fwd_stash_bytes = 0
        self._prefetch_cache.clear()
        self._prefetch_cache_bytes = 0
        self._bwd_schedule = None
        self._prefetch_cursor = 0

    def _evict_fwd_stash(self) -> None:
        compute = torch.cuda.current_stream(self._device)
        while self._fwd_stash and self._fwd_stash_bytes > self._stash_budget_bytes:
            _tid, (tensor, event) = self._fwd_stash.popitem(last=False)
            self._fwd_stash_bytes -= tensor.element_size() * tensor.nelement()
            compute.wait_event(event)

    def _pack_impl(self, tensor: torch.Tensor) -> int:
        tensor_id = self._next_id
        self._next_id += 1

        tensor_num_bytes = tensor.element_size() * tensor.nelement()
        if _is_weight(tensor) or tensor_num_bytes <= self._min_offload_size:
            self._tracker[tensor_id] = (tensor, False)
            return tensor_id

        if self._gpu_used_bytes + tensor_num_bytes <= self._gpu_limit_bytes:
            self._gpu_used_bytes += tensor_num_bytes
            self._bytes_kept_on_gpu += tensor_num_bytes
            self._tracker[tensor_id] = (tensor, False)
            return tensor_id

        self._ensure_stream(tensor.device)
        compute = torch.cuda.current_stream(tensor.device)
        self._offload_stream.wait_stream(compute)

        with torch.cuda.stream(self._offload_stream):
            cpu_tensor = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=True,
                device="cpu",
            )
            cpu_tensor.copy_(tensor, non_blocking=True)
        event = self._offload_stream.record_event()

        self._fwd_stash[tensor_id] = (tensor, event)
        self._fwd_stash_bytes += tensor_num_bytes
        self._evict_fwd_stash()

        self._bytes_offloaded += tensor_num_bytes
        self._tracker[tensor_id] = (cpu_tensor, True)
        self._offloaded_ids.append(tensor_id)
        return tensor_id

    def _unpack_impl(self, tensor_id: int) -> torch.Tensor:
        data, was_offloaded = self._tracker[tensor_id]

        if not was_offloaded:
            tensor_num_bytes = data.element_size() * data.nelement()
            self._gpu_used_bytes = max(0, self._gpu_used_bytes - tensor_num_bytes)
            del self._tracker[tensor_id]
            return data

        if tensor_id in self._fwd_stash:
            gpu_tensor, _event = self._fwd_stash.pop(tensor_id)
            self._fwd_stash_bytes -= gpu_tensor.element_size() * gpu_tensor.nelement()
            del self._tracker[tensor_id]
            self._trigger_prefetch()
            return gpu_tensor

        compute = torch.cuda.current_stream(self._device)

        if tensor_id in self._prefetch_cache:
            gpu_tensor, prefetch_event = self._prefetch_cache.pop(tensor_id)
            self._prefetch_cache_bytes -= gpu_tensor.element_size() * gpu_tensor.nelement()
            compute.wait_event(prefetch_event)
            del self._tracker[tensor_id]
            self._trigger_prefetch()
            return gpu_tensor

        gpu_tensor = torch.empty(data.size(), dtype=data.dtype, device=self._device)
        self._offload_stream.wait_stream(compute)
        with torch.cuda.stream(self._offload_stream):
            gpu_tensor.copy_(data, non_blocking=True)
        compute.wait_stream(self._offload_stream)

        del self._tracker[tensor_id]
        self._trigger_prefetch()
        return gpu_tensor

    def _trigger_prefetch(self) -> None:
        if self._prefetch_count <= 0 or not self._offloaded_ids:
            return

        if self._bwd_schedule is None:
            self._bwd_schedule = list(reversed(self._offloaded_ids))
            self._prefetch_cursor = 0

        compute = torch.cuda.current_stream(self._device)
        issued = 0
        while issued < self._prefetch_count and self._prefetch_cursor < len(self._bwd_schedule):
            if self._prefetch_cache_bytes >= self._prefetch_budget_bytes:
                break

            tensor_id = self._bwd_schedule[self._prefetch_cursor]
            self._prefetch_cursor += 1

            if tensor_id not in self._tracker:
                continue
            if tensor_id in self._prefetch_cache or tensor_id in self._fwd_stash:
                continue
            cpu_data, was_offloaded = self._tracker[tensor_id]
            if not was_offloaded:
                continue

            tensor_num_bytes = cpu_data.element_size() * cpu_data.nelement()
            gpu_tensor = torch.empty(cpu_data.size(), dtype=cpu_data.dtype, device=self._device)
            self._offload_stream.wait_stream(compute)
            with torch.cuda.stream(self._offload_stream):
                gpu_tensor.copy_(cpu_data, non_blocking=True)
            event = self._offload_stream.record_event()

            self._prefetch_cache[tensor_id] = (gpu_tensor, event)
            self._prefetch_cache_bytes += tensor_num_bytes
            issued += 1


def build_activation_offloading_context(
    enable_activation_offload: bool = False,
    enable_gradient_checkpointing: bool = False,
    activation_gpu_limit: Optional[float] = 0.0,
    prefetch_count: int = 2,
) -> Tuple[Union["saved_tensors_hooks", "nullcontext"], Union["saved_tensors_hooks", "nullcontext"]]:
    model_fwd_context, model_bwd_context = nullcontext(), nullcontext()
    activation_gpu_limit = 0.0 if activation_gpu_limit is None else activation_gpu_limit
    if enable_activation_offload:
        if enable_gradient_checkpointing:
            model_fwd_context = ActivationOffloader(gpu_limit_in_gb=0.0, prefetch_count=prefetch_count)
            model_bwd_context = ActivationOffloader(gpu_limit_in_gb=activation_gpu_limit, prefetch_count=0)
        else:
            model_fwd_context = ActivationOffloader(
                gpu_limit_in_gb=activation_gpu_limit,
                prefetch_count=prefetch_count,
            )

    return model_fwd_context, model_bwd_context
