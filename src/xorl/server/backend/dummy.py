"""
Dummy Backend - In-process mock for testing.

Replaces DummyModelWorker (270 lines + ZMQ + threads) with ~80 lines.
No ZMQ sockets, no separate processes, no threads. Just returns dummy results.
"""

import logging
import random

from xorl.server.backend.base import Backend


logger = logging.getLogger(__name__)


class DummyBackend(Backend):
    """In-process mock backend for testing. No ZMQ, no threads."""

    def __init__(self, processing_delay: float = 0.0, failure_rate: float = 0.0):
        self.processing_delay = processing_delay
        self.failure_rate = failure_rate
        self._ready = False
        self._step = 0

    async def start(self) -> None:
        self._ready = True
        logger.info("DummyBackend started")

    async def stop(self) -> None:
        self._ready = False
        logger.info("DummyBackend stopped")

    def is_ready(self) -> bool:
        return self._ready

    def _maybe_fail(self, operation: str):
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Simulated failure in {operation} (failure_rate={self.failure_rate})")

    async def forward_backward(
        self, batches, loss_fn="causallm_loss", loss_fn_params=None, model_id=None, routed_experts=None, request_id=None
    ):
        self._maybe_fail("forward_backward")
        valid_tokens = sum(len(batch.get("input_ids", [])) for batch in (batches or []))
        return {
            "total_loss": random.uniform(0.5, 5.0),
            "global_valid_tokens": valid_tokens,
            "num_batches": len(batches or []),
        }

    async def forward(self, batches, loss_fn="causallm_loss", loss_fn_params=None, model_id=None, request_id=None):
        self._maybe_fail("forward")
        valid_tokens = sum(len(batch.get("input_ids", [])) for batch in (batches or []))
        return {
            "total_loss": random.uniform(0.5, 5.0),
            "global_valid_tokens": valid_tokens,
            "num_batches": len(batches or []),
        }

    async def optim_step(
        self, lr, gradient_clip=None, beta1=None, beta2=None, eps=None, model_id=None, request_id=None
    ):
        self._maybe_fail("optim_step")
        self._step += 1
        return {
            "grad_norm": random.uniform(0.1, 2.0),
            "step": self._step,
            "learning_rate": lr,
        }

    async def save_state(
        self, checkpoint_path=None, save_optimizer=True, use_timestamp=False, model_id=None, request_id=None
    ):
        self._maybe_fail("save_state")
        return {"checkpoint_path": checkpoint_path or "/tmp/dummy_ckpt", "success": True}

    async def load_state(self, checkpoint_path=None, load_optimizer=True, model_id=None, request_id=None):
        self._maybe_fail("load_state")
        return {"checkpoint_path": checkpoint_path or "/tmp/dummy_ckpt", "success": True}

    async def save_lora_only(self, lora_path=None, model_id=None, request_id=None):
        self._maybe_fail("save_lora_only")
        return {"lora_path": lora_path or "/tmp/dummy_lora", "success": True}

    async def save_full_weights(
        self, output_path=None, dtype="bfloat16", base_model_path=None, model_id=None, request_id=None
    ):
        self._maybe_fail("save_full_weights")
        return {"output_path": output_path or "/tmp/dummy_weights", "success": True, "num_shards": 1}

    async def sleep(self, request_id=None):
        return {"status": "sleeping", "offload_time": 0.0}

    async def wake_up(self, request_id=None):
        return {"status": "awake", "load_time": 0.0}

    async def sync_inference_weights(
        self, endpoints, master_address="localhost", master_port=0, request_id=None, **kwargs
    ):
        return {
            "success": True,
            "message": "dummy sync",
            "transfer_time": 0.0,
            "total_bytes": 0,
            "num_parameters": 0,
            "num_buckets": 0,
            "endpoint_results": [],
        }

    async def register_adapter(self, model_id="default", lr=1e-5, request_id=None):
        return {"model_id": model_id, "lr": lr, "registered": True}

    async def save_adapter_state(self, model_id="default", path=None, save_optimizer=True, request_id=None):
        return {"success": True}

    async def load_adapter_state(self, model_id="default", path=None, load_optimizer=True, lr=None, request_id=None):
        return {"success": True}

    async def get_adapter_info(self, request_id=None):
        return {"adapters": {}}

    async def kill_session(self, model_id="default", save_checkpoint=True, request_id=None):
        return {"success": True, "message": "dummy session killed"}

    async def health_check(self, request_id=None):
        return {"status": "healthy"}
