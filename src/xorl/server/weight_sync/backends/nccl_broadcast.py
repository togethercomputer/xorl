"""NCCL broadcast backend for training-to-inference weight transfer.

Provides two layers:

- :class:`NCCLWeightSynchronizer` — low-level NCCL group management and tensor
  broadcast.  Handles TCPStore rendezvous, bucket-wise ``dist.broadcast``,
  and HTTP coordination with SGLang endpoints.

- :class:`NCCLBroadcastBackend` — thin :class:`~backends.base.WeightTransportBackend`
  adapter that wraps ``NCCLWeightSynchronizer`` so the handler can use it through
  the pluggable backend interface.
"""

import inspect
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import requests
import torch
import torch.distributed as dist
from torch.distributed import PrefixStore, TCPStore
from torch.distributed.distributed_c10d import (
    Backend,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
)


logger = logging.getLogger(__name__)

_DEFAULT_FP8_CPU_H2D_CHUNK_BYTES = 8 * 1024 * 1024

# Reusable session for HTTP connection pooling
_http_session: Optional[requests.Session] = None


def _nccl_broadcast_chunk_bytes() -> int:
    value = os.environ.get("XORL_WEIGHT_SYNC_NCCL_CHUNK_BYTES")
    if value is None or value == "":
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning("Invalid XORL_WEIGHT_SYNC_NCCL_CHUNK_BYTES=%r; disabling NCCL chunking", value)
        return 0


def _fp8_cpu_h2d_chunk_bytes() -> int:
    value = os.environ.get("XORL_WEIGHT_SYNC_FP8_CPU_H2D_CHUNK_BYTES")
    if value is None or value == "":
        return _DEFAULT_FP8_CPU_H2D_CHUNK_BYTES
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning(
            "Invalid XORL_WEIGHT_SYNC_FP8_CPU_H2D_CHUNK_BYTES=%r; using default %d",
            value,
            _DEFAULT_FP8_CPU_H2D_CHUNK_BYTES,
        )
        return _DEFAULT_FP8_CPU_H2D_CHUNK_BYTES


def _sglang_load_format() -> str | None:
    value = os.environ.get("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _receiver_load_format(load_format: str | None) -> str | None:
    chunk_bytes = _nccl_broadcast_chunk_bytes()
    if load_format == "direct" and chunk_bytes > 0:
        return f"direct_chunked:{chunk_bytes}"
    if load_format == "flattened_bucket" and chunk_bytes > 0:
        return f"flattened_bucket_chunked:{chunk_bytes}"
    return load_format


def _is_direct_load_format(load_format: str | None) -> bool:
    return load_format == "direct" or (load_format is not None and load_format.startswith("direct_chunked:"))


def _is_flattened_bucket_load_format(load_format: str | None) -> bool:
    return load_format == "flattened_bucket" or (
        load_format is not None and load_format.startswith("flattened_bucket_chunked:")
    )


def _is_hybrid_flattened_load_format(load_format: str | None) -> bool:
    return load_format == "hybrid_flattened"


def _requires_stacked_qwen_loader(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            ".self_attn.q_proj.",
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
            ".mlp.gate_proj.",
            ".mlp.up_proj.",
        )
    )


def _reinit_per_bucket() -> bool:
    return os.environ.get("XORL_WEIGHT_SYNC_REINIT_PER_BUCKET", "0").strip().lower() in {"1", "true", "yes", "on"}


def _wait_after_receiver() -> bool:
    return os.environ.get("XORL_WEIGHT_SYNC_WAIT_AFTER_RECEIVER", "0").strip().lower() in {"1", "true", "yes", "on"}


def _use_two_phase_update() -> bool:
    return os.environ.get("XORL_WEIGHT_SYNC_NCCL_TWO_PHASE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _sync_after_waited_broadcast() -> bool:
    return os.environ.get("XORL_WEIGHT_SYNC_NCCL_SYNC_AFTER_WAIT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _hold_receiver_fenced_refs() -> bool:
    return os.environ.get("XORL_WEIGHT_SYNC_HOLD_RECEIVER_FENCED_REFS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_http_session() -> requests.Session:
    """Get or create a reusable HTTP session with connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        # Increase pool size for multiple endpoints
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0,
        )
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
    return _http_session


@dataclass
class EndpointInfo:
    """Information about an inference endpoint."""

    host: str
    port: int
    world_size: int  # tensor_parallel_size for this endpoint


def _endpoint_update_result(endpoint: EndpointInfo, body: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "host": endpoint.host,
        "port": endpoint.port,
        "endpoint": f"{endpoint.host}:{endpoint.port}",
        "success": body.get("success", False),
        "message": body.get("message", ""),
    }
    cache_epoch = body["cache_epoch"] if "cache_epoch" in body else body.get("cache_version")
    if cache_epoch is not None:
        result["cache_epoch"] = cache_epoch
    for key in ("fp8_kv_cache_postprocess_ran", "fp8_kv_cache_static_scales_updated"):
        if key in body:
            result[key] = body[key]
    return result


@dataclass
class SyncResult:
    """Result of weight synchronization."""

    success: bool
    message: str
    transfer_time: float = 0.0
    total_bytes: int = 0
    num_parameters: int = 0
    num_buckets: int = 0
    endpoint_results: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.endpoint_results is None:
            self.endpoint_results = []


class NCCLWeightSynchronizer:
    """
    Handles weight synchronization from training to inference via NCCL.

    This class manages the entire weight transfer process:
    1. Initialize NCCL process groups on training and inference sides
    2. Transfer weights in buckets via NCCL broadcast
    3. Cleanup process groups
    """

    def __init__(
        self,
        endpoints: List[EndpointInfo],
        master_address: str = "localhost",
        master_port: int = 0,
        group_name: str = "weight_sync_group",
        buffer_size_mb: int = 1024,
        device: str = "cuda:0",
        run_post_process_weights: bool = False,
        fp8_kv_cache_enabled: bool = False,
        fp8_kv_cache_postprocess_required: bool = False,
        fp8_kv_cache_static_scales: bool = False,
    ):
        """
        Initialize the weight synchronizer.

        Args:
            endpoints: List of inference endpoints to sync weights to
            master_address: Address for NCCL rendezvous (training server)
            master_port: Port for NCCL rendezvous (0 selects an ephemeral port)
            group_name: Name of the NCCL process group
            buffer_size_mb: Size of each transfer bucket in MB
            device: Device to use for NCCL operations
        """
        self.endpoints = endpoints
        self.master_address = master_address
        self.master_port = master_port
        self.group_name = group_name
        self.buffer_size_bytes = buffer_size_mb * 1024 * 1024
        self.device = device
        self.run_post_process_weights = run_post_process_weights
        self.fp8_kv_cache_enabled = fp8_kv_cache_enabled
        self.fp8_kv_cache_postprocess_required = fp8_kv_cache_postprocess_required
        self.fp8_kv_cache_static_scales = fp8_kv_cache_static_scales

        # Calculate world size: 1 (training) + sum of all endpoint world_sizes
        self.world_size = 1 + sum(ep.world_size for ep in endpoints)

        # Process group (initialized during sync)
        self.process_group: Optional[dist.ProcessGroup] = None
        self._training_raw_store = None
        self._training_prefix_store = None
        self._active_master_port = master_port
        self._receiver_fenced_refs: list[tuple[Any, torch.Tensor, torch.Tensor | None]] = []

        logger.info(
            f"NCCLWeightSynchronizer initialized: "
            f"endpoints={len(endpoints)}, world_size={self.world_size}, "
            f"master={master_address}:{master_port}, two_phase={_use_two_phase_update()}"
        )

    # ========================================================================
    # NCCL process group management
    # ========================================================================

    @contextmanager
    def _without_torchelastic_agent_store(self):
        """Temporarily disable the elastic agent store override for custom groups."""
        old_agent_store = os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        try:
            yield
        finally:
            if old_agent_store is not None:
                os.environ["TORCHELASTIC_USE_AGENT_STORE"] = old_agent_store

    def _create_training_store(self) -> None:
        """Create and retain the rendezvous store used by the training process group."""
        self._cleanup_training_store()

        requested_port = self.master_port
        logger.info(f"[Training] Creating TCPStore (requested_port={requested_port}, is_master=True)...")

        with self._without_torchelastic_agent_store():
            # wait_for_workers=False: master must start listening without blocking.
            # /init_weights_update_group is only sent to inference endpoints later,
            # in init_inference (started after this call returns), and the actual
            # NCCL rendezvous is completed inside _init_training_process_group via
            # _new_process_group_helper. With the default wait_for_workers=True,
            # construction blocks waiting for workers that cannot connect yet,
            # deadlocking sync_inference_weights.
            raw_store = TCPStore(
                host_name=self.master_address,
                port=requested_port,
                world_size=self.world_size,
                is_master=True,
                timeout=default_pg_timeout,
                wait_for_workers=False,
            )

        self._training_raw_store = raw_store
        self._active_master_port = raw_store.port
        self._training_prefix_store = PrefixStore(self.group_name, raw_store)
        logger.info(
            f"[Training] Rendezvous store ready: tcp://{self.master_address}:{self._active_master_port}, "
            f"group_name={self.group_name}"
        )

    def _cleanup_training_store(self) -> None:
        """Drop references to the rendezvous store so the TCP listener can be reclaimed."""
        raw_store = self._training_raw_store
        self._training_prefix_store = None
        self._training_raw_store = None
        self._active_master_port = self.master_port

        close = getattr(raw_store, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning(f"[Training] Failed to close rendezvous store cleanly: {exc}")

    def _init_training_process_group(self) -> dist.ProcessGroup:
        """
        Initialize NCCL process group on the training side (rank 0).

        Uses TCPStore + _new_process_group_helper to create a separate process group
        that doesn't interfere with existing torch.distributed.

        Returns:
            The initialized process group
        """

        # Important: Only set NCCL_CUMEM_ENABLE=0
        os.environ["NCCL_CUMEM_ENABLE"] = "0"

        rank = 0  # Training is always rank 0

        logger.info(
            f"[Training] Initializing process group: tcp://{self.master_address}:{self._active_master_port}, "
            f"rank={rank}, world_size={self.world_size}, group_name={self.group_name}"
        )

        # Check existing distributed state
        logger.info(f"[Training] Existing dist.is_initialized(): {dist.is_initialized()}")
        if dist.is_initialized():
            logger.info(f"[Training] Existing default world_size: {dist.get_world_size()}")
            logger.info(f"[Training] Existing default rank: {dist.get_rank()}")

        backend_obj = Backend("nccl")
        timeout = default_pg_timeout

        if self._training_prefix_store is None:
            self._create_training_store()

        logger.info("[Training] TCPStore created, creating process group...")

        # Handle different PyTorch versions by inspecting the actual signature

        _pg_params = inspect.signature(_new_process_group_helper).parameters
        if "backend_options" in _pg_params:
            pg_options_param_name = "backend_options"
        else:
            pg_options_param_name = "pg_options"

        # Set CUDA device and pass device_id for proper NCCL comm initialization
        torch.cuda.set_device(self.device)
        device_id = torch.device(self.device)

        with self._without_torchelastic_agent_store():
            pg, _ = _new_process_group_helper(
                self.world_size,
                rank,
                [],
                backend_obj,
                self._training_prefix_store,
                group_name=self.group_name,
                **{pg_options_param_name: None},
                timeout=timeout,
                device_id=device_id,
            )

        _world.pg_group_ranks[pg] = {i: i for i in range(self.world_size)}

        torch.cuda.synchronize()
        logger.info(f"[Training] Process group '{self.group_name}' initialized successfully")

        return pg

    def init_nccl_group(self) -> bool:
        """
        Initialize NCCL process group with all inference endpoints.

        Pattern: start inference endpoint init in background, sleep 2s for
        HTTP round-trip, then init training side (rank 0) which completes
        the NCCL rendezvous.

        Returns:
            True if initialization succeeded, False otherwise
        """
        logger.info("=" * 70)
        logger.info("Initializing NCCL process groups...")
        logger.info("=" * 70)
        logger.info(f"[Training] World size: {self.world_size} (1 training + {self.world_size - 1} inference)")

        training_error = None
        inference_error = None
        init_results = []

        def init_inference():
            nonlocal init_results, inference_error
            try:
                init_results = self._init_inference_endpoints()
            except Exception as e:
                inference_error = e

        # Pattern from working example:
        # 1. Start inference in background thread (will block waiting for rank 0)
        # 2. Wait a bit for the HTTP request to reach inference endpoints
        # 3. Run training (rank 0) in main thread - this completes NCCL rendezvous
        # 4. Join inference thread after training completes

        try:
            self._create_training_store()
        except Exception as exc:
            logger.error(f"[Training] Failed to create rendezvous store: {exc}")
            return False

        logger.info(f"[Training] NCCL rendezvous: tcp://{self.master_address}:{self._active_master_port}")
        logger.info("[Training] Starting inference endpoint initialization in background...")
        inference_thread = Thread(target=init_inference)
        inference_thread.start()

        # Give inference endpoints time to:
        # 1. Receive the HTTP request
        # 2. Process the request and start their NCCL init
        # 3. Call rendezvous and wait for rank 0
        # This needs to be long enough for the HTTP round-trip + processing
        logger.info("[Training] Waiting 2 seconds for inference endpoints to start NCCL init...")
        time.sleep(2.0)

        # Initialize training process group (rank 0) in MAIN thread
        # This will block until NCCL rendezvous completes with all ranks
        logger.info("[Training] Initializing training side process group (rank 0)...")
        try:
            self.process_group = self._init_training_process_group()
        except Exception as e:
            training_error = e

        # Wait for inference thread to complete
        inference_thread.join()

        logger.info("[Training] Both sides completed NCCL initialization")

        # Check for errors
        if training_error:
            logger.error(f"[Training] Process group init failed: {training_error}")
            return False

        if inference_error:
            logger.error(f"[Training] Inference endpoint init failed: {inference_error}")
            if self.process_group:
                dist.destroy_process_group(self.process_group)
                self.process_group = None
            return False

        if not all(r["success"] for r in init_results):
            failed = [r for r in init_results if not r["success"]]
            logger.error(f"[Training] Some inference endpoints failed: {failed}")
            if self.process_group:
                dist.destroy_process_group(self.process_group)
                self.process_group = None
            return False

        # Success - all process groups initialized
        logger.info("=" * 70)
        logger.info("[Training] NCCL process group initialized successfully!")
        logger.info(f"[Training] Training rank 0 connected to {len(self.endpoints)} inference endpoint(s)")
        for r in init_results:
            logger.info(f"  - {r['endpoint']}: rank_offset={r.get('rank_offset', 'N/A')}")
        logger.info("=" * 70)

        return True

    def destroy_nccl_group(self) -> None:
        """Destroy NCCL process group on both training and inference sides."""
        logger.info("Destroying NCCL process groups...")

        try:
            self._destroy_inference_endpoints()
        except Exception as e:
            logger.error(f"Failed to destroy inference endpoints: {e}")

        if self.process_group:
            try:
                # Use the non-cooperative abort() rather than
                # dist.destroy_process_group(), which calls pg.shutdown() and
                # then waits for the inference-side comm to finalize. The
                # inference side aborted its comm immediately on receiving
                # /destroy_weights_update_group, so a cooperative shutdown
                # here hangs until the engine timeout fires.
                self.process_group.abort()
            except Exception as e:
                logger.warning(f"process_group.abort() failed ({e}); falling back to destroy_process_group")
                try:
                    dist.destroy_process_group(self.process_group)
                except Exception as e2:
                    logger.error(f"Failed to destroy training process group: {e2}")
            self.process_group = None

        self._cleanup_training_store()
        held_refs = len(self._receiver_fenced_refs)
        self._receiver_fenced_refs.clear()
        if held_refs:
            logger.info("[Training] Released %d receiver-fenced broadcast reference(s)", held_refs)

    # ========================================================================
    # Inference endpoint management
    # ========================================================================

    def _init_inference_endpoints(self) -> List[Dict[str, Any]]:
        """
        Initialize weight update groups on all inference endpoints in parallel.

        Calls /init_weights_update_group API on each endpoint.

        Returns:
            List of results from each endpoint
        """
        session = _get_http_session()

        def init_single(rank_offset: int, endpoint: EndpointInfo) -> Dict[str, Any]:
            url = f"http://{endpoint.host}:{endpoint.port}/init_weights_update_group"
            payload = {
                "master_address": self.master_address,
                "master_port": self._active_master_port,
                "rank_offset": rank_offset,
                "world_size": self.world_size,
                "group_name": self.group_name,
                "backend": "nccl",
            }
            logger.info(f"[Training] Sending init request to {url} with payload: {payload}")
            try:
                # Use longer timeout - NCCL init can take a while, especially with CUDA_LAUNCH_BLOCKING=1
                response = session.post(url, json=payload, timeout=600)
                logger.info(f"[Training] Received response from {endpoint.host}:{endpoint.port}")
                result = response.json()
                return {
                    "endpoint": f"{endpoint.host}:{endpoint.port}",
                    "rank_offset": rank_offset,
                    "success": result.get("success", False),
                    "message": result.get("message", ""),
                }
            except Exception as e:
                return {
                    "endpoint": f"{endpoint.host}:{endpoint.port}",
                    "rank_offset": rank_offset,
                    "success": False,
                    "message": str(e),
                }

        results = []
        with ThreadPoolExecutor(max_workers=len(self.endpoints)) as executor:
            # Calculate rank_offset for each endpoint
            # Rank 0 = training, ranks 1..N = inference
            rank_offset = 1
            futures = {}
            for endpoint in self.endpoints:
                future = executor.submit(init_single, rank_offset, endpoint)
                futures[future] = endpoint
                rank_offset += endpoint.world_size

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(f"[{result['endpoint']}] Init (rank_offset={result.get('rank_offset')}): {result}")

        return results

    def _destroy_inference_endpoints(self) -> List[Dict[str, Any]]:
        """
        Destroy weight update groups on all inference endpoints.

        Calls /destroy_weights_update_group API on each endpoint.
        """
        session = _get_http_session()

        def destroy_single(endpoint: EndpointInfo) -> Dict[str, Any]:
            url = f"http://{endpoint.host}:{endpoint.port}/destroy_weights_update_group"
            payload = {"group_name": self.group_name}
            try:
                response = session.post(url, json=payload, timeout=30)
                result = response.json()
                return {
                    "endpoint": f"{endpoint.host}:{endpoint.port}",
                    "success": result.get("success", False),
                    "message": result.get("message", ""),
                }
            except Exception as e:
                return {
                    "endpoint": f"{endpoint.host}:{endpoint.port}",
                    "success": False,
                    "message": str(e),
                }

        results = []
        with ThreadPoolExecutor(max_workers=len(self.endpoints)) as executor:
            futures = {executor.submit(destroy_single, ep): ep for ep in self.endpoints}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(f"[{result['endpoint']}] Destroy: {result}")

        return results

    # ========================================================================
    # Pause / resume inference
    # ========================================================================

    def _endpoint_request_with_retry(
        self,
        endpoint: EndpointInfo,
        url_path: str,
        operation: str,
        payload: Dict[str, Any],
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """POST to an endpoint with retry logic. Returns result dict."""
        session = _get_http_session()
        endpoint_label = f"{endpoint.host}:{endpoint.port}"
        endpoint_result = None

        for attempt in range(max_retries):
            try:
                url = f"http://{endpoint_label}{url_path}"
                response = session.post(url, json=payload, timeout=timeout)
                result = response.json()
                success = result.get("status") == "ok" or result.get("success", False)

                endpoint_result = {
                    "endpoint": endpoint_label,
                    "success": success,
                    "message": result.get("message", ""),
                    "attempts": attempt + 1,
                }

                if success:
                    logger.info(f"[{endpoint_label}] {operation} succeeded (attempt {attempt + 1}/{max_retries})")
                    return endpoint_result
                else:
                    logger.warning(
                        f"[{endpoint_label}] {operation} failed "
                        f"(attempt {attempt + 1}/{max_retries}): {result.get('message', '')}"
                    )

            except Exception as e:
                endpoint_result = {
                    "endpoint": endpoint_label,
                    "success": False,
                    "message": str(e),
                    "attempts": attempt + 1,
                }
                logger.warning(f"[{endpoint_label}] {operation} error (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)

        return endpoint_result

    def pause_inference_endpoints(
        self,
        pause_mode: str = "retract",
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Pause inference on all endpoints in parallel with retry logic.

        Returns:
            Tuple of (results, all_succeeded)
        """
        with ThreadPoolExecutor(max_workers=max(len(self.endpoints), 1)) as executor:
            futures = {
                executor.submit(
                    self._endpoint_request_with_retry,
                    ep,
                    "/pause_generation",
                    "Pause",
                    {"mode": pause_mode},
                    timeout=60,
                    max_retries=max_retries,
                    retry_delay_seconds=retry_delay_seconds,
                ): ep
                for ep in self.endpoints
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        all_succeeded = all(r["success"] for r in results)
        return results, all_succeeded

    def resume_inference_endpoints(
        self,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Resume inference on all endpoints in parallel with retry logic.

        Returns:
            List of results from each endpoint
        """
        with ThreadPoolExecutor(max_workers=max(len(self.endpoints), 1)) as executor:
            futures = {
                executor.submit(
                    self._endpoint_request_with_retry,
                    ep,
                    "/continue_generation",
                    "Resume",
                    {},
                    timeout=30,
                    max_retries=max_retries,
                    retry_delay_seconds=retry_delay_seconds,
                ): ep
                for ep in self.endpoints
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        for r in results:
            if not r["success"]:
                logger.error(
                    f"[{r['endpoint']}] Failed to resume after {max_retries} attempts. "
                    f"Endpoint may require manual intervention."
                )

        return results

    # ========================================================================
    # Weight transfer
    # ========================================================================

    def _transfer_single_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transfer a single bucket of parameters.

        Optimized for performance:
        1. API calls and GPU transfers happen in parallel
        2. Uses HTTP connection pooling
        3. Pre-transfers all tensors to GPU before NCCL broadcasts
        4. Uses non_blocking transfers where possible

        Args:
            bucket: List of (name, tensor) tuples
            flush_cache: Whether to flush cache after this bucket (only for last bucket)
            weight_version: Optional weight version to apply with this bucket.

        Returns:
            List of results from each endpoint
        """
        # Prepare metadata
        names = [name for name, _ in bucket]
        dtypes = [str(param.dtype).replace("torch.", "") for _, param in bucket]
        shapes = [list(param.shape) for _, param in bucket]
        logger.info(f"[Training] Bucket param names: {names[:5]}{'...' if len(names) > 5 else ''}")
        load_format = _sglang_load_format()
        receiver_load_format = _receiver_load_format(load_format)
        if _is_direct_load_format(receiver_load_format) and any(
            endpoint.world_size != 1 for endpoint in self.endpoints
        ):
            raise RuntimeError("XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT=direct requires SGLang world_size=1")
        if receiver_load_format is not None:
            logger.info("[Training] Receiver load_format=%s", receiver_load_format)

        # Results container - use list for thread-safe appends
        update_errors = []
        update_results = []
        session = _get_http_session()
        two_phase = _use_two_phase_update()
        if two_phase and _is_hybrid_flattened_load_format(receiver_load_format):
            raise RuntimeError("XORL_WEIGHT_SYNC_NCCL_TWO_PHASE=1 does not support hybrid_flattened load_format")
        if two_phase and _is_direct_load_format(receiver_load_format) and _nccl_broadcast_chunk_bytes() > 0:
            raise RuntimeError("XORL_WEIGHT_SYNC_NCCL_TWO_PHASE=1 does not support direct_chunked load_format")

        def prepare_single_endpoint(endpoint: EndpointInfo) -> Dict[str, Any]:
            """Ask one endpoint to start its background NCCL recv thread."""
            payload = {
                "names": names,
                "dtypes": dtypes,
                "shapes": shapes,
                "group_name": self.group_name,
                "transport": "nccl_broadcast",
            }
            if receiver_load_format is not None:
                payload["load_format"] = receiver_load_format
            url = f"http://{endpoint.host}:{endpoint.port}/prepare_weights_update"
            try:
                response = session.post(url, json=payload, timeout=600)
                result = response.json()
                return _endpoint_update_result(endpoint, result)
            except Exception as e:
                return {
                    "host": endpoint.host,
                    "port": endpoint.port,
                    "endpoint": f"{endpoint.host}:{endpoint.port}",
                    "success": False,
                    "message": str(e),
                }

        def complete_single_endpoint(endpoint: EndpointInfo) -> Dict[str, Any]:
            """Ask one endpoint to apply the weights received by the background thread."""
            payload = {
                "group_name": self.group_name,
                "flush_cache": flush_cache,
                "weight_version": weight_version,
                "transport": "nccl_broadcast",
                "run_post_process_weights": self.run_post_process_weights,
            }
            for key, value in (
                ("fp8_kv_cache_enabled", self.fp8_kv_cache_enabled),
                ("fp8_kv_cache_postprocess_required", self.fp8_kv_cache_postprocess_required),
                ("fp8_kv_cache_static_scales", self.fp8_kv_cache_static_scales),
            ):
                if value:
                    payload[key] = True
            if receiver_load_format is not None:
                payload["load_format"] = receiver_load_format
            url = f"http://{endpoint.host}:{endpoint.port}/complete_weights_update"
            try:
                response = session.post(url, json=payload, timeout=600)
                result = response.json()
                return _endpoint_update_result(endpoint, result)
            except Exception as e:
                return {
                    "host": endpoint.host,
                    "port": endpoint.port,
                    "endpoint": f"{endpoint.host}:{endpoint.port}",
                    "success": False,
                    "message": str(e),
                }

        def call_single_endpoint(endpoint: EndpointInfo, endpoint_idx: int):
            """Call update_weights_from_distributed on a single endpoint."""
            try:
                payload = {
                    "names": names,
                    "dtypes": dtypes,
                    "shapes": shapes,
                    "group_name": self.group_name,
                    "flush_cache": flush_cache,
                    "weight_version": weight_version,
                }
                if receiver_load_format is not None:
                    payload["load_format"] = receiver_load_format
                response = session.post(
                    f"http://{endpoint.host}:{endpoint.port}/update_weights_from_distributed",
                    json=payload,
                    timeout=600,
                )
                result = response.json()
                update_results.append(_endpoint_update_result(endpoint, result))
                if not result.get("success"):
                    update_errors.append(f"API failed on {endpoint.host}:{endpoint.port}: {result}")
            except Exception as e:
                update_errors.append(f"Exception calling {endpoint.host}:{endpoint.port}: {e}")

        if two_phase:
            logger.info("[Training] Preparing two-phase NCCL receiver update...")
            with ThreadPoolExecutor(max_workers=len(self.endpoints)) as executor:
                prepare_results = list(executor.map(prepare_single_endpoint, self.endpoints))
            update_results.extend(prepare_results)
            update_errors.extend(
                f"Prepare failed on {result['endpoint']}: {result}"
                for result in prepare_results
                if not result["success"]
            )
            if update_errors:
                raise RuntimeError(f"Weight update prepare failed: {update_errors}")

        # Start API calls in parallel threads (one per endpoint) for the legacy
        # single-phase path. The two-phase path has already prepared receiver
        # background threads above and will call /complete_weights_update after
        # the sender broadcasts.
        api_threads = []
        if not two_phase:
            for i, endpoint in enumerate(self.endpoints):
                t = Thread(target=call_single_endpoint, args=(endpoint, i))
                t.start()
                api_threads.append(t)

        # Set device once for all operations
        torch.cuda.set_device(self.device)

        # Broadcast each tensor. Direct receivers must fence the sender while
        # the peer is still executing the matching HTTP update call. Deferring
        # sender Work.wait() until after the receiver response can hang in NCCL,
        # while skipping the sender fence can leave corrupted receiver weights
        # when the group is aborted.
        deferred_broadcast_refs: list[tuple[Any, torch.Tensor, torch.Tensor | None]] = []
        flattened_refs: list[torch.Tensor | None] = []
        if _is_hybrid_flattened_load_format(receiver_load_format):
            direct_bucket = [(name, param) for name, param in bucket if not _requires_stacked_qwen_loader(name)]
            flattened_bucket_items = [(name, param) for name, param in bucket if _requires_stacked_qwen_loader(name)]
            logger.info(
                "[Training] Hybrid flattened bucket: direct=%d flattened=%d",
                len(direct_bucket),
                len(flattened_bucket_items),
            )
            for name, param in direct_bucket:
                staging_ref = None
                if param.device.type == "cpu":
                    param_data, staging_ref = self._stage_cpu_tensor_for_broadcast(param)
                else:
                    param_data = param.to(self.device).contiguous()
                logger.info(
                    "[Training] Hybrid direct tensor start: name=%s shape=%s dtype=%s nbytes=%d",
                    name,
                    list(param_data.shape),
                    param_data.dtype,
                    param_data.nbytes,
                )
                deferred_works = self._broadcast_tensor(name, param_data, wait=True)
                if deferred_works:
                    deferred_broadcast_refs.extend((work, param_data, staging_ref) for work in deferred_works)
                    logger.info("[Training] Hybrid direct tensor enqueued: name=%s", name)
                else:
                    logger.info("[Training] Hybrid direct tensor done: name=%s", name)
                    del staging_ref
            if flattened_bucket_items:
                flattened_bucket, flattened_refs = self._flatten_bucket_for_broadcast(flattened_bucket_items)
                logger.info(
                    "[Training] Hybrid flattened fallback broadcast start: params=%d nbytes=%d",
                    len(flattened_bucket_items),
                    flattened_bucket.nbytes,
                )
                deferred_works = self._broadcast_tensor("hybrid_flattened_fallback", flattened_bucket, wait=False)
                if deferred_works:
                    deferred_broadcast_refs.extend((work, flattened_bucket, None) for work in deferred_works)
                    logger.info(
                        "[Training] Hybrid flattened fallback broadcast enqueued: params=%d",
                        len(flattened_bucket_items),
                    )
                else:
                    logger.info(
                        "[Training] Hybrid flattened fallback broadcast done: params=%d",
                        len(flattened_bucket_items),
                    )
        elif _is_flattened_bucket_load_format(receiver_load_format):
            flattened_bucket, flattened_refs = self._flatten_bucket_for_broadcast(bucket)
            logger.info(
                "[Training] Flattened bucket broadcast start: params=%d nbytes=%d",
                len(bucket),
                flattened_bucket.nbytes,
            )
            self._broadcast_tensor("flattened_bucket", flattened_bucket, wait=True)
            logger.info("[Training] Flattened bucket broadcast done: params=%d", len(bucket))
        else:
            for i, (name, param) in enumerate(bucket):
                if i == 0:
                    logger.info(
                        f"[Training] First param: name={name}, shape={param.shape}, "
                        f"dtype={param.dtype}, device={param.device}"
                    )
                # Ensure tensor is on the right device and contiguous
                staging_ref = None
                if param.device.type == "cpu":
                    param_data, staging_ref = self._stage_cpu_tensor_for_broadcast(param)
                else:
                    param_data = param.to(self.device).contiguous()

                logger.info(
                    "[Training] Broadcast tensor start: name=%s shape=%s dtype=%s nbytes=%d",
                    name,
                    list(param_data.shape),
                    param_data.dtype,
                    param_data.nbytes,
                )
                deferred_works = self._broadcast_tensor(name, param_data, wait=True)
                if deferred_works:
                    deferred_broadcast_refs.extend((work, param_data, staging_ref) for work in deferred_works)
                    logger.info("[Training] Broadcast tensor enqueued: name=%s", name)
                else:
                    logger.info("[Training] Broadcast tensor done: name=%s", name)
                    del staging_ref

        if two_phase:
            logger.info("[Training] Completing two-phase NCCL receiver update...")
            with ThreadPoolExecutor(max_workers=len(self.endpoints)) as executor:
                complete_results = list(executor.map(complete_single_endpoint, self.endpoints))
            update_results = complete_results
            update_errors.extend(
                f"Complete failed on {result['endpoint']}: {result}"
                for result in complete_results
                if not result["success"]
            )
        else:
            # Wait for all API calls to complete
            for t in api_threads:
                t.join()

        # Check for errors
        if update_errors:
            raise RuntimeError(f"Weight update failed: {update_errors}")

        # Verify all endpoints succeeded
        for r in update_results:
            if not r["success"]:
                raise RuntimeError(f"Weight update failed: {r}")

        if deferred_broadcast_refs:
            if _wait_after_receiver():
                logger.info(
                    "[Training] Waiting on %d receiver-fenced broadcast work handle(s) after receiver completion",
                    len(deferred_broadcast_refs),
                )
                for work, _, _ in deferred_broadcast_refs:
                    work.wait()
            elif _hold_receiver_fenced_refs():
                logger.info(
                    "[Training] Holding %d receiver-fenced broadcast reference(s) until NCCL group teardown",
                    len(deferred_broadcast_refs),
                )
                self._receiver_fenced_refs.extend(deferred_broadcast_refs)
            logger.info(
                "[Training] Receiver completed %d receiver-fenced broadcast(s)",
                len(deferred_broadcast_refs),
            )
            deferred_broadcast_refs.clear()
        flattened_refs.clear()

        return update_results

    def _flatten_bucket_for_broadcast(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        flattened_parts: list[torch.Tensor] = []
        refs: list[torch.Tensor | None] = []
        for name, param in bucket:
            if param.device.type == "cpu":
                param_data, staging_ref = self._stage_cpu_tensor_for_broadcast(param)
            else:
                param_data = param.to(self.device).contiguous()
                staging_ref = None
            logger.info(
                "[Training] Flatten part: name=%s shape=%s dtype=%s nbytes=%d",
                name,
                list(param_data.shape),
                param_data.dtype,
                param_data.nbytes,
            )
            flattened_parts.append(param_data.flatten().view(torch.uint8))
            refs.extend((param_data, staging_ref))

        return torch.cat(flattened_parts, dim=0), refs

    def _broadcast_tensor(self, name: str, tensor: torch.Tensor, *, wait: bool = True) -> list[Any]:
        transfer_tensor = tensor
        if tensor.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
            transfer_tensor = tensor.view(torch.uint8).reshape(-1)
            logger.info("[Training] Broadcasting FP8 tensor as uint8 payload: name=%s", name)

        chunk_bytes = _nccl_broadcast_chunk_bytes()
        if chunk_bytes <= 0 or transfer_tensor.nbytes <= chunk_bytes:
            work = dist.broadcast(transfer_tensor, src=0, group=self.process_group, async_op=True)
            if wait:
                work.wait()
                self._synchronize_waited_broadcast(name)
                return []
            return [work]

        if not transfer_tensor.is_contiguous():
            raise RuntimeError("NCCL chunked broadcast requires a contiguous tensor")
        tensor_flat = transfer_tensor.reshape(-1)
        chunk_elems = max(1, chunk_bytes // transfer_tensor.element_size())
        chunks = (tensor_flat.numel() + chunk_elems - 1) // chunk_elems
        logger.info(
            "[Training] Chunked NCCL broadcast: name=%s nbytes=%d chunk_bytes=%d chunks=%d",
            name,
            transfer_tensor.nbytes,
            chunk_bytes,
            chunks,
        )
        deferred_works = []
        for start in range(0, tensor_flat.numel(), chunk_elems):
            end = min(start + chunk_elems, tensor_flat.numel())
            work = dist.broadcast(tensor_flat[start:end], src=0, group=self.process_group, async_op=True)
            if wait:
                work.wait()
            else:
                deferred_works.append(work)
        if wait:
            self._synchronize_waited_broadcast(name)
        return deferred_works

    def _synchronize_waited_broadcast(self, name: str) -> None:
        device = torch.device(self.device)
        if not _sync_after_waited_broadcast() or device.type != "cuda":
            return
        logger.debug("[Training] CUDA synchronize after waited NCCL broadcast start: name=%s", name)
        torch.cuda.synchronize(device)
        logger.debug("[Training] CUDA synchronize after waited NCCL broadcast done: name=%s", name)

    def _stage_cpu_tensor_for_broadcast(self, param: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Move a CPU tensor to the NCCL device.

        FP8 tensors are staged as raw bytes so the exact payload is preserved.
        Other CPU tensors use the same async staging discipline to avoid host
        synchronizations between long FP8 transfer streams.
        """
        contiguous = param.contiguous()
        async_h2d = os.environ.get("XORL_WEIGHT_SYNC_FP8_CPU_H2D_ASYNC", "1") != "0"
        copy_stream = torch.cuda.Stream(device=self.device) if async_h2d else None
        stream_context = torch.cuda.stream(copy_stream) if copy_stream is not None else nullcontext()

        if param.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
            cpu_source = contiguous.view(torch.uint8).reshape(-1)
            if torch.cuda.is_available() and os.environ.get("XORL_WEIGHT_SYNC_PIN_FP8_CPU_BYTES", "1") != "0":
                cpu_source = cpu_source.pin_memory()

            cuda_bytes = torch.empty(cpu_source.numel(), dtype=torch.uint8, device=self.device)
            chunk_bytes = _fp8_cpu_h2d_chunk_bytes()
            with stream_context:
                if chunk_bytes == 0 or cpu_source.numel() <= chunk_bytes:
                    cuda_bytes.copy_(cpu_source, non_blocking=async_h2d)
                else:
                    for start in range(0, cpu_source.numel(), chunk_bytes):
                        end = min(start + chunk_bytes, cpu_source.numel())
                        cuda_bytes[start:end].copy_(cpu_source[start:end], non_blocking=async_h2d)
            cuda_tensor = cuda_bytes.view(param.dtype).reshape(contiguous.shape)
        else:
            cpu_source = contiguous
            if async_h2d and torch.cuda.is_available():
                cpu_source = cpu_source.pin_memory()

            cuda_tensor = torch.empty_like(contiguous, device=self.device)
            with stream_context:
                cuda_tensor.copy_(cpu_source, non_blocking=async_h2d)

        if copy_stream is not None:
            event = torch.cuda.Event()
            event.record(copy_stream)
            torch.cuda.current_stream(self.device).wait_event(event)
            return cuda_tensor, cpu_source

        return cuda_tensor, None

    def sync_weights(
        self,
        state_dict: Dict[str, torch.Tensor],
        tie_word_embeddings: bool = False,
    ) -> SyncResult:
        """
        Synchronize weights to all inference endpoints.

        This is the main entry point for weight synchronization.
        Initializes NCCL, transfers weights in buckets, then cleans up.

        Args:
            state_dict: Model state dictionary to transfer
            tie_word_embeddings: If True, skip lm_head.weight (tied with embed_tokens)

        Returns:
            SyncResult with transfer statistics
        """
        if not self.endpoints:
            return SyncResult(
                success=False,
                message="No inference endpoints registered",
            )

        start_time = time.perf_counter()

        try:
            # Step 1: Initialize NCCL process groups
            if not self.init_nccl_group():
                return SyncResult(
                    success=False,
                    message="Failed to initialize NCCL process groups",
                )

            # Step 2: Prepare parameters for transfer
            logger.info("=" * 70)
            logger.info("Step 2: Preparing weights for transfer...")
            logger.info("=" * 70)

            # Filter parameters
            param_names = list(state_dict.keys())
            if tie_word_embeddings and "lm_head.weight" in param_names:
                param_names.remove("lm_head.weight")
                logger.info("[Training] Skipping lm_head.weight (tied with embed_tokens)")

            total_params = sum(state_dict[n].numel() for n in param_names)
            total_bytes = sum(state_dict[n].numel() * state_dict[n].element_size() for n in param_names)

            logger.info(f"[Training] Transferring {len(param_names)} parameters")
            logger.info(f"[Training] Total parameters: {total_params:,}")
            logger.info(f"[Training] Total size: {total_bytes / 1e9:.2f} GB")
            logger.info(f"[Training] Bucket size: {self.buffer_size_bytes / 1e6:.0f} MB")

            # Build buckets
            buckets: List[List[Tuple[str, torch.Tensor]]] = []
            current_bucket: List[Tuple[str, torch.Tensor]] = []
            current_bucket_size = 0

            for name in param_names:
                param = state_dict[name]
                param_size = param.numel() * param.element_size()

                if current_bucket and current_bucket_size + param_size >= self.buffer_size_bytes:
                    buckets.append(current_bucket)
                    current_bucket = []
                    current_bucket_size = 0

                current_bucket.append((name, param))
                current_bucket_size += param_size

            if current_bucket:
                buckets.append(current_bucket)

            logger.info(f"[Training] Split into {len(buckets)} buckets")

            # Step 3: Transfer weights
            logger.info("=" * 70)
            logger.info("Step 3: Transferring weights...")
            logger.info("=" * 70)

            transfer_start = time.perf_counter()
            total_transferred = 0

            for i, bucket in enumerate(buckets):
                bucket_size = sum(p.numel() * p.element_size() for _, p in bucket)
                is_last_bucket = i == len(buckets) - 1

                logger.info(
                    f"[Training] Bucket {i + 1}/{len(buckets)} "
                    f"({len(bucket)} params, {bucket_size / 1e6:.1f} MB)"
                    f"{' [final, flush_cache=True]' if is_last_bucket else ''}"
                )

                self._transfer_single_bucket(
                    bucket=bucket,
                    flush_cache=is_last_bucket,
                )

                total_transferred += len(bucket)

            transfer_time = time.perf_counter() - transfer_start
            throughput_gbps = (total_bytes / transfer_time) / (1024**3)

            logger.info(f"[Training] Transfer completed in {transfer_time:.2f}s")
            logger.info(f"[Training] Throughput: {throughput_gbps:.2f} GB/s")

            # Step 4: Cleanup
            logger.info("=" * 70)
            logger.info("Step 4: Cleaning up...")
            logger.info("=" * 70)

            self.destroy_nccl_group()

            total_time = time.perf_counter() - start_time

            return SyncResult(
                success=True,
                message=f"Successfully synced weights to {len(self.endpoints)} endpoints",
                transfer_time=total_time,
                total_bytes=total_bytes,
                num_parameters=len(param_names),
                num_buckets=len(buckets),
                endpoint_results=[{"host": ep.host, "port": ep.port, "success": True} for ep in self.endpoints],
            )

        except Exception as e:
            logger.error(f"Weight sync failed: {e}", exc_info=True)

            # Cleanup on error
            self.destroy_nccl_group()

            return SyncResult(
                success=False,
                message=f"Weight sync failed: {str(e)}",
                transfer_time=time.perf_counter() - start_time,
            )


# ---------------------------------------------------------------------------
# WeightTransportBackend adapter
# ---------------------------------------------------------------------------

from .base import TransportConfig, WeightTransportBackend  # noqa: E402


class NCCLBroadcastBackend(WeightTransportBackend):
    """Single-rank NCCL broadcast transport.

    Wraps :class:`NCCLWeightSynchronizer` to implement the
    :class:`WeightTransportBackend` interface used by the handler.

    Architecture::

        Training rank 0  ──NCCL broadcast──►  SGLang TP workers (ranks 1..N)

    Only rank 0 participates in the NCCL group; other training ranks only
    participate in training-side collectives (FSDP unshard/reshard, QLoRA
    full_tensor, EP gather).
    """

    def __init__(self, config: TransportConfig, **kwargs) -> None:
        super().__init__(config)
        self._synchronizer: Optional[NCCLWeightSynchronizer] = None
        self._process_group = None
        self._reinit_generation = 0

    def initialize(self) -> bool:
        cfg = self.config
        group_name = cfg.group_name
        if _reinit_per_bucket():
            self._reinit_generation += 1
            group_name = f"{cfg.group_name}_rb{self._reinit_generation}"
        ep_infos = [EndpointInfo(host=e.host, port=e.port, world_size=e.world_size) for e in cfg.endpoints]
        self._synchronizer = NCCLWeightSynchronizer(
            endpoints=ep_infos,
            master_address=cfg.master_address,
            master_port=cfg.master_port,
            group_name=group_name,
            buffer_size_mb=cfg.buffer_size_mb,
            device=cfg.device,
            run_post_process_weights=bool(cfg.backend_config.get("run_post_process_weights", False)),
            fp8_kv_cache_enabled=bool(cfg.backend_config.get("fp8_kv_cache_enabled", False)),
            fp8_kv_cache_postprocess_required=bool(
                cfg.backend_config.get("fp8_kv_cache_postprocess_required", False)
            ),
            fp8_kv_cache_static_scales=bool(cfg.backend_config.get("fp8_kv_cache_static_scales", False)),
        )
        logger.info(f"[NCCLBroadcast] Initializing NCCL sync group ({len(ep_infos)} endpoints, device={cfg.device})")
        ok = self._synchronizer.init_nccl_group()
        if ok:
            self._process_group = self._synchronizer.process_group
        return ok

    def destroy(self, *, complete_receiver: bool = True) -> None:
        _ = complete_receiver
        if self._synchronizer is not None:
            try:
                self._synchronizer.destroy_nccl_group()
            except Exception as e:
                logger.warning(f"[NCCLBroadcast] destroy_nccl_group failed: {e}")
            self._synchronizer = None
            self._process_group = None

    def transfer_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        src_rank: int = 0,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ) -> None:
        if src_rank != 0:
            raise ValueError(f"NCCLBroadcastBackend only supports src_rank=0, got {src_rank}")
        if self._synchronizer is None:
            if not _reinit_per_bucket():
                raise RuntimeError("Backend not initialized — call initialize() first")
            if not self.initialize():
                raise RuntimeError("Failed to initialize NCCL backend for transfer bucket")
        try:
            self.endpoint_results = self._synchronizer._transfer_single_bucket(
                bucket,
                flush_cache=flush_cache,
                weight_version=weight_version,
            )
        finally:
            if _reinit_per_bucket():
                logger.info("[NCCLBroadcast] Reinitializing per bucket: destroying NCCL sync group after transfer")
                self.destroy()

    @property
    def sender_ranks(self) -> FrozenSet[int]:
        return frozenset({0})

    @property
    def supports_direct_ep_transfer(self) -> bool:
        return False

    @property
    def supports_direct_pp_transfer(self) -> bool:
        return False
