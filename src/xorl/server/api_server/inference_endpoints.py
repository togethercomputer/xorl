"""Inference endpoints, LoRA adapter management, and sampling session mixin."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any, Dict, List

import httpx
import requests
from fastapi import HTTPException, status
from huggingface_hub import hf_hub_download

from xorl.server.api_server.api_types import (
    AddInferenceEndpointRequest,
    AddInferenceEndpointResponse,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    EndpointSyncResult,
    InferenceEndpoint,
    InferenceEndpointServerInfo,
    ListInferenceEndpointsResponse,
    RemoveInferenceEndpointRequest,
    RemoveInferenceEndpointResponse,
    SetSyncQuantizationRequest,
    SetSyncQuantizationResponse,
    SyncInferenceWeightsRequest,
    SyncInferenceWeightsResponse,
)
from xorl.server.api_server.utils import validate_model_id
from xorl.server.protocol.api_orchestrator import OrchestratorRequest
from xorl.server.protocol.operations import SyncWeightsData
from xorl.server.weight_sync.quantization_config import (
    SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY,
    UnsupportedSyncQuantizationError,
    normalize_sync_quantization_config,
)


logger = logging.getLogger(__name__)


class InferenceEndpointsMixin:
    """Mixin for inference endpoints, LoRA adapter management, and sampling sessions."""

    @staticmethod
    def _endpoint_worker_url(endpoint: InferenceEndpoint) -> str:
        """Return the inference worker URL used for LoRA adapter management."""
        worker_port = endpoint.worker_port if endpoint.worker_port is not None else endpoint.port
        return f"http://{endpoint.host}:{worker_port}"

    @staticmethod
    async def _check_endpoint_health(client: httpx.AsyncClient, endpoint_url: str, endpoint_name: str) -> bool:
        """Check whether an HTTP endpoint responds on one of the supported health paths."""
        for health_endpoint in ("/health", "/v1/models"):
            try:
                response = await client.get(f"{endpoint_url}{health_endpoint}")
                response.raise_for_status()
                logger.info(f"✓ {endpoint_name} health check passed for {endpoint_url} (via {health_endpoint})")
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _normalize_modules_to_not_convert(entries: List[str]) -> List[str]:
        """Translate receiver-side module names into trainer-side names.

        Multimodal HF checkpoints (e.g. Qwen3.6-35B-A3B-FP8) wrap the language
        model under `model.language_model.*`, while the xorl trainer loads only
        the text submodel and exposes its tensors as `model.*`. We strip that
        `language_model.` infix so the prefix-based skip-match in
        ``_should_quantize_fp8_weight`` actually fires on trainer tensor names.
        Vision-only entries (`model.visual.*`, `visual.*`) are dropped because
        they have no trainer-side counterpart.
        """
        out: List[str] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                continue
            # Drop vision-only entries — trainer doesn't have these tensors.
            if entry.startswith("visual.") or ".visual." in entry or entry.startswith("model.visual."):
                continue
            # Strip the multimodal language_model nesting.
            normalized = entry.replace("model.language_model.", "model.", 1)
            if normalized.startswith("language_model."):
                normalized = normalized[len("language_model.") :]
            if normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
        return out

    @staticmethod
    def _positive_int_config_value(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @staticmethod
    def _truthy_server_info_value(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return None

    @classmethod
    def _server_info_bool(cls, info_data: Dict[str, Any], *keys: str) -> bool | None:
        for key in keys:
            if key not in info_data:
                continue
            parsed = cls._truthy_server_info_value(info_data.get(key))
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _server_info_kv_cache_dtype(cls, info_data: Dict[str, Any]) -> str | None:
        value = info_data.get("kv_cache_dtype")
        if value is None:
            value = info_data.get("kv_cache_dtype_str")
        if value is None:
            return None
        return str(value).strip().lower()

    @classmethod
    def _server_info_fp8_kv_cache_enabled(cls, info_data: Dict[str, Any]) -> bool | None:
        explicit = cls._server_info_bool(info_data, "fp8_kv_cache_enabled", "enable_fp8_kv_cache")
        if explicit is not None:
            return explicit
        dtype = cls._server_info_kv_cache_dtype(info_data)
        if dtype is None:
            return None
        return dtype in {"fp8", "fp8_e4m3", "e4m3", "float8_e4m3fn"}

    @staticmethod
    def _normalize_receiver_kv_cache_dtype(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized not in {"auto", "fp8", "fp8_e4m3"}:
            raise ValueError(f"receiver_kv_cache_dtype must be one of auto, fp8, fp8_e4m3; got {value!r}")
        return normalized

    @staticmethod
    def _is_fp8_kv_cache_dtype(value: str | None) -> bool:
        return value in {"fp8", "fp8_e4m3", "e4m3", "float8_e4m3fn"}

    @staticmethod
    def _cache_epoch_from_mapping(data: Dict[str, Any]) -> Any:
        if "cache_epoch" in data:
            return data["cache_epoch"]
        return data.get("cache_version")

    def _resolve_receiver_kv_cache_dtype_requirement(
        self,
        request: AddInferenceEndpointRequest,
    ) -> tuple[str | None, str | None]:
        try:
            train_config = getattr(self, "train_config", {}) or {}
            server_expected = self._normalize_receiver_kv_cache_dtype(
                train_config.get("receiver_kv_cache_dtype")
            )
            request_expected = self._normalize_receiver_kv_cache_dtype(request.receiver_kv_cache_dtype)
        except ValueError as exc:
            return None, str(exc)

        expectations = [value for value in (server_expected, request_expected) if value is not None]
        if not expectations:
            return None, None
        fp8_expectations = [value for value in expectations if value in {"fp8", "fp8_e4m3"}]
        if fp8_expectations:
            return fp8_expectations[0], None
        return "auto", None

    def _receiver_kv_cache_dtype_error(
        self,
        server_info: InferenceEndpointServerInfo | None,
        expected: str | None,
        endpoint_url: str,
    ) -> str | None:
        if expected is None or expected == "auto":
            return None
        if not self._is_fp8_kv_cache_dtype(expected):
            return None
        if server_info is None:
            return (
                f"Endpoint {endpoint_url} must report FP8 KV cache for receiver_kv_cache_dtype={expected!r}, "
                "but /server_info was unavailable."
            )
        if server_info.fp8_kv_cache_enabled is True or self._is_fp8_kv_cache_dtype(server_info.kv_cache_dtype):
            return None
        return (
            f"Endpoint {endpoint_url} does not match receiver_kv_cache_dtype={expected!r}: "
            f"/server_info kv_cache_dtype={server_info.kv_cache_dtype!r}, "
            f"fp8_kv_cache_enabled={server_info.fp8_kv_cache_enabled!r}."
        )

    @staticmethod
    def _looks_like_mtp_module_name(entry: str) -> bool:
        parts = entry.split(".")
        return any(part == "mtp" or part.startswith("mtp_") for part in parts)

    @classmethod
    def _unsupported_mtp_low_precision_reason(
        cls, config_dict: dict[str, Any], quant_config: dict[str, Any]
    ) -> str | None:
        """Return an explicit unsupported reason when an HF FP8 receiver config advertises MTP.

        This guard is intentionally scoped to auto-detected FP8 receiver configs. BF16
        base configs can still be used with explicit SGLang runtime FP8 modes while the
        MTP/speculative tensor contract remains undefined.
        """
        evidence: list[str] = []
        mtp_count_fields = ("mtp_num_hidden_layers", "mtp_num_layers", "num_nextn_predict_layers")
        for prefix, section in (("config", config_dict), ("text_config", config_dict.get("text_config"))):
            if not isinstance(section, dict):
                continue
            for key in mtp_count_fields:
                value = section.get(key)
                if cls._positive_int_config_value(value):
                    evidence.append(f"{prefix}.{key}={value}")

        modules_to_not_convert = quant_config.get("modules_to_not_convert")
        if isinstance(modules_to_not_convert, list):
            mtp_entries = [
                entry
                for entry in modules_to_not_convert
                if isinstance(entry, str) and cls._looks_like_mtp_module_name(entry)
            ]
            if mtp_entries:
                preview = ", ".join(mtp_entries[:3])
                suffix = ", ..." if len(mtp_entries) > 3 else ""
                evidence.append(f"modules_to_not_convert includes {preview}{suffix}")

        if not evidence:
            return None
        return (
            "MTP/speculative low-precision sync is not implemented. "
            f"Detected {evidence[0]}. Enumerate the receiver-visible MTP tensors and add a same-weight "
            "speculative SGLang validation gate before enabling this receiver."
        )

    @classmethod
    def _detect_quantization_from_hf_config(cls, model_path: str) -> dict | None:
        """Detect quantization config from HF model's config.json.

        SGLang /server_info returns quantization=None for auto-detected FP8 models.
        Reads the full quantization_config dict from config.json.
        Supports both local paths and HuggingFace repo IDs.

        Returns the sync-compatible HF quantization_config dict, or None. Any
        `modules_to_not_convert` list is normalized to trainer-side tensor names,
        and supported FP8 receiver configs are normalized to XoRL's sender-side
        sync contract.
        """

        # Try local path first
        config_path = Path(model_path) / "config.json"
        config_dict = None
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_dict = json.load(f)
            except Exception:
                pass

        # Try HuggingFace hub (for repo IDs like "Qwen/Qwen3-8B-FP8")
        if config_dict is None:
            try:
                cached_path = hf_hub_download(model_path, "config.json")
                with open(cached_path) as f:
                    config_dict = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load config.json for {model_path}: {e}")
                return None

        quant_config = config_dict.get("quantization_config")
        if quant_config and quant_config.get("quant_method") == "fp8":
            mtnc = quant_config.get("modules_to_not_convert")
            if isinstance(mtnc, list):
                quant_config = {
                    **quant_config,
                    "modules_to_not_convert": cls._normalize_modules_to_not_convert(mtnc),
                }
            try:
                normalized_config = normalize_sync_quantization_config(
                    quant_config,
                    context="auto-detected FP8 receiver quantization",
                )
            except UnsupportedSyncQuantizationError as exc:
                normalized_config = {
                    **quant_config,
                    SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY: str(exc),
                }
            assert normalized_config is not None
            quant_config = normalized_config

            reason = cls._unsupported_mtp_low_precision_reason(config_dict, quant_config)
            if reason is not None:
                quant_config = {
                    **quant_config,
                    SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY: reason,
                }
            return quant_config
        return None

    def _get_endpoint_quantization(self) -> dict | None:
        """Get quantization config: user-set default > auto-detected from endpoint HF config."""
        if self._default_sync_quantization is not None:
            return self._default_sync_quantization
        # Fall back to auto-detected quantization stored on endpoint
        for ep in self.inference_endpoints:
            if ep.server_info and ep.server_info.quantization_config:
                return ep.server_info.quantization_config
        return None

    def _auto_detected_endpoint_quantization(self) -> dict | None:
        """Return the auto-detected receiver quantization config, ignoring the
        user-set default. Used to enrich a user-supplied per-call config that
        omits `modules_to_not_convert`.
        """
        for ep in self.inference_endpoints:
            if ep.server_info and ep.server_info.quantization_config:
                return ep.server_info.quantization_config
        return None

    def _enrich_quantization_with_receiver_skip_list(self, quantization: dict | None) -> dict | None:
        """If the caller passes an FP8 quant config without `modules_to_not_convert`,
        fall back to the receiver's auto-detected skip list (already normalized
        to trainer-side names by `_detect_quantization_from_hf_config`). This is
        what saves clients from having to hard-code per-model skip lists for
        block-FP8 receivers.
        """
        if not quantization or quantization.get("quant_method") != "fp8":
            return quantization
        detected = self._auto_detected_endpoint_quantization()
        if not detected or detected.get("quant_method") != "fp8":
            return quantization
        unsupported_reason = detected.get(SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY)
        if isinstance(unsupported_reason, str) and unsupported_reason.strip():
            quantization = {
                **quantization,
                SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY: unsupported_reason,
            }
        if "modules_to_not_convert" in quantization:
            return quantization
        detected_skip = detected.get("modules_to_not_convert")
        if not isinstance(detected_skip, list) or not detected_skip:
            return quantization
        return {**quantization, "modules_to_not_convert": list(detected_skip)}

    def set_sync_quantization(self, request: SetSyncQuantizationRequest) -> SetSyncQuantizationResponse:
        """Set the default quantization format for weight sync."""
        try:
            config = normalize_sync_quantization_config(
                request.quantization,
                context="set_sync_quantization.quantization",
            )
        except UnsupportedSyncQuantizationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        self._default_sync_quantization = config
        if config is not None:
            fmt = f"{config.get('quant_method')} (block_size={config.get('weight_block_size', [128, 128])})"
        else:
            fmt = "bf16 (no quantization)"
        logger.info(f"Set default sync quantization to: {fmt}")
        return SetSyncQuantizationResponse(
            quantization=config,
            message=f"Sync quantization set to {fmt}",
        )

    def _resolve_sync_cache_behavior(
        self,
        request: SyncInferenceWeightsRequest,
        quantization: dict | None,
    ) -> Dict[str, Any]:
        fp8_weight_sync = bool(quantization and quantization.get("quant_method") == "fp8")
        fp8_kv_cache_enabled = False
        fp8_kv_cache_requires_postprocess = False
        fp8_kv_cache_static_scales = False
        cache_epoch = None

        for ep in self.inference_endpoints:
            info = ep.server_info
            if info is None:
                continue
            if info.fp8_kv_cache_enabled is True or self._is_fp8_kv_cache_dtype(info.kv_cache_dtype):
                fp8_kv_cache_enabled = True
            if info.fp8_kv_cache_requires_postprocess is True:
                fp8_kv_cache_requires_postprocess = True
            if info.fp8_kv_cache_static_scales is True:
                fp8_kv_cache_static_scales = True
            if cache_epoch is None and info.cache_epoch is not None:
                cache_epoch = info.cache_epoch

        mode = request.cache_invalidation_mode
        flush_cache = bool(request.flush_cache)
        if mode == "flush":
            flush_cache = True
        elif mode == "auto" and fp8_weight_sync and fp8_kv_cache_enabled:
            flush_cache = True

        postprocess_required = bool(
            fp8_weight_sync
            and fp8_kv_cache_enabled
            and (fp8_kv_cache_requires_postprocess or fp8_kv_cache_static_scales)
        )
        return {
            "cache_invalidation_mode": mode,
            "flush_cache": flush_cache,
            "fp8_kv_cache_enabled": fp8_kv_cache_enabled,
            "fp8_kv_cache_postprocess_required": postprocess_required,
            "fp8_kv_cache_static_scales": fp8_kv_cache_static_scales,
            "cache_epoch": cache_epoch,
        }

    async def _sync_weights_to_endpoints(
        self,
        endpoints: List[Dict[str, Any]],
        master_address: str,
        master_port: int,
        group_name: str,
        buffer_size_mb: int,
        quantization: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Internal method to sync weights to specific endpoints.

        Args:
            endpoints: List of endpoint dicts with host, port, world_size
            master_address: NCCL rendezvous address
            master_port: NCCL rendezvous port
            group_name: NCCL process group name
            buffer_size_mb: Transfer bucket size
            quantization: HF quantization_config dict (e.g. {"quant_method": "fp8", ...})

        Returns:
            Dict with success status and details
        """
        quantization = normalize_sync_quantization_config(
            quantization,
            context="_sync_weights_to_endpoints.quantization",
        )
        engine_request = OrchestratorRequest(
            operation="sync_inference_weights",
            payload=SyncWeightsData(
                endpoints=endpoints,
                master_address=master_address,
                master_port=master_port,
                group_name=group_name,
                buffer_size_mb=buffer_size_mb,
                sync_method=self.sync_inference_method,
                quantization=quantization,
            ),
        )

        # Send to engine and get future for response
        response_future = await self.orchestrator_client.send_request(engine_request)

        # Wait for output with extended timeout (weight sync can take a while)
        sync_timeout = max(self.default_timeout, 600.0)
        output = await asyncio.wait_for(response_future, timeout=sync_timeout)

        if output.error:
            return {"success": False, "message": f"Engine error: {output.error}"}

        result = output.outputs[0] if output.outputs else {}
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "transfer_time": result.get("transfer_time", 0),
            "total_bytes": result.get("total_bytes", 0),
        }

    async def add_inference_endpoint(self, request: AddInferenceEndpointRequest) -> AddInferenceEndpointResponse:
        """
        Add an inference endpoint to the registry.

        This endpoint:
        1. Checks if the endpoint is healthy via /health
        2. Fetches server info via /server_info to get model and quantization config
        3. Detects FP8 quantization from HF config if not reported by SGLang
        4. Validates endpoint consistency (model_path, quantization, tp_size)
        5. If sync_weights is True, syncs weights to the new endpoint

        Args:
            request: Add inference endpoint request with host, port, and world_size

        Returns:
            Response indicating success/failure and endpoint info
        """
        endpoint_url = f"http://{request.host}:{request.port}"
        worker_port = request.worker_port if request.worker_port is not None else request.port
        worker_url = f"http://{request.host}:{worker_port}"

        # Check if endpoint already exists
        for existing in self.inference_endpoints:
            if existing.host == request.host and existing.port == request.port:
                return AddInferenceEndpointResponse(
                    success=False,
                    message=f"Endpoint {endpoint_url} already registered",
                    endpoint=existing,
                )

        # Health check both SGLang server and inference worker
        # Try multiple health check endpoints - SGLang may not have /health
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if not await self._check_endpoint_health(client, endpoint_url, "SGLang server"):
                    raise Exception(f"All health endpoints failed for {endpoint_url}")

                if worker_url != endpoint_url and not await self._check_endpoint_health(
                    client, worker_url, "Inference worker"
                ):
                    raise Exception(f"All health endpoints failed for {worker_url}")

                is_healthy = True
        except Exception as e:
            logger.warning(f"Health check failed for {endpoint_url} or {worker_url}: {e}")
            is_healthy = False

        if not is_healthy:
            return AddInferenceEndpointResponse(
                success=False,
                message=f"Health check failed for SGLang server {endpoint_url} or inference worker {worker_url}",
                endpoint=None,
            )

        # Fetch server info to get model and LoRA configuration
        server_info = None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{endpoint_url}/server_info")
                response.raise_for_status()
                info_data = response.json()

                # Extract relevant fields from server_info
                server_info = InferenceEndpointServerInfo(
                    model_path=info_data.get("model_path"),
                    served_model_name=info_data.get("served_model_name"),
                    tp_size=info_data.get("tp_size"),
                    quantization=info_data.get("quantization"),
                    dtype=info_data.get("dtype"),
                    kv_cache_dtype=self._server_info_kv_cache_dtype(info_data),
                    fp8_kv_cache_enabled=self._server_info_fp8_kv_cache_enabled(info_data),
                    fp8_kv_cache_requires_postprocess=self._server_info_bool(
                        info_data,
                        "fp8_kv_cache_requires_postprocess",
                        "requires_fp8_kv_cache_postprocess",
                        "kv_cache_requires_postprocess",
                    ),
                    fp8_kv_cache_static_scales=self._server_info_bool(
                        info_data,
                        "fp8_kv_cache_static_scales",
                        "has_fp8_kv_cache_static_scales",
                        "kv_cache_static_scales",
                    ),
                    cache_epoch=self._cache_epoch_from_mapping(info_data),
                    enable_lora=info_data.get("enable_lora"),
                    max_lora_rank=info_data.get("max_lora_rank"),
                    version=info_data.get("version"),
                )

                logger.info(
                    f"Server info for {endpoint_url}: "
                    f"model={server_info.model_path}, "
                    f"quantization={server_info.quantization}, "
                    f"dtype={server_info.dtype}, "
                    f"tp_size={server_info.tp_size}, "
                    f"kv_cache_dtype={server_info.kv_cache_dtype}, "
                    f"fp8_kv_cache_enabled={server_info.fp8_kv_cache_enabled}, "
                    f"cache_epoch={server_info.cache_epoch}"
                )

        except Exception as e:
            logger.warning(f"Failed to fetch server_info from {endpoint_url}: {e}")
            # Continue without server_info - older SGLang versions may not have this endpoint

        # Detect quantization from HF config.json
        if server_info is not None and server_info.model_path:
            detected_config = self._detect_quantization_from_hf_config(server_info.model_path)
            if detected_config is not None:
                server_info.quantization_config = detected_config
                if server_info.quantization is None:
                    server_info.quantization = detected_config.get("quant_method")
                logger.info(
                    f"Detected quantization from HF config: quant_method={detected_config.get('quant_method')}, "
                    f"block_size={detected_config.get('weight_block_size')}, "
                    f"modules_to_not_convert={'yes' if detected_config.get('modules_to_not_convert') else 'no'}"
                )

        expected_kv_cache_dtype, expected_kv_cache_error = self._resolve_receiver_kv_cache_dtype_requirement(request)
        if expected_kv_cache_error is not None:
            return AddInferenceEndpointResponse(
                success=False,
                message=expected_kv_cache_error,
                endpoint=None,
            )
        kv_cache_error = self._receiver_kv_cache_dtype_error(server_info, expected_kv_cache_dtype, endpoint_url)
        if kv_cache_error is not None:
            return AddInferenceEndpointResponse(
                success=False,
                message=kv_cache_error,
                endpoint=None,
            )

        # Validate endpoint consistency (model_path, quantization, tp_size must match)
        if server_info is not None and self.inference_endpoints:
            existing_info = self.inference_endpoints[0].server_info
            if existing_info is not None:
                mismatches = []
                if (
                    existing_info.model_path
                    and server_info.model_path
                    and existing_info.model_path != server_info.model_path
                ):
                    mismatches.append(f"model_path: {existing_info.model_path} vs {server_info.model_path}")
                if existing_info.quantization != server_info.quantization:
                    mismatches.append(f"quantization: {existing_info.quantization} vs {server_info.quantization}")
                if existing_info.tp_size and server_info.tp_size and existing_info.tp_size != server_info.tp_size:
                    mismatches.append(f"tp_size: {existing_info.tp_size} vs {server_info.tp_size}")
                if mismatches:
                    return AddInferenceEndpointResponse(
                        success=False,
                        message=f"Endpoint config mismatch with existing endpoints: {'; '.join(mismatches)}",
                        endpoint=None,
                    )

        # Determine world_size: use server_info.tp_size if available, else request.world_size
        world_size = request.world_size
        if server_info is not None and server_info.tp_size is not None and server_info.tp_size > 1:
            world_size = server_info.tp_size
            logger.info(f"Using tp_size from server_info: {world_size}")

        # Create and add the endpoint
        endpoint = InferenceEndpoint(
            host=request.host,
            port=request.port,
            worker_port=worker_port,
            world_size=world_size,
            healthy=is_healthy,
            pool=request.pool,
            server_info=server_info,
        )
        self.inference_endpoints.append(endpoint)

        logger.info(f"Added inference endpoint: {endpoint_url} (world_size={world_size}, pool={request.pool})")

        # Auto-sync weights if requested
        weights_synced = False
        sync_message = None

        if request.sync_weights:
            if not self._running or not self.orchestrator_client:
                sync_message = "Cannot sync weights: API server not fully running"
                logger.warning(sync_message)
            else:
                try:
                    # Auto-detect master_address if not provided
                    master_address = request.master_address
                    if not master_address:
                        master_address = socket.getfqdn()
                        logger.info(f"Auto-detected master_address: {master_address}")

                    logger.info(f"Auto-syncing weights to new endpoint {endpoint_url}...")
                    sync_result = await self._sync_weights_to_endpoints(
                        endpoints=[
                            {
                                "host": request.host,
                                "port": request.port,
                                "world_size": world_size,
                            }
                        ],
                        master_address=master_address,
                        master_port=request.master_port,
                        group_name=request.group_name,
                        buffer_size_mb=request.buffer_size_mb,
                        quantization=self._get_endpoint_quantization(),
                    )
                    weights_synced = sync_result.get("success", False)
                    if weights_synced:
                        transfer_time = sync_result.get("transfer_time", 0)
                        total_bytes = sync_result.get("total_bytes", 0)
                        sync_message = (
                            f"Weights synced successfully in {transfer_time:.2f}s ({total_bytes / 1e9:.2f} GB)"
                        )
                        logger.info(sync_message)
                    else:
                        sync_message = f"Weight sync failed: {sync_result.get('message', 'Unknown error')}"
                        logger.error(sync_message)
                except asyncio.TimeoutError:
                    sync_message = "Weight sync timed out"
                    logger.error(sync_message)
                except Exception as e:
                    sync_message = f"Weight sync error: {str(e)}"
                    logger.error(sync_message, exc_info=True)

        return AddInferenceEndpointResponse(
            success=True,
            message=f"Successfully added endpoint {endpoint_url}",
            endpoint=endpoint,
            weights_synced=weights_synced,
            sync_message=sync_message,
        )

    async def list_inference_endpoints(self) -> ListInferenceEndpointsResponse:
        """
        List all registered inference endpoints.

        Performs health checks on all endpoints and removes unhealthy ones.

        Returns:
            Response with list of healthy endpoints and count
        """
        if not self.inference_endpoints:
            return ListInferenceEndpointsResponse(
                endpoints=[],
                count=0,
            )

        # Check health of all endpoints concurrently
        async def check_endpoint_health(endpoint: InferenceEndpoint) -> tuple[InferenceEndpoint, bool]:
            endpoint_url = f"http://{endpoint.host}:{endpoint.port}"
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    endpoint_healthy = await self._check_endpoint_health(client, endpoint_url, "Inference endpoint")
                    if not endpoint_healthy:
                        raise RuntimeError("Inference endpoint health check failed")
                    return endpoint, True
            except Exception as e:
                logger.warning(f"Health check failed for endpoint {endpoint_url}: {e}")
                return endpoint, False

        results = await asyncio.gather(
            *[check_endpoint_health(ep) for ep in self.inference_endpoints], return_exceptions=True
        )

        # Filter out unhealthy endpoints
        healthy_endpoints = []
        removed_count = 0
        for result in results:
            if isinstance(result, Exception):
                # If the health check task itself failed, skip this endpoint
                removed_count += 1
                continue
            endpoint, is_healthy = result
            if is_healthy:
                endpoint.healthy = True
                healthy_endpoints.append(endpoint)
            else:
                removed_count += 1
                logger.info(f"Removing unhealthy endpoint: http://{endpoint.host}:{endpoint.port}")

        # Update the endpoints list
        self.inference_endpoints = healthy_endpoints

        if removed_count > 0:
            logger.info(f"Removed {removed_count} unhealthy endpoint(s), {len(healthy_endpoints)} remaining")

        return ListInferenceEndpointsResponse(
            endpoints=healthy_endpoints,
            count=len(healthy_endpoints),
        )

    def remove_inference_endpoint(self, request: RemoveInferenceEndpointRequest) -> RemoveInferenceEndpointResponse:
        """
        Remove an inference endpoint from the registry.

        Args:
            request: Remove inference endpoint request with host and port

        Returns:
            Response indicating success/failure
        """
        endpoint_url = f"http://{request.host}:{request.port}"

        # Find and remove the endpoint
        for i, endpoint in enumerate(self.inference_endpoints):
            if endpoint.host == request.host and endpoint.port == request.port:
                self.inference_endpoints.pop(i)
                if not self.inference_endpoints and self.loaded_sampling_loras:
                    self.loaded_sampling_loras.clear()
                    logger.info("Cleared tracked sampling adapters after removing the last inference endpoint")
                logger.info(f"Removed inference endpoint: {endpoint_url}")
                return RemoveInferenceEndpointResponse(
                    success=True,
                    message=f"Successfully removed endpoint {endpoint_url}",
                )

        # Endpoint not found
        return RemoveInferenceEndpointResponse(
            success=False,
            message=f"Endpoint {endpoint_url} not found in registry",
        )

    async def sync_inference_weights(self, request: SyncInferenceWeightsRequest) -> SyncInferenceWeightsResponse:
        """
        Synchronize model weights to all registered inference endpoints via NCCL.

        This triggers the training worker (rank 0) to:
        1. Extract model weights (state_dict)
        2. Initialize NCCL process group with inference endpoints
        3. Transfer weights in buckets via NCCL broadcast
        4. Cleanup process groups

        Args:
            request: Sync request with NCCL configuration

        Returns:
            Response with sync status and statistics
        """
        self._require_engine()

        if not self.inference_endpoints:
            return SyncInferenceWeightsResponse(
                success=False,
                message="No inference endpoints registered. Use /add_inference_endpoint first.",
                endpoints_synced=[],
            )

        try:
            # Convert inference endpoints to list format for the engine.
            # Deduplicate by (host, port) to avoid inflating world_size when
            # the same endpoint is registered more than once (e.g. via
            # add_inference_endpoint called from multiple paths).
            # request.pools (None = all) restricts the sync to matching pool
            # tags, so a dedicated eval pool can stay on frozen weights while
            # the per-step sync covers only the training samplers.
            seen: set[tuple[str, int]] = set()
            endpoints_data = []
            for ep in self.inference_endpoints:
                if request.pools is not None and ep.pool not in request.pools:
                    continue
                key = (ep.host, ep.port)
                if key not in seen:
                    seen.add(key)
                    endpoints_data.append({"host": ep.host, "port": ep.port, "world_size": ep.world_size})
            if not endpoints_data:
                return SyncInferenceWeightsResponse(
                    success=False,
                    message=f"No registered endpoints match pools={request.pools}.",
                    endpoints_synced=[],
                )

            # Auto-detect master_address if localhost or empty (for cross-node NCCL)
            master_address = request.master_address
            if not master_address or master_address == "localhost":
                master_address = socket.getfqdn()
                logger.info(f"Auto-detected master_address: {master_address}")

            if "quantization" in request.model_fields_set:
                requested_quantization = request.quantization
            else:
                requested_quantization = self._get_endpoint_quantization()

            try:
                quantization = normalize_sync_quantization_config(
                    self._enrich_quantization_with_receiver_skip_list(requested_quantization),
                    context="sync_inference_weights.quantization",
                )
            except UnsupportedSyncQuantizationError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            cache_behavior = self._resolve_sync_cache_behavior(request, quantization)

            engine_request = OrchestratorRequest(
                operation="sync_inference_weights",
                payload=SyncWeightsData(
                    model_id=request.model_id,
                    endpoints=endpoints_data,
                    master_address=master_address,
                    master_port=request.master_port,
                    group_name=request.group_name,
                    buffer_size_mb=request.buffer_size_mb,
                    sync_method=self.sync_inference_method,
                    flush_cache=cache_behavior["flush_cache"],
                    cache_invalidation_mode=cache_behavior["cache_invalidation_mode"],
                    fp8_kv_cache_enabled=cache_behavior["fp8_kv_cache_enabled"],
                    fp8_kv_cache_postprocess_required=cache_behavior["fp8_kv_cache_postprocess_required"],
                    fp8_kv_cache_static_scales=cache_behavior["fp8_kv_cache_static_scales"],
                    pause_mode=request.pause_mode,
                    weight_version=request.weight_version,
                    quantization=quantization,
                    sparse_delta_paths=request.sparse_delta_paths,
                    sparse_delta_config=request.sparse_delta_config,
                ),
            )

            # Send to engine and get future for response
            response_future = await self.orchestrator_client.send_request(engine_request)

            # Wait for output with extended timeout (weight sync can take a while)
            sync_timeout = max(self.default_timeout, 600.0)  # At least 10 minutes
            try:
                output = await asyncio.wait_for(response_future, timeout=sync_timeout)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"Weight sync timeout after {sync_timeout}s"
                )

            if output.error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Engine error: {output.error}"
                )

            # Extract results
            result = output.outputs[0] if output.outputs else {}

            # Build endpoint sync results
            endpoint_results = []
            for ep_result in result.get("endpoint_results", []):
                endpoint_results.append(
                    EndpointSyncResult(
                        host=ep_result.get("host", ""),
                        port=ep_result.get("port", 0),
                        success=ep_result.get("success", False),
                        message=ep_result.get("message", ""),
                        cache_epoch=self._cache_epoch_from_mapping(ep_result),
                        fp8_kv_cache_postprocess_ran=ep_result.get("fp8_kv_cache_postprocess_ran"),
                        fp8_kv_cache_static_scales_updated=ep_result.get(
                            "fp8_kv_cache_static_scales_updated"
                        ),
                    )
                )
            result_cache_epoch = self._cache_epoch_from_mapping(result)
            if result_cache_epoch is None:
                for ep_result in endpoint_results:
                    if ep_result.cache_epoch is not None:
                        result_cache_epoch = ep_result.cache_epoch
                        break
            if result_cache_epoch is None:
                result_cache_epoch = cache_behavior["cache_epoch"]

            return SyncInferenceWeightsResponse(
                success=result.get("success", False),
                message=result.get("message", ""),
                transfer_time=result.get("transfer_time", 0.0),
                total_bytes=result.get("total_bytes", 0),
                num_parameters=result.get("num_parameters", 0),
                num_buckets=result.get("num_buckets", 0),
                timing_breakdown=result.get("timing_breakdown", {}),
                p2p_rank_summaries=result.get("p2p_rank_summaries", []),
                cache_invalidation_mode=result.get(
                    "cache_invalidation_mode", cache_behavior["cache_invalidation_mode"]
                ),
                flush_cache=result.get("flush_cache", cache_behavior["flush_cache"]),
                fp8_kv_cache_enabled=result.get(
                    "fp8_kv_cache_enabled", cache_behavior["fp8_kv_cache_enabled"]
                ),
                fp8_kv_cache_postprocess_requested=result.get(
                    "fp8_kv_cache_postprocess_requested",
                    cache_behavior["fp8_kv_cache_postprocess_required"],
                ),
                fp8_kv_cache_static_scales=result.get(
                    "fp8_kv_cache_static_scales", cache_behavior["fp8_kv_cache_static_scales"]
                ),
                cache_epoch=result_cache_epoch,
                endpoints_synced=endpoint_results,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Sync inference weights failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sync inference weights failed: {e}"
            )

    # =========================================================================
    # Sampling Session Management (LoRA Adapter Loading)
    # =========================================================================

    def _resolve_model_path(self, model_path: str) -> tuple[str | None, str, str]:
        """
        Resolve model_path to (source_model_id, lora_name, absolute_path).

        Sampler weights are stored flat under output_dir/sampler_weights/{name}
        without model_id subdirectories, because inference endpoints don't know about model_id.

        Supported formats:
            - "xorl://model_id/sampler_weights/adapter_name" -> adapter_name
            - "sampler_weights/adapter_name" -> adapter_name
            - "adapter_name" -> adapter_name

        Args:
            model_path: Path to the LoRA adapter (can be xorl:// URI or relative path)

        Returns:
            Tuple of (source_model_id, lora_name, absolute_path)

        Raises:
            HTTPException: If path format is invalid or path doesn't exist
        """
        source_model_id: str | None = None
        if model_path.startswith("xorl://"):
            # Format: xorl://model_id/sampler_weights/adapter_name
            # Parse: remove "xorl://", split by "/", extract adapter name
            parts = model_path[7:].split("/")  # Remove "xorl://"
            if len(parts) >= 3 and parts[1] == "sampler_weights":
                # xorl://model_id/sampler_weights/adapter_name
                source_model_id = validate_model_id(parts[0])
                lora_name = "/".join(parts[2:])  # In case adapter name has /
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid xorl:// URI format for sampler weights: {model_path}. "
                    f"Expected: xorl://model_id/sampler_weights/adapter_name",
                )
        elif model_path.startswith("sampler_weights/"):
            # Format: sampler_weights/adapter_name
            lora_name = model_path[len("sampler_weights/") :]
        else:
            # Just the adapter name
            lora_name = model_path

        # Construct absolute path - sampler_weights are stored flat under output_dir/sampler_weights/
        absolute_path = os.path.abspath(os.path.join(self.output_dir, "sampler_weights", lora_name))

        # Check if path exists
        if not os.path.exists(absolute_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Model path does not exist: {absolute_path}"
            )

        return source_model_id, lora_name, absolute_path

    async def _load_lora_on_inference_endpoints(self, lora_name: str, lora_path: str) -> bool:
        """
        Load a LoRA adapter on all inference endpoints via SGLang's /load_lora_adapter.

        Args:
            lora_name: Name for the LoRA adapter
            lora_path: Absolute path to the adapter

        Returns:
            True if successful on all endpoints

        Raises:
            HTTPException: If loading fails
        """
        if not self.inference_endpoints:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No inference endpoints registered. Use /add_inference_endpoint first.",
            )

        async def load_on_endpoint(endpoint: InferenceEndpoint) -> tuple[str, bool, str]:
            endpoint_url = self._endpoint_worker_url(endpoint)
            try:

                def do_post() -> requests.Response:
                    return requests.post(
                        f"{endpoint_url}/load_lora_adapter",
                        json={
                            "lora_name": lora_name,
                            "lora_path": lora_path,
                            "pinned": False,  # Allow eviction for memory management
                        },
                        headers={"Connection": "close"},
                        timeout=300.0,
                    )

                response = await asyncio.to_thread(do_post)
                result = response.json()

                if response.status_code == 200 and result.get("success", False):
                    logger.info(f"Loaded LoRA adapter '{lora_name}' on {endpoint_url}")
                    return endpoint_url, True, ""

                # Check for errors
                error_msg = result.get("error_message", "")

                # "already loaded" is not a fatal error - treat it as success
                # This can happen when create_sampling_session is called after save_weights_for_sampler
                if "already loaded" in error_msg.lower():
                    logger.warning(f"LoRA adapter '{lora_name}' already loaded on {endpoint_url}, continuing")
                    return endpoint_url, True, ""

                if not error_msg:
                    error_msg = f"HTTP {response.status_code}"
                logger.error(f"Failed to load LoRA adapter on {endpoint_url}: {error_msg}")
                return endpoint_url, False, error_msg

            except Exception as e:
                logger.error(f"Failed to load LoRA adapter on {endpoint_url}: {e}")
                return endpoint_url, False, str(e)

        # Load on all endpoints concurrently
        results = await asyncio.gather(
            *[load_on_endpoint(ep) for ep in self.inference_endpoints], return_exceptions=True
        )

        # Check results
        all_success = True
        errors = []
        for result in results:
            if isinstance(result, Exception):
                all_success = False
                errors.append(str(result))
            else:
                url, success, error = result
                if not success:
                    all_success = False
                    errors.append(f"{url}: {error}")

        if not all_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load LoRA adapter on some endpoints: {'; '.join(errors)}",
            )

        return True

    async def _unload_lora_on_inference_endpoints(self, lora_name: str) -> bool:
        """
        Unload a LoRA adapter from all inference endpoints via SGLang's /unload_lora_adapter.

        Args:
            lora_name: Name of the LoRA adapter to unload

        Returns:
            True if successful on all endpoints
        """
        if not self.inference_endpoints:
            logger.warning("No inference endpoints to unload LoRA from")
            return True

        async def unload_on_endpoint(endpoint: InferenceEndpoint) -> tuple[str, bool, str]:
            endpoint_url = self._endpoint_worker_url(endpoint)
            try:

                def do_post() -> requests.Response:
                    return requests.post(
                        f"{endpoint_url}/unload_lora_adapter",
                        json={"lora_name": lora_name},
                        headers={"Connection": "close"},
                        timeout=30.0,
                    )

                response = await asyncio.to_thread(do_post)
                response.raise_for_status()
                result = response.json()

                if result.get("success", False):
                    logger.info(f"Unloaded LoRA adapter '{lora_name}' from {endpoint_url}")
                    return endpoint_url, True, ""
                else:
                    error_msg = result.get("error_message", "Unknown error")
                    logger.warning(f"Failed to unload LoRA adapter from {endpoint_url}: {error_msg}")
                    return endpoint_url, False, error_msg

            except Exception as e:
                logger.warning(f"Failed to unload LoRA adapter from {endpoint_url}: {e}")
                return endpoint_url, False, str(e)

        # Unload from all endpoints concurrently
        results = await asyncio.gather(
            *[unload_on_endpoint(ep) for ep in self.inference_endpoints], return_exceptions=True
        )

        # Log any failures but don't raise
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Exception during LoRA unload: {result}")

        return True

    async def _get_loaded_adapters_from_endpoint(self, endpoint: "InferenceEndpoint") -> list[str] | None:
        """
        Get list of currently loaded LoRA adapters from an inference endpoint.

        Args:
            endpoint: The inference endpoint to query

        Returns:
            List of adapter names currently loaded, or None if the endpoint could not be queried
        """
        endpoint_url = self._endpoint_worker_url(endpoint)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{endpoint_url}/v1/models")
                response.raise_for_status()
                result = response.json()

                # Adapters have a 'parent' field set, base model does not
                adapters = []
                for model in result.get("data", []):
                    if model.get("parent"):
                        adapters.append(model.get("id"))
                return adapters
        except Exception as e:
            logger.warning(f"Error getting loaded adapters from {endpoint_url}: {e}")
            return None

    async def _reconcile_tracked_adapters(self, model_id: str) -> list[set[str]]:
        """
        Reconcile tracked adapter state with what inference endpoints actually have loaded.

        This prunes stale tracking entries left behind by endpoint restarts or failed
        create_sampling_session attempts, while preserving adapters that are still loaded
        on at least one endpoint and may just need reloading on the others.

        Returns:
            Per-endpoint sets of currently loaded adapter names.
        """
        if not self.inference_endpoints:
            return []

        results = await asyncio.gather(
            *[self._get_loaded_adapters_from_endpoint(endpoint) for endpoint in self.inference_endpoints],
            return_exceptions=True,
        )

        loaded_by_endpoint: list[set[str]] = []
        unknown_queries = False
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Exception while querying loaded adapters: {result}")
                unknown_queries = True
                loaded_by_endpoint.append(set())
            elif result is None:
                unknown_queries = True
                loaded_by_endpoint.append(set())
            else:
                loaded_by_endpoint.append(set(result))

        loaded_anywhere = set().union(*loaded_by_endpoint) if loaded_by_endpoint else set()
        tracked = self.loaded_sampling_loras.get(model_id, [])
        if tracked and not unknown_queries:
            stale = [name for name, _ in tracked if name not in loaded_anywhere]
            if stale:
                self.loaded_sampling_loras[model_id] = [
                    (name, path) for name, path in tracked if name in loaded_anywhere
                ]
                logger.info(f"Pruned {len(stale)} stale tracked adapter(s) for model_id={model_id}: {stale}")
        elif tracked:
            logger.info(
                f"Skipped stale adapter pruning for model_id={model_id} because one or more endpoints "
                "could not report loaded adapters"
            )

        return loaded_by_endpoint

    async def _unload_all_adapters_from_endpoints(self) -> int:
        """
        Unload all currently loaded LoRA adapters from all inference endpoints.

        This handles edge cases like Ctrl+C interruptions that leave stale adapters.

        Returns:
            Total number of adapters successfully unloaded across all endpoints
        """
        if not self.inference_endpoints:
            return 0

        total_unloaded = 0
        for endpoint in self.inference_endpoints:
            adapters = await self._get_loaded_adapters_from_endpoint(endpoint)
            if adapters:
                endpoint_url = f"http://{endpoint.host}:{endpoint.port}"
                logger.info(f"Found {len(adapters)} loaded adapter(s) on {endpoint_url}: {adapters}")
                for adapter_name in adapters:
                    await self._unload_lora_on_inference_endpoints(adapter_name)
                    total_unloaded += 1

        # Clear all tracking lists since we unloaded everything
        if total_unloaded > 0:
            self.loaded_sampling_loras.clear()

        return total_unloaded

    async def _unload_adapters_for_model(self, model_id: str) -> int:
        """
        Unload all adapters belonging to a specific model_id.

        This is used for session cleanup when a training session ends or times out.
        It unloads ALL adapters for the model_id, not LRU eviction.

        Args:
            model_id: The model_id whose adapters should be unloaded

        Returns:
            Number of adapters unloaded
        """
        if model_id not in self.loaded_sampling_loras:
            return 0

        unloaded = 0
        for lora_name, lora_path in self.loaded_sampling_loras[model_id]:
            await self._unload_lora_on_inference_endpoints(lora_name)
            unloaded += 1

        # Clear the list for this model_id
        self.loaded_sampling_loras[model_id] = []

        logger.info(f"Unloaded {unloaded} adapter(s) for model_id={model_id}")
        return unloaded

    def _track_adapter(
        self,
        lora_name: str,
        lora_path: str,
        model_id: str = "default",
        *,
        add_if_missing: bool = True,
    ) -> bool:
        """
        Track a loaded adapter in the LRU list.

        If the adapter is already tracked, moves it to MRU position.
        If not tracked, optionally adds it to the tracking list.

        Args:
            lora_name: Name of the LoRA adapter
            lora_path: Path to the adapter files
            model_id: The model/session ID for per-session tracking (default: "default")
            add_if_missing: When False, only touches existing entries and does not create
                a new tracking entry.

        Returns:
            True if adapter was already tracked, False otherwise
        """

        # Initialize tracking list if not exists
        if model_id not in self.loaded_sampling_loras:
            self.loaded_sampling_loras[model_id] = []

        adapters = self.loaded_sampling_loras[model_id]

        # Check if already in list
        for existing_name, existing_path in adapters:
            if existing_name == lora_name:
                # Move to end (most recently used)
                adapters.remove((existing_name, existing_path))
                adapters.append((lora_name, lora_path))
                logger.info(f"LoRA adapter '{lora_name}' already tracked, moved to MRU")
                return True

        if not add_if_missing:
            return False

        # Add new adapter to tracking
        adapters.append((lora_name, lora_path))
        logger.info(f"LoRA adapter '{lora_name}' added to tracking list (count={len(adapters)})")
        return False

    async def create_sampling_session(self, request: CreateSamplingSessionRequest) -> CreateSamplingSessionResponse:
        """
        Create a sampling session by loading a LoRA adapter on inference workers.

        This method:
        1. Validates the model_path format and existence
        2. Checks if adapter is already loaded (skips loading if so)
        3. If at max capacity, unloads the oldest adapter (LRU eviction)
        4. Loads the adapter on inference workers if not already loaded

        Args:
            request: Request with model_path

        Returns:
            Response with session info
        """
        model_path = request.model_path
        requested_model_id = validate_model_id(request.model_id)
        logger.info(f"Creating sampling session for model_path: {model_path}")

        # Resolve path and validate
        path_model_id, lora_name, absolute_path = self._resolve_model_path(model_path)
        model_id = path_model_id or requested_model_id
        logger.info(f"Sampling session will be tracked under model_id={model_id}")

        loaded_by_endpoint = await self._reconcile_tracked_adapters(model_id)
        loaded_on_all_endpoints = bool(loaded_by_endpoint) and all(
            lora_name in loaded_names for loaded_names in loaded_by_endpoint
        )

        # Touch existing tracking entry if present, but do not create a new entry until we
        # know the adapter is actually loaded or the load succeeds.
        already_tracked = self._track_adapter(
            lora_name,
            absolute_path,
            model_id=model_id,
            add_if_missing=False,
        )

        if loaded_on_all_endpoints:
            if not already_tracked:
                self._track_adapter(lora_name, absolute_path, model_id=model_id)
            logger.info(f"LoRA adapter '{lora_name}' already loaded, skipping duplicate load")
        else:
            adapters = self.loaded_sampling_loras.get(model_id, [])
            if not already_tracked and len(adapters) >= self.max_adapters_per_model:
                oldest_name, _oldest_path = adapters[0]
                logger.info(
                    f"Max LoRA adapters exceeded ({self.max_adapters_per_model}), unloading oldest: {oldest_name}"
                )
                await self._unload_lora_on_inference_endpoints(oldest_name)
                adapters.pop(0)

            await self._load_lora_on_inference_endpoints(lora_name, absolute_path)
            if not already_tracked:
                self._track_adapter(lora_name, absolute_path, model_id=model_id)

        total_adapters = len(self.loaded_sampling_loras.get(model_id, []))
        logger.info(f"Sampling session created: lora_name={lora_name}, total_adapters={total_adapters}")

        return CreateSamplingSessionResponse(
            success=True,
            model_path=model_path,
            lora_name=lora_name,
            message=f"LoRA adapter '{lora_name}' loaded successfully",
        )
