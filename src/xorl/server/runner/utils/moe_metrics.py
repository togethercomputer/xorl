"""
MoE expert metrics tracking for Qwen3 MoE models.

Extracted from ModelRunner to keep the main runner focused on
forward/backward/optimizer operations.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import torch

try:
    from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import (
        enable_expert_metrics,
        get_expert_metrics,
    )
    _HAS_MOE_METRICS = True
except ImportError:
    _HAS_MOE_METRICS = False


logger = logging.getLogger(__name__)


class MoeMetricsTracker:
    """
    Tracks MoE expert load metrics (imbalance ratios, per-layer stats, memory).

    Usage:
        tracker = MoeMetricsTracker(model_config_obj, train_config, rank)
        # After each forward-backward:
        if tracker.enabled:
            metrics = tracker.collect(step, forward_backward_time)
            tracker.write(metrics)
    """

    def __init__(self, model_config_obj, train_config: Dict[str, Any], rank: int):
        self.rank = rank
        self.enabled = False
        self._metrics_file: Optional[str] = None

        # Check if MoE metrics collection is enabled in config
        if not train_config.get("enable_moe_metrics", True):
            return

        # Check if this is an MoE model
        num_experts = getattr(model_config_obj, "num_experts", 0)
        if num_experts <= 0:
            return

        # Setup metrics file path (only rank 0 writes)
        output_dir = train_config.get("output_dir", "outputs")
        self._metrics_file = os.path.join(output_dir, "moe_metrics.jsonl")

        # Create output directory on rank 0
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"MoE metrics will be written to: {self._metrics_file}")

        # Enable expert metrics collection for Qwen3 MoE
        if _HAS_MOE_METRICS:
            enable_expert_metrics(True)
            self.enabled = True
            logger.info(f"MoE expert metrics collection enabled (num_experts={num_experts})")
        else:
            logger.warning("Could not enable Qwen3 MoE expert metrics - module not available")

    def collect(self, step: int, forward_backward_time: float) -> dict:
        """
        Collect MoE expert metrics and memory stats after forward pass.

        Args:
            step: Current training step
            forward_backward_time: Time taken for forward_backward pass

        Returns:
            Dictionary of collected metrics
        """
        metrics: Dict[str, Any] = {
            "step": step,
            "forward_backward_time": forward_backward_time,
            "timestamp": time.time(),
        }

        # Collect expert load metrics
        if self.enabled and _HAS_MOE_METRICS:
            expert_metrics = get_expert_metrics()
            if expert_metrics:
                # Aggregate metrics across all layers
                all_imbalance_ratios = []
                all_max_loads = []
                all_min_loads = []
                total_tokens = 0
                for layer_key, layer_metrics in expert_metrics.items():
                    all_imbalance_ratios.append(layer_metrics["imbalance_ratio"])
                    all_max_loads.append(layer_metrics["max_load"])
                    all_min_loads.append(layer_metrics["min_load"])
                    total_tokens += layer_metrics["total_tokens"]

                metrics["expert_load"] = {
                    "num_moe_layers": len(expert_metrics),
                    "total_tokens": total_tokens,
                    "mean_imbalance_ratio": sum(all_imbalance_ratios) / len(all_imbalance_ratios) if all_imbalance_ratios else 0,
                    "max_imbalance_ratio": max(all_imbalance_ratios) if all_imbalance_ratios else 0,
                    "mean_max_load": sum(all_max_loads) / len(all_max_loads) if all_max_loads else 0,
                    "mean_min_load": sum(all_min_loads) / len(all_min_loads) if all_min_loads else 0,
                    # Include per-layer details for the first and last MoE layers (for debugging)
                    "per_layer": expert_metrics,
                }

        # Collect memory stats
        if torch.cuda.is_available():
            metrics["memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            # Get more detailed stats if available
            try:
                stats = torch.cuda.memory_stats()
                metrics["memory"]["num_alloc_retries"] = stats.get("num_alloc_retries", 0)
                metrics["memory"]["num_ooms"] = stats.get("num_ooms", 0)
            except Exception:
                pass

        return metrics

    def write(self, metrics: dict) -> None:
        """
        Write MoE metrics to JSONL file (only rank 0).

        Args:
            metrics: Dictionary of metrics to write
        """
        if self.rank != 0 or self._metrics_file is None:
            return

        try:
            with open(self._metrics_file, "a") as f:
                f.write(json.dumps(metrics, separators=(",", ":")) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write MoE metrics: {e}")
