"""
Validation and self-test utilities for ModelRunner.

Standalone functions extracted from ModelRunner so the runner can stay
focused on forward/backward/optimizer operations.
"""

import logging
import traceback
from typing import Any, Dict, List

import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.distributed.parallel_state import get_parallel_state


logger = logging.getLogger(__name__)


def validate_token_ids(micro_batches: List[Dict[str, Any]], vocab_size: int) -> None:
    """
    Validate that all token IDs and labels are within vocab range.

    This prevents CUDA device-side asserts when invalid token IDs are passed
    to the embedding layer or loss function.

    Args:
        micro_batches: List of dictionaries containing input tensors
        vocab_size: Model vocabulary size

    Raises:
        ValueError: If any token ID or label is out of vocab range
    """
    logger.debug(f"Validating {len(micro_batches)} micro-batches against vocab_size={vocab_size}")

    for batch_idx, micro_batch in enumerate(micro_batches):
        logger.debug(f"Validating batch {batch_idx}, keys: {list(micro_batch.keys())}")
        # Check input_ids
        input_ids = micro_batch.get("input_ids")
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                max_id = input_ids.max().item()
                min_id = input_ids.min().item()
            elif isinstance(input_ids, list):
                flat_ids = []
                for item in input_ids:
                    if isinstance(item, list):
                        flat_ids.extend(item)
                    else:
                        flat_ids.append(item)
                if flat_ids:
                    max_id = max(flat_ids)
                    min_id = min(flat_ids)
                else:
                    max_id = 0
                    min_id = 0
            else:
                max_id = 0
                min_id = 0

            if max_id >= vocab_size:
                error_msg = (
                    f"input_ids contain token ID {max_id} which is >= vocab_size {vocab_size}. "
                    f"This will cause a CUDA device-side assert. "
                    f"Please ensure all token IDs are in range [0, {vocab_size - 1}]."
                )
                logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                raise ValueError(error_msg)
            if min_id < 0:
                error_msg = f"input_ids contain negative token ID {min_id}. Please ensure all token IDs are >= 0."
                logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                raise ValueError(error_msg)

        # Check labels (excluding IGNORE_INDEX which is -100)
        labels = micro_batch.get("labels")
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                valid_labels = labels[labels != IGNORE_INDEX]
                if valid_labels.numel() > 0:
                    max_label = valid_labels.max().item()
                    min_label = valid_labels.min().item()
                    if max_label >= vocab_size:
                        error_msg = (
                            f"labels contain token ID {max_label} which is >= vocab_size {vocab_size}. "
                            f"This will cause a CUDA device-side assert. "
                            f"Please ensure all label IDs are in range [0, {vocab_size - 1}] or {IGNORE_INDEX}."
                        )
                        logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                        raise ValueError(error_msg)
                    if min_label < 0:
                        error_msg = (
                            f"labels contain invalid negative token ID {min_label}. "
                            f"This will cause a CUDA device-side assert. "
                            f"Only {IGNORE_INDEX} is allowed as a negative label value."
                        )
                        logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                        raise ValueError(error_msg)
            elif isinstance(labels, list):
                flat_labels = []
                for item in labels:
                    if isinstance(item, list):
                        flat_labels.extend(item)
                    else:
                        flat_labels.append(item)
                valid_labels = [l for l in flat_labels if l != IGNORE_INDEX]
                if valid_labels:
                    max_label = max(valid_labels)
                    min_label = min(valid_labels)
                    if max_label >= vocab_size:
                        error_msg = (
                            f"labels contain token ID {max_label} which is >= vocab_size {vocab_size}. "
                            f"This will cause a CUDA device-side assert. "
                            f"Please ensure all label IDs are in range [0, {vocab_size - 1}] or {IGNORE_INDEX}."
                        )
                        logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                        raise ValueError(error_msg)
                    if min_label < 0:
                        error_msg = (
                            f"labels contain invalid negative token ID {min_label}. "
                            f"This will cause a CUDA device-side assert. "
                            f"Only {IGNORE_INDEX} is allowed as a negative label value."
                        )
                        logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                        raise ValueError(error_msg)

        # Check logprobs if provided (used in importance sampling / RL)
        logprobs = micro_batch.get("logprobs")
        if logprobs is not None:
            if isinstance(logprobs, torch.Tensor):
                max_logprob = logprobs.max().item()
                if max_logprob > 0:
                    error_msg = (
                        f"logprobs contain positive value {max_logprob}. "
                        f"Log probabilities should be <= 0. "
                        f"Please check your logprobs computation."
                    )
                    logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                    raise ValueError(error_msg)
            elif isinstance(logprobs, list):
                flat_logprobs = []
                for item in logprobs:
                    if isinstance(item, list):
                        flat_logprobs.extend(item)
                    else:
                        flat_logprobs.append(item)
                if flat_logprobs:
                    max_logprob = max(flat_logprobs)
                    if max_logprob > 0:
                        error_msg = (
                            f"logprobs contain positive value {max_logprob}. "
                            f"Log probabilities should be <= 0. "
                            f"Please check your logprobs computation."
                        )
                        logger.error(f"Validation failed in batch {batch_idx}: {error_msg}")
                        raise ValueError(error_msg)


def run_self_test(runner) -> None:
    """
    Run a quick self-test to verify model and optimizer are functional.

    Performs a dummy forward-backward pass and optimizer step.

    Args:
        runner: ModelRunner instance (needs .model, .rank, .forward_backward,
                .optim_step, .optimizer attributes)
    """
    if runner.rank == 0:
        logger.info("=" * 60)
        logger.info("Running ModelRunner self-test...")
        logger.info("=" * 60)

    try:
        # Create a minimal dummy batch for testing
        test_seq_len = 128000
        test_batch_size = 1
        vocab_size = runner.model.config.vocab_size

        dummy_input_ids = torch.randint(
            1,
            min(vocab_size, 1000),
            (test_batch_size, test_seq_len),
            dtype=torch.long,
        )
        dummy_labels = dummy_input_ids.clone()
        dummy_position_ids = torch.arange(test_seq_len).unsqueeze(0).expand(test_batch_size, -1)

        # For Ulysses: shard input_ids and labels
        if get_parallel_state().cp_enabled:
            cp_size = get_parallel_state().cp_size
            cp_rank = get_parallel_state().cp_rank

            cp_chunk_size = (test_seq_len + cp_size - 1) // cp_size
            pad_len = cp_chunk_size * cp_size - test_seq_len

            if pad_len > 0:
                pad_ids = torch.zeros(test_batch_size, pad_len, dtype=torch.long)
                dummy_input_ids = torch.cat([dummy_input_ids, pad_ids], dim=-1)

                pad_labels = torch.full((test_batch_size, pad_len), IGNORE_INDEX, dtype=torch.long)
                dummy_labels = torch.cat([dummy_labels, pad_labels], dim=-1)

                pad_positions = torch.arange(test_seq_len, test_seq_len + pad_len).unsqueeze(0)
                dummy_position_ids = torch.cat([dummy_position_ids, pad_positions], dim=-1)

            start_idx = cp_rank * cp_chunk_size
            end_idx = start_idx + cp_chunk_size
            dummy_input_ids = dummy_input_ids[:, start_idx:end_idx]
            dummy_labels = dummy_labels[:, start_idx:end_idx]

            logger.info(
                f"  Ulysses self-test: rank {cp_rank}/{cp_size}, "
                f"input_shape={dummy_input_ids.shape}, position_ids_shape={dummy_position_ids.shape}"
            )

        test_micro_batches = [
            {
                "input_ids": dummy_input_ids,
                "labels": dummy_labels,
                "position_ids": dummy_position_ids,
            }
        ]

        # Test 1: Forward-backward pass
        if runner.rank == 0:
            logger.info("  [1/2] Testing forward_backward pass...")

        fb_result = runner.forward_backward(test_micro_batches, loss_fn="cross_entropy")

        if runner.rank == 0:
            logger.info("     Forward-backward completed")
            logger.info(f"       Loss: {fb_result['total_loss']:.6f}")
            logger.info(f"       Time: {fb_result['forward_backward_time']:.3f}s")

        # Test 2: Optimizer step (lr=0 so no actual parameter updates)
        if runner.rank == 0:
            logger.info("  [2/2] Testing optimizer step...")

        opt_result = runner.optim_step(gradient_clip=1.0, lr=0.0)

        if runner.rank == 0:
            logger.info("     Optimizer step completed")
            logger.info(f"       Grad norm: {opt_result['grad_norm']:.6f}")
            logger.info(f"       Time: {opt_result['optim_step_time']:.3f}s")

        # Reset optimizer state to avoid polluting real training
        runner.optimizer.state.clear()
        runner.global_step = 0

        if runner.rank == 0:
            logger.info("     Optimizer state reset")
            logger.info("=" * 60)
            logger.info("Self-test PASSED - ModelRunner is functional!")
            logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Self-test FAILED on rank {runner.rank}")
        logger.error(f"  Error: {e}")
        logger.error("=" * 60)
        logger.error(traceback.format_exc())
        raise RuntimeError(f"ModelRunner self-test failed: {e}")
