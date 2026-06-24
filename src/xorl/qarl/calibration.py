"""QARL calibration data loading and warmup helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from xorl.qarl.fake_quant import summarize_qarl_model


def _load_calibration_payload(path: str | Path) -> Any:
    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    if calibration_path.suffix.lower() in {".pt", ".pth"}:
        return torch.load(calibration_path, map_location="cpu", weights_only=True)
    if calibration_path.suffix.lower() in {".json", ".jsonl"}:
        text = calibration_path.read_text(encoding="utf-8")
        if calibration_path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)
    raise ValueError("QARL calibration data must be a .json, .jsonl, .pt, or .pth file")


def _slice_sequence(tensor: torch.Tensor, sequence_length: int | None) -> torch.Tensor:
    if sequence_length is None:
        return tensor
    if tensor.ndim == 0:
        raise ValueError("QARL calibration tensors must include a sequence dimension")
    return tensor[..., :sequence_length]


def _to_long_tensor(value: Any, *, sequence_length: int | None) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"QARL calibration input_ids must be 1D or 2D token ids, got shape={tuple(tensor.shape)}")
    return _slice_sequence(tensor.to(dtype=torch.long), sequence_length)


def _coerce_batch(raw_batch: Any, *, sequence_length: int | None) -> dict[str, torch.Tensor]:
    if isinstance(raw_batch, Mapping):
        if "input_ids" not in raw_batch:
            raise ValueError("QARL calibration mapping batches must include 'input_ids'")
        batch: dict[str, torch.Tensor] = {"input_ids": _to_long_tensor(raw_batch["input_ids"], sequence_length=sequence_length)}
        for key in ("attention_mask", "position_ids"):
            if key in raw_batch and raw_batch[key] is not None:
                value = raw_batch[key].detach().cpu() if isinstance(raw_batch[key], torch.Tensor) else torch.as_tensor(raw_batch[key])
                if value.ndim == 1:
                    value = value.unsqueeze(0)
                batch[key] = _slice_sequence(value, sequence_length)
        return batch
    return {"input_ids": _to_long_tensor(raw_batch, sequence_length=sequence_length)}


def load_qarl_calibration_batches(
    calibration_data: str | Path,
    *,
    calibration_size: int = 0,
    sequence_length: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load token-id calibration batches for dynamic QARL metadata warmup."""

    if calibration_size < 0:
        raise ValueError("qarl_calib_size must be non-negative")
    if sequence_length is not None and sequence_length <= 0:
        raise ValueError("qarl_quant_sequence_length must be positive when set")

    payload = _load_calibration_payload(calibration_data)
    if isinstance(payload, Mapping):
        if "batches" in payload:
            raw_batches = payload["batches"]
        elif "samples" in payload:
            raw_batches = payload["samples"]
        elif "input_ids" in payload:
            raw_batches = payload["input_ids"]
        else:
            raise ValueError("QARL calibration JSON mapping must include input_ids, samples, or batches")
    else:
        raw_batches = payload

    if isinstance(raw_batches, torch.Tensor):
        raw_sequence: Sequence[Any] = [raw_batches]
    elif isinstance(raw_batches, Sequence) and not isinstance(raw_batches, (str, bytes)):
        raw_sequence = raw_batches
    else:
        raise ValueError("QARL calibration data must be a tensor, list, or mapping")

    limit = calibration_size or len(raw_sequence)
    batches = [_coerce_batch(raw_batch, sequence_length=sequence_length) for raw_batch in list(raw_sequence)[:limit]]
    if not batches:
        raise ValueError("QARL calibration data produced no batches")
    return batches


def calibrate_qarl_model(
    model: nn.Module,
    calibration_data: str | Path,
    *,
    calibration_size: int = 0,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Run token-id calibration batches through a QARL model and return summary metadata."""

    batches = load_qarl_calibration_batches(
        calibration_data,
        calibration_size=calibration_size,
        sequence_length=sequence_length,
    )
    try:
        first_param = next(model.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            model(**{key: value.to(device=device) for key, value in batch.items()})
    model.train(was_training)

    summary = summarize_qarl_model(model)
    summary["calibration_batches"] = len(batches)
    summary["calibration_samples"] = sum(int(batch["input_ids"].shape[0]) for batch in batches)
    return summary
