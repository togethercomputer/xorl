from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


@dataclass(frozen=True)
class TeacherFiles:
    heads: dict[str, str]
    hidden_caches: dict[str, str]


def save_tensor_file(path: str | Path, key: str, tensor: torch.Tensor) -> str:
    path = Path(path)
    save_file({key: tensor.detach().cpu().contiguous()}, str(path))
    return str(path)


def make_teacher_files(
    tmp_path: Path,
    teacher_heads: Mapping[str, torch.Tensor],
    teacher_hidden_caches: Mapping[str, torch.Tensor],
) -> TeacherFiles:
    head_paths: dict[str, str] = {}
    cache_paths: dict[str, str] = {}
    for teacher_id in teacher_heads:
        head_paths[teacher_id] = save_tensor_file(
            tmp_path / f"teacher_{teacher_id}_head.safetensors",
            "lm_head.weight",
            teacher_heads[teacher_id],
        )
        cache_paths[teacher_id] = save_tensor_file(
            tmp_path / f"teacher_{teacher_id}_hidden.safetensors",
            "hidden_states",
            teacher_hidden_caches[teacher_id],
        )
    return TeacherFiles(heads=head_paths, hidden_caches=cache_paths)


def save_teacher_hidden_cache(hiddens: list[torch.Tensor], path: str | Path) -> list[list[int]]:
    """Concatenate per-sample hidden states and return per-sample cache indices."""
    cache_indices: list[list[int]] = []
    offset = 0
    for hidden in hiddens:
        cache_indices.append(list(range(offset, offset + hidden.shape[0])))
        offset += hidden.shape[0]
    save_tensor_file(path, "hidden_states", torch.cat(hiddens, dim=0))
    return cache_indices


def reference_opd_loss(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    teacher_weights: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
    normalization_denominator: torch.Tensor | int | float | None = None,
) -> torch.Tensor:
    h = hidden_states.reshape(-1, hidden_states.size(-1))
    th = teacher_hidden_states.reshape(-1, teacher_hidden_states.size(-1))
    labels_flat = labels.reshape(-1)
    valid = labels_flat != ignore_index

    if not valid.any():
        return h.float().sum() * 0.0 + weight.float().sum() * 0.0

    student_logits = h[valid] @ weight.t()
    teacher_logits = th[valid] @ teacher_weight.t()
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    token_kl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    if teacher_weights is not None:
        token_kl = token_kl * teacher_weights.reshape(-1)[valid].to(token_kl.device)

    if normalization_denominator is None:
        denom = valid.sum().to(device=token_kl.device, dtype=torch.float32).clamp(min=1.0)
    elif torch.is_tensor(normalization_denominator):
        denom = normalization_denominator.to(device=token_kl.device, dtype=torch.float32).clamp(min=1.0)
    else:
        denom = torch.tensor(float(normalization_denominator), device=token_kl.device, dtype=torch.float32).clamp(
            min=1.0
        )
    return token_kl.sum() / denom


def reference_grouped_opd_loss(
    batch: Mapping[str, object],
    student_hidden_table: torch.Tensor,
    student_head: torch.Tensor,
    teacher_hidden_caches: Mapping[str, torch.Tensor],
    teacher_heads: Mapping[str, torch.Tensor],
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
    labels = torch.tensor(batch["labels"], dtype=torch.long)
    teacher_ids = torch.tensor(batch["teacher_ids"], dtype=torch.long)
    cache_indices = torch.tensor(batch["teacher_cache_indices"], dtype=torch.long)
    teacher_weights = torch.tensor(batch["teacher_weights"], dtype=torch.float32)

    hidden_states = student_hidden_table[input_ids].reshape(-1, student_hidden_table.shape[-1])
    labels_flat = labels.reshape(-1)
    teacher_ids_flat = teacher_ids.reshape(-1)
    cache_indices_flat = cache_indices.reshape(-1)
    weights_flat = teacher_weights.reshape(-1)

    valid = labels_flat != ignore_index
    token_losses = []
    for idx in valid.nonzero(as_tuple=True)[0].tolist():
        teacher_id = str(int(teacher_ids_flat[idx].item()))
        student_logits = hidden_states[idx] @ student_head.t()
        teacher_hidden = teacher_hidden_caches[teacher_id][cache_indices_flat[idx]]
        teacher_logits = teacher_hidden @ teacher_heads[teacher_id].t()
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
        token_kl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum()
        token_losses.append(token_kl * weights_flat[idx])
    return torch.stack(token_losses).sum() / valid.sum().clamp(min=1)
