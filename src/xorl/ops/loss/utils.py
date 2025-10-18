"""
Utility functions for loss computation.

This module contains shared helper functions used by various loss functions.
"""

import torch
import torch.nn.functional as F


def _compute_eager_jsd(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence loss between student and teacher distributions.

    JSD_beta(P, Q) = beta * KL(P || M) + (1 - beta) * KL(Q || M)
    where M = beta * P + (1 - beta) * Q

    Args:
        student_hidden_states: Flattened student hidden states, shape (batch * seq_len, hidden_dim)
        student_weight: Student LM head weight, shape (vocab_size, hidden_dim)
        teacher_hidden_states: Flattened teacher hidden states, shape (batch * seq_len, hidden_dim)
        teacher_weight: Teacher LM head weight, shape (vocab_size, hidden_dim)
        labels: Flattened target labels, shape (batch * seq_len,)
        beta: Balance parameter. 0.0 for forward KL, 1.0 for reverse KL, 0.5 for symmetric JSD
        temperature: Temperature for softmax (default: 1.0)
        ignore_index: Index to ignore in loss computation (default: -100)

    Returns:
        Scalar JSD loss value
    """
    # Compute logits
    student_logits = (student_hidden_states @ student_weight.t()).float() / temperature
    teacher_logits = (teacher_hidden_states @ teacher_weight.t()).float() / temperature

    # Compute log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Compute probabilities for mixture
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()

    # Mixture distribution: M = beta * P_student + (1 - beta) * P_teacher
    mixture_probs = beta * student_probs + (1 - beta) * teacher_probs
    mixture_log_probs = mixture_probs.log()

    # KL(student || mixture) = sum(student_probs * (student_log_probs - mixture_log_probs))
    kl_student_mixture = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)

    # KL(teacher || mixture) = sum(teacher_probs * (teacher_log_probs - mixture_log_probs))
    kl_teacher_mixture = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)

    # JSD = beta * KL(student || M) + (1 - beta) * KL(teacher || M)
    jsd = beta * kl_student_mixture + (1 - beta) * kl_teacher_mixture

    # Mask ignored tokens
    valid_mask = labels != ignore_index
    jsd = jsd.masked_fill(~valid_mask, 0.0)

    # Return mean loss over valid tokens
    n_valid = valid_mask.sum().clamp(min=1)
    return jsd.sum() / n_valid
