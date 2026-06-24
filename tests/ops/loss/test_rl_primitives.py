import pytest
import torch

from xorl.rl import (
    compute_gspo_kl,
    compute_kl_estimate,
    compute_opsm_mask,
    compute_policy_clip_loss,
    compute_sequence_kl,
    reduce_token_or_sample_mean,
)


pytestmark = [pytest.mark.cpu]


def _slime_compute_approx_kl(log_probs, log_probs_base, kl_loss_type, importance_ratio=None):
    """Reference formula from Slime's slime/utils/ppo_utils.py."""
    log_ratio = log_probs.float() - log_probs_base.float()
    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")
    if importance_ratio is not None:
        kl = importance_ratio * kl
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)
    return kl


@pytest.mark.parametrize("kind", ["k1", "k2", "k3", "low_var_kl"])
def test_compute_kl_estimate_matches_slime_reference(kind):
    policy = torch.tensor([[-2.0, -1.0, 12.0], [-3.0, -4.5, -25.0]])
    base = torch.tensor([[-2.5, -0.25, -3.0], [-4.0, -4.0, 3.0]])
    importance_ratio = torch.tensor([[1.0, 0.5, 2.0], [1.5, 1.0, 0.25]])

    expected = _slime_compute_approx_kl(policy, base, kind, importance_ratio=importance_ratio)

    torch.testing.assert_close(compute_kl_estimate(policy, base, kind, importance_ratio), expected)


def test_compute_sequence_and_gspo_kl_match_slime_reference():
    current = torch.tensor([[-2.0, -1.0, -3.0], [-4.0, -2.5, -1.0]])
    old = torch.tensor([[-1.5, -1.25, -2.0], [-3.0, -3.5, -1.0]])
    masks = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)

    expected_seq = torch.tensor([(0.5 - 0.25) / 2.0, (1.0 - 1.0 + 0.0) / 3.0])
    expected_gspo = expected_seq.unsqueeze(-1).expand_as(current)

    torch.testing.assert_close(compute_sequence_kl(current, old, masks), expected_seq)
    torch.testing.assert_close(compute_gspo_kl(current, old, masks), expected_gspo)


def test_compute_policy_clip_loss_matches_slime_reference_with_dual_clip():
    ppo_kl = torch.tensor([-0.3, 0.4, -0.1, 0.2])
    advantages = torch.tensor([1.0, 1.0, -2.0, -0.5])
    eps_clip = 0.2
    eps_clip_high = 0.25
    eps_clip_c = 3.0

    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -eps_clip_c * advantages
    expected_loss = torch.where(advantages < 0, torch.minimum(pg_losses3, clip_pg_losses1), clip_pg_losses1)
    expected_clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    actual_loss, actual_clipfrac, actual_ratio = compute_policy_clip_loss(
        ppo_kl,
        advantages,
        eps_clip,
        eps_clip_high,
        eps_clip_c,
    )

    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_clipfrac, expected_clipfrac)
    torch.testing.assert_close(actual_ratio, ratio)


def test_compute_opsm_mask_matches_slime_reference():
    current = torch.tensor([[-2.0, -2.5, -2.0], [-1.0, -1.0, -1.0]])
    old = torch.tensor([[-1.0, -1.0, -2.0], [-1.2, -1.4, -1.0]])
    advantages = torch.tensor([[-0.5, 0.2, -1.0], [-0.1, -0.2, 0.3]])
    masks = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)

    opsm_mask, clipfrac = compute_opsm_mask(current, old, advantages, masks, delta=0.3)

    expected_mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    expected_clipfrac = torch.tensor(1.0 / 2.0)

    torch.testing.assert_close(opsm_mask, expected_mask)
    torch.testing.assert_close(clipfrac, expected_clipfrac)


def test_reduce_token_or_sample_mean_modes_are_explicit():
    values = torch.tensor([[1.0, 3.0, 100.0], [2.0, 8.0, 10.0], [7.0, 11.0, 13.0]])
    masks = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.float32)

    torch.testing.assert_close(reduce_token_or_sample_mean(values, masks, "token_sum"), torch.tensor(24.0))
    torch.testing.assert_close(reduce_token_or_sample_mean(values, masks, "token_mean"), torch.tensor(24.0 / 5.0))
    torch.testing.assert_close(
        reduce_token_or_sample_mean(values, masks, "slime_sum_of_sample_mean"),
        torch.tensor(2.0 + 20.0 / 3.0),
    )
    torch.testing.assert_close(
        reduce_token_or_sample_mean(values, masks, "sample_mean"),
        torch.tensor((2.0 + 20.0 / 3.0) / 2.0),
    )
