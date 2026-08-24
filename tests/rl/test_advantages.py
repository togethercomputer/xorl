import math
import random

import pytest

from xorl.rl import compute_skip_observation_gae, explained_variance


pytestmark = pytest.mark.cpu


def reference_gae(rewards, values, gamma, lam, bootstrap):
    """Textbook GAE over a dense action-only sequence."""
    n = len(rewards)
    advantages = [0.0] * n
    next_adv = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else bootstrap
        delta = rewards[t] + gamma * next_value - values[t]
        next_adv = delta + gamma * lam * next_adv
        advantages[t] = next_adv
    return advantages


def test_matches_standard_gae_without_mask():
    rewards = [0.0, 0.0, 0.5, 0.0, 1.0]
    values = [0.2, -0.1, 0.4, 0.3, 0.1]
    gamma, lam = 0.99, 0.95
    adv, ret = compute_skip_observation_gae(rewards, values, gamma=gamma, lam=lam)
    expected = reference_gae(rewards, values, gamma, lam, 0.0)
    for a, e in zip(adv, expected):
        assert math.isclose(a, e, rel_tol=1e-12, abs_tol=1e-12)
    for r, a, v in zip(ret, adv, values):
        assert math.isclose(r, a + v, rel_tol=1e-12, abs_tol=1e-12)


def test_skips_observation_tokens():
    # Layout: [action, obs, obs, action, obs, action]
    rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    values = [0.5, 9.9, -9.9, 0.4, 9.9, 0.2]
    action_mask = [1, 0, 0, 1, 0, 1]
    gamma, lam = 1.0, 1.0
    adv, ret = compute_skip_observation_gae(rewards, values, action_mask, gamma=gamma, lam=lam)

    # Equivalent to dense GAE over the action-token subsequence only.
    sub_adv = reference_gae([0.0, 0.0, 1.0], [0.5, 0.4, 0.2], gamma, lam, 0.0)
    action_indices = [0, 3, 5]
    for k, t in enumerate(action_indices):
        assert math.isclose(adv[t], sub_adv[k], rel_tol=1e-12)
        assert math.isclose(ret[t], sub_adv[k] + values[t], rel_tol=1e-12)

    # Observation tokens carry exactly 0.0 (the packer's mask convention) and
    # their (garbage) values never influence any action's advantage.
    for t in (1, 2, 4):
        assert adv[t] == 0.0
        assert ret[t] == 0.0


def test_observation_values_do_not_leak():
    rewards = [0.0, 0.0, 1.0]
    values = [0.3, 123.0, 0.1]
    base_adv, _ = compute_skip_observation_gae(rewards, [0.3, 0.0, 0.1], [1, 0, 1])
    leak_adv, _ = compute_skip_observation_gae(rewards, values, [1, 0, 1])
    assert base_adv == leak_adv


def test_bootstrap_value_for_truncated_trajectory():
    rewards = [0.0, 0.0]
    values = [0.1, 0.2]
    adv, ret = compute_skip_observation_gae(rewards, values, gamma=1.0, lam=1.0, bootstrap_value=0.7)
    # Last action bootstraps from 0.7: delta = 0 + 0.7 - 0.2 = 0.5
    assert math.isclose(adv[1], 0.5, rel_tol=1e-12)
    assert math.isclose(adv[0], (0.2 - 0.1) + 0.5, rel_tol=1e-12)
    for r, a, v in zip(ret, adv, values):
        assert math.isclose(r, a + v, rel_tol=1e-12)


def test_perfect_critic_terminal_reward_yields_zero_advantage():
    # gamma=1, lam=1, terminal reward 1.0; V(s_t)=1.0 at every action token is
    # exactly correct, so all advantages vanish and returns equal 1.0.
    rewards = [0.0, 0.0, 1.0]
    values = [1.0, 1.0, 1.0]
    adv, ret = compute_skip_observation_gae(rewards, values, gamma=1.0, lam=1.0)
    assert all(math.isclose(a, 0.0, abs_tol=1e-12) for a in adv)
    assert all(math.isclose(r, 1.0, rel_tol=1e-12) for r in ret)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_skip_observation_gae([0.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="action_mask"):
        compute_skip_observation_gae([0.0, 0.0], [0.0, 0.0], [1])


def test_explained_variance_matches_direct_computation():
    rng = random.Random(3)
    returns = [rng.gauss(0.5, 1.3) for _ in range(500)]
    values = [r + rng.gauss(0.0, 0.4) for r in returns]  # decent critic
    n = len(returns)
    mean_r = sum(returns) / n
    var_r = sum((r - mean_r) ** 2 for r in returns) / n
    mse = sum((v - r) ** 2 for v, r in zip(values, returns)) / n
    direct = 1.0 - mse / var_r

    ev = explained_variance(
        value_error_sq_mean=mse,
        return_mean=mean_r,
        return_sq_mean=sum(r * r for r in returns) / n,
    )
    assert math.isclose(ev, direct, rel_tol=1e-9)
    assert 0.5 < ev < 1.0


def test_explained_variance_perfect_and_mean_critic():
    returns = [0.0, 1.0, 2.0, 3.0]
    mean_r = 1.5
    sq_mean = sum(r * r for r in returns) / 4
    # Perfect critic: zero error -> EV = 1.
    assert explained_variance(0.0, mean_r, sq_mean) == pytest.approx(1.0)
    # Mean-predicting critic: error variance == return variance -> EV = 0.
    var_r = sq_mean - mean_r**2
    assert explained_variance(var_r, mean_r, sq_mean) == pytest.approx(0.0)


def test_explained_variance_undefined_for_constant_returns():
    assert math.isnan(explained_variance(0.1, 1.0, 1.0))
