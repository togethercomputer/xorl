"""RL utilities: advantage estimation for value-model (critic) training."""

from xorl.rl.advantages import compute_skip_observation_gae, explained_variance


__all__ = ["compute_skip_observation_gae", "explained_variance"]
