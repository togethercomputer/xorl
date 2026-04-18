# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported from torchforge tests for GRPOLoss.

import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss import drgrpo_loss_function


@pytest.fixture
def inputs():
    """Test inputs for DRGRPO loss.

    Note: xorl's drgrpo_loss_function takes hidden_states and weight instead of logits.
    We create hidden_states and weight such that hidden_states @ weight.T produces
    logits with similar scale to torch.randn(B, S, V).
    """
    torch.manual_seed(42)
    B, S, V = 2, 4, 10
    H = 16  # Hidden dimension

    # Create hidden_states and weight matrix
    # Scale by 1/sqrt(H) so that logits = hidden_states @ weight.T has variance ~1
    # (similar to torch.randn logits in torchforge tests)
    hidden_states = torch.randn(B, S, H) / (H**0.5)
    weight = torch.randn(V, H)

    # Compute effective logits for reference (should have variance ~1)
    logits = hidden_states @ weight.T

    # Target IDs (called 'labels' in xorl API)
    labels = torch.randint(0, V, (B, S))

    # Seq 0: mild divergence, Seq 1: high divergence (triggers clipping)
    old_logprobs = torch.tensor(
        [
            [-2.0, -2.1, -1.9, -2.0],
            [-6.0, -1.0, -5.0, -0.5],
        ]
    )
    ref_logprobs = torch.randn(B, S) * 0.5 - 2.0
    advantages = torch.randn(B, S)

    # Interleaved mask: use ignore_index to mark non-loss positions
    # loss_mask in torchforge: [[1, 0, 1, 0], [1, 1, 0, 0]]
    # For xorl, we use ignore_index=-100 in labels for masked positions
    ignore_index = -100
    labels_with_mask = labels.clone()
    mask_pattern = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    labels_with_mask[~mask_pattern] = ignore_index

    return {
        "B": B,
        "S": S,
        "V": V,
        "H": H,
        "hidden_states": hidden_states,
        "weight": weight,
        "logits": logits,
        "labels": labels,
        "labels_with_mask": labels_with_mask,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
        "ignore_index": ignore_index,
        "mask_pattern": mask_pattern,
    }


class TestDRGRPOLoss:
    """Tests for drgrpo_loss_function.

    Note: test_forward and test_backward contain regression tests with exact expected
    values. If the implementation changes intentionally, update the expected values by
    running the tests with pytest -v and recording the actual values:

        pytest tests/ops/loss/test_drgrpo_loss.py -v -k "test_forward or test_backward"

    Then update the assert_close(...) calls with the new values.
    """

    def test_forward(self, inputs):
        """Forward pass produces expected loss value (regression test)."""
        d = inputs
        hidden_states = d["hidden_states"].clone().requires_grad_(True)

        output = drgrpo_loss_function(
            hidden_states=hidden_states,
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ref_logprobs=d["ref_logprobs"],
            ignore_index=d["ignore_index"],
            clip_low=0.2,
            clip_high=0.2,
            beta=0.1,
        )

        assert output.loss.isfinite()
        assert output.loss.shape == ()
        # Regression test: expected value computed with seed=42 fixture inputs
        assert_close(output.loss, torch.tensor(0.363678))

    def test_backward(self, inputs):
        """Backward pass produces expected gradient norm (regression test)."""
        d = inputs
        hidden_states = d["hidden_states"].clone().requires_grad_(True)

        output = drgrpo_loss_function(
            hidden_states=hidden_states,
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ref_logprobs=d["ref_logprobs"],
            ignore_index=d["ignore_index"],
            clip_low=0.2,
            clip_high=0.2,
            beta=0.1,
        )

        output.loss.backward()
        assert hidden_states.grad is not None
        assert hidden_states.grad.isfinite().all()
        # Regression test: expected value computed with seed=42 fixture inputs
        assert_close(hidden_states.grad.norm(), torch.tensor(1.514308))

    def test_zero_advantages(self, inputs):
        """Zero advantages produce finite (near-zero) loss."""
        d = inputs
        advantages = torch.zeros_like(d["advantages"])

        output = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=advantages,
            ignore_index=d["ignore_index"],
            beta=0.0,
        )

        assert output.loss.isfinite()
        # With zero advantages, policy gradient loss should be zero
        assert output.loss.abs() < 1e-5

    def test_all_ignored_labels(self, inputs):
        """Loss should be finite (zero) when all labels are ignored (no trainable tokens)."""
        d = inputs
        # Set all labels to ignore_index
        all_ignored = torch.full_like(d["labels"], d["ignore_index"])

        output = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=all_ignored,
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ignore_index=d["ignore_index"],
            beta=0.0,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0

    def test_empty_sequence(self):
        """Loss should be zero when sequence length is 0."""
        B, V, H = 2, 10, 16
        hidden_states = torch.empty(B, 0, H)
        weight = torch.randn(V, H)
        labels = torch.empty(B, 0, dtype=torch.long)
        advantages = torch.empty(B, 0)
        old_logprobs = torch.empty(B, 0)

        output = drgrpo_loss_function(
            hidden_states=hidden_states,
            weight=weight,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=advantages,
            beta=0.0,
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0

    def test_requires_ref_logprobs_when_beta_positive(self, inputs):
        """ValueError raised when beta > 0 but ref_logprobs is None."""
        d = inputs

        with pytest.raises(ValueError, match="ref_logprobs required"):
            drgrpo_loss_function(
                hidden_states=d["hidden_states"],
                weight=d["weight"],
                labels=d["labels_with_mask"],
                old_logprobs=d["old_logprobs"],
                advantages=d["advantages"],
                ref_logprobs=None,
                ignore_index=d["ignore_index"],
                beta=0.1,
            )

    def test_positive_advantages_encourage_high_prob(self, inputs):
        """With positive advantages, higher target probability yields lower loss."""
        d = inputs
        B, S, V, H = d["B"], d["S"], d["V"], d["H"]

        # Create two scenarios with same structure but different logit magnitudes
        torch.manual_seed(123)
        labels = torch.randint(0, V, (B, S))
        old_logprobs = torch.zeros(B, S)
        positive_advantages = torch.ones(B, S) * 2.0

        # High probability scenario: hidden states that produce high logits for targets
        hidden_high = torch.randn(B, S, H)
        weight_high = torch.randn(V, H)
        # Bias the logits toward target tokens
        for b in range(B):
            for s in range(S):
                weight_high[labels[b, s]] += hidden_high[b, s] * 5.0

        # Low probability scenario
        hidden_low = torch.randn(B, S, H)
        weight_low = torch.randn(V, H)
        # Bias away from target tokens
        for b in range(B):
            for s in range(S):
                weight_low[labels[b, s]] -= hidden_low[b, s] * 5.0

        loss_high = drgrpo_loss_function(
            hidden_states=hidden_high,
            weight=weight_high,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=positive_advantages,
            beta=0.0,
        )

        loss_low = drgrpo_loss_function(
            hidden_states=hidden_low,
            weight=weight_low,
            labels=labels,
            old_logprobs=old_logprobs,
            advantages=positive_advantages,
            beta=0.0,
        )

        # Higher probability should yield lower (more negative) loss
        assert loss_high.loss < loss_low.loss

    def test_kl_penalty_affects_loss(self, inputs):
        """KL penalty modifies loss when beta > 0."""
        d = inputs

        loss_no_kl = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ignore_index=d["ignore_index"],
            beta=0.0,
        )

        loss_with_kl = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ref_logprobs=d["ref_logprobs"],
            ignore_index=d["ignore_index"],
            beta=0.1,
        )

        assert loss_no_kl.loss != loss_with_kl.loss
        assert loss_with_kl.loss.isfinite()
        # KL metrics should be present when beta > 0
        assert "loss/kl_ref/mean" in loss_with_kl.metrics

    def test_metrics_present(self, inputs):
        """Output includes expected metrics."""
        d = inputs

        output = drgrpo_loss_function(
            hidden_states=d["hidden_states"],
            weight=d["weight"],
            labels=d["labels_with_mask"],
            old_logprobs=d["old_logprobs"],
            advantages=d["advantages"],
            ref_logprobs=d["ref_logprobs"],
            ignore_index=d["ignore_index"],
            clip_low=0.2,
            clip_high=0.2,
            beta=0.1,
        )

        expected_keys = [
            "loss/ratio/mean",
            "loss/kl_policy/mean",
            "loss/clip/clipped_ratio/mean",
            "loss/clip/high_fraction",
            "loss/clip/low_fraction",
            "loss/kl_ref/mean",
            "loss/aggregate/active_fraction",
        ]

        for key in expected_keys:
            assert key in output.metrics, f"Missing metric: {key}"

    def test_aggregation_types(self, inputs):
        """Different aggregation types produce different losses."""
        d = inputs

        losses = {}
        for agg_type in ["token_mean", "fixed_horizon", "sequence_mean"]:
            output = drgrpo_loss_function(
                hidden_states=d["hidden_states"],
                weight=d["weight"],
                labels=d["labels_with_mask"],
                old_logprobs=d["old_logprobs"],
                advantages=d["advantages"],
                ignore_index=d["ignore_index"],
                agg_type=agg_type,
                beta=0.0,
            )
            losses[agg_type] = output.loss.item()
            assert output.loss.isfinite()

        # At least some aggregation types should produce different values
        unique_losses = set(round(v, 6) for v in losses.values())
        assert len(unique_losses) >= 2, "Aggregation types should produce different losses"
